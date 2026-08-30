"""Export a run as one self-contained interactive 3-D brain viewer.

``videocortex-spark export`` packs the fsaverage5 inflated mesh, the whole
prediction matrix (uint8-quantised), the colour LUT, and the Destrieux region
table into a single ``brain.html``. That file is the deliverable you email:
no server, no JavaScript dependencies, no network — the WebGL renderer,
orbit controls, timeline and vertex picker are all in the page.

Two house rules carry over from ``render.py``:

* Colour limits are computed **once over the whole run** (``compute_limits``),
  never per frame. The viewer gets ``vmax``/``threshold``/``ramp_frac`` and
  reproduces the soft-threshold alpha ramp exactly; a quiet second stays
  quieter than a loud one.
* Everything degrades gracefully: if the Destrieux atlas cannot be fetched
  (offline), the viewer simply ships without region names.
"""

from __future__ import annotations

import base64
import json
import logging
import typing as tp
from pathlib import Path

import numpy as np

from videocortex_spark.config import FSAVERAGE5_VERTICES, FSAVERAGE5_VERTICES_PER_HEMI
from videocortex_spark.render import MeshMismatch, compute_limits, energy_curve

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).resolve().parent / "export_template.html"
#: Placeholder inside the template's ``application/json`` data island.
DATA_TOKEN = "__VCX_DATA__"
#: Written into the payload so future viewers can refuse what they can't read.
FORMAT_ID = "videocortex-spark-brain@1"


class ExportOutput(tp.NamedTuple):
    path: Path
    n_tr: int
    n_vertices: int
    n_bytes: int
    regions: bool


def _b64(arr: np.ndarray) -> str:
    """Little-endian raw bytes of ``arr`` as base64 text."""
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _vertex_normals(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals (the weighted cross product trick).

    fsaverage faces are wound consistently, so these point outward; the
    shader still lights two-sided so a flipped input degrades to flat
    shading rather than black.
    """
    c = np.asarray(coords, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.cross(c[f[:, 1]] - c[f[:, 0]], c[f[:, 2]] - c[f[:, 0]])
    out = np.zeros_like(c)
    for corner in range(3):
        np.add.at(out, f[:, corner], n)
    norm = np.linalg.norm(out, axis=1)
    norm[norm == 0] = 1.0
    return (out / norm[:, None]).astype(np.float32)


def _normalise_sulc(sulc: np.ndarray) -> np.ndarray:
    """Sulcal map → uint8 in [0, 255], robust 1st–99th percentile stretch."""
    s = np.asarray(sulc, dtype=np.float32)
    lo, hi = (float(v) for v in np.percentile(s, (1.0, 99.0)))
    if hi <= lo:
        return np.zeros(s.shape, dtype=np.uint8)
    return (np.clip((s - lo) / (hi - lo), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def quantize_activations(preds: np.ndarray, vmax: float) -> np.ndarray:
    """(n_tr, n_vertices) float → uint8, 128 = 0, one unit = vmax / 127.

    Quantisation error is at most vmax/254 per vertex — two orders of
    magnitude below the soft-threshold ramp, so the viewer and the PNG
    plates tell the same story.
    """
    q = np.clip(np.asarray(preds, dtype=np.float32) / float(vmax), -1.0, 1.0)
    return (np.round(q * 127.0) + 128.0).astype(np.uint8)


def dequantize_activation(q: np.ndarray, vmax: float) -> np.ndarray:
    """Inverse of :func:`quantize_activations` (used by tests)."""
    return (np.asarray(q, dtype=np.float32) - 128.0) * (float(vmax) / 127.0)


def colormap_lut(cmap: str, n: int = 256) -> list[list[int]]:
    """Sample the base colormap into an ``n``-entry RGB LUT (0-255 ints).

    The soft-threshold alpha ramp is *not* baked in here — the viewer
    recomputes it from ``threshold``/``ramp_frac`` so the shipped numbers
    stay honest and inspectable.
    """
    xs = np.linspace(0.0, 1.0, n)
    if cmap == "cold_hot":
        from nilearn.plotting.cm import cold_hot as base
    else:
        import matplotlib.pyplot as plt

        base = plt.get_cmap(cmap)
    rgba = np.asarray(base(xs), dtype=np.float64)
    return (np.clip(rgba[:, :3], 0.0, 1.0) * 255.0 + 0.5).astype(int).tolist()


def _load_mesh() -> dict[str, np.ndarray]:
    """Inflated fsaverage5 geometry, per hemisphere. Bundled — no network."""
    from nilearn import surface

    from videocortex_spark.render import load_fsaverage5

    fs = load_fsaverage5()
    out: dict[str, np.ndarray] = {}
    for hemi in ("left", "right"):
        mesh = surface.load_surf_mesh(fs[hemi]["mesh"])
        coords = np.asarray(mesh.coordinates, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.max() >= 2**16:
            raise MeshMismatch(f"{hemi} mesh has too many vertices for uint16 faces")
        out[f"coords_{hemi[0]}"] = coords
        out[f"norms_{hemi[0]}"] = _vertex_normals(coords, faces)
        out[f"faces_{hemi[0]}"] = faces.astype(np.uint16)
        out[f"sulc_{hemi[0]}"] = _normalise_sulc(
            np.asarray(surface.load_surf_data(fs[hemi]["bg"]), dtype=np.float32)
        )
    return out


def build_payload(
    preds: np.ndarray,
    *,
    timestamps: tp.Sequence[float] | None = None,
    title: str = "brain",
    cmap: str = "cold_hot",
    percentile: float = 99.0,
    threshold_frac: float = 0.25,
    ramp_frac: float = 0.5,
    regions: bool = True,
    progress: tp.Callable[[str], None] | None = None,
) -> dict[str, tp.Any]:
    """Assemble the JSON payload the viewer consumes.

    Binary arrays travel as base64 strings under ``"b64"``; everything else
    is plain JSON. Little-endian throughout (every browser TypedArray is).
    """
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = preds[None, :]
    if preds.ndim != 2 or preds.shape[1] != FSAVERAGE5_VERTICES:
        raise MeshMismatch(
            f"expected a 2-D (timesteps x {FSAVERAGE5_VERTICES}) array, "
            f"got {preds.shape}. Is this really a TRIBE v2 cortical prediction?"
        )
    note = progress or (lambda msg: None)

    note("colour limits")
    vmax, threshold = compute_limits(preds, percentile, threshold_frac)

    note("fsaverage5 mesh")
    mesh = _load_mesh()

    note("quantising activations")
    acts = quantize_activations(preds, vmax)
    energy = energy_curve(preds)

    n_tr = int(preds.shape[0])
    if timestamps is None:
        timestamps = [round(i * 1.49, 3) for i in range(n_tr)]
    ts = [float(t) for t in timestamps][:n_tr]
    if len(ts) < n_tr:  # short timestamps file — pad at TR cadence
        step = ts[1] - ts[0] if len(ts) > 1 else 1.49
        ts += [ts[-1] + step * k for k in range(1, n_tr - len(ts) + 1)] if ts else []
    tr_s = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.49

    region_names: list[str] = []
    region_ids = np.zeros(FSAVERAGE5_VERTICES, dtype=np.uint16)
    top_regions: list[str] = []
    if regions:
        try:
            from videocortex_spark.regions import load_region_labels, top_regions_per_tr

            note("Destrieux atlas")
            rid, names = load_region_labels()
            region_names = [
                "unlabelled" if i == 0 else f"{'L' if h == 'left' else 'R'} {n}"
                for i, (h, n) in enumerate(names)
            ]
            region_ids = rid.astype(np.uint16)
            note("top regions per TR")
            top_regions = top_regions_per_tr(preds, rid, names, k=3)
        except Exception as exc:  # atlas fetch can fail offline — not fatal
            logger.warning("region labels skipped: %s", exc)

    b64: dict[str, str] = {key: _b64(val) for key, val in mesh.items()}
    b64["acts"] = _b64(acts)
    b64["region_ids"] = _b64(region_ids) if region_names else ""

    return {
        "format": FORMAT_ID,
        "title": str(title),
        "n_tr": n_tr,
        "n_vertices": FSAVERAGE5_VERTICES,
        "hemi_split": FSAVERAGE5_VERTICES_PER_HEMI,
        "tr_s": tr_s,
        "timestamps": ts,
        "vmax": vmax,
        "threshold": threshold,
        "ramp_frac": float(ramp_frac),
        "percentile": float(percentile),
        "cmap": cmap,
        "energy": [round(float(e), 6) for e in energy],
        "top_regions": top_regions,
        "region_names": region_names,
        "lut": colormap_lut(cmap),
        "b64": b64,
    }


def render_html(payload: dict[str, tp.Any]) -> str:
    """Inject the payload into the template. One substitution, no templating."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if DATA_TOKEN not in template:
        raise RuntimeError(f"{TEMPLATE_PATH.name} lost its {DATA_TOKEN} placeholder")
    raw = json.dumps(payload, separators=(",", ":"))
    # The data island is XML-text: an unescaped "</" could forge a </script>
    # out of a run title and break out of the island.
    raw = raw.replace("</", "<\\/")
    return template.replace(DATA_TOKEN, raw, 1)


def export_viewer(
    preds: np.ndarray,
    out_path: Path,
    *,
    timestamps: tp.Sequence[float] | None = None,
    title: str = "brain",
    cmap: str = "cold_hot",
    percentile: float = 99.0,
    threshold_frac: float = 0.25,
    ramp_frac: float = 0.5,
    regions: bool = True,
    progress: tp.Callable[[str], None] | None = None,
) -> ExportOutput:
    """Write ``out_path``: one HTML file, whole run inside, zero dependencies."""
    payload = build_payload(
        preds,
        timestamps=timestamps,
        title=title,
        cmap=cmap,
        percentile=percentile,
        threshold_frac=threshold_frac,
        ramp_frac=ramp_frac,
        regions=regions,
        progress=progress,
    )
    if progress:
        progress("writing html")
    html = render_html(payload)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    n_bytes = out_path.stat().st_size
    logger.info("3-D viewer -> %s (%.1f MB)", out_path, n_bytes / 1e6)
    return ExportOutput(
        path=out_path,
        n_tr=int(payload["n_tr"]),
        n_vertices=int(payload["n_vertices"]),
        n_bytes=n_bytes,
        regions=bool(payload["region_names"]),
    )
