"""3D spinning-brain PIP: bake a pose atlas once, recolor per TR.

The 2×2 still card is a paper plate. This is one inflated fsaverage5, both
hemispheres, yawing in a dark void. Nilearn ``plot_surf_stat_map`` per video
frame would be hours; instead we screenshot a face-ID buffer at K azimuths
(matplotlib, antialiased off — IDs survive), then every output tick is a
numpy palette lookup.

No VTK. ``render.py`` stays nilearn-only.
"""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import numpy as np

from videocortex_spark.config import FSAVERAGE5_VERTICES, OverlayConfig
from videocortex_spark.overlay import OverlayError, plate_index_at
from videocortex_spark.render import compute_limits, format_pip_label, load_fsaverage5

logger = logging.getLogger(__name__)

_AMBIENT = 0.38
#: Extra midline gap in mesh units so the medial wall can flash through.
#: Inflated fsaverage already has a fissure; only widen if it is tighter than this.
_GAP = 3.0
#: matplotlib azim: eye in the x-y plane. -90 is posterior, left hemi on the left.
_AZIM0 = -90.0


class SpinAtlas(tp.NamedTuple):
    face_id: np.ndarray  # uint16 (K, H, W), 0 = background
    shade: np.ndarray  # float32 (K, n_faces)
    faces: np.ndarray  # int32 (n_faces, 3)
    sulc: np.ndarray  # float32 (n_verts,)
    azims: np.ndarray  # float32 (K,)
    elev: float
    size: int


def clamp_dps(dps: float) -> float:
    return float(min(48.0, max(12.0, dps)))


def n_poses(az_step: int) -> int:
    step = max(1, int(az_step))
    return int(round(360 / step))


def pose_index(t: float, *, dps: float, az_step: int) -> int:
    k0, _, _ = pose_blend(t, dps=dps, az_step=az_step)
    return k0


def pose_blend(t: float, *, dps: float, az_step: int) -> tuple[int, int, float]:
    """(k0, k1, frac) — azimuth as a blend of two atlas poses."""
    k_n = n_poses(az_step)
    az = (clamp_dps(dps) * float(t)) % 360.0
    x = az / (360.0 / k_n)
    k0 = int(np.floor(x)) % k_n
    k1 = (k0 + 1) % k_n
    frac = float(x - np.floor(x))
    return k0, k1, frac


def load_joined_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(coords, faces, sulc)`` for both inflated hemispheres."""
    from nilearn import surface

    fs = load_fsaverage5()
    lc, lf = surface.load_surf_mesh(fs["left"]["mesh"])
    rc, rf = surface.load_surf_mesh(fs["right"]["mesh"])
    ls = surface.load_surf_data(fs["left"]["bg"]).astype(np.float32).reshape(-1)
    rs = surface.load_surf_data(fs["right"]["bg"]).astype(np.float32).reshape(-1)
    lc = np.asarray(lc, dtype=np.float32)
    rc = np.asarray(rc, dtype=np.float32)
    lf = np.asarray(lf, dtype=np.int32)
    rf = np.asarray(rf, dtype=np.int32)

    # Native infl already has a midline gap; widen it a little so the medial
    # wall is visible as the globe turns, without turning into two organs.
    current = float(rc[:, 0].min() - lc[:, 0].max())
    extra = (_GAP - current) / 2.0
    if extra > 0:
        lc = lc.copy()
        rc = rc.copy()
        lc[:, 0] -= extra
        rc[:, 0] += extra

    coords = np.vstack([lc, rc])
    faces = np.vstack([lf, rf + lc.shape[0]])
    sulc = np.concatenate([ls, rs])
    if coords.shape[0] != FSAVERAGE5_VERTICES:
        raise OverlayError(
            f"joined mesh has {coords.shape[0]} verts, expected {FSAVERAGE5_VERTICES}"
        )
    return coords, faces, sulc


def _normalize_sulc(sulc: np.ndarray) -> np.ndarray:
    s = np.asarray(sulc, dtype=np.float32)
    lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    if hi <= lo:
        return np.zeros_like(s)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0)


def vertex_palette(
    values: np.ndarray,
    sulc: np.ndarray,
    *,
    cmap: str,
    vmax: float,
    threshold: float,
    ramp_frac: float = 0.5,
) -> np.ndarray:
    """RGB per vertex, nilearn's mix: Greys(sulc) under cmap at 0.7 near peak.

    Under threshold the globe stays sulcal grey — colour hides, mesh does not.
    ``ramp_frac`` fades colour in linearly across
    ``[threshold·(1-ramp), threshold]`` instead of popping on at a hard edge;
    during a spin a hard edge strobes as vertices cross threshold per frame.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from nilearn.plotting.cm import mix_colormaps

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    bg = plt.get_cmap("Greys")(_normalize_sulc(sulc))
    fg = plt.get_cmap(cmap)(Normalize(vmin=-vmax, vmax=vmax)(values))
    a = np.abs(values)
    lo = abs(threshold) * (1.0 - max(0.0, min(1.0, ramp_frac)))
    hi = abs(threshold)
    if hi > 0 and ramp_frac > 0:
        fg[..., 3] = 0.7 * np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    else:
        fg[..., 3] = np.where(a < hi, 0.0, 0.7)
    mix = mix_colormaps(fg, bg)
    return np.nan_to_num(mix[:, :3], nan=0.0).astype(np.float32)


def _rot_azim_elev(azim_deg: float, elev_deg: float) -> np.ndarray:
    """Approximate matplotlib ``view_init`` rotation (world → view)."""
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    cz, sz = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], dtype=np.float32)
    return rx @ rz


def face_shade(
    coords: np.ndarray,
    faces: np.ndarray,
    azim: float,
    elev: float,
    *,
    ambient: float = _AMBIENT,
) -> np.ndarray:
    R = _rot_azim_elev(azim, elev)
    view = coords @ R.T
    tri = view[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
    # After this rotation, camera looks along -Y; light from the camera.
    lambert = np.clip(-n[:, 1], 0.0, 1.0)
    return (ambient + (1.0 - ambient) * lambert).astype(np.float32)


def _encode_face_colors(n_faces: int) -> np.ndarray:
    fid = np.arange(1, n_faces + 1, dtype=np.int32)
    col = np.zeros((n_faces, 4), dtype=np.float32)
    col[:, 0] = ((fid >> 16) & 255) / 255.0
    col[:, 1] = ((fid >> 8) & 255) / 255.0
    col[:, 2] = (fid & 255) / 255.0
    col[:, 3] = 1.0
    return col


def _decode_face_id(rgba: np.ndarray, n_faces: int) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.int32)
    fid = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    fid = np.where((fid > 0) & (fid <= n_faces), fid, 0).astype(np.uint16)
    return fid


def bake_atlas(
    *,
    elev: float,
    az_step: int,
    size: int,
    azim0: float = _AZIM0,
    progress=None,
) -> SpinAtlas:
    """Screenshot K face-ID views of the joined inflated mesh."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    coords, faces, sulc = load_joined_mesh()
    n_faces = int(faces.shape[0])
    colors = _encode_face_colors(n_faces)
    k_n = n_poses(az_step)
    spin_az = (np.arange(k_n, dtype=np.float32) * (360.0 / k_n))
    mpl_az = azim0 + spin_az
    px = int(size)
    face_id = np.zeros((k_n, px, px), dtype=np.uint16)
    shade = np.zeros((k_n, n_faces), dtype=np.float32)

    fig = plt.figure(figsize=(px / 96.0, px / 96.0), dpi=96)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    coll = Poly3DCollection(
        coords[faces],
        facecolors=colors,
        edgecolors="none",
        antialiaseds=False,
        linewidths=0,
        shade=False,
    )
    ax.add_collection3d(coll)
    c = coords.mean(axis=0)
    r = 1.08 * float(np.max(np.linalg.norm(coords - c, axis=1)))
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_proj_type("ortho")
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.85)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    except (AttributeError, NotImplementedError):
        pass
    fig.subplots_adjust(0, 0, 1, 1)

    logger.info("baking spin atlas: %d poses at %d px", k_n, px)
    for i, az in enumerate(mpl_az):
        ax.view_init(elev=float(elev), azim=float(az))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        if buf.shape[0] != px or buf.shape[1] != px:
            buf = _center_square(buf, px)
        face_id[i] = _decode_face_id(buf, n_faces)
        shade[i] = face_shade(coords, faces, float(az), float(elev))
        if progress:
            progress(i + 1, k_n)
    plt.close(fig)
    return SpinAtlas(
        face_id=face_id,
        shade=shade,
        faces=faces.astype(np.int32),
        sulc=sulc,
        azims=spin_az,
        elev=float(elev),
        size=px,
    )


def _center_square(buf: np.ndarray, px: int) -> np.ndarray:
    h, w = buf.shape[:2]
    if h == px and w == px:
        return buf
    canvas = np.zeros((px, px, buf.shape[2]), dtype=buf.dtype)
    y0 = max(0, (px - h) // 2)
    x0 = max(0, (px - w) // 2)
    ys = slice(0, min(h, px))
    xs = slice(0, min(w, px))
    canvas[y0 : y0 + ys.stop, x0 : x0 + xs.stop] = buf[ys, xs]
    return canvas


def palette_index_frac(t: float, ts: np.ndarray, *, lag_s: float = 0.0) -> tuple[int, int, float]:
    """(i0, i1, frac) — TR bracket straddling ``t`` for temporal blending.

    Adjacent TRs of a TRIBE run correlate at 0.94+; holding one palette for a
    whole second and then cutting reads as a slideshow with the odd jump.
    Linear interpolation between the bracketing palettes turns the same data
    into a continuous flow. Out-of-range times clamp to the first/last TR.
    """
    t_show = np.asarray(ts, dtype=np.float64) + float(lag_s)
    if t_show.size == 0:
        return -1, -1, 0.0
    if t < t_show[0]:
        return -1, -1, 0.0
    i = int(np.searchsorted(t_show, t, side="right") - 1)
    if i >= t_show.size - 1:
        return int(t_show.size - 1), int(t_show.size - 1), 0.0
    span = t_show[i + 1] - t_show[i]
    frac = 0.0 if span <= 0 else float(np.clip((t - t_show[i]) / span, 0.0, 1.0))
    return i, i + 1, frac


def blend_palettes(
    palettes: np.ndarray, i0: int, i1: int, frac: float
) -> np.ndarray:
    """Lerp two per-vertex palettes. ~250k floats — microseconds."""
    if i0 == i1 or frac <= 0.0:
        return palettes[i0]
    if frac >= 1.0:
        return palettes[i1]
    return (
        palettes[i0].astype(np.float32) * (1.0 - frac)
        + palettes[i1].astype(np.float32) * frac
    )


def feather_alpha(rgba: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian-blur the alpha channel only.

    The face-ID atlas bakes with antialiasing off (IDs must survive), so the
    globe's silhouette is stair-stepped. Blurring alpha by ~1 px smooths the
    edge against the plate without touching colour.
    """
    if sigma <= 0:
        return rgba
    from scipy.ndimage import gaussian_filter1d

    out = rgba.copy()
    a = out[..., 3].astype(np.float32) / 255.0
    a = gaussian_filter1d(a, sigma, axis=0, mode="nearest")
    a = gaussian_filter1d(a, sigma, axis=1, mode="nearest")
    out[..., 3] = np.clip(a * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


def composite_frame_smooth(
    atlas: SpinAtlas,
    palette: np.ndarray,
    k0: int,
    k1: int,
    frac: float,
) -> np.ndarray:
    a = composite_frame(atlas, palette, k0)
    if frac < 0.02 or k0 == k1:
        return a
    if frac > 0.98:
        return composite_frame(atlas, palette, k1)
    b = composite_frame(atlas, palette, k1)
    out = a.astype(np.float32) * (1.0 - frac) + b.astype(np.float32) * frac
    return np.clip(out, 0, 255).astype(np.uint8)


def composite_frame(
    atlas: SpinAtlas,
    palette: np.ndarray,
    pose: int,
) -> np.ndarray:
    """RGBA uint8, ``palette`` is (n_verts, 3) float in [0, 1]."""
    fid = atlas.face_id[pose]
    out = np.zeros((fid.shape[0], fid.shape[1], 4), dtype=np.uint8)
    valid = fid > 0
    if not np.any(valid):
        return out
    fidx = fid[valid].astype(np.int32) - 1
    tri = atlas.faces[fidx]
    rgb = (palette[tri[:, 0]] + palette[tri[:, 1]] + palette[tri[:, 2]]) / 3.0
    rgb *= atlas.shade[pose, fidx, None]
    rgb = np.clip(rgb, 0.0, 1.0)
    pix = out[valid]
    pix[:, :3] = (rgb * 255.0 + 0.5).astype(np.uint8)
    pix[:, 3] = 255
    out[valid] = pix
    return out


def _blit_label(rgba: np.ndarray, *lines: str) -> np.ndarray:
    """Lower-left text inside the plate; empty lines are skipped."""
    lines = tuple(ln for ln in lines if ln)
    if not lines:
        return rgba
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return rgba
    im = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=13)
        font_sm = ImageFont.load_default(size=11)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
        font_sm = font
    y = 10
    for i, ln in enumerate(lines):
        f = font if i == 0 else font_sm
        draw.text((10, y), ln, fill=(232, 232, 238, 210), font=f)
        y += 16 if i == 0 else 14
    return np.asarray(im)


def save_atlas(path: Path, atlas: SpinAtlas) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        face_id=atlas.face_id,
        shade=atlas.shade,
        faces=atlas.faces,
        sulc=atlas.sulc,
        azims=atlas.azims,
        elev=np.float32(atlas.elev),
        size=np.int32(atlas.size),
    )


def load_atlas(path: Path) -> SpinAtlas:
    z = np.load(path)
    return SpinAtlas(
        face_id=z["face_id"],
        shade=z["shade"],
        faces=z["faces"],
        sulc=z["sulc"],
        azims=z["azims"],
        elev=float(z["elev"]),
        size=int(z["size"]),
    )


def atlas_fingerprint(cfg: OverlayConfig) -> dict:
    return {
        "layout": "pip-spin-v1",
        "elev": float(cfg.elev),
        "az_step": int(cfg.az_step),
        "atlas_px": int(cfg.atlas_px),
        "gap": _GAP,
        "azim0": _AZIM0,
    }


def build_palettes(
    preds: np.ndarray,
    sulc: np.ndarray,
    cfg: OverlayConfig,
    *,
    vmax: float,
    threshold: float,
) -> np.ndarray:
    """(n_tr, n_verts, 3) float32."""
    n = preds.shape[0]
    pal = np.empty((n, preds.shape[1], 3), dtype=np.float32)
    for i in range(n):
        pal[i] = vertex_palette(
            preds[i], sulc, cmap=cfg.cmap, vmax=vmax, threshold=threshold,
            ramp_frac=cfg.ramp_frac,
        )
    return pal


def iter_spin_frames(
    *,
    atlas: SpinAtlas,
    palettes: np.ndarray,
    timestamps: np.ndarray,
    duration: float,
    cfg: OverlayConfig,
    lag_s: float,
    stride_idx: np.ndarray | None = None,
):
    """Yield ``(path_key, rgba_uint8, duration)`` coalesced by (tr, pose).

    ``path_key`` is ``('blank',)`` or ``(tr, pose)``.
    """
    fps = max(1.0, float(cfg.fps))
    dt = 1.0 / fps
    dps = clamp_dps(cfg.dps)
    az_step = max(1, int(cfg.az_step))
    ts = timestamps if stride_idx is None else timestamps[stride_idx]
    n_tr = palettes.shape[0]
    t = 0.0
    prev: tuple | None = None
    # We yield coalesced runs in a second pass; first collect keys.
    keys: list[tuple] = []
    while t < duration - 1e-9:
        i = plate_index_at(t, ts, lag_s=lag_s)
        if i is None or i < 0 or i >= n_tr:
            key: tuple = ("blank",)
        else:
            key = (int(i), pose_index(t, dps=dps, az_step=az_step))
        keys.append(key)
        t += dt
    if not keys:
        return
    run_key = keys[0]
    run_n = 1
    for key in keys[1:]:
        if key == run_key:
            run_n += 1
        else:
            yield run_key, run_n * dt
            run_key = key
            run_n = 1
    yield run_key, run_n * dt
