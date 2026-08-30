"""Export tests — the self-contained 3-D brain viewer (brain.html).

Like the renderer tests these use the real bundled fsaverage5 geometry:
mocking the mesh would test nothing worth testing. The Destrieux atlas is
*not* bundled (it downloads on first use), so only the ``slow`` test lets
``regions=True``; everything else proves the fast offline path.
"""

import base64
import json
import re
import shutil
import subprocess

import numpy as np
import pytest

from videocortex_spark.config import FSAVERAGE5_VERTICES
from videocortex_spark.export import (
    DATA_TOKEN,
    TEMPLATE_PATH,
    build_payload,
    colormap_lut,
    dequantize_activation,
    export_viewer,
    quantize_activations,
    render_html,
    _load_mesh,
    _vertex_normals,
)
from videocortex_spark.render import MeshMismatch, compute_limits


def _preds(n_tr: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(n_tr, FSAVERAGE5_VERTICES)).astype(np.float32)


def _island_json(html: str) -> dict:
    m = re.search(
        r'<script type="application/json" id="vcx-data">(.*?)</script>',
        html,
        re.S,
    )
    assert m, "data island missing"
    return json.loads(m.group(1))


# -- quantisation ------------------------------------------------------------


def test_quantize_roundtrip_stays_under_half_step():
    preds = _preds()
    # percentile 100: nothing clips, so only the half-step rounding remains
    vmax, _ = compute_limits(preds, percentile=100.0)
    q = quantize_activations(preds, vmax)
    assert q.dtype == np.uint8
    err = np.abs(dequantize_activation(q, vmax) - preds)
    assert err.max() <= vmax / 254 + 1e-6


def test_quantize_clips_beyond_vmax():
    preds = np.array([[0.0] * (FSAVERAGE5_VERTICES - 2) + [-5.0, 5.0]])
    q = quantize_activations(preds, vmax=1.0)
    assert q[0, -1] == 255 and q[0, -2] == 1  # ±127 shifted by 128
    assert q[0, 0] == 128                     # exact zero lands mid-scale


# -- colour ------------------------------------------------------------------


def test_lut_is_cold_hot_shaped():
    # nilearn's cold_hot: white tips, blue quarter, black centre, red 3/4.
    lut = colormap_lut("cold_hot")
    assert len(lut) == 256 and all(len(c) == 3 for c in lut)
    r_cold, _, b_cold = lut[64]
    r_hot, _, b_hot = lut[192]
    assert b_cold > r_cold      # negative side is blue
    assert r_hot > b_hot        # positive side is red
    assert sum(lut[128]) < 60   # diverges through near-black, not white


# -- geometry ----------------------------------------------------------------


def test_vertex_normals_are_unit_and_outward():
    mesh = _load_mesh()
    for hemi in ("l", "r"):
        n = mesh[f"norms_{hemi}"]
        c = mesh[f"coords_{hemi}"]
        assert n.shape == c.shape
        assert np.linalg.norm(n, axis=1) == pytest.approx(1.0, abs=1e-4)
        # outward: normals point away from the hemisphere's own centroid
        dots = np.einsum("ij,ij->i", n, c - c.mean(axis=0))
        assert (dots > 0).mean() > 0.95


def test_faces_fit_uint16_per_hemisphere():
    mesh = _load_mesh()
    for hemi in ("l", "r"):
        f = mesh[f"faces_{hemi}"]
        assert f.dtype == np.uint16
        assert int(f.max()) < 10242


def test_vertex_normal_helper_handles_degenerate_faces():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], float)
    faces = np.array([[0, 1, 2], [0, 3, 2]], int)  # second face is degenerate
    n = _vertex_normals(coords, faces)
    assert np.isfinite(n).all()


# -- payload -----------------------------------------------------------------


def test_payload_rejects_wrong_vertex_count():
    with pytest.raises(MeshMismatch, match="20484"):
        build_payload(np.zeros((4, 100)), regions=False)


def test_payload_carries_run_wide_limits():
    preds = _preds()
    payload = build_payload(preds, regions=False)
    vmax, threshold = compute_limits(preds)
    assert payload["vmax"] == pytest.approx(vmax)
    assert payload["threshold"] == pytest.approx(threshold)
    assert payload["n_tr"] == preds.shape[0]
    acts = base64.b64decode(payload["b64"]["acts"])
    assert len(acts) == preds.size  # one byte per (TR, vertex)


def test_payload_synthesises_and_pads_timestamps():
    payload = build_payload(_preds(4), regions=False)
    assert payload["timestamps"] == [0.0, 1.49, 2.98, 4.47]
    payload = build_payload(_preds(4), timestamps=[10.0, 11.0], regions=False)
    assert payload["timestamps"] == [10.0, 11.0, 12.0, 13.0]
    assert payload["tr_s"] == pytest.approx(1.0)


def test_payload_without_regions_ships_no_atlas_blob():
    payload = build_payload(_preds(), regions=False)
    assert payload["region_names"] == []
    assert payload["top_regions"] == []
    assert payload["b64"]["region_ids"] == ""


@pytest.mark.slow
def test_payload_with_regions_names_both_hemispheres():
    """Needs the Destrieux atlas (nilearn download); offline it may ship []."""
    payload = build_payload(_preds(3), regions=True)
    names = payload["region_names"]
    if not names:  # atlas unfetchable — the honest offline degradation
        return
    assert len(names) == 151  # 2 × 75 Destrieux + unlabelled
    assert names[0] == "unlabelled"
    assert any(n.startswith("L ") for n in names)
    assert any(n.startswith("R ") for n in names)
    assert len(payload["top_regions"]) == 3
    ids = np.frombuffer(base64.b64decode(payload["b64"]["region_ids"]), "<u2")
    assert ids.shape == (FSAVERAGE5_VERTICES,)
    assert ids.max() < len(names)


# -- html --------------------------------------------------------------------


def test_template_still_has_its_token():
    assert DATA_TOKEN in TEMPLATE_PATH.read_text(encoding="utf-8")


def test_render_html_fills_the_island_exactly_once():
    html = render_html(build_payload(_preds(2), title="demo", regions=False))
    assert DATA_TOKEN not in html
    data = _island_json(html)
    assert data["title"] == "demo"
    assert data["format"].startswith("videocortex-spark-brain@")


def test_run_title_cannot_break_out_of_the_data_island():
    evil = 'x</script><script>alert(1)</script>'
    html = render_html(build_payload(_preds(2), title=evil, regions=False))
    # exactly the island's and the viewer script's own closing tags survive
    assert html.count("</script>") == 2
    # and the payload itself round-trips with the hostile string intact
    assert _island_json(html)["title"] == evil


def test_exported_html_is_self_contained(tmp_path):
    out = export_viewer(_preds(2), tmp_path / "brain.html", regions=False)
    html = out.path.read_text(encoding="utf-8")
    assert not re.search(r'<script[^>]+src=', html)
    assert not re.search(r'<link[^>]+href=', html)
    assert not re.search(r'https?://', html)
    assert out.n_tr == 2 and out.n_vertices == FSAVERAGE5_VERTICES
    assert out.n_bytes == out.path.stat().st_size
    # the mesh dominates size; even a long run must stay single-digit MB-ish
    assert out.n_bytes < 3_000_000


def test_exported_viewer_js_is_valid_syntax(tmp_path):
    """If node is on PATH, the embedded viewer script must parse."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    out = export_viewer(_preds(2), tmp_path / "brain.html", regions=False)
    html = out.path.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    probe = tmp_path / "viewer.js"
    probe.write_text(scripts[0], encoding="utf-8")
    subprocess.run([node, "--check", str(probe)], check=True)
