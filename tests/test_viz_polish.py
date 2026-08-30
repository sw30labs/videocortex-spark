"""Region naming, temporal blending, soft ramp, ribbon — the new-viz maths.

The geometry-touching paths are covered by the slow render tests; these are
the units that must never regress and cost milliseconds.
"""

import numpy as np
import pytest

from videocortex_spark.config import FSAVERAGE5_VERTICES


# -- Destrieux region names --------------------------------------------------


def test_region_map_is_unique_and_complete():
    from videocortex_spark.regions import load_region_labels

    rid, names = load_region_labels()
    assert rid.shape == (FSAVERAGE5_VERTICES,)
    assert len(names) > 100  # Destrieux is ~75 regions per hemi
    assert rid.min() >= 0 and rid.max() < len(names)
    # both hemispheres must be representable
    hemis = {h for h, _ in names}
    assert hemis == {"left", "right"}


def test_top_regions_pick_plausible_names():
    from videocortex_spark.regions import load_region_labels, top_regions_per_tr

    rid, names = load_region_labels()
    preds = np.zeros((2, FSAVERAGE5_VERTICES), dtype=np.float32)
    # light up one left-hemi region strongly
    target = int(np.unique(rid[(rid > 0) & (rid < 100)])[0])
    preds[0, rid == target] = 1.0
    top = top_regions_per_tr(preds, rid, names, k=1)
    assert top[0].startswith("L ")
    assert "unlabelled" not in top[0]
    # a silent TR names nothing
    assert top[1] == ""


def test_top_regions_rejects_wrong_vertex_count():
    from videocortex_spark.regions import load_region_labels, top_regions_per_tr

    rid, names = load_region_labels()
    with pytest.raises(ValueError, match="20484"):
        top_regions_per_tr(np.zeros((2, 10)), rid, names)


# -- temporal interpolation --------------------------------------------------


def test_palette_index_brackets_and_clamps():
    from videocortex_spark.spin import palette_index_frac

    ts = np.array([0.0, 1.0, 2.0])
    assert palette_index_frac(-0.1, ts) == (-1, -1, 0.0)
    assert palette_index_frac(0.0, ts) == (0, 1, 0.0)
    i0, i1, f = palette_index_frac(0.5, ts)
    assert (i0, i1) == (0, 1) and abs(f - 0.5) < 1e-9
    i0, i1, f = palette_index_frac(2.5, ts)  # past the end: hold last
    assert (i0, i1, f) == (2, 2, 0.0)
    i0, i1, f = palette_index_frac(6.25, ts, lag_s=5.0)  # first show at t=5
    assert (i0, i1) == (1, 2) and abs(f - 0.25) < 1e-9


def test_blend_palettes_is_a_lerp():
    from videocortex_spark.spin import blend_palettes

    pals = np.stack([np.zeros((4, 3), np.float32), np.ones((4, 3), np.float32)])
    assert np.array_equal(blend_palettes(pals, 0, 1, 0.0), pals[0])
    assert np.array_equal(blend_palettes(pals, 0, 1, 1.0), pals[1])
    mid = blend_palettes(pals, 0, 1, 0.25)
    assert np.allclose(mid, 0.25)


# -- soft threshold ramp -----------------------------------------------------


def test_soft_ramp_fades_where_hard_cut_pops():
    from videocortex_spark.render import soft_threshold_cmap

    cmap = soft_threshold_cmap("cold_hot", vmax=1.0, threshold=0.4, ramp_frac=0.5)
    assert cmap((0.15 + 1) / 2)[3] == pytest.approx(0.0)   # below the band
    assert 0.0 < cmap((0.30 + 1) / 2)[3] < 1.0             # mid-band is translucent
    assert cmap((0.40 + 1) / 2)[3] == pytest.approx(1.0)   # at threshold: opaque
    assert cmap((0.90 + 1) / 2)[3] == pytest.approx(1.0)   # above: opaque


def test_ramp_frac_zero_is_the_hard_cut():
    from videocortex_spark.render import soft_threshold_cmap

    cmap = soft_threshold_cmap("cold_hot", vmax=1.0, threshold=0.4, ramp_frac=0.0)
    assert cmap((0.39 + 1) / 2)[3] == pytest.approx(0.0)
    assert cmap((0.41 + 1) / 2)[3] == pytest.approx(1.0)


def test_vertex_palette_ramp_is_monotone_in_alpha_mix():
    from videocortex_spark.spin import vertex_palette

    sulc = np.zeros(3)
    pbg = vertex_palette(np.zeros(3), sulc, cmap="cold_hot", vmax=1.0, threshold=0.4, ramp_frac=0.5)
    pq = vertex_palette(np.full(3, 0.20), sulc, cmap="cold_hot", vmax=1.0, threshold=0.4, ramp_frac=0.5)
    pm = vertex_palette(np.full(3, 0.30), sulc, cmap="cold_hot", vmax=1.0, threshold=0.4, ramp_frac=0.5)
    pl = vertex_palette(np.full(3, 0.75), sulc, cmap="cold_hot", vmax=1.0, threshold=0.4, ramp_frac=0.5)
    # (not 1.0: cold_hot's ends are white, indistinguishable from the grey floor)
    d = lambda a, b: float(np.abs(a - b).mean())
    sat = lambda a: float((a.max(axis=-1) - a.min(axis=-1)).mean())
    assert d(pq, pbg) == pytest.approx(0.0, abs=1e-6)      # below floor = grey
    assert sat(pm) > sat(pbg)                              # band starts moving
    assert sat(pm) < sat(pl)                               # monotone toward loud
    # (saturation, not distance-from-floor: cold_hot's pale near-threshold
    # colours are *closer* to the white floor than its saturated mid-range)


# -- energy ribbon -----------------------------------------------------------


def test_energy_curve_is_mean_abs_per_tr():
    from videocortex_spark.render import energy_curve

    preds = np.array([[-1.0, 1.0], [0.0, 0.0], [2.0, -2.0]])
    assert np.allclose(energy_curve(preds), [1.0, 0.0, 2.0])


def test_ribbon_blits_and_playshead_moves():
    from videocortex_spark.render import blit_ribbon, draw_energy_ribbon, energy_curve

    rng = np.random.default_rng(0)
    ribbon = draw_energy_ribbon(
        energy_curve(rng.normal(0, 1, (40, 8))), width_px=200, height_px=40
    )
    assert ribbon.shape[-1] == 4 and ribbon.dtype == np.uint8
    frame = np.zeros((160, 160, 4), dtype=np.uint8)
    frame[..., 3] = 255
    a = blit_ribbon(frame, ribbon, playhead=0.1)
    b = blit_ribbon(frame, ribbon, playhead=0.9)
    assert a.shape == frame.shape
    assert not np.array_equal(a, b), "playhead position must change the frame"
    assert a[..., 1].sum() > 0, "the green curve must actually blit"
    # tiny frames bail out untouched
    assert blit_ribbon(np.zeros((32, 32, 4), np.uint8), ribbon, playhead=0.5).shape == (32, 32, 4)


def test_feather_softens_the_silhouette():
    from videocortex_spark.spin import feather_alpha

    img = np.zeros((40, 40, 4), dtype=np.uint8)
    img[10:30, 10:30, 3] = 255
    out = feather_alpha(img, sigma=1.2)
    # hard edge had exactly one step; feathered must have partial alpha
    edge_row = out[20, 6:14, 3]
    assert ((edge_row > 10) & (edge_row < 245)).any()
    # and colour untouched
    assert np.array_equal(out[..., :3], img[..., :3])
