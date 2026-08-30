"""Renderer tests. These use the real fsaverage5 geometry — it's 20 MB and
cached after the first run, and mocking it would test nothing worth testing.
"""

import numpy as np
import pytest

from videocortex_spark.config import FSAVERAGE5_VERTICES, RenderConfig
from videocortex_spark.render import (
    MeshMismatch,
    auto_columns,
    build_contact_sheet,
    compute_limits,
    render_series,
    select_frames,
    split_hemispheres,
)


def test_split_hemispheres_halves_the_vector():
    v = np.arange(FSAVERAGE5_VERTICES, dtype=float)
    h = split_hemispheres(v)
    assert h["left"].shape == h["right"].shape == (10242,)
    # left comes first, matching TRIBE's concatenation order
    assert h["left"][0] == 0 and h["right"][0] == 10242


def test_wrong_vertex_count_is_rejected_loudly():
    with pytest.raises(MeshMismatch, match="20484"):
        split_hemispheres(np.zeros(1000))
    with pytest.raises(MeshMismatch, match="1-D"):
        split_hemispheres(np.zeros((4, 4, 4)))


def test_limits_are_symmetric_and_robust_to_a_single_outlier():
    rng = np.random.default_rng(0)
    preds = rng.normal(0, 1, size=(20, FSAVERAGE5_VERTICES))
    vmax_clean, _ = compute_limits(preds)
    preds[3, 17] = 1e6
    vmax_spiked, _ = compute_limits(preds)
    # a percentile, not a max: one berserk vertex must not flatten every frame
    assert vmax_spiked < vmax_clean * 1.05


def test_threshold_tracks_vmax():
    preds = np.linspace(-2, 2, FSAVERAGE5_VERTICES)[None, :]
    vmax, thr = compute_limits(preds, threshold_frac=0.25)
    assert thr == pytest.approx(vmax * 0.25)


def test_all_nonfinite_predictions_raise():
    with pytest.raises(ValueError, match="no finite"):
        compute_limits(np.full((2, 10), np.nan))


def test_select_frames_widens_stride_instead_of_truncating():
    """A 4000-TR film must be sampled across its whole length, not its first minute."""
    idx = select_frames(4000, stride=1, max_frames=60)
    assert len(idx) <= 60
    assert idx[0] == 0
    assert idx[-1] > 3800, "sampling must reach the end of the clip"


def test_select_frames_honours_stride_when_it_already_fits():
    assert select_frames(20, stride=2, max_frames=60) == list(range(0, 20, 2))
    assert select_frames(0, 1, 60) == []


def test_auto_columns_targets_a_readable_aspect():
    # A 4-panel tile is ~4:1; six of them across would be a 30-inch strip.
    wide = auto_columns(6, tile_w=10.4, tile_h=2.6)
    assert wide <= 3
    # Tall, narrow tiles can afford more columns.
    assert auto_columns(6, tile_w=2.6, tile_h=3.0) >= wide
    assert auto_columns(1, 10.4, 2.6) == 1


@pytest.mark.slow
def test_render_series_writes_png_per_frame_and_one_sheet(tmp_path):
    rng = np.random.default_rng(1)
    preds = rng.normal(0, 1, size=(3, FSAVERAGE5_VERTICES))
    cfg = RenderConfig(views="lateral", dpi=60, max_frames=3)
    out = render_series(preds, tmp_path / "frames", cfg)

    assert len(out.frames) == 3
    assert all(p.exists() and p.stat().st_size > 5_000 for p in out.frames)
    assert out.contact_sheet is not None and out.contact_sheet.exists()
    # The sheet lands beside the frames directory, not inside it.
    assert out.contact_sheet.parent == tmp_path
    assert out.vmax > 0 and out.threshold == pytest.approx(out.vmax * 0.25)


@pytest.mark.slow
def test_contact_sheet_can_be_disabled(tmp_path):
    preds = np.random.default_rng(2).normal(0, 1, size=(2, FSAVERAGE5_VERTICES))
    cfg = RenderConfig(views="lateral", dpi=60, contact_sheet=False)
    out = render_series(preds, tmp_path / "frames", cfg)
    assert out.contact_sheet is None


@pytest.mark.slow
def test_colour_scale_is_shared_across_frames(tmp_path):
    """A quiet frame and a loud frame must not render identically."""
    rng = np.random.default_rng(7)
    pattern = rng.normal(0, 1, size=FSAVERAGE5_VERTICES)
    preds = np.stack([pattern * 0.02, pattern * 1.0])  # same map, 50x apart
    cfg = RenderConfig(views="lateral", dpi=60, max_frames=2, contact_sheet=False)
    out = render_series(preds, tmp_path / "f", cfg)

    import matplotlib.image as mpimg

    a, b = (mpimg.imread(p)[..., :3] for p in out.frames)
    assert a.shape == b.shape
    assert not np.allclose(a, b), "frames are identical — per-frame rescaling crept in"


def test_contact_sheet_of_nothing_is_none(tmp_path):
    assert (
        build_contact_sheet(
            [], [], tmp_path / "x.png", RenderConfig(), vmax=1.0, threshold=0.25
        )
        is None
    )
