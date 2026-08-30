import json
from pathlib import Path

import pytest

from videocortex_spark.config import (
    FEATURE_EXTRACTORS,
    VIEW_PRESETS,
    OverlayConfig,
    RenderConfig,
    RunConfig,
    write_manifest,
)
from videocortex_spark.model import build_config_overrides


def test_every_view_preset_is_a_valid_nilearn_pair():
    hemis = {"left", "right"}
    views = {"lateral", "medial", "dorsal", "ventral", "anterior", "posterior"}
    for name, pairs in VIEW_PRESETS.items():
        assert pairs, f"{name} is empty"
        for hemi, view in pairs:
            assert hemi in hemis, (name, hemi)
            assert view in views, (name, view)


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown view preset"):
        RenderConfig(views="sideways").view_pairs()


def test_stimulus_requires_exactly_one_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        RunConfig().stimulus()
    with pytest.raises(ValueError, match="exactly one"):
        RunConfig(video=tmp_path / "a.mp4", audio=tmp_path / "a.wav").stimulus()
    kind, path = RunConfig(video=tmp_path / "a.mp4").stimulus()
    assert kind == "video" and path.name == "a.mp4"


def test_overlay_config_cannot_leak_max_frames_into_draw():
    ov = OverlayConfig()
    r = ov.as_render()
    assert r.max_frames == 0
    assert r.contact_sheet is False
    assert ov.view_pairs() == VIEW_PRESETS["standard"]


def test_overrides_cover_every_extractor_that_hardcodes_cuda():
    """The checkpoint ships device: cuda four times. Miss one and it crashes."""
    ov = build_config_overrides("cuda")
    for spec in FEATURE_EXTRACTORS.values():
        assert ov[spec["device_key"]] == "cuda"
    assert ov["data.num_workers"] == 4
    assert ov["data.batch_size"] == 4
    assert ov["data.video_feature.image.batch_size"] == 2


def test_overrides_use_dotted_paths_matching_the_published_config():
    ov = build_config_overrides("cpu")
    assert "data.video_feature.image.device" in ov  # nested one level deeper
    assert "data.audio_feature.device" in ov
    assert all("." in k for k in ov)


def test_manifest_round_trips(tmp_path: Path):
    p = write_manifest(
        tmp_path / "manifest.json",
        run=RunConfig(video=tmp_path / "clip.mp4"),
        render=RenderConfig(),
        extra={"result": {"n_timesteps": 7}},
    )
    data = json.loads(p.read_text())
    assert data["model"]["n_vertices"] == 20484
    assert data["model"]["mesh"] == "fsaverage5"
    assert data["result"]["n_timesteps"] == 7
    assert data["run"]["video"].endswith("clip.mp4")
