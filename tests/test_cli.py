import pytest

from videocortex_spark.cli import build_parser


def test_render_requires_a_stimulus():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["render"])


def test_render_rejects_two_stimuli():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["render", "--video", "a.mp4", "--text", "b.txt"])


def test_render_defaults_are_spark_shaped():
    a = build_parser().parse_args(["render", "--video", "clip.mp4"])
    # Upstream ships batch_size=8 / num_workers=20, tuned for a Slurm node.
    # Spark: 128 GB UMA, 20 hybrid Arm cores, 273 GB/s LPDDR5x.
    assert a.batch_size == 4
    assert a.num_workers == 4
    assert a.feature_batch_size == 2
    assert a.device == "auto"
    assert a.views == "standard"
    # slots dataclass trap: RunConfig.checkpoint is a member_descriptor.
    assert a.checkpoint == "facebook/tribev2"
    assert isinstance(a.checkpoint, str)


def test_draw_needs_no_model_arguments():
    a = build_parser().parse_args(["draw", "runs/clip/predictions.npy", "--views", "full"])
    assert a.command == "draw"
    assert a.views == "full"
    assert not hasattr(a, "device")


def test_doctor_offline_flag():
    assert build_parser().parse_args(["doctor", "--offline"]).offline is True


def test_doctor_renderer_flag():
    a = build_parser().parse_args(["doctor", "--renderer"])
    assert a.renderer is True
    assert a.offline is False


def test_overlay_requires_run():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["overlay"])


def test_overlay_defaults_match_the_spec():
    a = build_parser().parse_args(["overlay", "--run", "runs/clip"])
    assert a.command == "overlay"
    assert a.position == "top-right"
    assert a.lag_mode == "stimulus"
    assert a.label == "time"
    assert a.size == 0.24
    assert a.views == "standard"
    assert a.stride == 1
    assert not a.fast
    assert not a.spin
    assert a.dps == 24.0


def test_overlay_spin_flag():
    a = build_parser().parse_args(["overlay", "--run", "runs/clip", "--spin", "--dps", "18"])
    assert a.spin is True
    assert a.dps == 18.0
    assert a.fps == 24.0
    assert a.az_step == 2


def test_render_rejects_mps():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["render", "--video", "clip.mp4", "--device", "mps"])


def test_serve_defaults_are_loopback():
    a = build_parser().parse_args(["serve"])
    assert a.command == "serve"
    assert a.host == "127.0.0.1"
    assert a.port == 8730
    assert a.no_browser is False
    assert a.runs is None


def test_serve_flags():
    a = build_parser().parse_args(
        ["serve", "--port", "9001", "--no-browser", "--runs", "/tmp/runs"]
    )
    assert a.port == 9001
    assert a.no_browser is True
    assert str(a.runs) == "/tmp/runs"


def test_export_parses_run_and_flags():
    a = build_parser().parse_args(
        ["export", "--run", "runs/clip", "--percentile", "95", "--no-regions"]
    )
    assert a.command == "export"
    assert str(a.run) == "runs/clip"
    assert a.predictions is None
    assert a.percentile == 95.0
    assert a.no_regions is True
    assert a.out is None


def test_export_accepts_a_positional_predictions_path():
    a = build_parser().parse_args(["export", "runs/clip/predictions.npy", "-o", "b.html"])
    assert str(a.predictions) == "runs/clip/predictions.npy"
    assert str(a.out) == "b.html"


def test_sonify_parses_run_and_defaults():
    a = build_parser().parse_args(["sonify", "--run", "runs/clip"])
    assert a.command == "sonify"
    assert a.lag_mode == "stimulus"
    assert a.video is None
    assert a.out is None
    assert a.percentile == 99.0
    assert a.threshold_frac == 0.25


def test_overlay_events_and_sonify_flags_default_off():
    a = build_parser().parse_args(["overlay", "--run", "runs/clip"])
    assert a.events is None
    assert a.no_caption is False
    assert a.sonify is False
    assert a.sonify_only is False


def test_overlay_accepts_events_and_sonify():
    a = build_parser().parse_args(
        ["overlay", "--run", "runs/clip", "--events", "runs/clip/events.json",
         "--sonify"]
    )
    assert str(a.events) == "runs/clip/events.json"
    assert a.sonify is True


def test_export_cli_needs_run_or_predictions(capsys):
    from videocortex_spark.cli import main

    assert main(["export"]) == 2
    assert "--run" in capsys.readouterr().err


def test_export_cli_inherits_manifest_render_defaults(tmp_path, monkeypatch):
    """cmd_export must reuse the run's colour choices, not house defaults."""
    import json

    import numpy as np

    run = tmp_path / "clip"
    run.mkdir()
    np.save(run / "predictions.npy", np.zeros((2, 20484), dtype=np.float32))
    np.save(run / "timestamps.npy", np.array([0.0, 1.49]))
    (run / "manifest.json").write_text(
        json.dumps({"render": {"cmap": "viridis", "percentile": 90.0,
                               "threshold_frac": 0.4, "ramp_frac": 0.1}}),
        encoding="utf-8",
    )
    seen = {}
    monkeypatch.setattr(
        "videocortex_spark.export.export_viewer",
        lambda preds, out, **kw: seen.update(kw, out=out, shape=preds.shape)
        or type("R", (), {"path": out, "n_tr": 2, "n_vertices": 20484,
                          "n_bytes": 1, "regions": True})(),
    )
    from videocortex_spark.cli import main

    assert main(["export", "--run", str(run)]) == 0
    assert seen["cmap"] == "viridis"
    assert seen["percentile"] == 90.0
    assert seen["threshold_frac"] == 0.4
    assert seen["ramp_frac"] == 0.1
    assert seen["timestamps"] == [0.0, 1.49]
    assert seen["title"] == "clip"
    assert seen["out"] == run / "brain.html"
    assert seen["shape"] == (2, 20484)
