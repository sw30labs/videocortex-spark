"""PIP overlay: clock math, geometry, concat, ffmpeg compose."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from videocortex_spark.config import OverlayConfig
from videocortex_spark.overlay import (
    OverlayError,
    hold_schedule,
    load_run,
    overlay_from_run,
    pip_box,
    plate_index_at,
    probe_video,
    write_concat,
)
from videocortex_spark.render import format_pip_label, rounded_rect_alpha


def test_pip_box_landscape_is_top_right_and_small():
    x, y, w, h = pip_box(1920, 1080, size=0.24, position="top-right")
    assert w == pytest.approx(461, abs=2)
    assert x + w < 1920
    assert y < 80
    assert x > 1000
    # roughly 8% of the frame
    assert (w * h) / (1920 * 1080) < 0.12


def test_pip_box_portrait_keeps_top_inset_for_a_notch():
    x, y, w, h = pip_box(1080, 1920, size=0.24, position="top-right")
    assert y >= int(1920 * 0.07)
    assert w > 300
    assert x + w <= 1080


def test_pip_box_top_left():
    x, y, w, h = pip_box(1280, 720, position="top-left")
    assert x < 50
    assert y < 50


def test_pip_box_spin_is_square():
    x, y, w, h = pip_box(1920, 1080, size=0.24, square=True)
    assert w == h


def test_plate_index_forward_fills_and_hides_before_start():
    ts = np.array([0.0, 1.0, 2.0, 5.0])  # gap between 2 and 5
    assert plate_index_at(-0.1, ts) is None
    assert plate_index_at(0.0, ts) == 0
    assert plate_index_at(0.9, ts) == 0
    assert plate_index_at(1.0, ts) == 1
    assert plate_index_at(3.5, ts) == 2  # hold across the hole
    assert plate_index_at(5.0, ts) == 3
    assert plate_index_at(100.0, ts) == 3


def test_scanner_lag_hides_the_first_five_seconds():
    ts = np.array([0.0, 1.0, 2.0])
    assert plate_index_at(4.9, ts, lag_s=5.0) is None
    assert plate_index_at(5.0, ts, lag_s=5.0) == 0
    assert plate_index_at(6.5, ts, lag_s=5.0) == 1


def test_hold_schedule_last_plate_covers_eof():
    ts = np.array([0.0, 1.0, 2.0])
    start, durs = hold_schedule(ts, video_duration=5.4, lag_s=0.0)
    assert start == 0.0
    assert len(durs) == 3
    assert durs[0] == pytest.approx(1.0)
    assert durs[1] == pytest.approx(1.0)
    assert durs[2] == pytest.approx(3.4)  # hold through EOF


def test_hold_schedule_rejects_empty_and_backwards():
    with pytest.raises(OverlayError, match="empty"):
        hold_schedule(np.array([]), video_duration=1.0)
    with pytest.raises(OverlayError, match="monotonic"):
        hold_schedule(np.array([0.0, 2.0, 1.0]), video_duration=3.0)


def test_hold_schedule_scanner_lead():
    start, durs = hold_schedule(
        np.array([0.0, 1.0]), video_duration=10.0, lag_s=5.0
    )
    assert start == pytest.approx(5.0)
    assert durs[0] == pytest.approx(1.0)


def test_format_pip_label():
    assert format_pip_label("none", 3, 65.2) is None
    assert format_pip_label("tr", 3, 65.2) == "TR 3"
    assert format_pip_label("time", 3, 65.2) == "1:05"
    assert format_pip_label("time", 0, 0.0) == "0:00"


def test_monitor_frame_is_black_plate_with_green_bezel():
    from videocortex_spark.render import apply_monitor_frame

    src = np.zeros((80, 80, 4), dtype=np.uint8)
    src[30:50, 30:50, 0] = 255
    src[30:50, 30:50, 3] = 255
    out = apply_monitor_frame(src, border_px=5)
    assert out[0, 0, 3] < 10
    # interior plate, away from the red blob: black, opaque
    plate = out[12, 40]
    assert plate[3] > 200
    assert plate[1] < 40 and plate[0] < 40
    # bezel is green-dominant
    edge = out[3, 40]
    assert edge[3] > 200
    assert edge[1] > edge[0] and edge[1] > edge[2]
    # brain still shows through
    assert out[40, 40, 0] > 200


def test_rounded_rect_corners_are_transparent_centre_is_not():
    mask = rounded_rect_alpha(100, 80, 12)
    assert mask[50, 40] == pytest.approx(1.0)
    assert mask[0, 0] == pytest.approx(0.0)
    assert mask[0, 40] == pytest.approx(1.0)  # top edge, not a corner


def test_write_concat_repeats_the_last_file(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    p = write_concat(tmp_path / "c.txt", [a, b], np.array([1.0, 2.5]))
    text = p.read_text()
    lines = [ln for ln in text.splitlines() if ln]
    assert lines[0] == "ffconcat version 1.0"
    assert text.count("duration") == 2
    b_esc = str(b.resolve()).replace("\\", "/")
    assert lines[-1] == f"file '{b_esc}'"
    assert lines[-2].startswith("duration 2.5")


def test_load_run_requires_timestamps(tmp_path):
    np.save(tmp_path / "predictions.npy", np.zeros((2, 10)))
    with pytest.raises(OverlayError, match="timestamps"):
        load_run(tmp_path)


def test_load_run_length_mismatch(tmp_path):
    np.save(tmp_path / "predictions.npy", np.zeros((3, 8)))
    np.save(tmp_path / "timestamps.npy", np.arange(2))
    with pytest.raises(OverlayError, match="do not match"):
        load_run(tmp_path)


def test_overlay_from_run_refuses_audio_only_manifest(tmp_path):
    np.save(tmp_path / "predictions.npy", np.zeros((2, 8)))
    np.save(tmp_path / "timestamps.npy", np.array([0.0, 1.0]))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"run": {"video": None, "audio": "x.wav"}})
    )
    with pytest.raises(OverlayError, match="nothing to overlay"):
        overlay_from_run(tmp_path, OverlayConfig())


def test_overlay_config_without_events_is_today():
    """No --events -> no caption, no ticks, no cortex: bit-identical config."""
    cfg = OverlayConfig()
    assert cfg.events is None
    assert cfg.caption is True  # only consulted when events are present
    assert cfg.sonify is False
    assert cfg.sonify_only is False


def _lavfi_clip(path: Path, seconds: float = 2.0, audio: bool = False) -> Path:
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=blue:s=640x360:d={seconds}",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
        cmd += ["-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", str(seconds), str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_overlay_from_run_missing_events_is_a_hard_error(tmp_path):
    src = _lavfi_clip(tmp_path / "src.mp4", 2.0)
    run = tmp_path / "run"
    run.mkdir()
    np.save(run / "predictions.npy", np.zeros((2, 8), dtype=np.float32))
    np.save(run / "timestamps.npy", np.array([0.0, 1.0]))
    with pytest.raises(OverlayError, match="events file not found"):
        overlay_from_run(
            run, OverlayConfig(events=tmp_path / "nope.json"), video=src
        )


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_caption_lands_in_the_lower_third(tmp_path):
    """Same compose with and without the caption: the lower third must differ."""
    from videocortex_spark.events import render_caption_png
    from videocortex_spark.overlay import compose_ffmpeg, _write_blank_png

    src = _lavfi_clip(tmp_path / "src.mp4", 1.0)
    card = tmp_path / "card.png"
    _write_blank_png(card, 32, 32)
    concat = write_concat(tmp_path / "c.txt", [card], np.array([1.0]))
    caption_png = render_caption_png(tmp_path / "caption.png", 640)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    means = []
    for name, cap in (("plain", None), ("captioned", caption_png)):
        out = tmp_path / f"{name}.mp4"
        compose_ffmpeg(
            video=src, concat=concat, box=(10, 10, 80, 72), out=out,
            has_audio=False, crf=28, caption_png=cap,
        )
        frame = tmp_path / f"{name}.png"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(out), "-vframes", "1", str(frame)],
            check=True, capture_output=True,
        )
        pix = mpimg.imread(frame)
        # lower-third band, away from the PIP corner
        means.append(float(pix[300:350, 100:540, :3].mean()))
    assert abs(means[0] - means[1]) > 0.02


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_sonify_mix_keeps_audio_for_the_whole_clip(tmp_path):
    from videocortex_spark.overlay import compose_ffmpeg, _write_blank_png
    from videocortex_spark.sonify import write_wav

    src = _lavfi_clip(tmp_path / "src.mp4", 2.0, audio=True)
    card = tmp_path / "card.png"
    _write_blank_png(card, 32, 32)
    concat = write_concat(tmp_path / "c.txt", [card], np.array([2.0]))
    cortex = tmp_path / "cortex.wav"
    t = np.arange(2 * 48000, dtype=np.float32) / 48000
    bed = np.stack([0.2 * np.sin(2 * np.pi * 196 * t)] * 2, axis=1)
    write_wav(cortex, bed)

    out = tmp_path / "mixed.mp4"
    compose_ffmpeg(
        video=src, concat=concat, box=(10, 10, 80, 72), out=out,
        has_audio=True, crf=28, cortex_wav=cortex,
    )
    info = probe_video(out)
    assert info["has_audio"]
    # the mix must last the whole clip, not stop at some first-stream boundary
    assert info["duration"] == pytest.approx(2.0, abs=0.2)
    # re-encoded aac, never stream-copied
    fmt = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(out)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert fmt == "aac"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_sonify_only_replaces_the_original_audio(tmp_path):
    from videocortex_spark.overlay import compose_ffmpeg, _write_blank_png
    from videocortex_spark.sonify import write_wav

    src = _lavfi_clip(tmp_path / "src.mp4", 2.0, audio=True)
    card = tmp_path / "card.png"
    _write_blank_png(card, 32, 32)
    concat = write_concat(tmp_path / "c.txt", [card], np.array([2.0]))
    cortex = tmp_path / "cortex.wav"
    t = np.arange(2 * 48000, dtype=np.float32) / 48000
    write_wav(cortex, np.stack([0.5 * np.sin(2 * np.pi * 880 * t)] * 2, axis=1))

    out = tmp_path / "only.mp4"
    compose_ffmpeg(
        video=src, concat=concat, box=(10, 10, 80, 72), out=out,
        has_audio=True, crf=28, cortex_wav=cortex, sonify_only=True,
    )
    # decode the output audio: the 880 Hz bed must dominate the 440 Hz original
    raw = subprocess.run(
        ["ffmpeg", "-y", "-i", str(out), "-f", "f32le", "-ac", "1", "-"],
        check=True, capture_output=True,
    ).stdout
    sig = np.frombuffer(raw, dtype=np.float32)
    spectrum = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
    freqs = np.fft.rfftfreq(sig.size, 1 / 48000)
    def power_at(f):
        return spectrum[np.abs(freqs - f) < 30].sum()
    assert power_at(880) > 5 * power_at(440)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_compose_ffmpeg_overlays_a_tiny_clip(tmp_path):
    from videocortex_spark.overlay import compose_ffmpeg, _write_blank_png

    src = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=640x360:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", "1", str(src),
        ],
        check=True, capture_output=True,
    )
    card = tmp_path / "card.png"
    _write_blank_png(card, 32, 32)
    # make it opaque red so we can see it landed
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    rgba = np.zeros((32, 32, 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 3] = 1.0
    mpimg.imsave(card, rgba)
    concat = write_concat(tmp_path / "c.txt", [card], np.array([1.0]))
    out = tmp_path / "out.mp4"
    compose_ffmpeg(
        video=src, concat=concat, box=(10, 10, 80, 72),
        out=out, has_audio=False, fast=False, crf=28,
    )
    assert out.is_file() and out.stat().st_size > 500
    info = probe_video(out)
    assert info["width"] == 640 and info["height"] == 360
    assert info["duration"] == pytest.approx(1.0, abs=0.15)
    frame = tmp_path / "frame.png"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out), "-vframes", "1", str(frame)],
        check=True, capture_output=True,
    )
    import matplotlib.image as mpimg

    pix = mpimg.imread(frame)
    pip = pix[10:80, 10:90, :3]
    bg = pix[200:250, 200:250, :3]
    # PIP is red on a blue field — mapping the filter output, not the raw source.
    assert float(pip[..., 0].mean()) > float(pip[..., 2].mean())
    assert float(bg[..., 2].mean()) > float(bg[..., 0].mean())


@pytest.mark.slow
def test_render_pip_card_is_rgba_with_clear_corners(tmp_path):
    from videocortex_spark.config import FSAVERAGE5_VERTICES, RenderConfig
    from videocortex_spark.render import render_pip_card

    rng = np.random.default_rng(3)
    values = rng.normal(0, 1, size=FSAVERAGE5_VERTICES)
    path = tmp_path / "card.png"
    render_pip_card(
        values, path, RenderConfig(views="standard", dpi=50),
        vmax=2.0, threshold=0.5, label="0:01",
    )
    import matplotlib.image as mpimg

    img = mpimg.imread(path)
    assert img.ndim == 3 and img.shape[-1] == 4
    assert img[0, 0, 3] < 0.05
    assert img[img.shape[0] // 2, img.shape[1] // 2, 3] > 0.9


def test_pose_index_wraps_and_respects_dps():
    from videocortex_spark.spin import clamp_dps, n_poses, pose_blend, pose_index

    assert clamp_dps(1) == 12.0
    assert clamp_dps(99) == 48.0
    assert n_poses(10) == 36
    assert n_poses(2) == 180
    assert pose_index(0.0, dps=24, az_step=10) == 0
    # 15 s at 24 deg/s is a full turn
    assert pose_index(15.0, dps=24, az_step=10) == 0
    assert pose_index(7.5, dps=24, az_step=10) == n_poses(10) // 2
    k0, k1, frac = pose_blend(0.0, dps=24, az_step=2)
    assert k0 == 0 and frac == pytest.approx(0.0)
    k0, k1, frac = pose_blend(1.0 / 24.0, dps=24, az_step=2)  # 1° into a 2° bin
    assert k0 == 0 and k1 == 1
    assert 0.4 < frac < 0.6


def test_vertex_palette_quiet_stays_quieter_than_loud():
    from videocortex_spark.spin import vertex_palette

    sulc = np.linspace(0, 1, 64)
    quiet = np.zeros(64)
    loud = np.ones(64)
    pq = vertex_palette(quiet, sulc, cmap="cold_hot", vmax=1.0, threshold=0.25)
    pl = vertex_palette(loud, sulc, cmap="cold_hot", vmax=1.0, threshold=0.25)
    assert pq.shape == (64, 3)
    assert not np.allclose(pq, pl)


def test_composite_smooth_blends_two_poses():
    from videocortex_spark.spin import SpinAtlas, composite_frame_smooth

    fid = np.zeros((2, 4, 4), dtype=np.uint16)
    fid[0, :, :] = 1
    fid[1, :, :] = 1
    atlas = SpinAtlas(
        face_id=fid,
        shade=np.array([[1.0], [0.0]], dtype=np.float32),
        faces=np.array([[0, 0, 0]], dtype=np.int32),
        sulc=np.zeros(1, dtype=np.float32),
        azims=np.array([0.0, 2.0], dtype=np.float32),
        elev=18.0,
        size=4,
    )
    pal = np.ones((1, 3), dtype=np.float32)
    a = composite_frame_smooth(atlas, pal, 0, 1, 0.0)
    b = composite_frame_smooth(atlas, pal, 0, 1, 1.0)
    m = composite_frame_smooth(atlas, pal, 0, 1, 0.5)
    assert a[1, 1, 0] > 200
    assert b[1, 1, 0] < 30  # shade 0
    assert 80 < m[1, 1, 0] < 180


def test_composite_frame_lookup_and_alpha():
    from videocortex_spark.spin import SpinAtlas, composite_frame

    fid = np.zeros((1, 8, 8), dtype=np.uint16)
    fid[0, 2:6, 2:6] = 1
    atlas = SpinAtlas(
        face_id=fid,
        shade=np.ones((1, 1), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        sulc=np.zeros(3, dtype=np.float32),
        azims=np.array([0.0], dtype=np.float32),
        elev=18.0,
        size=8,
    )
    pal = np.zeros((3, 3), dtype=np.float32)
    pal[:, 0] = 1.0
    rgba = composite_frame(atlas, pal, 0)
    assert rgba[0, 0, 3] == 0
    assert rgba[4, 4, 3] == 255
    assert rgba[4, 4, 0] > 200


def test_iter_spin_coalesces_identical_keys():
    from videocortex_spark.config import OverlayConfig
    from videocortex_spark.spin import SpinAtlas, iter_spin_frames

    fid = np.zeros((36, 4, 4), dtype=np.uint16)
    atlas = SpinAtlas(
        face_id=fid, shade=np.ones((36, 1), np.float32),
        faces=np.array([[0, 0, 0]], np.int32), sulc=np.zeros(1, np.float32),
        azims=np.arange(36, dtype=np.float32), elev=18.0, size=4,
    )
    pal = np.zeros((2, 1, 3), np.float32)
    cfg = OverlayConfig(spin=True, fps=12, dps=24, az_step=10)
    runs = list(
        iter_spin_frames(
            atlas=atlas, palettes=pal, timestamps=np.array([0.0, 1.0]),
            duration=1.0, cfg=cfg, lag_s=0.0,
        )
    )
    assert runs
    assert all(d > 0 for _, d in runs)
    assert abs(sum(d for _, d in runs) - 1.0) < 0.08


@pytest.mark.slow
def test_bake_atlas_ids_and_transparency(tmp_path):
    from videocortex_spark.spin import bake_atlas, composite_frame, vertex_palette

    atlas = bake_atlas(elev=18, az_step=90, size=96)
    assert atlas.face_id.shape[0] == 4
    assert atlas.face_id.shape[1] == atlas.face_id.shape[2]
    # Corners of an ortho brain render should be empty.
    assert float((atlas.face_id[0, 0, 0] == 0))
    assert atlas.face_id[0].max() > 0
    pal = vertex_palette(
        np.zeros(atlas.sulc.shape[0], dtype=np.float32),
        atlas.sulc, cmap="cold_hot", vmax=1.0, threshold=0.25,
    )
    rgba = composite_frame(atlas, pal, 0)
    assert rgba[0, 0, 3] == 0
    assert rgba[..., 3].max() == 255
