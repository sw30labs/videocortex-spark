"""Picture-in-picture overlay: brain-map animation locked to the source video.

A second product next to ``draw``. Inference is already done; this re-renders
every TR as a compact card and asks ffmpeg to composite it. ``frames/`` from a
contact-sheet run is the wrong geometry and the wrong sampling — never reused.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import typing as tp
from pathlib import Path

import numpy as np

from videocortex_spark.config import OverlayConfig
from videocortex_spark.spark import video_encode_args

logger = logging.getLogger(__name__)


class OverlayError(RuntimeError):
    """The overlay cannot be composed from this run / video."""


class OverlayResult(tp.NamedTuple):
    out: Path
    pip_dir: Path
    n_cards: int
    seconds: float


# ---------------------------------------------------------------------------
# geometry / time
# ---------------------------------------------------------------------------


def pip_box(
    width: int,
    height: int,
    *,
    size: float = 0.24,
    position: str = "top-right",
    square: bool = False,
) -> tuple[int, int, int, int]:
    """Return ``(x, y, pip_w, pip_h)`` in pixels, origin top-left.

    Landscape: ``size`` is a fraction of width. Portrait gets a larger
    fraction so the card stays readable, plus extra top inset for a notch.
    """
    if width <= 0 or height <= 0:
        raise OverlayError(f"bad video size {width}x{height}")
    if position not in ("top-right", "top-left"):
        raise OverlayError(f"unknown PIP position {position!r}")
    size = min(0.5, max(0.12, float(size)))
    portrait = height > width * 1.2
    if portrait:
        pip_w = int(round(width * max(size, 0.36)))
        top = int(round(height * 0.08))
        side = int(round(width * 0.04))
    else:
        pip_w = int(round(width * size))
        # 360p sources: 24% is a postage stamp. Floor without changing 1080p.
        pip_w = max(pip_w, min(int(width * 0.32), 280))
        short = min(width, height)
        top = int(round(short * 0.032))
        side = int(round(short * 0.032))
    pip_h = pip_w if square else int(round(pip_w / 1.11))
    pip_h = min(pip_h, height - 2 * top)
    pip_w = min(pip_w, width - 2 * side)
    if square:
        side_px = min(pip_w, pip_h)
        pip_w = pip_h = side_px
    y = top
    x = side if position == "top-left" else width - side - pip_w
    return x, y, pip_w, pip_h


def plate_index_at(
    t: float, timestamps: np.ndarray, *, lag_s: float = 0.0
) -> int | None:
    """Forward-fill: last plate whose show-time is ≤ t. None before the first."""
    t_show = np.asarray(timestamps, dtype=np.float64) + float(lag_s)
    if t_show.size == 0 or t < t_show[0]:
        return None
    return int(np.searchsorted(t_show, t, side="right") - 1)


def hold_schedule(
    timestamps: np.ndarray,
    *,
    video_duration: float,
    lag_s: float = 0.0,
) -> tuple[float, np.ndarray]:
    """``(first_show_time, duration_per_plate)`` on the source clock.

    Last plate holds through EOF. Gaps in timestamps become longer holds,
    never a black flash.
    """
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        raise OverlayError("timestamps.npy is empty")
    if not np.all(np.isfinite(ts)):
        raise OverlayError("timestamps.npy contains non-finite values")
    if ts.size >= 2 and np.any(np.diff(ts) < -1e-9):
        raise OverlayError("timestamps.npy is not monotonic")
    t_show = ts + float(lag_s)
    start = float(max(t_show[0], 0.0))
    if ts.size == 1:
        tr = 1.0
        last = max(tr, float(video_duration) - t_show[0])
        return start, np.array([max(last, 1e-3)], dtype=np.float64)
    durs = np.diff(t_show)
    tr = float(np.median(durs))
    if not math.isfinite(tr) or tr <= 0:
        tr = 1.0
    last = max(tr, float(video_duration) - t_show[-1])
    durs = np.append(durs, max(last, 1e-3))
    durs = np.maximum(durs, 1e-3)
    return start, durs


# ---------------------------------------------------------------------------
# run dir / ffprobe
# ---------------------------------------------------------------------------


def load_run(run_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    run_dir = Path(run_dir)
    pred = run_dir / "predictions.npy"
    ts_path = run_dir / "timestamps.npy"
    if not pred.is_file():
        raise OverlayError(f"no predictions.npy in {run_dir}")
    if not ts_path.is_file():
        raise OverlayError(
            f"no timestamps.npy in {run_dir} — overlay needs the TR clock. "
            "Re-run `videocortex-spark render` (newer than the overlay feature)."
        )
    preds = np.load(pred)
    timestamps = np.load(ts_path).astype(np.float64).reshape(-1)
    if preds.ndim != 2:
        raise OverlayError(f"predictions.npy has shape {preds.shape}, expected 2-D")
    if timestamps.shape[0] != preds.shape[0]:
        raise OverlayError(
            f"timestamps ({timestamps.shape[0]}) do not match "
            f"predictions ({preds.shape[0]} TRs)"
        )
    manifest: dict = {}
    man_path = run_dir / "manifest.json"
    if man_path.is_file():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    return preds, timestamps, manifest


def resolve_video(run_dir: Path, manifest: dict, video: Path | None) -> Path:
    if video is not None:
        path = Path(video)
    else:
        raw = (manifest.get("run") or {}).get("video")
        if not raw:
            raise OverlayError(
                "no --video given and manifest.run.video is empty "
                "(audio/text runs have nothing to overlay onto)"
            )
        path = Path(raw)
    if not path.is_file():
        raise OverlayError(f"source video not found: {path}")
    return path


def probe_video(path: Path) -> dict[str, tp.Any]:
    if shutil.which("ffprobe") is None:
        raise OverlayError("ffprobe not on PATH (comes with ffmpeg): sudo apt install ffmpeg")
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OverlayError(f"ffprobe failed on {path}: {exc}") from exc
    info = json.loads(out.stdout or "{}")
    streams = info.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise OverlayError(f"no video stream in {path}")
    width = int(v.get("width") or 0)
    height = int(v.get("height") or 0)
    rotate = 0
    tags = v.get("tags") or {}
    if "rotate" in tags:
        try:
            rotate = int(tags["rotate"]) % 360
        except (TypeError, ValueError):
            rotate = 0
    if rotate in (90, 270):
        width, height = height, width
    duration = _parse_duration(v.get("duration")) or _parse_duration(
        (info.get("format") or {}).get("duration")
    )
    if not duration or duration <= 0:
        raise OverlayError(f"could not read duration of {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return {
        "width": width,
        "height": height,
        "duration": float(duration),
        "has_audio": has_audio,
        "rotate": rotate,
    }


def _parse_duration(value: tp.Any) -> float | None:
    if value is None or value == "N/A":
        return None
    try:
        d = float(value)
    except (TypeError, ValueError):
        return None
    return d if math.isfinite(d) and d > 0 else None


# ---------------------------------------------------------------------------
# pip card cache
# ---------------------------------------------------------------------------


def _pip_meta(cfg: OverlayConfig, *, vmax: float, threshold: float, n: int) -> dict:
    return {
        "layout": "pip-v2",
        "views": cfg.views,
        "cmap": cfg.cmap,
        "percentile": cfg.percentile,
        "threshold_frac": cfg.threshold_frac,
        "ramp_frac": cfg.ramp_frac,
        "dpi": cfg.dpi,
        "label": cfg.label,
        "darkbg": cfg.darkbg,
        "stride": cfg.stride,
        "vmax": round(float(vmax), 6),
        "threshold": round(float(threshold), 6),
        "n": int(n),
    }


def _cache_ok(pip_dir: Path, meta: dict, n_cards: int) -> bool:
    path = pip_dir / "meta.json"
    if not path.is_file():
        return False
    try:
        got = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if got != meta:
        return False
    for i in range(0, n_cards, max(1, int(meta.get("stride") or 1))):
        if not (pip_dir / f"frame_{i:05d}.png").is_file():
            return False
    return True


def render_pip_cards(
    preds: np.ndarray,
    timestamps: np.ndarray,
    pip_dir: Path,
    cfg: OverlayConfig,
    *,
    progress=None,
) -> tuple[list[Path], float, float]:
    from videocortex_spark import render

    render_cfg = cfg.as_render()
    vmax, threshold = render.compute_limits(
        preds, cfg.percentile, cfg.threshold_frac
    )
    stride = max(1, int(cfg.stride))
    idx = list(range(0, preds.shape[0], stride))
    meta = _pip_meta(cfg, vmax=vmax, threshold=threshold, n=preds.shape[0])
    if not cfg.force and _cache_ok(pip_dir, meta, preds.shape[0]):
        logger.info("reusing cached PIP cards in %s", pip_dir)
        return [pip_dir / f"frame_{i:05d}.png" for i in idx], vmax, threshold

    if pip_dir.exists() and cfg.force:
        for leftover in pip_dir.glob("frame_*.png"):
            leftover.unlink()

    logger.info(
        "rendering %d PIP cards (every %sTR, not a contact-sheet sample)",
        len(idx),
        "" if stride == 1 else f"{stride}th ",
    )
    out = render.render_pip_series(
        preds,
        pip_dir,
        render_cfg,
        timestamps=timestamps,
        label_kind=cfg.label,
        stride=stride,
        progress=progress,
    )
    (pip_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return out.frames, out.vmax, out.threshold


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------


def write_concat(
    path: Path,
    cards: list[Path],
    durations: np.ndarray,
    *,
    lead_s: float = 0.0,
    blank: Path | None = None,
) -> Path:
    if len(cards) != len(durations):
        raise OverlayError("concat: cards and durations length mismatch")
    lines = ["ffconcat version 1.0"]
    if lead_s > 0.02 and blank is not None:
        lines += [f"file {_concat_escape(blank)}", f"duration {lead_s:.6f}"]
    for card, dur in zip(cards, durations):
        lines += [f"file {_concat_escape(card)}", f"duration {float(dur):.6f}"]
    # ffmpeg image concat eats the last duration unless the file is repeated.
    last = blank if (lead_s > 0.02 and not cards) else cards[-1]
    lines.append(f"file {_concat_escape(last)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _concat_escape(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/").replace("'", r"'\''")
    return f"'{s}'"


def _video_chain(
    box: tuple[int, int, int, int],
    *,
    cap_idx: int | None,
) -> str:
    """PIP composite, then the honesty lower-third when events are in play.

    The caption arrives as a pre-rendered PNG (drawtext needs libfreetype,
    which not every ffmpeg build ships; ``overlay`` is always there). Cards
    stay events-free so the cache survives a changed caption.
    """
    x, y, pip_w, pip_h = box
    parts = [
        f"[1:v]format=rgba,scale={pip_w}:{pip_h}:flags=lanczos[pip]",
        f"[0:v][pip]overlay={x}:{y}:format=auto:eof_action=repeat[v0]",
    ]
    cur = "v0"
    if cap_idx is not None:
        parts.append(f"[{cap_idx}:v]format=rgba[cap]")
        parts.append(f"[{cur}][cap]overlay=(W-w)/2:H-h-0.03*H:format=auto[vc]")
        cur = "vc"
    # ffmpeg's rawvideo tag defaults leak through as smpte170m on the encode;
    # untagged 8-bit 4:2:0 gets guessed as bt601 by players, which desaturates
    # red and green — the two colours this overlay lives on. Pin the window.
    parts.append(
        f"[{cur}]setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709[outv]"
    )
    return ";".join(parts)


def _audio_args(
    has_audio: bool,
    cortex_idx: int | None,
    sonify_only: bool,
) -> tuple[str, list[str]]:
    """``(audio filter chain, map/codec args)``.

    Default path is today's: copy the source audio untouched. Once cortex.wav
    is in the graph the audio is decoded and re-encoded — never ``-c:a copy``:
    the mix ducks the original ~-6 dB and lays the cortex bed at ~-18 dB
    under a limiter, so speech stays intelligible over it.
    """
    if cortex_idx is None:
        return "", (["-map", "0:a?", "-c:a", "copy"] if has_audio else [])
    if has_audio and not sonify_only:
        filt = (
            f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.501[aorg];"
            f"[{cortex_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            "volume=0.126[acx];"
            "[aorg][acx]amix=inputs=2:duration=first:dropout_transition=0,"
            "alimiter=limit=0.95[outa]"
        )
        return filt, ["-map", "[outa]", "-c:a", "aac"]
    return "", ["-map", f"{cortex_idx}:a", "-c:a", "aac"]


def _compose_inputs(
    video: Path,
    cards_args: list[str],
    caption_png: Path | None,
    cortex_wav: Path | None,
) -> tuple[list[str], int | None, int | None]:
    """Flat argv inputs; returns (argv, caption index, cortex index)."""
    groups: list[list[str]] = [["-i", str(video)], cards_args]
    cap_idx = None
    if caption_png is not None:
        cap_idx = len(groups)
        # A single frame, no -loop: overlay's eof_action=repeat holds it for
        # the whole clip. -loop 1 would make the caption an infinite stream
        # and the output timeline follows it — the video never ends.
        groups.append(["-i", str(caption_png)])
    cortex_idx = None
    if cortex_wav is not None:
        cortex_idx = len(groups)
        groups.append(["-i", str(cortex_wav)])
    return [a for g in groups for a in g], cap_idx, cortex_idx


def _write_blank_png(path: Path, w: int = 16, h: int = 16) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    rgba = np.zeros((h, w, 4), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(path, rgba)
    return path


def compose_ffmpeg(
    *,
    video: Path,
    concat: Path,
    box: tuple[int, int, int, int],
    out: Path,
    has_audio: bool,
    fast: bool = False,
    crf: int = 16,
    caption_png: Path | None = None,
    cortex_wav: Path | None = None,
    sonify_only: bool = False,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise OverlayError("ffmpeg not on PATH: sudo apt install ffmpeg")
    inputs, cap_idx, cortex_idx = _compose_inputs(
        video,
        ["-f", "concat", "-safe", "0", "-fflags", "+genpts", "-i", str(concat)],
        caption_png,
        cortex_wav,
    )
    ov = _video_chain(box, cap_idx=cap_idx)
    audio_filt, audio_args = _audio_args(has_audio, cortex_idx, sonify_only)
    if audio_filt:
        ov += ";" + audio_filt
    cmd = [
        ffmpeg, "-y",
        *inputs,
        "-filter_complex", ov,
        "-map", "[outv]",
        *audio_args,
    ]
    # NVENC (and some libx264 builds) skip automatic bt709 tagging; untagged
    # 8-bit 4:2:0 gets guessed as bt601 by players, which desaturates red
    # and green — the two colours this overlay lives on. The setparams
    # filter above is what pins the window.
    cmd += video_encode_args(fast=fast, crf=crf)
    cmd += ["-movflags", "+faststart", str(out)]
    logger.info("ffmpeg overlay %s", out)
    logger.debug("ffmpeg %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise OverlayError(f"ffmpeg failed to start: {exc}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise OverlayError(f"ffmpeg overlay failed:\n{err[-4000:]}")
    return out


def compose_ffmpeg_pipe(
    *,
    video: Path,
    box: tuple[int, int, int, int],
    out: Path,
    has_audio: bool,
    width: int,
    height: int,
    fps: float,
    fast: bool = False,
    crf: int = 16,
    caption_png: Path | None = None,
    cortex_wav: Path | None = None,
    sonify_only: bool = False,
) -> subprocess.Popen:
    """ffmpeg that reads RGBA frames on stdin. Caller writes then waits."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise OverlayError("ffmpeg not on PATH: sudo apt install ffmpeg")
    inputs, cap_idx, cortex_idx = _compose_inputs(
        video,
        [
            "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{int(width)}x{int(height)}",
            "-framerate", f"{fps:g}",
            "-thread_queue_size", "1024",
            "-i", "pipe:0",
        ],
        caption_png,
        cortex_wav,
    )
    ov = _video_chain(box, cap_idx=cap_idx)
    audio_filt, audio_args = _audio_args(has_audio, cortex_idx, sonify_only)
    if audio_filt:
        ov += ";" + audio_filt
    cmd = [
        ffmpeg, "-y",
        *inputs,
        "-filter_complex", ov,
        "-map", "[outv]",
        *audio_args,
    ]
    # See _video_chain: bt709 is pinned via the setparams filter or
    # players guess bt601 and eat the overlay's reds and greens.
    cmd += video_encode_args(fast=fast, crf=crf)
    cmd += ["-movflags", "+faststart", "-shortest", str(out)]
    logger.info("ffmpeg overlay (raw %dx%d @ %g fps) %s", width, height, fps, out)
    try:
        err = open(out.with_suffix(out.suffix + ".ffmpeg.log"), "w")
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err,
        )
        proc._videocortex_spark_err = err  # type: ignore[attr-defined]
        return proc
    except OSError as exc:
        raise OverlayError(f"ffmpeg failed to start: {exc}") from exc


# ---------------------------------------------------------------------------
# public entry
# ---------------------------------------------------------------------------


def overlay_from_run(
    run_dir: Path,
    cfg: OverlayConfig,
    *,
    video: Path | None = None,
    out: Path | None = None,
    progress=None,
) -> OverlayResult:
    import time

    t0 = time.time()
    run_dir = Path(run_dir)
    preds, timestamps, manifest = load_run(run_dir)
    video_path = resolve_video(run_dir, manifest, video)
    probe = probe_video(video_path)
    lag = cfg.hemodynamic_offset_s if cfg.lag_mode == "scanner" else 0.0

    if cfg.lag_mode == "scanner" and probe["duration"] < cfg.hemodynamic_offset_s:
        logger.warning(
            "clip is shorter than the 5 s haemodynamic lag — "
            "scanner-mode PIP will not appear"
        )

    dest = Path(out) if out is not None else run_dir / "overlay.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Events fail fast — before a single card is rendered. A missing events
    # file is a typo, not a hint to skip the annotation.
    events = None
    caption = False
    if cfg.events is not None:
        from videocortex_spark import events as events_mod

        try:
            events = events_mod.load_events(cfg.events)
        except events_mod.EventsError as exc:
            raise OverlayError(str(exc)) from exc
        caption = cfg.caption

    cortex_wav = None
    if cfg.sonify or cfg.sonify_only:
        from videocortex_spark import sonify

        try:
            cortex_wav = sonify.sonify_from_run(
                run_dir,
                duration_s=probe["duration"],
                lag_mode=cfg.lag_mode,
                percentile=cfg.percentile,
                threshold_frac=cfg.threshold_frac,
            ).wav
        except sonify.SonifyError as exc:
            raise OverlayError(str(exc)) from exc

    if cfg.spin:
        result = _overlay_spin(
            run_dir, cfg, preds, timestamps, video_path, probe, lag,
            dest, progress=progress, events=events, caption=caption,
            cortex_wav=cortex_wav,
        )
        result = result._replace(seconds=time.time() - t0)
        logger.info(
            "overlay -> %s  (%d spin frames, %.1fs)",
            dest, result.n_cards, result.seconds,
        )
        return result

    stride = max(1, int(cfg.stride))
    idx = list(range(0, len(timestamps), stride))
    start, durs = hold_schedule(
        timestamps[idx], video_duration=probe["duration"], lag_s=lag
    )

    box = pip_box(
        probe["width"], probe["height"], size=cfg.size, position=cfg.position
    )
    logger.info(
        "PIP %dx%d at (%d,%d) on %dx%d · %d TRs · lag=%s",
        box[2], box[3], box[0], box[1],
        probe["width"], probe["height"],
        preds.shape[0], cfg.lag_mode,
    )

    pip_dir = run_dir / "pip"
    cards, _, _ = render_pip_cards(
        preds, timestamps, pip_dir, cfg, progress=progress
    )

    concat = pip_dir / "concat.txt"
    blank = pip_dir / "blank.png"
    _write_blank_png(blank)
    write_concat(concat, cards, durs, lead_s=start, blank=blank)

    caption_png = None
    if caption:
        from videocortex_spark.events import render_caption_png

        caption_png = render_caption_png(pip_dir / "caption.png", probe["width"])

    compose_ffmpeg(
        video=video_path,
        concat=concat,
        box=box,
        out=dest,
        has_audio=probe["has_audio"],
        fast=cfg.fast,
        crf=cfg.crf,
        caption_png=caption_png,
        cortex_wav=cortex_wav,
        sonify_only=cfg.sonify_only,
    )
    elapsed = time.time() - t0
    logger.info("overlay -> %s  (%d cards, %.1fs)", dest, len(cards), elapsed)
    return OverlayResult(
        out=dest, pip_dir=pip_dir, n_cards=len(cards), seconds=elapsed
    )


def _overlay_spin(
    run_dir: Path,
    cfg: OverlayConfig,
    preds: np.ndarray,
    timestamps: np.ndarray,
    video_path: Path,
    probe: dict,
    lag: float,
    dest: Path,
    *,
    progress=None,
    events=None,
    caption: bool = False,
    cortex_wav: Path | None = None,
) -> OverlayResult:
    from videocortex_spark import spin
    from videocortex_spark.render import blit_ribbon, compute_limits, format_pip_label

    pip_dir = run_dir / "pip_spin"
    pip_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = pip_dir / "atlas.npz"
    fp = spin.atlas_fingerprint(cfg)
    atlas_meta = pip_dir / "atlas_meta.json"
    atlas = None
    if atlas_path.is_file() and atlas_meta.is_file() and not cfg.force:
        try:
            if json.loads(atlas_meta.read_text(encoding="utf-8")) == fp:
                atlas = spin.load_atlas(atlas_path)
                logger.info("reusing spin atlas %s", atlas_path)
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            atlas = None
    if atlas is None:
        atlas = spin.bake_atlas(
            elev=cfg.elev, az_step=cfg.az_step, size=cfg.atlas_px,
            progress=progress,
        )
        spin.save_atlas(atlas_path, atlas)
        atlas_meta.write_text(json.dumps(fp, indent=2), encoding="utf-8")

    vmax, threshold = compute_limits(preds, cfg.percentile, cfg.threshold_frac)
    stride = max(1, int(cfg.stride))
    idx = np.arange(0, preds.shape[0], stride)
    palettes = spin.build_palettes(
        preds[idx], atlas.sulc, cfg, vmax=vmax, threshold=threshold,
    )
    ts = timestamps[idx]

    region_names: list[str] = []
    if cfg.regions:
        try:
            from videocortex_spark.regions import load_region_labels, top_regions_per_tr

            rid, names = load_region_labels()
            region_names = top_regions_per_tr(preds[idx], rid, names, k=3)
        except Exception as exc:  # atlas fetch can fail offline — not fatal
            logger.warning("region labels skipped: %s", exc)

    ribbon = None
    if cfg.ribbon:
        from videocortex_spark.render import draw_energy_ribbon, energy_curve

        ribbon = draw_energy_ribbon(
            energy_curve(preds[idx]), width_px=atlas.size, height_px=40,
            darkbg=cfg.darkbg,
        )
        if events is not None:
            from videocortex_spark.events import blit_event_ticks

            ribbon = blit_event_ticks(ribbon, events, duration_s=probe["duration"])
    fps = max(8.0, float(cfg.fps))
    n_frames = max(1, int(math.ceil(probe["duration"] * fps - 1e-9)))
    blank = np.zeros((atlas.size, atlas.size, 4), dtype=np.uint8)
    monitor = None
    if cfg.monitor:
        from videocortex_spark.render import apply_monitor_frame

        monitor = apply_monitor_frame
        blank = monitor(blank)

    box = pip_box(
        probe["width"], probe["height"],
        size=cfg.size, position=cfg.position, square=True,
    )
    logger.info(
        "PIP spin %dx%d at (%d,%d) · %d TRs · %.0f deg/s · %g fps · az %d°",
        box[2], box[3], box[0], box[1],
        palettes.shape[0], spin.clamp_dps(cfg.dps), fps, cfg.az_step,
    )
    caption_png = None
    if caption:
        from videocortex_spark.events import render_caption_png

        caption_png = render_caption_png(pip_dir / "caption.png", probe["width"])
    proc = compose_ffmpeg_pipe(
        video=video_path, box=box, out=dest, has_audio=probe["has_audio"],
        width=atlas.size, height=atlas.size, fps=fps,
        fast=cfg.fast, crf=cfg.crf,
        caption_png=caption_png, cortex_wav=cortex_wav,
        sonify_only=cfg.sonify_only,
    )
    assert proc.stdin is not None
    try:
        for n in range(n_frames):
            t = n / fps
            i0, i1, tfrac = spin.palette_index_frac(t, ts, lag_s=lag)
            if i0 < 0 or i0 >= palettes.shape[0]:
                frame = blank
            else:
                k0, k1, frac = spin.pose_blend(
                    t, dps=cfg.dps, az_step=cfg.az_step
                )
                pal = spin.blend_palettes(palettes, i0, i1, tfrac)
                frame = spin.feather_alpha(
                    spin.composite_frame_smooth(atlas, pal, k0, k1, frac)
                )
                orig_i = int(idx[i0])
                label = format_pip_label(
                    cfg.label, orig_i,
                    float(timestamps[orig_i]) if orig_i < len(timestamps) else t,
                )
                regions = region_names[i0] if i0 < len(region_names) else ""
                if label or regions:
                    frame = spin._blit_label(frame, label or "", regions)
                if ribbon is not None:
                    frame = blit_ribbon(
                        frame, ribbon, playhead=t / max(probe["duration"], 1e-6)
                    )
                if monitor is not None:
                    frame = monitor(frame)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if progress and (n + 1 == n_frames or n % max(1, int(fps)) == 0):
                progress(n + 1, n_frames)
    except BrokenPipeError as exc:
        raise OverlayError("ffmpeg pipe broke (see overlay.mp4.ffmpeg.log)") from exc
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        err_f = getattr(proc, "_videocortex_spark_err", None)
        if err_f is not None:
            err_f.close()
    rc = proc.wait()
    log_path = dest.with_suffix(dest.suffix + ".ffmpeg.log")
    if rc != 0:
        err = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        raise OverlayError(f"ffmpeg overlay failed:\n{err[-4000:]}")
    return OverlayResult(
        out=dest, pip_dir=pip_dir, n_cards=n_frames, seconds=0.0
    )
