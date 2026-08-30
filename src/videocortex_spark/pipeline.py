"""The end-to-end run: stimulus file in, brain-map stills out."""

from __future__ import annotations

import logging
import time
import typing as tp
from pathlib import Path

import numpy as np

from videocortex_spark.config import RenderConfig, RunConfig, write_manifest
from videocortex_spark.patches import whisperx_compat

logger = logging.getLogger(__name__)


class RunResult(tp.NamedTuple):
    out_dir: Path
    frames: list[Path]
    contact_sheet: Path | None
    predictions: Path | None
    n_timesteps: int
    seconds: float


def run_dir_for(out_root: Path, stimulus: Path) -> Path:
    """One directory per stimulus, so repeat runs overwrite rather than pile up."""
    return Path(out_root) / stimulus.stem


def predict(run: RunConfig) -> tuple[np.ndarray, list[float]]:
    """Run TRIBE v2 over the stimulus and return ``(preds, timestamps)``.

    ``preds`` is (n_timesteps x 20484) on the fsaverage5 surface. Note the
    5-second offset upstream applies to compensate for haemodynamic lag: the
    prediction at index i is the response to what happened ~5s earlier.
    """
    from videocortex_spark.model import load_model

    kind, path = run.stimulus()
    model = load_model(run)

    logger.info("building events from %s (%s)", path.name, kind)
    t0 = time.time()
    # Transcription happens in here, which is where the whisperx CPU issue bites.
    with whisperx_compat():
        events = model.get_events_dataframe(**{f"{kind}_path": str(path)})
    logger.info("events ready in %.1fs (%d rows)", time.time() - t0, len(events))

    t0 = time.time()
    preds, segments = model.predict(events)
    logger.info(
        "predicted %s in %.1fs", "x".join(str(d) for d in preds.shape), time.time() - t0
    )

    tr = float(getattr(model.data, "TR", 1.49))
    timestamps = [
        float(getattr(seg, "start", i * tr)) for i, seg in enumerate(segments)
    ]
    return preds, timestamps


def render_only(
    predictions: Path, out_dir: Path, render_cfg: RenderConfig
) -> RunResult:
    """Re-render a saved prediction. No torch, no model, no re-inference."""
    from videocortex_spark import render

    t0 = time.time()
    preds = np.load(predictions)
    logger.info("loaded %s from %s", preds.shape, predictions)

    out = render.render_series(
        preds, out_dir / "frames", render_cfg, progress=_progress
    )
    return RunResult(
        out_dir=out_dir,
        frames=out.frames,
        contact_sheet=out.contact_sheet,
        predictions=predictions,
        n_timesteps=int(preds.shape[0]),
        seconds=time.time() - t0,
    )


def run(run_cfg: RunConfig, render_cfg: RenderConfig) -> RunResult:
    """Full pipeline: predict, then render."""
    from videocortex_spark import render

    t0 = time.time()
    _, stimulus = run_cfg.stimulus()
    out_dir = run_dir_for(run_cfg.out_dir, stimulus)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, timestamps = predict(run_cfg)

    pred_path: Path | None = None
    if run_cfg.save_predictions:
        # Inference costs minutes; re-rendering should not. Keeping the raw
        # matrix means `videocortex-spark draw` can retry the picture for free.
        pred_path = out_dir / "predictions.npy"
        np.save(pred_path, preds)
        np.save(out_dir / "timestamps.npy", np.asarray(timestamps))
        logger.info("saved predictions -> %s", pred_path)

    out = render.render_series(
        preds, out_dir / "frames", render_cfg,
        timestamps=timestamps,
        stimulus_video=stimulus if run_cfg.video is not None else None,
        progress=_progress,
    )

    elapsed = time.time() - t0
    tr_s = None
    if len(timestamps) >= 2:
        tr_s = float(np.median(np.diff(np.asarray(timestamps, dtype=float))))
    elif timestamps:
        tr_s = 1.0
    write_manifest(
        out_dir / "manifest.json",
        run=run_cfg,
        render=render_cfg,
        extra={
            "result": {
                "n_timesteps": int(preds.shape[0]),
                "n_vertices": int(preds.shape[1]),
                "n_frames": len(out.frames),
                "seconds": round(elapsed, 1),
                "hemodynamic_offset_s": 5.0,
                "tr_s": tr_s,
            }
        },
    )

    return RunResult(
        out_dir=out_dir,
        frames=out.frames,
        contact_sheet=out.contact_sheet,
        predictions=pred_path,
        n_timesteps=int(preds.shape[0]),
        seconds=elapsed,
    )


def _progress(i: int, n: int) -> None:
    end = "\n" if i == n else "\r"
    print(f"  rendering {i}/{n}", end=end, flush=True)
