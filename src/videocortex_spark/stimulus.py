"""Grab frames from the stimulus video — the filmstrip in the contact sheet.

"Stimulus → brain response" is the whole story of this pipeline, and the
contact sheet used to tell only half of it. This pulls one small JPEG per
requested timestamp through ffmpeg, then decodes each to numpy for the sheet
composer. Fast seeking (``-ss`` before ``-i``) makes a seek-per-tile cheap
enough that a 60-tile sheet is seconds, and boring-and-correct beats a
fragile select-expression doing it in one pass.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import typing as tp
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def extract_stimulus_frames(
    video: Path,
    times_s: tp.Sequence[float],
    out_dir: Path,
    *,
    width: int = 256,
) -> dict[int, np.ndarray]:
    """Decode ``video``; return ``{index: uint8 RGB array}`` for each timestamp.

    Indexes follow ``times_s`` order; a timestamp ffmpeg cannot seek to
    (past EOF, corrupt stream) is simply missing from the dict — a sheet
    with a gap beats a crash, and the tile keeps its label either way.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH: sudo apt install ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.image as mpimg

    result: dict[int, np.ndarray] = {}
    for i, t in enumerate(times_s):
        t = max(0.0, float(t))
        out = out_dir / f"stim_{i:03d}.jpg"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{t:.3f}", "-i", str(video),
            "-frames:v", "1",
            "-vf", f"scale={width}:-2",
            "-q:v", "3",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out.is_file():
            logger.warning(
                "stimulus frame at %.1fs failed: %s", t, (proc.stderr or "")[-200:]
            )
            out.unlink(missing_ok=True)
            continue
        try:
            img = mpimg.imread(out)
        except Exception:
            out.unlink(missing_ok=True)
            continue
        result[i] = (img[..., :3] * 255).astype(np.uint8) if img.max() <= 1.5 \
            else img[..., :3].astype(np.uint8)
    return result
