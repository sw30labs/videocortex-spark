"""Generate the synthetic sample run.

This is *not* model output. It is a scripted sequence — early visual cortex
throughout, motion-sensitive cortex waxing and waning, superior temporal
joining once someone starts talking — built so the renderer can be exercised
end to end without a 20 GB download.

Deterministic: same seed, same file, every time.

    python examples/make_sample.py
    videocortex-spark draw examples/sample_run/predictions.npy --max-frames 6
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from videocortex_spark.config import FSAVERAGE5_VERTICES
from videocortex_spark.render import load_fsaverage5

OUT = Path(__file__).parent / "sample_run" / "predictions.npy"
N_TR = 12
SEED = 11

# Rough fsaverage5 coordinates, right-hemisphere convention; mirrored for left.
SEEDS = {
    "hMT+": np.array([-45.0, -70.0, 5.0]),
    "STS": np.array([-52.0, -18.0, 2.0]),
}


def _bump(coords, centre, sigma, weight):
    """A Gaussian blob on the surface, both hemispheres, left then right."""
    out = []
    for hemi in ("left", "right"):
        c = np.asarray(centre) * np.array([1 if hemi == "right" else -1, 1, 1])
        d = np.linalg.norm(coords[hemi] - c, axis=1)
        out.append(weight * np.exp(-(d**2) / (2 * sigma**2)))
    return np.concatenate(out)


def build() -> np.ndarray:
    fs = load_fsaverage5()
    coords = {h: nib.load(fs[h]["mesh"]).darrays[0].data for h in ("left", "right")}

    left = coords["left"]
    occipital = left[np.argmin(left[:, 1])] * np.array([-1, 1, 1])

    rng = np.random.default_rng(SEED)
    frames = []
    for t in range(N_TR):
        visual = 1.0 if t < 9 else 0.3           # video runs most of the clip
        motion = np.sin(t / 2.0) ** 2            # motion energy comes and goes
        speech = 0.0 if t < 3 else min(1.0, (t - 3) / 3.0)   # talking starts at TR 3
        frames.append(
            _bump(coords, occipital, 20, 1.0 * visual)
            + _bump(coords, SEEDS["hMT+"], 14, 0.75 * motion)
            + _bump(coords, SEEDS["STS"], 16, 0.9 * speech)
            + rng.normal(0, 0.05, size=FSAVERAGE5_VERTICES)
        )

    preds = np.stack(frames)
    return (preds - preds.mean(axis=1, keepdims=True)).astype(np.float32)


if __name__ == "__main__":
    preds = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, preds)
    print(f"wrote {OUT}  {preds.shape}  {preds.dtype}")
