"""Run and render configuration.

Two dataclasses, no magic. ``RunConfig`` covers the model half, ``RenderConfig``
the picture half; keeping them apart is what lets ``videocortex-spark draw``
re-render a saved prediction without torch anywhere in sight.

Loader defaults are sized for DGX Spark (GB10, 128 GB UMA), not a laptop
and not a discrete-HBM Slurm node. See ``spark.py``.
"""

from __future__ import annotations

import dataclasses
import json
import typing as tp
from pathlib import Path

from videocortex_spark.spark import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FEATURE_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
)

# fsaverage5: 10242 vertices per hemisphere.
FSAVERAGE5_VERTICES_PER_HEMI = 10242
FSAVERAGE5_VERTICES = FSAVERAGE5_VERTICES_PER_HEMI * 2  # 20484

# The four frozen feature extractors TRIBE v2 stacks in front of its fusion
# transformer. Pinned here because the checkpoint config hard-codes device=cuda
# for every one of them, and we have to override each by its exact dotted path.
FEATURE_EXTRACTORS: dict[str, dict[str, str]] = {
    "text": {
        "repo": "meta-llama/Llama-3.2-3B",
        "device_key": "data.text_feature.device",
        "batch_key": "data.text_feature.batch_size",
        "gated": "yes",
        "probe": "config.json",
    },
    "image": {
        "repo": "facebook/dinov2-large",
        "device_key": "data.image_feature.image.device",
        "batch_key": "data.image_feature.image.batch_size",
        "gated": "no",
        "probe": "config.json",
    },
    "audio": {
        "repo": "facebook/w2v-bert-2.0",
        "device_key": "data.audio_feature.device",
        "batch_key": "",
        "gated": "no",
        "probe": "config.json",
    },
    "video": {
        "repo": "facebook/vjepa2-vitg-fpc64-256",
        "device_key": "data.video_feature.image.device",
        "batch_key": "data.video_feature.image.batch_size",
        "gated": "no",
        "probe": "config.json",
    },
}

HF_CHECKPOINT_REPO = "facebook/tribev2"
#: A small file that must be fetchable to prove we really have access.
HF_CHECKPOINT_PROBE = "config.yaml"

# nilearn takes (hemi, view) pairs. These are the eight that are worth looking
# at; "standard" is the set most papers print.
VIEW_PRESETS: dict[str, tuple[tuple[str, str], ...]] = {
    "standard": (
        ("left", "lateral"),
        ("right", "lateral"),
        ("left", "medial"),
        ("right", "medial"),
    ),
    "lateral": (("left", "lateral"), ("right", "lateral")),
    "medial": (("left", "medial"), ("right", "medial")),
    "left": (("left", "lateral"), ("left", "medial")),
    "right": (("right", "lateral"), ("right", "medial")),
    "occipital": (("left", "posterior"), ("right", "posterior")),
    "full": (
        ("left", "lateral"),
        ("right", "lateral"),
        ("left", "medial"),
        ("right", "medial"),
        ("left", "dorsal"),
        ("left", "ventral"),
    ),
}


@dataclasses.dataclass(slots=True)
class RenderConfig:
    """How the (n_timesteps x n_vertices) matrix becomes pictures."""

    views: str = "standard"
    cmap: str = "cold_hot"
    #: Percentile used for the symmetric colour limits. Computed once over the
    #: whole run, never per-frame — per-frame rescaling makes a quiet moment
    #: look identical to a loud one, which is how you lie with a brain map.
    percentile: float = 99.0
    #: Vertices below this fraction of vmax are left transparent.
    threshold_frac: float = 0.25
    #: Soft-threshold ramp width as a fraction of threshold: colour fades in
    #: from ``(1 - ramp_frac) * threshold`` to ``threshold`` instead of popping
    #: on at a hard edge. 0 restores the hard cut.
    ramp_frac: float = 0.5
    dpi: int = 150
    #: Render every Nth TR. 1 = every predicted timepoint.
    stride: int = 1
    #: Cap on frames, so a feature film doesn't emit 4000 PNGs.
    max_frames: int = 60
    contact_sheet: bool = True
    #: None = pick a column count that lands the sheet near 16:9.
    contact_sheet_cols: int | None = None
    darkbg: bool = True
    #: Contact-sheet filmstrip: the stimulus frame above each tile. Needs the
    #: source video; skipped silently without it.
    filmstrip: bool = True
    #: Name the top regions per TR under each plate title (Destrieux).
    #: Skips with a warning if the atlas cannot be fetched.
    regions: bool = True

    def view_pairs(self) -> tuple[tuple[str, str], ...]:
        if self.views not in VIEW_PRESETS:
            raise ValueError(
                f"unknown view preset {self.views!r}; "
                f"choose from {sorted(VIEW_PRESETS)}"
            )
        return VIEW_PRESETS[self.views]


@dataclasses.dataclass(slots=True)
class OverlayConfig:
    """PIP overlay: dense TR cards composited onto the source video.

    Separate from ``RenderConfig`` so the contact-sheet ``max_frames`` cap
    cannot leak in and widen-stride the animation out of sync.
    """

    views: str = "standard"
    cmap: str = "cold_hot"
    percentile: float = 99.0
    threshold_frac: float = 0.25
    #: Soft-threshold ramp: colour fades in linearly from
    #: ``(1 - ramp_frac) * threshold`` up to ``threshold`` instead of popping
    #: on at a hard edge. 0 restores the hard cut.
    ramp_frac: float = 0.5
    darkbg: bool = True
    dpi: int = 90
    #: Fraction of frame *width* on landscape; portrait uses a larger fraction.
    size: float = 0.24
    position: str = "top-right"
    label: str = "time"  # time | tr | none
    lag_mode: str = "stimulus"  # stimulus | scanner
    hemodynamic_offset_s: float = 5.0
    crf: int = 16
    fast: bool = False
    force: bool = False
    stride: int = 1
    #: One joined inflated brain yawing in the PIP. Default overlay stays 2×2.
    spin: bool = False
    #: Yaw rate in degrees per second of *video* time. Clamped 12–48.
    dps: float = 24.0
    fps: float = 24.0
    elev: float = 18.0
    #: Discrete azimuth step for the pose atlas (degrees). 2° + lerp is smooth
    #: at 24 fps; 10° is a slide show.
    az_step: int = 2
    atlas_px: int = 384
    #: Black rounded plate + phosphor-green bezel under the PIP.
    monitor: bool = True
    #: Mean-energy trace with a playhead along the bottom of the PIP — tells
    #: the viewer "the brain went quiet" when the shared colour scale goes dark.
    ribbon: bool = True
    #: Name the top regions per TR in the PIP/card footer (Destrieux).
    #: Skips with a warning if the atlas cannot be fetched.
    regions: bool = True

    def view_pairs(self) -> tuple[tuple[str, str], ...]:
        return RenderConfig(views=self.views).view_pairs()

    def as_render(self) -> RenderConfig:
        return RenderConfig(
            views=self.views,
            cmap=self.cmap,
            percentile=self.percentile,
            threshold_frac=self.threshold_frac,
            ramp_frac=self.ramp_frac,
            dpi=self.dpi,
            darkbg=self.darkbg,
            contact_sheet=False,
            stride=1,
            max_frames=0,
        )


@dataclasses.dataclass(slots=True)
class RunConfig:
    """How the model gets loaded and fed."""

    video: Path | None = None
    audio: Path | None = None
    text: Path | None = None
    out_dir: Path = Path("runs")
    device: str = "auto"
    checkpoint: str = HF_CHECKPOINT_REPO
    cache_dir: Path = Path(".videocortex-spark-cache")
    #: Loader batch size. Upstream ships 8 with num_workers=20, tuned for a
    #: Slurm node with discrete HBM. Spark has 128 GB UMA but only 273 GB/s
    #: of LPDDR5x — 4 / 4 is the compromise that actually saturates GB10
    #: without crowding the OS out of the same memory pool.
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    #: Per-extractor batch size cap, applied to all four.
    feature_batch_size: int = DEFAULT_FEATURE_BATCH_SIZE
    save_predictions: bool = True

    def stimulus(self) -> tuple[str, Path]:
        """Return the single (kind, path) this run is driven by."""
        given = [
            (kind, val)
            for kind, val in (
                ("video", self.video),
                ("audio", self.audio),
                ("text", self.text),
            )
            if val is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "exactly one of --video / --audio / --text is required, "
                f"got {len(given)}"
            )
        return given[0][0], Path(given[0][1])


def write_manifest(
    path: Path,
    *,
    run: RunConfig,
    render: RenderConfig,
    extra: dict[str, tp.Any] | None = None,
) -> Path:
    """Drop a manifest next to the frames so a run can be reproduced."""
    payload: dict[str, tp.Any] = {
        "videocortex_spark_version": _version(),
        "platform": "dgx-spark-gb10",
        "run": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in dataclasses.asdict(run).items()
        },
        "render": dataclasses.asdict(render),
        "model": {
            "checkpoint": run.checkpoint,
            "mesh": "fsaverage5",
            "n_vertices": FSAVERAGE5_VERTICES,
            "feature_extractors": {
                k: v["repo"] for k, v in FEATURE_EXTRACTORS.items()
            },
        },
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _version() -> str:
    from videocortex_spark import __version__

    return __version__
