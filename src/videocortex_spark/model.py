"""Loading TRIBE v2 so that it runs on DGX Spark rather than a Slurm node.

The published checkpoint config was written on a cluster and it shows: every
one of the four frozen feature extractors carries ``device: cuda``, the loader
asks for 20 workers, and ``from_pretrained(device="auto")`` only ever resolves
to CUDA or CPU. CUDA is what we want on GB10 — but the worker count and
feature-extractor batch were sized for discrete HBM, and Llama loads in
fp32 unless we say otherwise. This module rewrites those before the model
is built.
"""

from __future__ import annotations

import logging
import typing as tp

from videocortex_spark.config import FEATURE_EXTRACTORS, RunConfig
from videocortex_spark.device import describe_device, resolve_device
from videocortex_spark.spark import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FEATURE_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
)

logger = logging.getLogger(__name__)


def build_config_overrides(
    device: str,
    *,
    feature_batch_size: int = DEFAULT_FEATURE_BATCH_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict[str, tp.Any]:
    """Dotted-path overrides handed to ``TribeModel.from_pretrained``.

    ``exca.ConfDict`` resolves dotted keys, so this maps onto the nested YAML
    without us having to rebuild the whole tree.

    >>> ov = build_config_overrides("cuda")
    >>> ov["data.text_feature.device"], ov["data.video_feature.image.device"]
    ('cuda', 'cuda')
    """
    overrides: dict[str, tp.Any] = {
        "data.batch_size": batch_size,
        "data.num_workers": num_workers,
    }
    for name, spec in FEATURE_EXTRACTORS.items():
        overrides[spec["device_key"]] = device
        if spec["batch_key"]:
            overrides[spec["batch_key"]] = feature_batch_size
    return overrides


def load_model(run: RunConfig, *, device: str | None = None):
    """Load TRIBE v2 onto the GB10 GPU (or CPU if that's all that's left).

    Returns the upstream ``TribeModel``; we deliberately don't wrap it, so
    anything upstream adds later is reachable without going through us.
    """
    device = device or resolve_device(run.device)
    logger.info("device: %s", describe_device(device))

    try:
        from tribev2.demo_utils import TribeModel
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "tribev2 is not installed. Install the prediction extra:\n"
            "    uv pip install -e '.[predict]'\n"
            f"(underlying error: {exc})"
        ) from exc

    overrides = build_config_overrides(
        device,
        feature_batch_size=run.feature_batch_size,
        batch_size=run.batch_size,
        num_workers=run.num_workers,
    )
    logger.info(
        "overriding %d checkpoint config keys (feature extractors -> %s, "
        "batch=%s workers=%s)",
        len(overrides),
        device,
        run.batch_size,
        run.num_workers,
    )

    if device == "cuda":
        from videocortex_spark.patches import llama_cuda_sdpa

        llama_cuda_sdpa()

    checkpoint = run.checkpoint
    if not isinstance(checkpoint, (str,)):
        raise TypeError(
            f"checkpoint must be a repo id or path string, got {type(checkpoint).__name__}"
        )

    model = TribeModel.from_pretrained(
        checkpoint,
        cache_folder=str(run.cache_dir),
        device=device,
        config_update=overrides,
    )
    logger.info("model loaded (TR=%.3gs)", getattr(model.data, "TR", float("nan")))
    return model
