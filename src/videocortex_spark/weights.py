"""Pre-download everything, so the first real run isn't also the first download.

TRIBE v2's own checkpoint is small (0.7 GB) — it is only the fusion transformer
and the surface head. The bulk is the four frozen encoders it sits on top of,
which HuggingFace pulls lazily the first time you run inference. That is a poor
moment to discover you need to accept a licence.
"""

from __future__ import annotations

import logging
from pathlib import Path

from videocortex_spark.config import FEATURE_EXTRACTORS, HF_CHECKPOINT_REPO

logger = logging.getLogger(__name__)


def fetch_surfaces() -> Path:
    """Load the fsaverage5 meshes and report where they live.

    They ship inside nilearn, so this verifies rather than downloads.
    """
    from videocortex_spark.render import load_fsaverage5

    fs = load_fsaverage5()
    return Path(fs["left"]["mesh"]).parent


def fetch_checkpoint(cache_dir: Path | None = None) -> list[Path]:
    """Download ``config.yaml`` and ``best.ckpt`` for TRIBE v2."""
    from huggingface_hub import hf_hub_download

    out = []
    for name in ("config.yaml", "best.ckpt"):
        p = hf_hub_download(
            HF_CHECKPOINT_REPO,
            name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        logger.info("fetched %s -> %s", name, p)
        out.append(Path(p))
    return out


def fetch_encoders(cache_dir: Path | None = None) -> dict[str, str]:
    """Download the four frozen feature extractors.

    Returns a ``{modality: status}`` map rather than raising, so one gated repo
    doesn't abort the other three.
    """
    from huggingface_hub import snapshot_download

    results: dict[str, str] = {}
    for modality, spec in FEATURE_EXTRACTORS.items():
        repo = spec["repo"]
        try:
            snapshot_download(
                repo,
                cache_dir=str(cache_dir) if cache_dir else None,
                allow_patterns=[
                    "*.json", "*.safetensors", "*.model", "*.txt", "*.bin"
                ],
            )
            results[modality] = "ok"
            logger.info("fetched %s (%s)", repo, modality)
        except Exception as exc:
            results[modality] = f"{type(exc).__name__}: {str(exc)[:120]}"
            logger.warning("could not fetch %s: %s", repo, exc)
    return results
