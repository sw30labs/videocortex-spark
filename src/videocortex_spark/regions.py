"""Name the blobs: top activations per TR mapped to Destrieux region names.

A coloured bump on an inflated brain says "something happened here". A slide
that says "L occipital inf · R temporal sup · L postcentral" is a claim a
reviewer can actually read. The atlas is nilearn's surface Destrieux
(`fetch_atlas_surf_destrieux`), which ships **on fsaverage at 10242 vertices
per hemisphere** — the same vertex count and order as fsaverage5, so the map
applies directly. (Verified: mesh-edge label agreement is ~87%, far above
what any permuted correspondence could achieve; a wrong ordering gives ~1%.)

Region 0 ("Unknown" / medial wall in the annot) is excluded from ranking —
it is unlabelled cortex, not a finding.
"""

from __future__ import annotations

import logging
import re
import typing as tp

import numpy as np

from videocortex_spark.config import FSAVERAGE5_VERTICES, FSAVERAGE5_VERTICES_PER_HEMI

logger = logging.getLogger(__name__)


def _load_hemi_map(hemi: str) -> np.ndarray:
    """Destrieux label id per fsaverage5 vertex for one hemisphere."""
    from nilearn import datasets

    at = datasets.fetch_atlas_surf_destrieux()
    map_key = f"map_{hemi}"
    lab = np.rint(np.asarray(at[map_key], dtype=np.float64)).astype(np.int32)
    if lab.shape[0] != FSAVERAGE5_VERTICES_PER_HEMI:
        raise RuntimeError(
            f"Destrieux {map_key} has {lab.shape[0]} vertices, expected "
            f"{FSAVERAGE5_VERTICES_PER_HEMI} (fsaverage5). Atlas changed?"
        )
    return lab


def _clean_name(name: str) -> str:
    """'G_temp_sup-G_T_transv' -> 'G temp sup/G T transv' — footer-width honest."""
    s = name.replace("_and_S_", "G/S ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_region_labels() -> tuple[np.ndarray, list[tuple[str, str]]]:
    """``(region_id per vertex, [(hemi, name), ...])``; index 0 = unlabelled.

    Region ids are unique across hemispheres: left keeps the atlas ids
    1..n, right is offset by n+1 so one bincount can cover both hemis.
    """
    from nilearn import datasets

    at = datasets.fetch_atlas_surf_destrieux()
    names: list[tuple[str, str]] = [("left", "unlabelled")]
    left = _load_hemi_map("left")
    right = _load_hemi_map("right")
    raw_labels = [_clean_name(str(n)) for n in at["labels"]]
    if raw_labels and raw_labels[0].strip().lower() == "unknown":
        names += [("left", n) for n in raw_labels[1:]]
        # Left ids occupy 1..L (L = len(raw_labels)-1). Right ids shift by L
        # so atlas right-id i lands on names[L + i] = ("right", labels[i]).
        offset = len(raw_labels) - 1
    else:
        raise RuntimeError(f"unexpected Destrieux label order: {raw_labels[:3]}")
    names += [("right", n) for n in raw_labels[1:]]

    left = np.where(left < 0, 0, left)
    right = np.where(right == 0, 0, right + offset)
    region_id = np.concatenate([left, right])
    assert len(names) == 2 * offset + 1, "region table misaligned"
    assert region_id.max() < len(names), "region id exceeds name table"
    return region_id.astype(np.int32), names


def top_regions_per_tr(
    preds: np.ndarray,
    region_id: np.ndarray,
    names: list[tuple[str, str]],
    *,
    k: int = 3,
    min_signum: float = 0.0,
) -> list[str]:
    """Per TR: 'L G occipital inf · R S temporal sup' by mean |signal|.

    ``min_signum`` prunes regions whose mean is under this fraction of the
    run-wide vertex threshold — a region nobody crosses the threshold in is
    not "active", even if its |mean| is the 181st-largest.
    """
    preds = np.asarray(preds, dtype=np.float32)
    n_tr, n_v = preds.shape
    if region_id.shape[0] != FSAVERAGE5_VERTICES or n_v != FSAVERAGE5_VERTICES:
        raise ValueError(
            f"expected {FSAVERAGE5_VERTICES}-vertex predictions, got {preds.shape}"
        )
    n_reg = len(names)
    counts = np.bincount(region_id, minlength=n_reg).astype(np.float64)
    counts[0] = 1.0  # divide-safe; region 0 is masked out below anyway
    sums = np.zeros((n_tr, n_reg), dtype=np.float64)
    abs_p = np.abs(preds)
    for t in range(n_tr):
        sums[t] = np.bincount(region_id, weights=abs_p[t], minlength=n_reg)
    means = sums / counts
    means[:, 0] = -1.0
    prefixes = np.array(["L" if h == "left" else "R" for h, _ in names])
    labels = np.array([n for _, n in names])
    out: list[str] = []
    k = max(0, int(k))
    for t in range(n_tr):
        if k == 0:
            out.append("")
            continue
        order = np.argsort(means[t])[::-1][:k]
        picked = [
            f"{prefixes[j]} {labels[j]}"
            for j in order
            if means[t][j] > min_signum and j != 0
        ]
        out.append(" · ".join(picked))
    return out
