"""Events file: what happened in the clip, and what a human said about it.

Schema ``videocortex.events.v1`` — a small JSON next to the run:

    {
      "schema": "videocortex.events.v1",
      "clip": "counting-task.mp4",
      "instruction": "count the passes",
      "unexpected": [{"label": "walker", "t0": 12.4, "t1": 17.1}],
      "human_report": [{"label": "noticed", "t": 14.0, "note": "optional"}]
    }

``unexpected`` windows are on the **stimulus clock** — the same clock the
overlay's ``--lag-mode stimulus`` shows. ``human_report`` is handwritten by a
person who watched the clip; nothing here is ever inferred from predictions.
If it is absent we draw no ticks — an absent report is not a "didn't notice".

The caption is the whole point of the events file, so it lives here, once,
exactly as it should read on a slide.
"""

from __future__ import annotations

import json
import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = "videocortex.events.v1"

#: The honesty sentence. Printed on the overlay lower-third and the contact
#: sheet footer whenever --events is in play. Do not paraphrase it away.
CAPTION = (
    "TRIBE predicts cortical drive from the pixels. "
    "It does not predict what a viewer counting passes will notice. "
    "Encoding is not attention."
)


class EventsError(ValueError):
    """The events file is missing, malformed, or off-schema. Hard stop."""


@dataclass(frozen=True, slots=True)
class EventWindow:
    label: str
    t0: float
    t1: float


@dataclass(frozen=True, slots=True)
class HumanReport:
    label: str
    t: float
    note: str = ""


@dataclass(frozen=True, slots=True)
class Events:
    clip: str
    instruction: str
    unexpected: tuple[EventWindow, ...]
    human_report: tuple[HumanReport, ...]


def _as_seconds(value: tp.Any, where: str) -> float:
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise EventsError(f"{where}: {value!r} is not a number of seconds") from None
    if not np.isfinite(t) or t < 0:
        raise EventsError(f"{where}: {value!r} is not a sane timestamp")
    return t


def load_events(path: Path) -> Events:
    """Parse and validate. Missing file, bad JSON, wrong schema: all fatal.

    Silently skipping a typo'd path is how a demo ships without its point.
    """
    path = Path(path)
    if not path.is_file():
        raise EventsError(f"events file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EventsError(f"{path}: not JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise EventsError(f"{path}: top level must be an object")
    if data.get("schema") != SCHEMA:
        raise EventsError(
            f"{path}: schema must be {SCHEMA!r}, got {data.get('schema')!r}"
        )

    unexpected: list[EventWindow] = []
    for i, raw in enumerate(data.get("unexpected") or []):
        if not isinstance(raw, dict):
            raise EventsError(f"{path}: unexpected[{i}] must be an object")
        label = str(raw.get("label") or "").strip()
        if not label:
            raise EventsError(f"{path}: unexpected[{i}] needs a label")
        t0 = _as_seconds(raw.get("t0"), f"unexpected[{i}].t0")
        t1 = _as_seconds(raw.get("t1"), f"unexpected[{i}].t1")
        if t1 < t0:
            raise EventsError(f"unexpected[{i}]: t1 ({t1}) before t0 ({t0})")
        unexpected.append(EventWindow(label, t0, t1))

    reports: list[HumanReport] = []
    for i, raw in enumerate(data.get("human_report") or []):
        if not isinstance(raw, dict):
            raise EventsError(f"{path}: human_report[{i}] must be an object")
        label = str(raw.get("label") or "").strip()
        if not label:
            raise EventsError(f"{path}: human_report[{i}] needs a label")
        t = _as_seconds(raw.get("t"), f"human_report[{i}].t")
        reports.append(HumanReport(label, t, str(raw.get("note") or "")))

    if not unexpected and not reports:
        logger.warning("%s: no unexpected windows and no human report", path)
    return Events(
        clip=str(data.get("clip") or ""),
        instruction=str(data.get("instruction") or ""),
        unexpected=tuple(unexpected),
        human_report=tuple(reports),
    )


def caption_lines() -> tuple[str, str, str]:
    """CAPTION split at sentence boundaries for a lower-third."""
    return (
        "TRIBE predicts cortical drive from the pixels.",
        "It does not predict what a viewer counting passes will notice.",
        "Encoding is not attention.",
    )


# Tick strip colours, RGBA. Amber band = the walker was on screen; the pale
# tick = a human said something. Neither is derived from predictions.
_TICK_BAND = np.array([0xF5, 0xB0, 0x41, 0xC0], dtype=np.uint8)
_TICK_MARK = np.array([0xE8, 0xE8, 0xEE, 0xE0], dtype=np.uint8)


def blit_event_ticks(
    ribbon: np.ndarray,
    events: Events,
    *,
    duration_s: float,
    band_px: int = 5,
) -> np.ndarray:
    """Paint a thin event row into the bottom of an energy-ribbon RGBA strip.

    Same mapping as the playhead: x = t / duration across the strip width.
    Unexpected windows become a filled band, human reports a taller tick.
    The ribbon's own scale is untouched — this is annotation, not data.
    """
    img = np.asarray(ribbon)
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"ribbon must be HxWx4, got {img.shape}")
    h, w = img.shape[:2]
    if h < 8 or w < 8 or duration_s <= 0:
        return img
    out = img.copy()
    band_px = max(2, min(int(band_px), h // 4))

    def x_of(t: float) -> int:
        return int(np.clip(round(t / duration_s * (w - 1)), 0, w - 1))

    for win in events.unexpected:
        x0, x1 = x_of(win.t0), x_of(win.t1)
        if x1 <= x0:
            x1 = min(w - 1, x0 + 1)
        out[h - band_px :, x0 : x1 + 1] = _TICK_BAND
    for rep in events.human_report:
        x = x_of(rep.t)
        out[h - 2 * band_px :, max(0, x - 1) : x + 1] = _TICK_MARK
    return out
