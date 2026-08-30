"""Events file: schema, hard errors, caption, ribbon ticks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from videocortex_spark.events import (
    CAPTION,
    EventsError,
    blit_event_ticks,
    caption_lines,
    load_events,
)

GOOD = {
    "schema": "videocortex.events.v1",
    "clip": "counting-task.mp4",
    "instruction": "count the passes",
    "unexpected": [{"label": "walker", "t0": 12.4, "t1": 17.1}],
    "human_report": [{"label": "noticed", "t": 14.0, "note": "saw it"}],
}


def _write(tmp_path, payload) -> "Path":
    from pathlib import Path

    p = tmp_path / "events.json"
    p.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return p


def test_load_events_roundtrip(tmp_path):
    ev = load_events(_write(tmp_path, GOOD))
    assert ev.clip == "counting-task.mp4"
    assert ev.instruction == "count the passes"
    assert ev.unexpected[0].label == "walker"
    assert ev.unexpected[0].t0 == 12.4
    assert ev.human_report[0].t == 14.0


def test_human_report_is_optional_and_never_invented(tmp_path):
    payload = {k: v for k, v in GOOD.items() if k != "human_report"}
    ev = load_events(_write(tmp_path, payload))
    assert ev.human_report == ()
    assert ev.unexpected  # windows still there


def test_missing_file_is_a_hard_error(tmp_path):
    with pytest.raises(EventsError, match="not found"):
        load_events(tmp_path / "nope.json")


def test_malformed_json_is_a_hard_error(tmp_path):
    with pytest.raises(EventsError, match="not JSON"):
        load_events(_write(tmp_path, "{not json"))


def test_wrong_schema_is_a_hard_error(tmp_path):
    payload = dict(GOOD, schema="something-else")
    with pytest.raises(EventsError, match="schema"):
        load_events(_write(tmp_path, payload))


def test_backwards_window_is_a_hard_error(tmp_path):
    payload = dict(GOOD, unexpected=[{"label": "x", "t0": 5.0, "t1": 2.0}])
    with pytest.raises(EventsError, match="before t0"):
        load_events(_write(tmp_path, payload))


def test_unlabelled_entries_are_a_hard_error(tmp_path):
    payload = dict(GOOD, unexpected=[{"t0": 1.0, "t1": 2.0}])
    with pytest.raises(EventsError, match="label"):
        load_events(_write(tmp_path, payload))


def test_caption_lines_preserve_the_exact_sentence():
    assert " ".join(caption_lines()) == CAPTION
    assert CAPTION.endswith("Encoding is not attention.")


def _ribbon(w=200, h=40):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[: h // 2, :, :3] = 40  # pretend there is a curve above
    img[: h // 2, :, 3] = 120
    return img


def test_blit_event_ticks_band_and_report():
    from videocortex_spark.events import Events, EventWindow, HumanReport

    ev = Events(
        clip="c",
        instruction="",
        unexpected=(EventWindow("walker", 10.0, 20.0),),
        human_report=(HumanReport("noticed", 15.0),),
    )
    out = blit_event_ticks(_ribbon(), ev, duration_s=40.0)
    # band spans t 10..20 s on a 40 s clip -> x 50..100 of 200
    band = out[-3, 75]
    assert band[3] > 150 and band[0] > 200  # amber, opaque-ish
    assert out[-3, 75, 1] > 120
    # outside the window the bottom row is untouched
    assert out[-3, 10, 3] == 0
    assert out[-3, 150, 3] == 0
    # human report tick at t=15 -> x=75, taller than the band
    tick = out[-9, 75]
    assert tick[3] > 150 and tick[0] > 200 and tick[1] > 200


def test_blit_event_ticks_empty_events_leave_strip_untouched():
    from videocortex_spark.events import Events

    ev = Events(clip="c", instruction="", unexpected=(), human_report=())
    ribbon = _ribbon()
    out = blit_event_ticks(ribbon, ev, duration_s=40.0)
    assert np.array_equal(out, ribbon)
