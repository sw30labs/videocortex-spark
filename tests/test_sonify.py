"""Sonify: shared scale, silence for silence, clock, lag. No torch, no nilearn
fetch — the region table is synthetic everywhere except the CLI smoke test."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from videocortex_spark.sonify import (
    SR,
    resample_envelopes,
    scale_envelopes,
    sonify_from_run,
    track_envelopes,
    voice_indices,
)

N_VERT = 20484
TR_S = 1.49

# A synthetic Destrieux table: id 0 is the medial wall, then four regions that
# hit each voice's match strings once, both hemispheres where it matters.
NAMES = [
    ("left", "unlabelled"),
    ("left", "S occipital inf"),
    ("right", "G cuneus"),
    ("left", "G oc-temp lat-fusifor"),
    ("right", "G oc-temp med-Parahip"),
    ("left", "G precentral"),  # matches nothing
]


def _region_id() -> np.ndarray:
    rid = np.zeros(N_VERT, dtype=np.int32)
    rid[100:300] = 1    # S occipital inf
    rid[300:500] = 2    # G cuneus
    rid[500:700] = 3    # fusiform
    rid[700:900] = 4    # parahippocampal
    rid[900:1100] = 5   # precentral
    return rid


def _run_dir(tmp_path, preds) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    np.save(run / "predictions.npy", np.asarray(preds, dtype=np.float32))
    np.save(run / "timestamps.npy", np.arange(preds.shape[0]) * TR_S)
    return run


def _read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as fh:
        n_ch, sw, sr = fh.getnchannels(), fh.getsampwidth(), fh.getframerate()
        raw = fh.readframes(fh.getnframes())
    assert sw == 2
    pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, n_ch).astype(np.float32) / 32768.0
    return pcm, sr, n_ch


def test_voice_matcher_hits_expected_regions_and_warns_on_miss(caplog):
    idx = voice_indices(NAMES)
    assert sorted(idx["occipital"]) == [1, 2]       # occipital inf + cuneus
    assert idx["fusiform"] == [3]
    assert idx["parahippocampal"] == [4]
    with caplog.at_level("WARNING"):
        idx2 = voice_indices([("left", "unlabelled"), ("left", "G precentral")])
    assert idx2["occipital"] == []
    assert "voice silent" in caplog.text


def test_track_envelopes_mean_abs_inside_regions():
    preds = np.zeros((4, N_VERT), dtype=np.float32)
    preds[:, 100:500] = 2.0    # occipital inf + cuneus loud — one voice, mean of both
    preds[:, 500:700] = -0.5   # fusiform, sign must not matter
    tracks = track_envelopes(preds, _region_id(), NAMES)
    assert tracks["occipital"].tolist() == [2.0] * 4
    assert tracks["fusiform"][0] == pytest.approx(0.5)
    assert "parahippocampal" in tracks  # matched, near zero
    assert tracks["parahippocampal"][0] == pytest.approx(0.0)
    assert tracks["whole_brain"][0] == pytest.approx((400 * 2.0 + 200 * 0.5) / N_VERT)


def test_zero_predictions_make_a_near_silent_wav(tmp_path):
    run = _run_dir(tmp_path, np.zeros((6, N_VERT), dtype=np.float32))
    res = sonify_from_run(run, duration_s=1.0, labels=(_region_id(), NAMES))
    pcm, sr, n_ch = _read_wav(res.wav)
    assert sr == SR and n_ch == 2
    assert float(np.abs(pcm).max()) < 1e-3


def test_shared_scale_quiet_track_stays_quiet(tmp_path):
    """Loud occipital, 10x quieter fusiform: no per-track normalisation."""
    preds = np.zeros((6, N_VERT), dtype=np.float32)
    preds[:, 100:500] = 1.0    # occipital inf + cuneus, >1% of vertices -> vmax 1.0
    preds[:, 500:700] = 0.1    # fusiform at a tenth
    run = _run_dir(tmp_path, preds)
    res = sonify_from_run(
        run, duration_s=2.0, threshold_frac=0.05, labels=(_region_id(), NAMES)
    )
    z = np.load(res.tracks)
    occ = float(z["occipital"].max())
    fus = float(z["fusiform"].max())
    assert occ == pytest.approx(1.0, abs=0.05)
    # sqrt(0.1) ~ 0.32 — audible, but never pushed up to the loud track
    assert fus < 0.5 * occ
    assert fus > 0.05


def test_threshold_gates_quiet_tracks(tmp_path):
    preds = np.zeros((6, N_VERT), dtype=np.float32)
    preds[:, 100:500] = 1.0
    preds[:, 500:700] = 0.1  # below the default 0.25 gate
    run = _run_dir(tmp_path, preds)
    res = sonify_from_run(run, duration_s=1.0, labels=(_region_id(), NAMES))
    z = np.load(res.tracks)
    assert float(z["fusiform"].max()) == 0.0


def test_duration_matches_request_and_wav_is_stereo_48k(tmp_path):
    run = _run_dir(tmp_path, np.random.default_rng(1).normal(0, 1, (6, N_VERT)))
    res = sonify_from_run(run, duration_s=2.5, labels=(_region_id(), NAMES))
    pcm, sr, n_ch = _read_wav(res.wav)
    assert sr == SR and n_ch == 2
    assert pcm.shape[0] / sr == pytest.approx(2.5, abs=0.05)


def test_scanner_lag_delays_the_envelope_by_five_seconds(tmp_path):
    n_tr = 10
    preds = np.zeros((n_tr, N_VERT), dtype=np.float32)
    preds[3, :] = 1.0  # one loud TR
    run = _run_dir(tmp_path, preds)
    stim = sonify_from_run(
        run, duration_s=20.0, lag_mode="stimulus",
        labels=(_region_id(), NAMES), out=run / "a.wav",
    )
    scan = sonify_from_run(
        run, duration_s=20.0, lag_mode="scanner",
        labels=(_region_id(), NAMES), out=run / "b.wav",
    )
    a = np.load(stim.tracks)["whole_brain"]
    b = np.load(scan.tracks)["whole_brain"]
    from scipy.signal import correlate

    corr = correlate(b - b.mean(), a - a.mean(), mode="full", method="fft")
    lag_s = (int(np.argmax(corr)) - (a.size - 1)) / SR
    assert lag_s == pytest.approx(5.0, abs=0.2)


def test_resample_smooths_the_tr_zipper():
    ts = np.arange(8) * TR_S
    env = (np.arange(8) % 2).astype(np.float32)  # worst-case 0/1 flicker
    times, out = resample_envelopes(
        {"whole_brain": env}, ts, duration_s=8 * TR_S, lag_s=0.0
    )
    y = out["whole_brain"]
    # post-smoothing, adjacent samples never jump the full range
    assert float(np.abs(np.diff(y)).max()) < 0.1


def test_sonify_requires_a_clock(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    np.save(run / "predictions.npy", np.zeros((4, N_VERT), dtype=np.float32))
    from videocortex_spark.sonify import SonifyError

    with pytest.raises(SonifyError, match="timestamps"):
        sonify_from_run(run)


def test_cli_smoke_writes_cortex_wav(tmp_path):
    from videocortex_spark.cli import main

    rng = np.random.default_rng(7)
    run = _run_dir(tmp_path, rng.normal(0, 1, (6, N_VERT)))
    assert main(["sonify", "--run", str(run)]) == 0
    wav = run / "cortex.wav"
    assert wav.is_file()
    pcm, sr, n_ch = _read_wav(wav)
    assert n_ch == 2 and sr == SR
    # duration falls back to last TR + one median TR
    assert pcm.shape[0] / sr == pytest.approx(6 * TR_S, abs=0.05)
    assert (run / "cortex_tracks.npz").is_file()
