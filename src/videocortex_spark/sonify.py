"""Sonify a run: |predicted BOLD| mapped onto amplitude, per region.

Eagleman's line run backwards — the brain does not care which channel, so put
occupancy on the audio channel. This is **not** "what the brain sounds like",
not a decoder, not EEG. It is the same numbers the plates draw, re-expressed
as loudness.

The honesty rule is the colour map's rule: ``compute_limits`` once over the
whole run, every track divided by that one vmax, gated at that one threshold.
No per-track normalisation — a quiet fusiform stays quieter than occipital,
the way it stays dimmer on the plates.

Region voices are Destrieux stand-ins, not a localizer: "fusiform" means the
fusiform gyrus labels of the atlas, which overlap but are not FFA. Names are
matched as case-insensitive substrings of the cleaned Destrieux table.

No torch, no audio framework: numpy + stdlib ``wave``, 48 kHz stereo PCM.
"""

from __future__ import annotations

import logging
import typing as tp
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from videocortex_spark.render import compute_limits, energy_curve

logger = logging.getLogger(__name__)

SR = 48000

#: Smoothing on the audio clock so the 1.49 s TR zipper is not audible.
SMOOTH_S = 0.05

#: Second harmonic on each carrier — a naked sine at 196 Hz is a tuning fork,
#: not a voice. Kept well under the fundamental so the carrier still names it.
HARMONIC_GAIN = 0.35

#: Relative bed/voice level into the mixer. The sum can reach ~2, the soft
#: clip takes it — but a quiet run must stay quiet, so these are constants,
#: not a normaliser.
VOICE_GAIN = 0.6
BED_GAIN = 0.4
#: Fixed output level after the soft clip. Never recomputed per run.
OUT_LEVEL = 0.8


@dataclass(frozen=True, slots=True)
class Voice:
    name: str
    carrier_hz: float
    pan: float  # -1 hard left … +1 hard right
    match: tuple[str, ...]  # substrings of cleaned Destrieux names, lowercased


#: G3, D4, A4 — a plain triad, far enough apart to tell apart by ear.
#: Whole-brain bed sits an octave below occipital, centre.
VOICES: tuple[Voice, ...] = (
    Voice("occipital", 196.0, -0.7, ("occipital", "cuneus", "pole occipital")),
    Voice("fusiform", 294.0, 0.0, ("fusifor", "oc-temp lat")),
    Voice("parahippocampal", 440.0, 0.7, ("parahip", "oc-temp med")),
)
BED_CARRIER_HZ = 98.0


class SonifyError(RuntimeError):
    """The run cannot be turned into audio (no predictions, no clock)."""


class SonifyResult(tp.NamedTuple):
    wav: Path
    tracks: Path
    duration: float
    voices: tuple[str, ...]  # voices that matched at least one region


def voice_indices(names: list[tuple[str, str]]) -> dict[str, list[int]]:
    """Voice name -> indices into the region table. Empty list = silent voice.

    Pure function of the names table — tests feed it a synthetic atlas, no
    nilearn fetch required.
    """
    out: dict[str, list[int]] = {}
    for voice in VOICES:
        hits = [
            i
            for i, (_, region) in enumerate(names)
            if i != 0  # region 0 is the medial wall, not a finding
            and any(sub in region.lower() for sub in voice.match)
        ]
        if not hits:
            logger.warning(
                "sonify: no Destrieux region matches %s (%s) — voice silent",
                voice.name,
                ", ".join(voice.match),
            )
        out[voice.name] = hits
    return out


def track_envelopes(
    preds: np.ndarray,
    region_id: np.ndarray | None,
    names: list[tuple[str, str]] | None,
) -> dict[str, np.ndarray]:
    """Per-TR mean |signal| for the whole brain and each matched voice."""
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim != 2:
        raise SonifyError(f"predictions must be 2-D, got {preds.shape}")
    tracks: dict[str, np.ndarray] = {"whole_brain": energy_curve(preds)}
    if region_id is None or names is None:
        return tracks
    region_id = np.asarray(region_id)
    if region_id.shape[0] != preds.shape[1]:
        raise SonifyError(
            f"region table has {region_id.shape[0]} vertices, "
            f"predictions have {preds.shape[1]}"
        )
    abs_p = np.abs(preds.astype(np.float64))
    for name, idx in voice_indices(names).items():
        if not idx:
            continue
        mask = np.isin(region_id, np.asarray(idx))
        tracks[name] = abs_p[:, mask].mean(axis=1).astype(np.float32)
    return tracks


def scale_envelopes(
    tracks: dict[str, np.ndarray], *, vmax: float, threshold: float
) -> dict[str, np.ndarray]:
    """sqrt(clip(env / vmax, 0, 1)), gated below threshold. One shared scale."""
    out: dict[str, np.ndarray] = {}
    for name, env in tracks.items():
        amp = np.sqrt(np.clip(env / vmax, 0.0, 1.0))
        amp = np.where(env < threshold, 0.0, amp)
        out[name] = amp.astype(np.float32)
    return out


def resample_envelopes(
    tracks: dict[str, np.ndarray],
    timestamps: np.ndarray,
    *,
    duration_s: float,
    lag_s: float = 0.0,
    sr: int = SR,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """TR-rate envelopes -> audio clock: linear interp, then a 50 ms smooth.

    ``lag_s`` is the scanner-mode haemodynamic delay: the same envelope,
    shifted later. Nothing invents a third clock.
    """
    from scipy.ndimage import uniform_filter1d

    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1) + float(lag_s)
    n = max(1, int(round(duration_s * sr)))
    times = np.arange(n, dtype=np.float64) / sr
    win = max(1, int(round(SMOOTH_S * sr)))
    out: dict[str, np.ndarray] = {}
    for name, env in tracks.items():
        # Before the first (lagged) TR there is no prediction — silence, not a
        # repeat of the first sample. After the last, hold: the BOLD response
        # to the final frame does not vanish at EOF.
        y = np.interp(times, ts, np.asarray(env, dtype=np.float64), left=0.0)
        out[name] = uniform_filter1d(y, size=win, mode="nearest").astype(np.float32)
    return times, out


def _oscillator(times: np.ndarray, carrier_hz: float) -> np.ndarray:
    ph = 2.0 * np.pi * carrier_hz * times
    return np.sin(ph) + HARMONIC_GAIN * np.sin(2.0 * ph)


def _pan_gains(pan: float) -> tuple[float, float]:
    """Constant-power pan; -1 left, +1 right."""
    ang = (float(pan) + 1.0) * np.pi / 4.0
    return float(np.cos(ang)), float(np.sin(ang))


def synth_pcm(
    times: np.ndarray,
    tracks: dict[str, np.ndarray],
) -> np.ndarray:
    """Envelopes -> stereo float PCM in [-1, 1]. Amplitude-modulated carriers,
    soft-clipped sum. Fixed gains throughout — the scale is the run's, set by
    ``scale_envelopes`` upstream."""
    n = times.shape[0]
    mix = np.zeros((n, 2), dtype=np.float64)
    for name, env in tracks.items():
        if name == "whole_brain":
            carrier, pan, gain = BED_CARRIER_HZ, 0.0, BED_GAIN
        else:
            voice = next(v for v in VOICES if v.name == name)
            carrier, pan, gain = voice.carrier_hz, voice.pan, VOICE_GAIN
        gl, gr = _pan_gains(pan)
        sig = _oscillator(times, carrier) * np.asarray(env, dtype=np.float64) * gain
        mix[:, 0] += gl * sig
        mix[:, 1] += gr * sig
    pcm = np.tanh(mix) * OUT_LEVEL
    return pcm.astype(np.float32)


def write_wav(path: Path, pcm: np.ndarray, *, sr: int = SR) -> Path:
    """Stereo 16-bit PCM via stdlib wave. No encoder, no dependency."""
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.ndim != 2 or pcm.shape[1] != 2:
        raise SonifyError(f"pcm must be (n, 2), got {pcm.shape}")
    frames = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(frames.tobytes())
    return path


def _resolve_duration(
    timestamps: np.ndarray,
    *,
    video: Path | None,
    duration_s: float | None,
) -> float:
    """Video duration when known, else last TR + one median TR."""
    if duration_s is not None:
        return float(duration_s)
    if video is not None:
        try:
            from videocortex_spark.overlay import probe_video

            return float(probe_video(video)["duration"])
        except Exception as exc:  # noqa: BLE001 — sonify must not need ffprobe
            logger.warning("could not probe %s (%s) — falling back to the TR clock", video, exc)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        raise SonifyError("timestamps.npy is empty — no clock to sonify against")
    tr = float(np.median(np.diff(ts))) if ts.size >= 2 else 1.49
    return float(ts[-1]) + max(tr, 1e-3)


def sonify_from_run(
    run_dir: Path,
    *,
    video: Path | None = None,
    duration_s: float | None = None,
    lag_mode: str = "stimulus",
    hemodynamic_offset_s: float = 5.0,
    percentile: float = 99.0,
    threshold_frac: float = 0.25,
    out: Path | None = None,
    labels: tuple[np.ndarray, list[tuple[str, str]]] | None = None,
) -> SonifyResult:
    """Write ``cortex.wav`` + ``cortex_tracks.npz`` for a saved run.

    ``labels`` injects a region table (tests); None fetches Destrieux via
    nilearn. An unreachable atlas is a warning and a whole-brain-only bed,
    never a crash.
    """
    run_dir = Path(run_dir)
    pred_path = run_dir / "predictions.npy"
    ts_path = run_dir / "timestamps.npy"
    if not pred_path.is_file():
        raise SonifyError(f"no predictions.npy in {run_dir}")
    if not ts_path.is_file():
        raise SonifyError(
            f"no timestamps.npy in {run_dir} — sonify needs the TR clock"
        )
    preds = np.load(pred_path)
    timestamps = np.load(ts_path).astype(np.float64).reshape(-1)
    if preds.ndim != 2 or timestamps.shape[0] != preds.shape[0]:
        raise SonifyError(
            f"timestamps ({timestamps.shape[0]}) do not match predictions "
            f"({preds.shape[0]} TRs)"
        )

    if labels is None:
        try:
            from videocortex_spark.regions import load_region_labels

            labels = load_region_labels()
        except Exception as exc:  # noqa: BLE001 — offline must not kill audio
            logger.warning("region atlas unavailable (%s) — whole-brain bed only", exc)
    region_id, names = labels if labels is not None else (None, None)

    if lag_mode not in ("stimulus", "scanner"):
        raise SonifyError(f"unknown lag mode {lag_mode!r}")
    lag_s = hemodynamic_offset_s if lag_mode == "scanner" else 0.0

    # One scale for the whole run — the same limits the plates use.
    vmax, threshold = compute_limits(preds, percentile, threshold_frac)
    tracks = track_envelopes(preds, region_id, names)
    scaled = scale_envelopes(tracks, vmax=vmax, threshold=threshold)

    duration = _resolve_duration(timestamps, video=video, duration_s=duration_s)
    times, audio = resample_envelopes(scaled, timestamps, duration_s=duration, lag_s=lag_s)
    pcm = synth_pcm(times, audio)

    wav_path = Path(out) if out is not None else run_dir / "cortex.wav"
    write_wav(wav_path, pcm)
    tracks_path = wav_path.with_name(wav_path.stem + "_tracks.npz")
    np.savez(
        tracks_path,
        sr=SR,
        duration=duration,
        lag_s=lag_s,
        times=times.astype(np.float32),
        **{k: v.astype(np.float32) for k, v in audio.items()},
    )
    voices = tuple(k for k in audio if k != "whole_brain")
    logger.info(
        "sonify -> %s (%.1fs, %s, voices: %s)",
        wav_path, duration, lag_mode, ", ".join(voices) or "whole-brain only",
    )
    return SonifyResult(wav=wav_path, tracks=tracks_path, duration=duration, voices=voices)
