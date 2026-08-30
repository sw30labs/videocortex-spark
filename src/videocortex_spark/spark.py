"""DGX Spark / GB10 facts used by the rest of the package.

The original ``videocortex`` tree is a laptop port: Metal, batch size 1,
VideoToolbox. This module is the Spark equivalent — Grace Blackwell, CUDA 13,
compute capability 12.1, 128 GB of coherent unified memory. Nothing here
imports torch; the decision logic is unit-testable on a machine that has
never seen a GPU driver.

GB10 notes worth keeping next to the code:

* The GPU is an iGPU. There is no discrete VRAM carve-out. ``nvidia-smi``
  prints ``Memory-Usage: Not Supported``; ``cudaMemGetInfo`` under-reports
  because it ignores pages the CPU could reclaim. Read ``/proc/meminfo``.
* sm_121 cubins are binary-compatible with sm_120. Official PyTorch wheels
  target 12.0+PTX; the "unsupported cuda capability 12.1" warning is noise.
* Triton's bundled ``ptxas`` is often CUDA 12.8 and dies on ``sm_121a``.
  Point it at the toolkit: ``TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas``.
* flash-attn wheels pull ``libcudart.so.12``. Do not install them. SDPA on
  Blackwell is the right attention path.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import typing as tp
from pathlib import Path

# GB10 / DGX Spark
SPARK_COMPUTE = (12, 1)
SPARK_ARCH = "sm_121"
SPARK_CUDA_MAJOR = 13
SPARK_CPU_CORES = 20  # 10x Cortex-X925 + 10x Cortex-A725
SPARK_UNIFIED_GIB = 128
SPARK_MEMORY_BANDWIDTH_GBS = 273

# Loader defaults. Upstream ships batch_size=8 / num_workers=20 for a Slurm
# node with discrete HBM. The Mac port dropped both to 1 / 0. Spark has the
# memory for a real batch, but LPDDR5x is 273 GB/s — not HBM — and the four
# frozen extractors (V-JEPA2 ViT-g + Llama 3.2 3B + DINOv2-L + w2v-BERT)
# sit in the same 128 GB as the OS, so we stay well under the cluster
# numbers. Hybrid Arm: don't spawn one worker per core.
DEFAULT_BATCH_SIZE = 4
DEFAULT_FEATURE_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 4

VALID_DEVICES = ("auto", "cuda", "cpu")

PTXAS_CANDIDATES = (
    "/usr/local/cuda/bin/ptxas",
    "/usr/bin/ptxas",
)

TORCH_CU130_INDEX = "https://download.pytorch.org/whl/cu130"
NGC_PYTORCH_IMAGE = "nvcr.io/nvidia/pytorch:25.12-py3"


def is_aarch64(machine: str | None = None) -> bool:
    m = (machine or platform.machine()).lower()
    return m in {"aarch64", "arm64"}


def is_gb10_name(name: str | None) -> bool:
    if not name:
        return False
    n = name.lower()
    return "gb10" in n or "dgx spark" in n


def looks_like_spark(
    *,
    machine: str | None = None,
    gpu_name: str | None = None,
    capability: tuple[int, int] | None = None,
) -> bool:
    if is_gb10_name(gpu_name):
        return True
    if capability == SPARK_COMPUTE and is_aarch64(machine):
        return True
    return False


def find_ptxas() -> Path | None:
    env = os.environ.get("TRITON_PTXAS_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for raw in PTXAS_CANDIDATES:
        p = Path(raw)
        if p.is_file():
            return p
    which = shutil.which("ptxas")
    return Path(which) if which else None


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].lstrip()
    if "=" not in line:
        return None
    key, _, val = line.partition("=")
    key = key.strip()
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        val = val[1:-1]
    if not key:
        return None
    return key, val


def _dotenv_candidates(start: Path | None = None) -> list[Path]:
    """cwd → parents (stop at checkout root), then this package's repo root."""
    out: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        try:
            r = p.resolve()
        except OSError:
            return
        if r in seen:
            return
        seen.add(r)
        out.append(p)

    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        add(p / ".env")
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            break
    try:
        add(Path(__file__).resolve().parents[2] / ".env")
    except IndexError:
        pass
    return out


def load_local_env(start: Path | None = None) -> Path | None:
    """Pull KEY=VAL from a checkout ``.env`` into ``os.environ`` (setdefault).

    huggingface_hub reads ``HF_TOKEN`` from the environment. A local ``.env``
    is the usual place to put it; we do not take a python-dotenv dependency
    for one file. Existing env vars win. Returns the file applied, or None.
    """
    for path in _dotenv_candidates(start):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            parsed = _parse_env_line(raw)
            if parsed is None:
                continue
            key, val = parsed
            os.environ.setdefault(key, val)
        return path
    return None


def apply_runtime_env() -> dict[str, str]:
    """Set the env vars Spark workloads need, without clobbering a user pin.

    Must run before Triton / torch CUDA lazy-init. Safe to call more than once.
    """
    load_local_env()
    applied: dict[str, str] = {}
    ptxas = find_ptxas()
    if ptxas is not None:
        applied["TRITON_PTXAS_PATH"] = os.environ.setdefault(
            "TRITON_PTXAS_PATH", str(ptxas)
        )
    applied["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    applied["CUDA_MODULE_LOADING"] = os.environ.setdefault(
        "CUDA_MODULE_LOADING", "LAZY"
    )
    # torch 2.6+ weights_only default breaks pyannote VAD pickles (whisperx).
    applied["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = os.environ.setdefault(
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1"
    )
    return applied


def read_meminfo(text: str | None = None) -> dict[str, int]:
    """Parse ``/proc/meminfo`` into a kib-keyed dict. UMA-safe.

    On DGX Spark the allocatable GPU pool is "whatever the CPU can give
    back", which is closer to ``MemAvailable`` (+ ``SwapFree`` if you are
    willing to swap) than to ``cudaMemGetInfo``.
    """
    raw = text
    if raw is None:
        try:
            raw = Path("/proc/meminfo").read_text(encoding="utf-8")
        except OSError:
            return {}
    out: dict[str, int] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        bits = rest.split()
        if not bits:
            continue
        try:
            kib = int(bits[0])
        except ValueError:
            continue
        out[key] = kib
    return out


def uma_report(mem: dict[str, int] | None = None) -> dict[str, tp.Any]:
    info = mem if mem is not None else read_meminfo()
    def _gb(key: str) -> float | None:
        kib = info.get(key)
        if kib is None:
            return None
        return kib / (1024 * 1024)

    total = _gb("MemTotal")
    available = _gb("MemAvailable")
    swap_free = _gb("SwapFree")
    return {
        "total_gb": total,
        "available_gb": available,
        "swap_free_gb": swap_free,
        "allocatable_no_swap_gb": available,
        "allocatable_with_swap_gb": (
            None if available is None or swap_free is None else available + swap_free
        ),
    }


def query_nvidia_smi() -> dict[str, tp.Any] | None:
    """Board identity via nvidia-smi. Memory fields are intentionally ignored."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    cmd = [
        exe,
        "--query-gpu=name,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return None
    line = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    cap = None
    if "." in parts[1]:
        try:
            major, minor = parts[1].split(".", 1)
            cap = (int(major), int(minor))
        except ValueError:
            cap = None
    return {
        "name": parts[0],
        "compute_cap": cap,
        "driver": parts[2],
    }


def ffmpeg_has_encoder(name: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return False
    blob = (proc.stdout or "") + (proc.stderr or "")
    # Match a whole encoder token, not a substring of a description.
    for line in blob.splitlines():
        cols = line.split()
        if len(cols) >= 2 and cols[1] == name:
            return True
    return False


def nvenc_h264_available() -> bool:
    return ffmpeg_has_encoder("h264_nvenc")


def nvenc_encode_args(*, bitrate: str = "8M") -> list[str]:
    """GB10 has 1x NVENC. p4 / hq is the quality/throughput compromise."""
    return [
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", "19",
        "-b:v", bitrate,
        "-maxrate", "12M",
        "-bufsize", "24M",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
    ]


def x264_encode_args(*, crf: int = 16) -> list[str]:
    return [
        "-c:v", "libx264",
        "-crf", str(int(crf)),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
    ]


def video_encode_args(*, fast: bool, crf: int = 16) -> list[str]:
    """``fast`` means NVENC when the encoder exists, else libx264."""
    if fast and nvenc_h264_available():
        return nvenc_encode_args()
    return x264_encode_args(crf=crf)
