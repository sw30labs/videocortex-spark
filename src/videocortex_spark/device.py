"""Device selection for DGX Spark.

Upstream's ``TribeModel.from_pretrained(device="auto")`` resolves to
``"cuda" if torch.cuda.is_available() else "cpu"``. On Spark that is the
right priority — there is no Metal — but the *wrong* CUDA is a silent
disaster: a cu12 / x86 wheel imports, reports no GPU, and you spend an
hour on CPU. This module refuses that, and is written so the decision
logic can be unit-tested without torch installed.
"""

from __future__ import annotations

import platform
import typing as tp
import warnings

from videocortex_spark.spark import (
    SPARK_COMPUTE,
    SPARK_CUDA_MAJOR,
    VALID_DEVICES,
    apply_runtime_env,
    is_aarch64,
    is_gb10_name,
    looks_like_spark,
    query_nvidia_smi,
    uma_report,
)

# Ops that TRIBE / whisperx / Triton touch. Set before torch CUDA init.
_RUNTIME_ENV_APPLIED = False


class DeviceError(RuntimeError):
    """Raised when an explicitly requested device is not available."""


def select_device(
    preferred: str = "auto",
    *,
    has_cuda: bool = False,
) -> str:
    """Pure decision function: given what's available, pick a device.

    >>> select_device("auto", has_cuda=True)
    'cuda'
    >>> select_device("auto", has_cuda=False)
    'cpu'
    """
    if preferred not in VALID_DEVICES:
        raise ValueError(
            f"device must be one of {VALID_DEVICES}, got {preferred!r}"
        )

    if preferred == "auto":
        return "cuda" if has_cuda else "cpu"

    if preferred == "cuda" and not has_cuda:
        raise DeviceError(
            "CUDA was requested but torch reports no CUDA device. "
            "On DGX Spark that almost always means a CPU-only or CUDA-12 "
            "wheel. Reinstall from the cu130 index "
            "(https://download.pytorch.org/whl/cu130) or use the NGC "
            "PyTorch container. Use --device auto to fall back."
        )
    return preferred


def enable_spark_runtime() -> None:
    """Env + warning filters that have to land before the first CUDA op."""
    global _RUNTIME_ENV_APPLIED
    apply_runtime_env()
    if _RUNTIME_ENV_APPLIED:
        return
    # sm_120 cubins JIT to sm_121. The warning is accurate and useless.
    warnings.filterwarnings(
        "ignore",
        message=r".*(cuda capability 12\.1|SM 12\.1|sm_121).*",
    )
    _RUNTIME_ENV_APPLIED = True


def configure_torch_cuda() -> None:
    """TF32 / matmul precision for Blackwell tensor cores. No-op without torch."""
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass
    # Variable-length video / audio chunks: benchmarking the first shape
    # and reusing it for the rest is a slowdown, not a win.
    torch.backends.cudnn.benchmark = False


def probe() -> dict[str, tp.Any]:
    """Report what this machine and torch can see. Never raises."""
    board = query_nvidia_smi() or {}
    uma = uma_report()
    info: dict[str, tp.Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "aarch64": is_aarch64(),
        "torch": None,
        "torch_cuda": None,
        "has_cuda": False,
        "device_name": board.get("name"),
        "capability": board.get("compute_cap"),
        "driver": board.get("driver"),
        "is_gb10": is_gb10_name(board.get("name")),
        "is_spark": looks_like_spark(
            gpu_name=board.get("name"),
            capability=board.get("compute_cap"),
        ),
        "uma": uma,
        "note": None,
    }

    try:
        import torch
    except ImportError:
        info["note"] = (
            "torch is not installed. On Spark: "
            "uv pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu130"
        )
        return info

    info["torch"] = torch.__version__
    info["torch_cuda"] = getattr(torch.version, "cuda", None)
    info["has_cuda"] = bool(torch.cuda.is_available())

    if info["has_cuda"]:
        try:
            info["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        try:
            major, minor = torch.cuda.get_device_capability(0)
            info["capability"] = (int(major), int(minor))
        except Exception:
            pass
        info["is_gb10"] = is_gb10_name(info.get("device_name"))
        info["is_spark"] = looks_like_spark(
            gpu_name=info.get("device_name"),
            capability=info.get("capability"),
        )

        cuda_ver = str(info["torch_cuda"] or "")
        if cuda_ver and not cuda_ver.startswith(str(SPARK_CUDA_MAJOR)):
            info["note"] = (
                f"torch was built against CUDA {cuda_ver}; DGX Spark ships "
                f"CUDA {SPARK_CUDA_MAJOR}. cu12 wheels see libcudart.so.12 "
                "and often report no GPU. Reinstall from the cu130 index."
            )
        cap = info.get("capability")
        if cap == SPARK_COMPUTE:
            pass  # expected
        elif cap is not None and cap[0] >= 12:
            info["note"] = info["note"] or (
                f"GPU compute capability {cap[0]}.{cap[1]} "
                f"(Spark is {SPARK_COMPUTE[0]}.{SPARK_COMPUTE[1]}; "
                "sm_120 cubins are binary-compatible with sm_121)."
            )
    else:
        if is_aarch64() and board.get("name"):
            info["note"] = (
                f"{board.get('name')} is visible to the driver but not to "
                "torch — this is a CPU-only or CUDA-12 wheel. Reinstall "
                "from https://download.pytorch.org/whl/cu130"
            )
        else:
            info["note"] = "torch reports no CUDA device"
    return info


def resolve_device(preferred: str = "auto") -> str:
    """Resolve ``preferred`` against the machine we're actually on."""
    enable_spark_runtime()
    info = probe()
    if info["torch"] is None:
        raise DeviceError(
            "torch is required to run the model. On DGX Spark:\n"
            "    uv pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu130\n"
            "    uv pip install -e '.[predict]'"
        )
    device = select_device(preferred, has_cuda=info["has_cuda"])
    if device == "cuda":
        configure_torch_cuda()
    return device


def describe_device(device: str) -> str:
    """One human-readable line about what we're about to run on."""
    if device == "cuda":
        info = probe()
        name = info.get("device_name") or "CUDA"
        cap = info.get("capability")
        cap_s = f" sm_{cap[0]}{cap[1]}" if cap else ""
        uma = info.get("uma") or {}
        avail = uma.get("available_gb")
        mem_s = f", {avail:.0f} GB UMA free" if isinstance(avail, float) else ""
        return f"cuda ({name}{cap_s}{mem_s})"
    return f"cpu ({platform.machine()}) — expect this to be slow on Spark"
