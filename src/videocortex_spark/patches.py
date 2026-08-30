"""Surgical fixes for running TRIBE v2 on DGX Spark.

Everything here is a workaround for an upstream assumption that the machine
is an x86 CUDA-12 box, or for a Spark-specific sharp edge (Triton ptxas,
pyannote pickles, uvx following Python 3.14). Each patch names the upstream
code it compensates for so that when Meta / PyTorch / CTranslate2 catch up,
the corresponding patch can be deleted rather than quietly rotting.

Written against facebookresearch/tribev2 @ main and CUDA 13.0 on GB10.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import typing as tp
from pathlib import Path

from videocortex_spark.spark import apply_runtime_env, find_ptxas, is_aarch64

logger = logging.getLogger(__name__)

_LLAMA_CUDA_PATCHED = False

# faster-whisper refuses float16 on CPU. CUDA (including GB10) wants float16.
_CPU_COMPUTE_TYPE = "int8"

# uvx without a pin follows the newest Python on PATH. DGX OS currently
# ships a 3.14 stub next to 3.12; pyannote and whisperx die there. Cap at 3.12.
# torch 2.6 flipped torch.load to weights_only=True. pyannote VAD checkpoints
# pickle omegaconf.ListConfig, which that unpickler rejects.
_WHISPERX_TORCH_ENV = {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}

# uv's torch selector. "cu130" is the Spark wheel; without it, uvx whisperx
# resolves torch from PyPI (2.8+cpu on aarch64) and torchcodec then dies
# looking for libc10_cuda.so. CTranslate2 is a separate problem — see
# _uvx_whisperx_force_cpu.
_WHISPERX_TORCH_BACKEND_CUDA = "cu130"
_WHISPERX_TORCH_BACKEND_CPU = "cpu"

# Upstream hard-codes --batch_size 16. GB10's 128 GB UMA can carry more;
# 32 is the value that still fits comfortably next to the four extractors.
_WHISPERX_CUDA_BATCH = "32"


def _whisperx_python() -> str:
    """Interpreter for the uvx whisperx env. Cap at 3.12; 3.13/3.14 break pyannote."""
    v = sys.version_info
    if (v.major, v.minor) >= (3, 13):
        return "3.12"
    if (v.major, v.minor) < (3, 11):
        return "3.11"
    return f"{v.major}.{v.minor}"


@contextlib.contextmanager
def whisperx_compat() -> tp.Iterator[None]:
    """Rewrite the whisperx command line at the subprocess boundary.

    ``tribev2.eventstransforms.ExtractWordsFromAudio`` builds a fixed command::

        uvx whisperx ... --device {cuda|cpu} --compute_type float16 ...

    On Spark the CUDA path is the one we want, but five things still bite:

    1. ``uvx`` with no ``--python`` follows 3.14 on a stock DGX OS image.
    2. Triton's bundled ``ptxas`` does not know ``sm_121a``.
    3. torch 2.6 ``weights_only=True`` cannot unpickle pyannote's VAD checkpoint.
    4. ``uvx`` without ``--torch-backend`` pulls PyPI torch (CPU on aarch64).
    5. CTranslate2's aarch64 manylinux wheels are CPU-only. Parent torch has
       GB10, so upstream passes ``--device cuda`` and ``WhisperModel`` dies.

    We intercept ``subprocess.run`` so transcript parsing stays upstream's.
    """
    import subprocess

    original_run = subprocess.run
    apply_runtime_env()

    def patched_run(cmd, *args, **kwargs):
        if _is_uvx_whisperx(cmd):
            cmd = _rewrite_uvx_whisperx(list(cmd))
            env = kwargs.get("env")
            env = dict(env) if env is not None else os.environ.copy()
            for key, value in _WHISPERX_TORCH_ENV.items():
                env.setdefault(key, value)
            ptxas = find_ptxas()
            if ptxas is not None:
                env.setdefault("TRITON_PTXAS_PATH", str(ptxas))
            kwargs["env"] = env
        return original_run(cmd, *args, **kwargs)

    subprocess.run = patched_run
    try:
        yield
    finally:
        subprocess.run = original_run


# Old name kept as an alias so a stray import from the Mac tree still works.
whisperx_cpu_compat = whisperx_compat


def _is_uvx_whisperx(cmd: tp.Any) -> bool:
    if not isinstance(cmd, (list, tuple)) or len(cmd) < 2:
        return False
    prog = Path(str(cmd[0])).name
    return prog in {"uvx", "uv"} and "whisperx" in cmd


def _whisperx_device(cmd: list[str]) -> str | None:
    try:
        return cmd[cmd.index("--device") + 1]
    except (ValueError, IndexError):
        return None


def _uvx_whisperx_force_cpu() -> bool:
    """CTranslate2 has no CUDA wheel on aarch64. Parent CUDA does not transfer.

    PyPI's manylinux aarch64 ctranslate2 is ~17 MB and links libgomp, not
    libcudart. The tribev2 process sees GB10 and passes ``--device cuda``;
    the isolated uvx env then raises ``CTranslate2 package was not compiled
    with CUDA support``. CPU int8 on 20 Grace cores is the working path
    until OpenNMT ships a CUDA 13 aarch64 wheel.
    """
    return is_aarch64()


def _rewrite_uvx_whisperx(cmd: list[str]) -> list[str]:
    tool = cmd.index("whisperx")
    prefix, rest = cmd[:tool], cmd[tool:]

    # Device lives on whisperx's CLI (rest). Settle it before picking
    # uvx's torch backend — otherwise we download 400 MB of cu130 into
    # an env that is about to run --device cpu.
    if _whisperx_device(rest) == "cuda" and _uvx_whisperx_force_cpu():
        i = rest.index("--device")
        rest[i + 1] = "cpu"
        logger.info(
            "whisperx via uvx: CTranslate2 aarch64 wheels are CPU-only; "
            "--device cuda -> cpu"
        )

    cuda = _whisperx_device(rest) == "cuda"
    inserts: list[str] = []
    if "--python" not in prefix:
        inserts += ["--python", _whisperx_python()]
        logger.info("whisperx via uvx: pinning python %s", _whisperx_python())
    if "--torch-backend" not in prefix:
        backend = (
            _WHISPERX_TORCH_BACKEND_CUDA if cuda else _WHISPERX_TORCH_BACKEND_CPU
        )
        inserts += ["--torch-backend", backend]
        logger.info("whisperx via uvx: --torch-backend %s", backend)

    cmd = prefix + inserts + rest

    if (not cuda) and "--compute_type" in cmd:
        i = cmd.index("--compute_type")
        if cmd[i + 1] != _CPU_COMPUTE_TYPE:
            logger.info(
                "whisperx on CPU: rewriting --compute_type %s -> %s",
                cmd[i + 1],
                _CPU_COMPUTE_TYPE,
            )
            cmd[i + 1] = _CPU_COMPUTE_TYPE

    if cuda and "--batch_size" in cmd:
        i = cmd.index("--batch_size")
        try:
            current = int(cmd[i + 1])
        except (ValueError, IndexError):
            current = 0
        if 0 < current < int(_WHISPERX_CUDA_BATCH):
            logger.info(
                "whisperx on CUDA: --batch_size %s -> %s (GB10 UMA)",
                cmd[i + 1],
                _WHISPERX_CUDA_BATCH,
            )
            cmd[i + 1] = _WHISPERX_CUDA_BATCH
    return cmd


def _is_cpu_whisperx(cmd: tp.Any) -> bool:
    if not _is_uvx_whisperx(cmd):
        return False
    if "--compute_type" not in cmd:
        return False
    return _whisperx_device(list(cmd)) != "cuda"


def cuda_text_from_pretrained_kwargs(device: str, kwargs: dict) -> dict:
    """Llama 3.2 on Blackwell: SDPA + bfloat16.

    The Mac port has to force eager + float32 because Metal's fused SDPA
    aborts on GQA (24 q-heads, 8 kv-heads). CUDA is fine. bf16 is the
    dtype 5th-gen tensor cores actually want; leaving the extractor in
    fp32 on a 3B LM is leaving a lot of GB10 on the table.
    """
    if device != "cuda":
        return kwargs
    import torch

    out = dict(kwargs)
    out.setdefault("attn_implementation", "sdpa")
    # bfloat16 if this torch build has it (all cu130 wheels do).
    dtype = getattr(torch, "bfloat16", None) or torch.float16
    out.setdefault("torch_dtype", dtype)
    return out


def llama_cuda_sdpa() -> None:
    """Force HuggingFaceText on CUDA onto SDPA + bfloat16.

    Idempotent. neuralset's ``_load_model`` otherwise does
    ``AutoModel.from_pretrained(name)`` then ``.to("cuda")``, which
    lands in whatever dtype the checkpoint shipped — often fp32.
    """
    global _LLAMA_CUDA_PATCHED
    if _LLAMA_CUDA_PATCHED:
        return
    from neuralset.extractors.text import HuggingFaceText

    original = HuggingFaceText._load_model

    def _load_model(self, **kwargs):
        kwargs = cuda_text_from_pretrained_kwargs(getattr(self, "device", ""), kwargs)
        if kwargs.get("attn_implementation") == "sdpa":
            logger.info(
                "Llama on CUDA: attn_implementation=sdpa torch_dtype=%s",
                kwargs.get("torch_dtype"),
            )
        return original(self, **kwargs)

    HuggingFaceText._load_model = _load_model  # type: ignore[method-assign]
    _LLAMA_CUDA_PATCHED = True
