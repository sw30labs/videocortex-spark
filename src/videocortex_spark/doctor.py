"""Preflight checks for DGX Spark.

The failure modes of this stack are almost all discoverable *before* you start
a twenty-gigabyte download or a forty-minute inference run: a CUDA-12 wheel
that silently has no GPU, Triton's 12.8 ptxas, a gated Llama you haven't
accepted. ``videocortex-spark doctor`` finds them in about five seconds.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import sys
import typing as tp
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from videocortex_spark.config import (
    FEATURE_EXTRACTORS,
    HF_CHECKPOINT_PROBE,
    HF_CHECKPOINT_REPO,
)
from videocortex_spark.spark import (
    SPARK_COMPUTE,
    SPARK_CUDA_MAJOR,
    SPARK_UNIFIED_GIB,
    TORCH_CU130_INDEX,
    apply_runtime_env,
    find_ptxas,
    is_aarch64,
    nvenc_h264_available,
    uma_report,
)

OK, WARN, FAIL, UNKNOWN = "ok", "warn", "fail", "unknown"

#: Rough on-disk cost of the checkpoint plus the four frozen encoders.
WEIGHTS_BUDGET_GB = 20


@dataclass(slots=True)
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""

    @property
    def glyph(self) -> str:
        return {OK: "✓", WARN: "!", FAIL: "✗", UNKNOWN: "?"}[self.status]


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@contextmanager
def _quiet_neuralset():
    """tribev2 / neuralset warn on import, before any events exist.

    neuralset.extractors.base sets its own logger level, so silencing the
    parent is not enough — disable() covers the whole tree for the import.
    """
    prev = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module=r"neuralset(\.|$)")
            yield
    finally:
        logging.disable(prev)


def check_python() -> Check:
    v = sys.version_info
    got = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < (3, 11):
        return Check(
            "python", FAIL, f"{got} — tribev2 requires >= 3.11",
            "uv venv --python 3.12 && source .venv/bin/activate",
        )
    return Check("python", OK, got)


def check_predict_python() -> Check:
    """whisperx/pyannote via uvx die on 3.13+; the renderer does not care."""
    v = sys.version_info
    got = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 13):
        return Check(
            "whisperx python", WARN, f"{got} — uvx whisperx is pinned to 3.12",
            "uv venv --python 3.12 && source .venv/bin/activate",
        )
    return Check("whisperx python", OK, got)


def check_platform() -> Check:
    machine = __import__("platform").machine()
    if is_aarch64(machine):
        return Check("arch", OK, f"{machine} (Grace / GB10)")
    return Check(
        "arch", WARN,
        f"{machine} — this port targets DGX Spark (aarch64 + GB10)",
        "run it on the Spark, or cross-compile; x86 CUDA-12 wheels will not work",
    )


def check_torch() -> list[Check]:
    from videocortex_spark.device import probe

    info = probe()
    if info["torch"] is None:
        return [
            Check(
                "torch", FAIL, "not installed",
                f"uv pip install torch torchvision torchaudio --index-url {TORCH_CU130_INDEX}",
            )
        ]

    checks = [Check("torch", OK, f"{info['torch']} on {info['machine']}")]

    cuda_ver = info.get("torch_cuda")
    if cuda_ver:
        major = str(cuda_ver).split(".")[0]
        if major != str(SPARK_CUDA_MAJOR):
            checks.append(
                Check(
                    "torch cuda", FAIL,
                    f"built against CUDA {cuda_ver} — Spark needs CUDA {SPARK_CUDA_MAJOR}",
                    f"uv pip install --force-reinstall torch torchvision torchaudio "
                    f"--index-url {TORCH_CU130_INDEX}",
                )
            )
        else:
            checks.append(Check("torch cuda", OK, f"CUDA {cuda_ver}"))

    if info["has_cuda"]:
        name = info.get("device_name") or "CUDA GPU"
        cap = info.get("capability")
        cap_s = f" sm_{cap[0]}{cap[1]}" if cap else ""
        gb10 = " (GB10)" if info.get("is_gb10") else ""
        checks.append(Check("accelerator", OK, f"cuda · {name}{cap_s}{gb10}"))
        if cap == SPARK_COMPUTE:
            checks.append(
                Check(
                    "compute", OK,
                    f"{cap[0]}.{cap[1]} — sm_120 cubins JIT here; ignore the warning",
                )
            )
        elif cap is not None:
            checks.append(
                Check(
                    "compute", WARN,
                    f"{cap[0]}.{cap[1]} (Spark is {SPARK_COMPUTE[0]}.{SPARK_COMPUTE[1]})",
                )
            )
    else:
        checks.append(
            Check(
                "accelerator", FAIL,
                info["note"] or "torch reports no CUDA device",
                f"uv pip install torch torchvision torchaudio --index-url {TORCH_CU130_INDEX}",
            )
        )
    return checks


def check_uma() -> Check:
    uma = uma_report()
    avail = uma.get("available_gb")
    total = uma.get("total_gb")
    if avail is None or total is None:
        return Check("uma", UNKNOWN, "/proc/meminfo not readable")
    # Four extractors + fusion transformer + video decode comfortably fit
    # in ~40 GB. Warn under 48 GB free; fail under 16 GB.
    detail = (
        f"{avail:.0f} GB available of {total:.0f} GB unified "
        f"(nvidia-smi memory is not meaningful on UMA)"
    )
    if avail < 16:
        return Check(
            "uma", FAIL, detail,
            "close other GPU/CPU jobs; UMA is shared with the desktop",
        )
    if avail < 48:
        return Check(
            "uma", WARN, detail,
            "drop --batch-size if the run OOMs; 128 GB is shared, not VRAM",
        )
    if total < SPARK_UNIFIED_GIB * 0.7:
        return Check(
            "uma", WARN,
            f"{total:.0f} GB total — a Founders Spark is {SPARK_UNIFIED_GIB} GB",
        )
    return Check("uma", OK, detail)


def check_ptxas() -> Check:
    path = find_ptxas()
    if path is None:
        return Check(
            "ptxas", WARN,
            "CUDA ptxas not found — Triton may fail on sm_121a",
            "export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas",
        )
    return Check("ptxas", OK, str(path))


def check_tribev2() -> Check:
    if not _has("tribev2"):
        return Check(
            "tribev2", FAIL, "not installed",
            "uv pip install -e '.[predict]'",
        )
    try:
        with _quiet_neuralset():
            import tribev2  # noqa: F401

        return Check("tribev2", OK, "importable")
    except Exception as exc:  # pragma: no cover - environment dependent
        return Check("tribev2", FAIL, f"import failed: {exc}", "")


def check_render_stack() -> Check:
    missing = [m for m in ("nilearn", "nibabel", "matplotlib", "scipy") if not _has(m)]
    if missing:
        return Check(
            "render stack", FAIL, f"missing {', '.join(missing)}",
            "uv pip install -e .",
        )
    return Check("render stack", OK, "nilearn, nibabel, matplotlib, scipy")


def check_ffmpeg() -> Check:
    if shutil.which("ffmpeg"):
        return Check("ffmpeg", OK, "on PATH")
    return Check(
        "ffmpeg", FAIL,
        "not found — needed to decode video and to compose the PIP overlay",
        "sudo apt install ffmpeg",
    )


def check_nvenc() -> Check:
    if not shutil.which("ffmpeg"):
        return Check("nvenc", UNKNOWN, "ffmpeg not on PATH")
    if nvenc_h264_available():
        return Check("nvenc", OK, "h264_nvenc (GB10 1x NVENC)")
    return Check(
        "nvenc", WARN,
        "this ffmpeg has no h264_nvenc — --fast will use libx264",
        "install an NVENC-enabled ffmpeg, or ignore and keep software encode",
    )


def check_binaries() -> list[Check]:
    out = [check_ffmpeg(), check_nvenc()]
    if shutil.which("uvx") or shutil.which("uv"):
        out.append(Check("uvx", OK, "on PATH (used to run whisperx)"))
    else:
        out.append(
            Check(
                "uvx", FAIL,
                "not found — upstream shells out to `uvx whisperx` to get word timings",
                "curl -LsSf https://astral.sh/uv/install.sh | sh",
            )
        )
    return out


def check_fsaverage5() -> Check:
    """Is the surface geometry available?

    nilearn ships fsaverage5 as package data — unlike the higher-resolution
    meshes, it never downloads. So this is really a "can we load it" check.
    """
    if not _has("nilearn"):
        return Check("fsaverage5", FAIL, "nilearn not installed", "uv pip install -e .")
    try:
        from videocortex_spark.render import load_fsaverage5

        mesh = Path(load_fsaverage5()["left"]["mesh"])
        where = "bundled with nilearn" if "site-packages" in str(mesh) else str(mesh.parent)
        return Check("fsaverage5", OK, where)
    except Exception as exc:
        return Check(
            "fsaverage5", FAIL, f"could not load ({type(exc).__name__}: {exc})",
            "pip install --upgrade nilearn",
        )


def check_hf_access(timeout: float = 10.0) -> list[Check]:
    """The one that actually bites: Llama-3.2-3B is a manually gated repo.

    ``model_info`` is not enough — it happily answers for a gated repo you have
    no access to (it reports ``gated='manual'`` and returns). The only honest
    test is to try to fetch a file, so for gated repos we pull one small JSON.
    """
    if not _has("huggingface_hub"):
        return [
            Check(
                "huggingface", UNKNOWN, "huggingface_hub not installed",
                "uv pip install -e .",
            )
        ]

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import GatedRepoError

    api = HfApi()
    checks: list[Check] = []
    repos = [(HF_CHECKPOINT_REPO, HF_CHECKPOINT_PROBE)] + [
        (spec["repo"], spec["probe"]) for spec in FEATURE_EXTRACTORS.values()
    ]

    for repo, probe in repos:
        name = f"hf:{repo}"
        try:
            info = api.model_info(repo, timeout=timeout)
        except Exception as exc:
            checks.append(
                Check(name, UNKNOWN, f"could not reach ({type(exc).__name__})")
            )
            continue

        if not getattr(info, "gated", False):
            checks.append(Check(name, OK, "reachable"))
            continue

        try:
            hf_hub_download(repo, probe)
            checks.append(Check(name, OK, f"gated ({info.gated}) — access granted"))
        except GatedRepoError:
            checks.append(
                Check(
                    name, FAIL,
                    f"gated ({info.gated}) — your account has NOT been granted access",
                    f"accept the licence at https://huggingface.co/{repo}, "
                    "then: hf auth login",
                )
            )
        except Exception as exc:
            checks.append(
                Check(
                    name, WARN,
                    f"gated ({info.gated}) — could not verify ({type(exc).__name__})",
                    "hf auth login",
                )
            )
    return checks


def check_hf_token() -> Check:
    """A token isn't optional once a gated repo is in the dependency chain."""
    if not _has("huggingface_hub"):
        return Check("hf token", UNKNOWN, "huggingface_hub not installed")
    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:
        token = None
    if token:
        return Check("hf token", OK, "authenticated")
    return Check(
        "hf token", WARN,
        "not logged in — required for the gated Llama-3.2-3B weights",
        "hf auth login",
    )


def check_disk(path: Path | None = None) -> Check:
    target = Path(path or Path.home())
    usage = shutil.disk_usage(target)
    free_gb = usage.free / 1e9
    if free_gb < WEIGHTS_BUDGET_GB:
        return Check(
            "disk", FAIL,
            f"{free_gb:.0f} GB free — the checkpoint plus four frozen encoders "
            f"need roughly {WEIGHTS_BUDGET_GB} GB",
            "free space, or point HF_HOME at a bigger volume",
        )
    return Check("disk", OK, f"{free_gb:.0f} GB free")


def run_all(*, network: bool = True, model: bool = True) -> list[Check]:
    """Run preflight checks.

    ``model=False`` is the draw/overlay path: python, nilearn, ffmpeg,
    fsaverage5. Missing torch is not a failure there — it is the default install.
    """
    checks: list[Check] = [check_python()]
    if model:
        apply_runtime_env()
        checks.append(check_predict_python())
        checks.append(check_platform())
        checks.extend(check_torch())
        checks.append(check_uma())
        checks.append(check_ptxas())
        checks.append(check_tribev2())
    checks.append(check_render_stack())
    if model:
        checks.extend(check_binaries())
    else:
        checks.append(check_ffmpeg())
    checks.append(check_fsaverage5())
    if model:
        checks.append(check_disk())
        if network:
            checks.append(check_hf_token())
            checks.extend(check_hf_access())
    return checks


def format_report(checks: tp.Sequence[Check]) -> str:
    width = max(len(c.name) for c in checks) + 2
    lines = []
    for c in checks:
        lines.append(f"  {c.glyph}  {c.name:<{width}} {c.detail}")
        if c.fix and c.status in (FAIL, WARN):
            lines.append(f"     {'':<{width}} → {c.fix}")

    n_fail = sum(c.status == FAIL for c in checks)
    n_warn = sum(c.status == WARN for c in checks)
    lines.append("")
    if n_fail:
        lines.append(f"  {n_fail} blocking problem(s), {n_warn} warning(s).")
    elif n_warn:
        lines.append(f"  Ready, with {n_warn} warning(s).")
    else:
        lines.append("  All clear.")
    return "\n".join(lines)


def exit_code(checks: tp.Sequence[Check]) -> int:
    return 1 if any(c.status == FAIL for c in checks) else 0
