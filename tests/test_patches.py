"""The whisperx patch is the difference between a run and a stack trace on Spark."""

import subprocess

from videocortex_spark.patches import (
    _is_cpu_whisperx,
    _rewrite_uvx_whisperx,
    _whisperx_python,
    whisperx_compat,
)

CPU_CMD = [
    "uvx", "whisperx", "in.wav", "--model", "large-v3",
    "--device", "cpu", "--compute_type", "float16", "--batch_size", "16",
]
CUDA_CMD = [*CPU_CMD[:6], "cuda", *CPU_CMD[7:]]


def _uvx_opts(cmd: list[str]) -> list[str]:
    return cmd[: cmd.index("whisperx")]


def test_detects_only_the_cpu_whisperx_invocation():
    assert _is_cpu_whisperx(CPU_CMD)
    assert not _is_cpu_whisperx(CUDA_CMD)
    assert not _is_cpu_whisperx(["ffmpeg", "-i", "a.mp4"])
    assert not _is_cpu_whisperx("uvx whisperx --device cpu")  # str, not list


def test_rewrites_float16_to_int8_on_cpu(monkeypatch):
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: seen.append(list(cmd)))
    with whisperx_compat():
        subprocess.run(CPU_CMD)
    assert seen[0][seen[0].index("--compute_type") + 1] == "int8"


def test_cpu_uvx_pins_python_and_cpu_torch():
    out = _rewrite_uvx_whisperx(list(CPU_CMD))
    opts = _uvx_opts(out)
    assert opts[opts.index("--python") + 1] == _whisperx_python()
    assert opts[opts.index("--torch-backend") + 1] == "cpu"
    assert out[out.index("--compute_type") + 1] == "int8"
    assert out[out.index("whisperx") + 1] == "in.wav"


def test_cuda_uvx_on_x86_keeps_float16_and_raises_batch(monkeypatch):
    monkeypatch.setattr("videocortex_spark.patches.is_aarch64", lambda: False)
    out = _rewrite_uvx_whisperx(list(CUDA_CMD))
    opts = _uvx_opts(out)
    assert opts[opts.index("--python") + 1] == _whisperx_python()
    assert opts[opts.index("--torch-backend") + 1] == "cu130"
    assert out[out.index("--device") + 1] == "cuda"
    assert out[out.index("--compute_type") + 1] == "float16"
    assert out[out.index("--batch_size") + 1] == "32"


def test_cuda_uvx_on_aarch64_falls_back_to_cpu_int8(monkeypatch):
    monkeypatch.setattr("videocortex_spark.patches.is_aarch64", lambda: True)
    out = _rewrite_uvx_whisperx(list(CUDA_CMD))
    opts = _uvx_opts(out)
    assert out[out.index("--device") + 1] == "cpu"
    assert out[out.index("--compute_type") + 1] == "int8"
    assert out[out.index("--batch_size") + 1] == "16"
    assert opts[opts.index("--torch-backend") + 1] == "cpu"


def test_leaves_cuda_invocations_compute_type_alone_on_x86(monkeypatch):
    monkeypatch.setattr("videocortex_spark.patches.is_aarch64", lambda: False)
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: seen.append(list(cmd)))
    with whisperx_compat():
        subprocess.run(CUDA_CMD)
    assert seen[0][seen[0].index("--compute_type") + 1] == "float16"
    assert seen[0][seen[0].index("--device") + 1] == "cuda"


def test_whisperx_subprocess_gets_spark_env(monkeypatch):
    seen = []

    def fake(cmd, *a, **k):
        seen.append(k.get("env"))

    monkeypatch.setattr(subprocess, "run", fake)
    with whisperx_compat():
        subprocess.run(CPU_CMD, env={"FOO": "1"})
    assert seen[0]["FOO"] == "1"
    assert seen[0]["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_unrelated_subprocess_calls_pass_through_untouched(monkeypatch):
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: seen.append(list(cmd)))
    with whisperx_compat():
        subprocess.run(["ffmpeg", "-version"])
    assert seen[0] == ["ffmpeg", "-version"]


def test_original_run_is_restored_even_after_an_exception(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(subprocess, "run", sentinel)
    try:
        with whisperx_compat():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert subprocess.run is sentinel
