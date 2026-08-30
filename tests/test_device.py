"""Device priority is pure logic, so it gets tested without torch present."""

import os

import pytest

from videocortex_spark.device import DeviceError, select_device
from videocortex_spark.spark import (
    VALID_DEVICES,
    load_local_env,
    looks_like_spark,
    read_meminfo,
    uma_report,
)


def test_auto_priority():
    assert select_device("auto", has_cuda=True) == "cuda"
    assert select_device("auto", has_cuda=False) == "cpu"


def test_explicit_device_is_honoured():
    assert select_device("cpu", has_cuda=True) == "cpu"
    assert select_device("cuda", has_cuda=True) == "cuda"


def test_explicit_unavailable_device_raises_rather_than_falling_back():
    with pytest.raises(DeviceError, match="CUDA"):
        select_device("cuda", has_cuda=False)


def test_unknown_device_rejected():
    with pytest.raises(ValueError):
        select_device("mps")
    with pytest.raises(ValueError):
        select_device("tpu")


def test_valid_devices_are_spark_shaped():
    assert "mps" not in VALID_DEVICES
    assert VALID_DEVICES == ("auto", "cuda", "cpu")


def test_looks_like_spark_from_name_or_cap():
    assert looks_like_spark(gpu_name="NVIDIA GB10")
    assert looks_like_spark(machine="aarch64", capability=(12, 1))
    assert not looks_like_spark(machine="x86_64", capability=(8, 9))


def test_fast_encode_never_asks_for_videotoolbox():
    from videocortex_spark.spark import video_encode_args

    args = video_encode_args(fast=True, crf=16)
    joined = " ".join(args)
    assert "videotoolbox" not in joined
    assert args[0] == "-c:v"
    assert args[1] in {"h264_nvenc", "libx264"}
    slow = video_encode_args(fast=False, crf=18)
    assert slow[1] == "libx264"
    assert "18" in slow


def test_uma_report_prefers_memavailable_over_smi():
    mem = read_meminfo(
        "MemTotal:       126877696 kB\n"
        "MemAvailable:   116391936 kB\n"
        "SwapFree:        15728640 kB\n"
    )
    report = uma_report(mem)
    assert report["total_gb"] == pytest.approx(121.0, abs=1.0)
    assert report["available_gb"] == pytest.approx(111.0, abs=1.0)
    assert report["allocatable_with_swap_gb"] > report["allocatable_no_swap_gb"]


def test_load_local_env_setdefaults_and_skips_comments(tmp_path, monkeypatch):
    envfile = tmp_path / ".env"
    envfile.write_text(
        "# not a secret\n"
        "export HF_TOKEN=hf_fromfile\n"
        "ALREADY=fromfile\n"
        "QUOTED='quoted-value'\n"
        "\n"
    )
    monkeypatch.setenv("ALREADY", "fromenv")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    applied = load_local_env(start=tmp_path)
    assert applied == envfile
    assert os.environ["HF_TOKEN"] == "hf_fromfile"
    assert os.environ["ALREADY"] == "fromenv"
    assert os.environ["QUOTED"] == "quoted-value"
