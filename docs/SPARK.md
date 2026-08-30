# DGX Spark port notes

Target: NVIDIA DGX Spark, GB10 Grace Blackwell Superchip.

| | |
|---|---|
| CPU | 20-core Armv9.2 (10× Cortex-X925 + 10× Cortex-A725) |
| GPU | Blackwell iGPU, 6,144 CUDA cores, sm_121 |
| Memory | 128 GB LPDDR5x **unified**, 273 GB/s, no VRAM carve-out |
| CUDA | 13.0 (driver R580) |
| OS | DGX OS 7 / Ubuntu 24.04, aarch64 |
| Video | 1× NVENC, 1× NVDEC |

This file is the delta against the laptop `videocortex` tree. The renderer
(`render.py`, `spin.py`, `regions.py`, contact sheets) is unchanged. Everything
that touched Metal, VideoToolbox, or "laptop RAM" was replaced.

## Why a separate project

The Mac tree's whole personality is "upstream assumed CUDA; we have Metal."
Patching that tree to also be "upstream assumed x86 CUDA-12; we have aarch64
CUDA-13 UMA" would leave both ports worse. Batch sizes, whisperx, overlay
encode, doctor, and the torch install path do not share a useful default.

## What changed

| Concern | laptop `videocortex` | this tree |
|---|---|---|
| Device | CUDA → MPS → CPU | CUDA, else CPU (no MPS) |
| Torch install | `torch>=2.5.1,<2.7` from PyPI | cu130 index / NGC 25.10+ |
| Llama attention | eager + float32 (Metal GQA abort) | SDPA + bfloat16 |
| whisperx | float16 → int8, pin torch 2.6 | uvx `--torch-backend cpu`, `--device cpu` int8 (CTranslate2 aarch64 is CPU-only). x86 CUDA path still float16 / batch 32 / `cu130` |
| Overlay `--fast` | `h264_videotoolbox` | `h264_nvenc`, else libx264 |
| batch / workers | 1 / 0 | 4 / 4 (feature batch 2) |
| Memory reporting | n/a | `/proc/meminfo` MemAvailable |
| ffmpeg hint | `brew install ffmpeg` | `sudo apt install ffmpeg` |

## Sharp edges that will waste an afternoon

1. **cu12 wheels.** `libcudart.so.12` vs `.13`. Torch imports. `cuda.is_available()` is False. You encode on CPU and blame TRIBE.
2. **Triton ptxas.** Bundled 12.8 dies on `sm_121a`. Toolkit ptxas at `/usr/local/cuda/bin/ptxas` is the fix.
3. **flash-attn.** Don't. libcudart 12, and SDPA is faster on Blackwell.
4. **Python 3.14.** On PATH. whisperx/pyannote want 3.12. The venv must pin it.
5. **`nvidia-smi` memory.** Not supported. Do not size batches off it.
6. **NGC `< 25.10`.** "Detected NVIDIA GB10 GPU, which is not yet supported."
7. **x86 containers.** They will not run. Pull arm64 tags only.
8. **GPUDirect RDMA / nvidia-peermem.** Not supported on this UMA. Irrelevant for this pipeline; noted so nobody "fixes" a hang with it.
9. **whisperx CUDA.** CTranslate2's aarch64 manylinux wheels are CPU-only. Parent torch has GB10, so upstream passes `--device cuda` and uvx dies in `WhisperModel`. We rewrite to cpu int8. Do not "fix" this by `pip install ctranslate2` from PyPI.

## References

- [DGX Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/)
- [UMA memory reporting](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html#guidance-for-reporting-memory-resources-with-unified-memory-architecture)
- [PyTorch cu130 wheels](https://download.pytorch.org/whl/cu130)
