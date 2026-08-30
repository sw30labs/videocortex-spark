# videocortex-spark

Video in, cortical activation maps out — on NVIDIA DGX Spark.

A local pipeline around [Meta's TRIBE v2](https://github.com/facebookresearch/tribev2),
the trimodal brain encoder that predicts fMRI responses to naturalistic stimuli.
This is a **separate project** from [`videocortex`](../videocortex), which is
the Apple Silicon / laptop port. Same renderer, same model, different machine.

GB10 is Grace Blackwell: 20 Arm cores, a Blackwell iGPU (compute capability
12.1), and 128 GB of coherent unified memory. CUDA 13. Not an x86 box with
a discrete GPU. Not a Mac.

```
./setup_and_run.sh                 # venv + tests + synthetic sample (no model)
./setup_and_run.sh --predict       # + tribev2 + cu130 torch
./setup_and_run.sh --deck          # then the loopback command deck
videocortex-spark render --video clip.mp4
videocortex-spark serve            # http://127.0.0.1:8730
```

*Synthetic — a seeded, scripted occipital → temporal sequence from
`examples/make_sample.py`, used to exercise the renderer without a 20 GB
download. Run `make sample` to reproduce it. Real output looks the same; the
blobs are just less tidy.*

---

## What this is not

It does not decode brains. If you came here from
[MinD-Vis](https://mind-vis.github.io) or MinD-Video, those run the other
direction: fMRI in, picture out. This is the **encoding** direction — stimulus
in, predicted brain response out.

It also does not read *your* brain. TRIBE v2 predicts an **average subject**, at
fMRI's temporal resolution (TR ≈ 1.49 s), haemodynamically lagged and smoothed.
Upstream shifts its predictions 5 seconds into the past to compensate for that
lag, so frame *i* is the response to what happened around *i* − 5 s. Read the
output as *"this clip drives these cortical regions"*, never as *"this is what
someone is thinking"*.

It is not a drop-in for the Mac tree. Device selection, whisperx, overlay
encode, batch sizes, and the preflight checks are all GB10-shaped.

---

## Install

Python **3.12** (3.11 also works). DGX OS currently puts 3.14 on PATH; do not
use it — pyannote / whisperx will not import. `uv` is required for the model
path (`uvx whisperx`).

### Native (recommended on the Spark itself)

```bash
git clone <this repo> && cd videocortex-spark
./setup_and_run.sh                 # renderer only — venv, tests, synthetic sample
./setup_and_run.sh --predict       # + tribev2 + torch from the cu130 index
videocortex-spark doctor
```

`--predict` installs torch from `https://download.pytorch.org/whl/cu130`.
A bare `pip install torch` on aarch64 will hand you a CPU wheel, or a CUDA-12
wheel that looks for `libcudart.so.12`. GB10 will then look like it has no GPU.

Do **not** `pip install flash-attn`. Those wheels pull CUDA 12. SDPA is the
attention path on Blackwell, and it is faster here anyway.

### NGC container

```bash
docker build -t videocortex-spark .
docker run --gpus all --ipc=host --rm -it \
  -v "$PWD":/workspace -w /workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p 8730:8730 \
  videocortex-spark
```

Base image: `nvcr.io/nvidia/pytorch:25.12-py3` (CUDA 13, GB10-tested from
25.10 onward). Older NGC tags warn `Detected NVIDIA GB10 GPU, which is not
yet supported` and then hang on the first interesting kernel.

`videocortex-spark doctor` checks python, aarch64, torch / CUDA 13, GB10
compute capability, UMA headroom (`/proc/meminfo`, not `nvidia-smi`), CUDA
`ptxas` (Triton), ffmpeg / NVENC, `uvx`, fsaverage5, free disk, and whether
HuggingFace will actually hand you each of the five model repos. Fix what it
flags before starting a run.

---

## The Spark notes

Four things in upstream assume an x86 CUDA-12 cluster. All four are handled
here, but they're worth knowing about because they're invisible until they bite.

**1. The wrong torch wheel is silent.** Upstream resolves `device="auto"` as
`"cuda" if torch.cuda.is_available() else "cpu"`. A CUDA-12 / CPU aarch64
wheel imports, reports no GPU, and you spend an hour on twenty Arm cores.
`videocortex-spark` refuses that combination in `doctor` and rewrites the
four frozen extractors onto `cuda` with Spark-sized batches.

**2. 128 GB is unified, not VRAM.** GB10 has no discrete memory carve-out.
`nvidia-smi` prints `Memory-Usage: Not Supported`. `cudaMemGetInfo`
under-reports because it ignores pages the CPU could reclaim. Doctor reads
`MemAvailable` from `/proc/meminfo`. The four extractors, the fusion
transformer, ffmpeg, and the Ubuntu desktop all sit in the same pool — so
the defaults are `batch_size=4`, `feature_batch_size=2`, `num_workers=4`,
not upstream's 8 / 20 and not the laptop's 1 / 0.

**3. sm_121 is binary-compatible with sm_120.** You will see:

```
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

Ignore it. Official cu130 wheels ship 12.0+PTX; the runtime JITs. Triton's
bundled `ptxas`, however, is often CUDA 12.8 and dies on `sm_121a`. We set
`TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` before anything CUDA-shaped
starts.

**4. Word timings come from `uvx whisperx`.** Parent torch sees GB10, so
upstream passes `--device cuda --compute_type float16`. That uvx env does
not inherit our cu130 wheel: PyPI torch on aarch64 is CPU, and CTranslate2's
aarch64 manylinux wheel was not built with CUDA. The patch pins
`--python 3.12`, `--torch-backend cpu`, `--device cpu`, `int8`, and
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. On an x86 CUDA box the rewrite
keeps float16, raises `--batch_size` 16 → 32, and uses `--torch-backend cu130`.

Llama 3.2 on CUDA uses SDPA + bfloat16. The Mac port has to force eager +
float32 because Metal's fused SDPA aborts on grouped-query attention; CUDA
does not.

`--fast` on overlay is NVENC (`h264_nvenc`), not VideoToolbox. Stock Ubuntu
ffmpeg often has no NVENC encoder; doctor warns and we fall back to libx264.
GB10 has 1× NVENC / 1× NVDEC.

---

## Usage

```bash
videocortex-spark doctor                       # preflight (model path)
videocortex-spark doctor --renderer            # draw path only — no torch expected
videocortex-spark fetch                        # pull all five repos up front

videocortex-spark render --video clip.mp4
videocortex-spark render --audio talk.wav --views full --max-frames 24
videocortex-spark render --text  essay.txt --device cpu

videocortex-spark draw runs/clip/predictions.npy --views lateral --light

videocortex-spark overlay --run runs/clip
videocortex-spark overlay --run runs/clip --spin
# → runs/clip/overlay.mp4  (PIP animation, every TR, audio copied)

videocortex-spark serve                    # loopback command deck, http://127.0.0.1:8730
```

Hub cache (`$HF_HOME/hub`, ~18 GB). `fetch` pulls the five TRIBE repos;
whisperx / pyannote land on the first audio pass. Llama and pyannote are gated.

| repo | size | role |
|---|---|---|
| `facebook/tribev2` | 677 MB | fusion checkpoint |
| `meta-llama/Llama-3.2-3B` | 6.0 GB | text (gated) |
| `facebook/dinov2-large` | 2.3 GB | image |
| `facebook/w2v-bert-2.0` | 2.2 GB | audio |
| `facebook/vjepa2-vitg-fpc64-256` | 3.9 GB | video |
| `Systran/faster-whisper-large-v3` | 2.9 GB | whisperx |
| `pyannote/speaker-diarization-3.1` + `segmentation-3.0` + `wespeaker-voxceleb-resnet34-LM` | ~44 MB | whisperx diarization (gated) |

Destrieux surface atlas is in `~/nilearn_data`.

The deck is a stdlib HTTP page (no FastAPI, no extra web deps). It binds
loopback, pins the `Host` header, and refuses non-loopback peers on `/api/`
and `/media/`. Doctor, launch encode/overlay, browse runs, watch the one
job.

### Options worth knowing

| flag | what it does |
|---|---|
| `--views` | `standard` (L/R lateral + medial), `lateral`, `medial`, `left`, `right`, `occipital`, `full` |
| `--stride N` | render every Nth TR |
| `--max-frames N` | hard cap; if stride overshoots it, the stride *widens* rather than truncating |
| `--percentile` | robust colour limit (default 99th of \|x\|) |
| `--threshold-frac` | hide vertices below this fraction of vmax (default 0.25) |
| `--ramp-frac` | fade colour in across that band below threshold (default 0.5; `0` = hard cut) |
| `--device` | `auto` \| `cuda` \| `cpu`. Explicit values raise if unavailable rather than silently falling back |
| `--batch-size` | event loader batch (default **4** on Spark) |
| `--feature-batch-size` | per-extractor batch (default **2**) |
| `--num-workers` | DataLoader workers (default **4**; do not set 20) |
| `--light` | light background instead of dark |

Overlay (`videocortex-spark overlay --run …`):

| flag | what it does |
|---|---|
| `--size` | PIP width as a fraction of the frame (default 0.24, landscape) |
| `--position` | `top-right` (default) or `top-left` |
| `--lag-mode` | `stimulus` matches the picture (default). `scanner` delays the PIP 5 s |
| `--stride N` | draft: every Nth TR (default 1) |
| `--fast` | NVENC encode when ffmpeg has `h264_nvenc`; else libx264 |
| `--force` | rebuild `pip/` cards |
| `--spin` | 3D inflated globe in the PIP (both hemispheres, yaw-only) |

Output layout is the same as the laptop tree: `runs/<stem>/{frames,contact_sheet.png,predictions.npy,timestamps.npy,manifest.json,overlay.mp4}`.

---

## One design decision worth defending

**Colour limits are computed once over the whole run, never per frame.**

Per-frame normalisation is the default in a lot of quick visualisation code and
it is quietly dishonest: it makes a resting moment render exactly as vividly as
a startling one, because each frame gets restretched to fill the colormap. Every
frame here shares one scale, derived from a robust percentile across the entire
prediction so a single berserk vertex can't flatten everything else. There's a
test that fails if per-frame rescaling ever creeps back in.

**UMA memory is reported from `/proc/meminfo`, never from `nvidia-smi`.**

On a discrete GPU those two agree. On GB10 they do not, and believing
`nvidia-smi` is how you conclude the machine is out of memory with 100 GB
sitting idle.

---

## Layout

```
src/videocortex_spark/
├── cli.py         six verbs: doctor, fetch, render, draw, overlay, serve
├── config.py      RunConfig / RenderConfig / OverlayConfig, view presets
├── device.py      CUDA (else CPU). Refuses a CUDA-12 wheel that looks empty.
├── doctor.py      preflight checks, including UMA / ptxas / NVENC
├── model.py       loads TRIBE v2 with Spark batch/worker/bf16 overrides
├── overlay.py     PIP animation; --fast is NVENC
├── patches.py     whisperx uvx pin, Triton ptxas, Llama SDPA+bf16
├── pipeline.py    predict → render
├── spark.py       GB10 constants, meminfo, encode args (no torch)
├── render.py      nilearn-backed plates (identical to the laptop tree)
└── web/           loopback command deck
```

`render.py` still depends on nilearn alone. `pip install -e .` without the
`predict` extra gives you a working `draw` command, testable without a GPU.

```bash
pytest              # everything
pytest -m "not slow"   # skip the ones that actually rasterise surfaces
```

---

## Licence

This wrapper is MIT — see `LICENSE`.

**The model is not.** TRIBE v2 and its weights are CC-BY-NC-4.0: research and
other non-commercial use only. `meta-llama/Llama-3.2-3B` carries its own
community licence and is gated; you must accept it on HuggingFace before
anything here will run. See `NOTICE.md`.

## Credit

TRIBE v2 — d'Ascoli, Rapin, Benchetrit, Brooks, Begany, Raugel, Banville and
King (Meta FAIR Brain & AI), *A foundation model of vision, audition, and
language for in-silico neuroscience*, 2026.
[paper](https://arxiv.org/abs/2605.04326) ·
[code](https://github.com/facebookresearch/tribev2) ·
[weights](https://huggingface.co/facebook/tribev2)

NVIDIA DGX Spark porting guide:
https://docs.nvidia.com/dgx/dgx-spark-porting-guide/
