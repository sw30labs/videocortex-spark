# Quickstart (DGX Spark)

## 0. Preflight

```bash
./setup_and_run.sh                 # renderer only — prove the drawing half
./setup_and_run.sh --predict       # + tribev2 + cu130 torch
make doctor
```

Resolve anything marked ✗ before going further. The ones that catch Spark
users:

- `hf:meta-llama/Llama-3.2-3B` — gated. Accept the licence, then `hf auth login`.
- `torch cuda` — a PyPI / cu12 wheel. Reinstall from the cu130 index.
- `ptxas` — `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`
- `python` 3.14 — recreate the venv with 3.12.

## 1. Pull the weights up front

```bash
videocortex-spark fetch
```

~0.7 GB for TRIBE v2 itself, plus roughly 15–20 GB across the four frozen
encoders. Doing this separately means a failed download doesn't waste an
inference run.

## 2. Start small

```bash
videocortex-spark render --video some_clip.mp4 --max-frames 12
```

A two-minute clip is no longer a two-minute-wait on GB10, but the frozen
encoders still dominate — V-JEPA2 ViT-g and a 3B language model over every
chunk. Watch the log line that reports the device:

```
INFO videocortex_spark.model: device: cuda (NVIDIA GB10 sm_121, 110 GB UMA free)
```

If that says `cpu` on a Spark, `videocortex-spark doctor` will tell you why
(usually a CPU-only or CUDA-12 wheel).

## 3. Iterate on the picture, not the model

Inference already saved `predictions.npy`. Re-render for free:

```bash
videocortex-spark overlay --run runs/some_clip
videocortex-spark overlay --run runs/some_clip --spin
videocortex-spark overlay --run runs/some_clip --fast    # NVENC if ffmpeg has it
videocortex-spark export --run runs/some_clip            # interactive 3-D brain.html
videocortex-spark draw runs/some_clip/predictions.npy --views full
```

`brain.html` is the show-and-tell artifact: one file, the whole run, a
WebGL brain you can orbit and scrub. It opens from disk; no server, no
dependencies.

## 4. Read it honestly

- Predictions are for an **average subject**, not anyone in particular.
- They sit on the fsaverage5 cortical surface: 20,484 vertices, no subcortex.
- Upstream offsets predictions 5 s into the past for haemodynamic lag.
- One TR ≈ 1.49 s. There is no finer temporal structure to read into it.

## 5. Command deck

```bash
videocortex-spark serve            # http://127.0.0.1:8730
./setup_and_run.sh --deck          # same, after the usual bootstrap
```

Local only. Doctor / Launch / Runs / Job. Launching an encode from the
browser still runs TRIBE on this Spark — it is not a remote API.

## Try it without the model

```bash
make sample
xdg-open examples/sample_run/contact_sheet.png
```

`examples/make_sample.py` builds a seeded, scripted occipital → temporal
sequence. It is **not** model output. It exists to prove the drawing half
works on this machine before you commit to a 20 GB download.
