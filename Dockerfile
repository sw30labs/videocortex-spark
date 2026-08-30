# NVIDIA DGX Spark (GB10 / sm_121 / CUDA 13 / aarch64)
#
# NGC PyTorch 25.10+ is the stack NVIDIA actually tests on GB10. Do not
# start from a CUDA 12 image — those wheels look for libcudart.so.12 and
# the GB10 driver will not pretend.
#
#   docker build -t videocortex-spark .
#   docker run --gpus all --ipc=host --rm -it \
#     -v "$PWD":/workspace -w /workspace \
#     -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
#     -p 8730:8730 \
#     videocortex-spark
#
# Pin linux/arm64 if you ever build this off-box. An amd64 image will not
# run here.

FROM nvcr.io/nvidia/pytorch:25.12-py3

ENV DEBIAN_FRONTEND=noninteractive \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_MODULE_LOADING=LAZY \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE NOTICE.md ./
COPY src ./src
COPY examples ./examples

# Torch is already in the NGC image. Installing from PyPI would clobber it
# with a CPU or cu12 wheel — fail the build if that happens.
RUN pip install --no-cache-dir -e '.[dev]' \
 && pip install --no-cache-dir 'tribev2 @ git+https://github.com/facebookresearch/tribev2.git' \
 && python -c "import torch; v=torch.version.cuda or ''; assert v.startswith('13'), f'torch lost CUDA 13 during install: {torch.__version__} cuda={v}'"

EXPOSE 8730
CMD ["videocortex-spark", "doctor"]
