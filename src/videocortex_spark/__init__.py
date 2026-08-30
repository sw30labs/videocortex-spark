"""videocortex-spark — video in, cortical activation maps out, on DGX Spark.

A thin, opinionated pipeline around Meta's TRIBE v2 brain encoder
(https://github.com/facebookresearch/tribev2), ported to NVIDIA GB10
(Grace Blackwell, CUDA 13, sm_121, 128 GB unified memory). Upstream owns
the model; this package owns everything around it: CUDA-13 / UMA device
selection, a preflight check that fails loudly instead of three gigabytes
into a download, and a renderer that turns the raw
(n_timesteps x n_vertices) matrix into brain-map stills you can actually
put in a slide.
"""

__version__ = "0.1.0"

from videocortex_spark.config import OverlayConfig, RenderConfig, RunConfig, VIEW_PRESETS
from videocortex_spark.device import describe_device, resolve_device, select_device

__all__ = [
    "__version__",
    "OverlayConfig",
    "RenderConfig",
    "RunConfig",
    "VIEW_PRESETS",
    "describe_device",
    "resolve_device",
    "select_device",
]
