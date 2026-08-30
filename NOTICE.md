# Licensing and provenance

`videocortex-spark` (this wrapper) is MIT licensed. What it drives is not.

## TRIBE v2 — CC-BY-NC-4.0

Both the code (`facebookresearch/tribev2`) and the weights
(`facebook/tribev2`) are released under Creative Commons
Attribution-NonCommercial 4.0. **Non-commercial use only.** Installing this
wrapper does not relicense them; if your use is commercial, neither the model
nor this pipeline around it is available to you.

## Frozen feature extractors

TRIBE v2 stacks four frozen encoders in front of its fusion transformer. Each
carries its own terms:

| modality | repo | note |
|---|---|---|
| text | `meta-llama/Llama-3.2-3B` | **Gated.** Llama 3.2 Community License — accept it on HuggingFace and authenticate before use. |
| image | `facebook/dinov2-large` | Apache-2.0 |
| audio | `facebook/w2v-bert-2.0` | MIT |
| video | `facebook/vjepa2-vitg-fpc64-256` | See the model card |

`videocortex-spark doctor` checks reachability for all five repos, and reports the
gated one distinctly, because it is the one that will stop you.

## Word timings

Transcription is performed by `whisperx`, run via `uvx` as a subprocess by
upstream. It is not vendored here and carries its own licence.

## Attribution

> d'Ascoli, S., Rapin, J., Benchetrit, Y., Brooks, T., Begany, K., Raugel, J.,
> Banville, H., & King, J.-R. (2026). *A foundation model of vision, audition,
> and language for in-silico neuroscience.* arXiv:2605.04326
