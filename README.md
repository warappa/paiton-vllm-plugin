# Paiton

Run chat, coding, image and video models on AMD Radeon with Paiton's native GPU
runtimes, integrated with vLLM, Diffusers and ComfyUI.

**Using Qwen3.8? Start with its [MXFP4 or 3-bit quickstart](models/Qwen3.8-MXFP4-DFlash2/README.md).**
It explains which weights to download, how to launch, and the optional context,
vision and cache settings.

Prefer a desktop app? [Paiton Studio](https://github.com/Eliovp-BV/paiton-studio)
provides chat, image and video tools for the supported models.

[Models](#models) · [Quick start](#quick-start) · [Requirements](#requirements) ·
[Your own vLLM](#use-your-own-vllm-environment) · [Docs](#where-to-find-things)

<a id="run-with-a-managed-container"></a><a id="managed-containers"></a><a id="launch-commands"></a>

## Quick start

The container launchers include each model's supported inference environment.
Choose a model below and follow its guide. For the smallest language model,
**MiniCPM5-2B**:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
./models/MiniCPM5-2B/serve-docker.sh
```

The first launch downloads its weights and prepares runtime components; later
launches reuse the caches. Its [guide](models/MiniCPM5-2B/README.md) includes
chat and API examples. Qwen3.8 and Meeting need weight preparation before launch.
Run one model at a time on the tested single-GPU setup.

<a id="model-library"></a><a id="reading-the-benchmarks"></a>

## Models

Model links lead to setup, launch options and support limits. Benchmark links
include the tested settings and quality tradeoffs; results from different models
or profiles are separate comparisons.

<a id="chat-reasoning-coding-and-visual-understanding"></a>

### Chat, reasoning and coding

These models serve an OpenAI-compatible API through vLLM.

| Model and setup | Use | Benchmark |
| --- | --- | --- |
| [MiniCPM5-2B](models/MiniCPM5-2B/README.md) | Lightweight chat, coding and tools · 8K | [Results](models/MiniCPM5-2B/BENCHMARKS.md) |
| [Qwen3.8 27B MXFP4 / 3-bit + DFlash2](models/Qwen3.8-MXFP4-DFlash2/README.md) | Chat and coding · 65K multi-request, or the full 262K context with up to eight requests · optional images | [Results](models/Qwen3.8-MXFP4-DFlash2/README.md#current-benchmark-results) |
| [Qwen3.8 27B Qronos](models/Qwen3.8/README.md) | Chat, coding and optional reasoning · 8K · text | [Results](https://eliovp.com/blog/paiton-qwen38-radeon-ai-pro-r9700) |
| [Qwen3.8 NEO CODER MAX 27B](models/Qwen3.8-NEO-CODER-MAX/README.md) | Coding and visual chat · 8K · text plus one image | [Results](models/Qwen3.8-NEO-CODER-MAX/BENCHMARKS.md) |
| [Qwen3-Coder 30B A3B](models/Qwen3-Coder-30B/README.md) | Code writing, review, testing and tools · 4K | [Results](models/Qwen3-Coder-30B/BENCHMARKS.md) |
| [GPT-OSS-20B](models/GPT-OSS-20B/README.md) | Reasoning, coding, tools and JSON schemas · 8K | [Results](models/GPT-OSS-20B/BENCHMARKS.md) |
| [Ornith 1.5 35B A3B](models/Ornith-1.5/README.md) | Chat and optional reasoning · 8K · text | [Results](models/Ornith-1.5/BENCHMARKS.md) |

### Image generation

| Model and setup | Use | Benchmark |
| --- | --- | --- |
| [FLUX.2 klein 4B](models/FLUX.2-klein/README.md) | Text to image · ComfyUI, web or CLI | [Results](models/FLUX.2-klein/BENCHMARKS.md) |
| [Qwen-Image-2.1 MXFP4](models/Qwen-Image-2.1/README.md) | Image generation and editing · HTTP API or CLI | [Results](models/Qwen-Image-2.1/BENCHMARKS.md) |

### Video generation

| Model and setup | Use | Benchmark |
| --- | --- | --- |
| [FastWan FullAttn 5B](models/FastWan/README.md) | Text to silent video · ComfyUI | [Results](models/Wan2.2/BENCHMARKS.md) |
| [Wan2.2 TI2V-5B](models/Wan2.2/README.md) | Text or image to silent video · ComfyUI | [Results](models/Wan2.2/BENCHMARKS.md) |
| [MiniMax H3](models/MiniMax-H3/README.md) | Video with stereo audio · optional first/last frame · ComfyUI | [Results](models/MiniMax-H3/BENCHMARKS.md) |

<a id="meeting-recordings"></a>

### Meeting notes

[Meeting](models/Meeting/README.md) is a review candidate for transcripts,
speaker labels and timestamped notes from recordings. Notes need review against
the recording; live Teams capture and Studio integration are unsupported.
[Benchmark](models/Meeting/BENCHMARKS.md).

<a id="run-in-your-vllm-environment"></a><a id="native-cli"></a><a id="native-serving-presets"></a><a id="existing-commands-remain-supported"></a>

## Use your own vLLM environment

The pip-installable plugin can run supported presets inside an existing
**matching vLLM build**. It checks the checkpoint, runtime and native artifacts;
it does not install or replace vLLM, PyTorch or ROCm.

Activate the [preset's supported environment](docs/NATIVE_EXECUTION.md#qualified-combinations), then:

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton doctor
paiton serve minicpm5
```

See [native execution](docs/NATIVE_EXECUTION.md) for all presets, exact runtime
requirements, local weights and offline use. The native `qwen38-nvfp4` preset
runs without DFlash2 or the 3-bit weights; use the [Qwen3.8 container guide](models/Qwen3.8-MXFP4-DFlash2/README.md)
for those release profiles and their benchmark results.

## Model weights and existing downloads

Cloning this repository gets the launchers and guides. Follow your model's guide
to download its pinned weights or reuse an existing copy. See
[weights and caches](docs/MODEL_WEIGHTS.md) for local folders and Docker mounts.

## Requirements

- **Tested GPU:** one Radeon AI PRO R9700, 32 GB, RDNA4 / `gfx1201`. Other GPUs have not been qualified.
- **Containers:** Linux, Docker and AMD device access through `/dev/kfd` and `/dev/dri`; ComfyUI launchers also need Docker Compose.
- **Native plugin:** the preset's exact supported environment, listed in [native execution](docs/NATIVE_EXECUTION.md).
- **Memory and storage:** depend on the model, context, concurrency and image/video settings; check the model guide before downloading.

## Where to find things

| Looking for | Go to |
| --- | --- |
| Weights and cache setup | [Model weights](docs/MODEL_WEIGHTS.md) |
| Plugin presets, runtime requirements and offline use | [Native execution](docs/NATIVE_EXECUTION.md) |
| Bundle packaging and the older compatibility launcher | [Native packaging](docs/NATIVE_PACKAGING.md) · [Existing vLLM](docs/EXISTING_VLLM.md) |
| Runtime and wheel downloads | [GitHub Releases](https://github.com/Eliovp-BV/paiton-vllm-plugin/releases) · [Containers](https://github.com/users/Eliovp/packages/container/package/paiton-vllm-plugin) |
| Published weights | [Hugging Face · EliovpAI](https://huggingface.co/EliovpAI) |
| Plugin and CLI source | [paiton_vllm_plugin](paiton_vllm_plugin) |
| Questions and bug reports | [Issues](https://github.com/Eliovp-BV/paiton-vllm-plugin/issues) |

## About Paiton

This repository contains public integrations and compiled runtime artifacts.
The compiler is proprietary. The plugin is [Apache-2.0 licensed](LICENSE);
model weights and bundled components retain their own licenses. See
[third-party notices](THIRD_PARTY_NOTICES.md) and each model's notices.

[More about Paiton](https://eliovp.com/products/paiton).
