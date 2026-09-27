<h1 align="center">Paiton</h1>

<p align="center">
  <strong>More from your AMD GPU.</strong><br>
  Faster local chat, coding, image and video models for AMD Radeon —<br>
  as ready-to-run containers or a plugin for your own vLLM.
</p>

<p align="center">
  <a href="#models"><img src="https://img.shields.io/badge/Tested_on-Radeon_AI_PRO_R9700-ED1C24?style=flat-square" alt="Tested on Radeon AI PRO R9700"></a>
  <a href="#requirements"><img src="https://img.shields.io/badge/Architecture-RDNA4_%C2%B7_32_GB-30363D?style=flat-square" alt="RDNA4 · 32 GB"></a>
  <a href="#chat-reasoning-and-coding"><img src="https://img.shields.io/badge/LLM_API-OpenAI_compatible-30363D?style=flat-square" alt="OpenAI-compatible language-model API"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Plugin_license-Apache_2.0-30363D?style=flat-square" alt="Plugin license: Apache 2.0"></a>
</p>

<p align="center">
  <a href="#quick-start"><strong>Quick start</strong></a> ·
  <a href="#models">Models</a> ·
  <a href="#launch-commands">Launch commands</a> ·
  <a href="#use-your-own-vllm-environment">Your own vLLM</a> ·
  <a href="#where-to-find-things">Docs</a> ·
  <a href="https://github.com/Eliovp-BV/paiton-vllm-plugin/releases">Releases</a> ·
  <a href="https://github.com/Eliovp-BV/paiton-studio">Paiton Studio</a>
</p>

Paiton makes open models run faster on AMD Radeon GPUs. It combines Paiton's
compiler-generated GPU kernels with established runtimes — **vLLM** for language
models, **Diffusers** and **ComfyUI** for images and video — and ships every model
with a setup guide, a defined support scope and a reproducible benchmark.

| This repository contains | Use it to |
| --- | --- |
| [**Model packages**](#models) — one folder per model with its guide, launchers and benchmark report | Pick a model and see exactly what is supported and measured |
| [**Containers**](#launch-commands) — pinned images that bring their own inference environment | Run a model with one command on Linux with Docker |
| [**`paiton` CLI and vLLM plugin**](#use-your-own-vllm-environment) — a pip-installable wheel | Serve a supported model by name in your existing vLLM environment |

Tested on one **Radeon AI PRO R9700** (32 GB, RDNA4, `gfx1201`). Other GPUs have not been qualified.

> [!TIP]
> Prefer a desktop app? [Paiton Studio](https://github.com/Eliovp-BV/paiton-studio)
> brings the supported models together for local chat, writing, images and video.

## Performance at a glance

Measured on one Radeon AI PRO R9700. Three examples from the model library:

<table>
  <tr>
    <td align="center" valign="top" width="33%">
      <h3>+70.1%</h3>
      <strong>Output throughput</strong>
      <p>Qwen3-Coder 30B A3B<br>vs stock · two concurrent requests</p>
      <a href="models/Qwen3-Coder-30B/BENCHMARKS.md">See the benchmark →</a>
    </td>
    <td align="center" valign="top" width="34%">
      <h3>184.4 tok/s</h3>
      <strong>Weighted decode</strong>
      <p>Qwen3.8 27B 3-bit W3A4 + DFlash2<br>65K profile · +19.9% vs the MXFP4 release</p>
      <a href="models/Qwen3.8-MXFP4-DFlash2/README.md#current-benchmark-results">See the benchmark →</a>
    </td>
    <td align="center" valign="top" width="33%">
      <h3>36.7% less</h3>
      <strong>Sampled GPU memory</strong>
      <p>FLUX.2 klein 4B vs stock<br>1024 × 1024 · four steps</p>
      <a href="models/FLUX.2-klein/BENCHMARKS.md">See the benchmark →</a>
    </td>
  </tr>
</table>

Different workloads — not a cross-model ranking or a universal speedup claim.
[How to read the numbers →](#reading-the-benchmarks)

<a id="run-with-a-managed-container"></a><a id="run-in-your-vllm-environment"></a>

## Quick start

Two ways to run a model; both serve the same OpenAI-compatible API. The example
uses **MiniCPM5-2B**, the smallest download in the library (2.11 GB).

**A. Run a container** — needs Linux, Docker and access to the AMD GPU. The
container brings its own inference environment.

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
./models/MiniCPM5-2B/serve-docker.sh
```

**B. Use your own vLLM environment** — first activate the model's
[supported vLLM build](#use-your-own-vllm-environment). Paiton does not change
your vLLM, PyTorch or ROCm installation.

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton serve minicpm5
```

The first launch downloads the weights (reusing your Hugging Face cache when it
already has them) and may build runtime components. When the server reports that
it is ready, send a request from another terminal:

```bash
curl http://127.0.0.1:8036/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minicpm5-2b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

For a terminal chat, run `python3 models/MiniCPM5-2B/chat.py` from the repository.

**Next:** [pick another model](#models) ·
[already have the weights?](#model-weights-and-existing-downloads) ·
[requirements](#requirements)

<a id="model-library"></a>

## Models

Each model name links to its guide: weights, launch options, API examples,
limits and the full benchmark report.

<sub>Context = input + generated tokens. GPU memory = sampled driver VRAM at the
tested settings, including runtime overhead; it is not a minimum requirement.
— = not reported. C1, C2, C8 = one, two, eight concurrent requests.</sub>

<a id="chat-reasoning-coding-and-visual-understanding"></a>

### Chat, reasoning and coding

All language models expose an OpenAI-compatible API through vLLM.

| Model | Best for · input | Context | GPU memory | Measured result |
| --- | --- | ---: | ---: | --- |
| [**MiniCPM5-2B**](models/MiniCPM5-2B/README.md) · W4A16 | Lightweight chat, coding and tools · text | 8K | ~4.75 GiB | [+54.4% output tok/s](models/MiniCPM5-2B/BENCHMARKS.md#sustained-generation-and-prefill) vs stock · C1 |
| [**Qwen3.8 27B MXFP4 + DFlash2**](models/Qwen3.8-MXFP4-DFlash2/README.md) · optional 3-bit W3A4 weights | Long-context chat, coding and tools · text | 65K / 200K | — | [184.4 tok/s weighted decode · 492.1 tok/s at C8 · 4,165 tok/s prefill at 8K](models/Qwen3.8-MXFP4-DFlash2/README.md#current-benchmark-results) with the 3-bit weights · 65K |
| [**Qwen3.8 27B Qronos**](models/Qwen3.8/README.md) | General chat, coding and optional reasoning · text | 8K | — | [+54.3% output tok/s](https://eliovp.com/blog/paiton-qwen38-radeon-ai-pro-r9700) vs stock · coding · C1 |
| [**Qwen3.8 NEO CODER MAX 27B**](models/Qwen3.8-NEO-CODER-MAX/README.md) · Q4_K_M GGUF | Coding and visual chat · text + one image | 8K | ~23.74 GiB | [6.4% lower request latency](models/Qwen3.8-NEO-CODER-MAX/BENCHMARKS.md#matched-text-comparison) vs llama.cpp · C1 |
| [**Qwen3-Coder 30B A3B**](models/Qwen3-Coder-30B/README.md) | Code writing, review, testing and tools · text | 4K | ~20.1 GiB | [+70.1% output tok/s at C2 · +21.3% at C1](models/Qwen3-Coder-30B/BENCHMARKS.md) vs stock |
| [**GPT-OSS-20B**](models/GPT-OSS-20B/README.md) | Reasoning, coding, tools and JSON schemas · text | 8K | ~17.0 GiB | [54.0% lower request latency](models/GPT-OSS-20B/BENCHMARKS.md) vs the fastest qualified stock run · C1 |
| [**Ornith 1.5 35B A3B**](models/Ornith-1.5/README.md) | Chat and optional reasoning · text | 8K | — | [+27.0% output tok/s](models/Ornith-1.5/BENCHMARKS.md) vs stock, including DFlash · C1 |

> [!NOTE]
> **Which Qwen3.8?** **MXFP4 + DFlash2** is the current release: 65K or 200K context
> with DFlash2 speculative decoding (the 200K profile is qualified for one active
> request). On the 65K profile, optional [3-bit W3A4 weights](https://huggingface.co/EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4)
> add +19.9% weighted decode
> for about 3 points of MMLU-Pro knowledge recall; the opt-in n-gram co-drafting
> measured +27% decode on an agentic coding session.
> **Qronos** is an 8K package with optional reasoning; **NEO CODER MAX**
> adds single-image input. The native `qwen38-nvfp4` preset runs the MXFP4 model in
> your own vLLM without DFlash2, so the DFlash2 benchmark does not apply to it.

- Reasoning tokens share the output budget with the answer; choose the mode
  explicitly when comparing responses or benchmarking. MiniCPM5's W4 thinking mode
  is experimental; direct answers are its default.
- Qronos and Ornith are text-only in these packages, although their upstream
  architectures include vision components.
- NEO runs the author's mixed GGUF weights directly through Paiton's vLLM
  integration: see its [image API](models/Qwen3.8-NEO-CODER-MAX/IMAGE_API.md) and
  [native GGUF notes](models/Qwen3.8-NEO-CODER-MAX/NATIVE_GGUF.md). Its MTP path is disabled.

### Image generation

| Model | What it does | GPU memory | Measured result |
| --- | --- | ---: | --- |
| [**FLUX.2 klein 4B**](models/FLUX.2-klein/README.md) | Text → image · 1024 × 1024, four steps · ComfyUI, web or CLI | ~14.6 GiB | [36.7% less GPU memory · 16.2% lower latency](models/FLUX.2-klein/BENCHMARKS.md) vs stock |
| [**Qwen-Image-2.1 MXFP4**](models/Qwen-Image-2.1/README.md) | Text → image (also transparent RGBA) at 2048 × 2048 · image editing at 1024 × 1024 · 40 steps · HTTP API or CLI | up to 25.62 GiB | [103.3 s per image, warm median](models/Qwen-Image-2.1/BENCHMARKS.md#release-v102) with the v1.0.2 default profile · 133.7 s bit-exact |

The Qwen-Image container can also serve Paiton's MXFP4 conversion of an
[uncensored fine-tune](https://huggingface.co/EliovpAI/Qwen_Image-2.1-Uncensored-MXFP4-Paiton)
(`--model uncensored`; its R9700 performance has not been measured). A
[portable MXFP4 package](https://huggingface.co/EliovpAI/Qwen_Image-2.1-MXFP4) covers
other hardware and runtimes. FLUX image editing is not qualified.

### Video generation

| Model | What it does | GPU memory | Measured result |
| --- | --- | ---: | --- |
| [**FastWan FullAttn 5B**](models/FastWan/README.md) | Text → silent video · 480/720-class presets | 23.2–30.4 GiB | [Up to 4.7% lower clip latency](models/Wan2.2/BENCHMARKS.md#complete-measured-results) vs stock · 832 × 480, 49 frames |
| [**Wan2.2 TI2V-5B**](models/Wan2.2/README.md) | Text + optional image → silent video | 23.0–24.4 GiB | [1.1% lower text-input latency](models/Wan2.2/BENCHMARKS.md#complete-measured-results) · image cases 0.1–0.8% slower; stock is the default |
| [**MiniMax H3**](models/MiniMax-H3/README.md) | Text + optional first/last frame → video with native stereo audio | up to 31.1 GiB | [16.7% lower clip latency](models/MiniMax-H3/BENCHMARKS.md#continuous-15-second-qualification) vs stock · 15-second Turbo8 clip |

Starting from an image? Use Wan2.2 or MiniMax H3; FastWan is qualified for text
input only. Video timings include file encoding. Gains from distillation or fewer
sampling steps are not counted as Paiton acceleration.

<a id="meeting-recordings"></a>

### Meeting notes

[**Meeting**](models/Meeting/README.md) *(review candidate)* turns a recording into a
transcript with anonymous speaker labels and timestamped notes, using Parakeet
speech recognition, speaker diarization and a compact Granite summary model, from
the CLI or a container.
[2.2% lower processing time](models/Meeting/BENCHMARKS.md#final-candidate-complete-matched-comparison)
vs stock: 291.0 → 284.5 s for a 39-minute meeting.

> [!IMPORTANT]
> Notes have incomplete coverage and must be reviewed against the recording. Live
> Teams capture and Paiton Studio integration are not supported, and processing
> times vary substantially between runs.

<a id="managed-containers"></a>

## Launch commands

Run launchers from the repository root. Run one model at a time on the tested
single-GPU setup (several language models use port 8000).

**Language models**

| Model | Container | Your own vLLM | API port · model name |
| --- | --- | --- | --- |
| MiniCPM5-2B | `./models/MiniCPM5-2B/serve-docker.sh` | `paiton serve minicpm5` | 8036 · `minicpm5-2b` |
| Qwen3.8 MXFP4 + DFlash2 | `bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10-65k.sh` ¹ | `paiton serve qwen38-nvfp4` ² | 18982 · `Qwen3.8` |
| Qwen3.8 Qronos | `./models/Qwen3.8/serve-docker.sh` | `paiton serve qwen38-qronos` | 8000 · `qwen38` |
| Qwen3.8 NEO CODER MAX | `./models/Qwen3.8-NEO-CODER-MAX/serve-docker.sh` | `paiton serve qwen38-neo` | 8000 · `qwen38-neo` |
| Qwen3-Coder 30B | `./models/Qwen3-Coder-30B/serve-docker.sh --chat` | `paiton serve qwen3-coder` | 8010 · `qwen3-coder` |
| GPT-OSS-20B | `./models/GPT-OSS-20B/serve-docker.sh` | `paiton serve gpt-oss-20b` | 8020 · `gpt-oss-20b` |
| Ornith 1.5 | `./models/Ornith-1.5/serve-docker.sh --chat` | `paiton serve ornith` | 8000 · `ornith` |

¹ Prepare the target and draft weights first
([how](models/Qwen3.8-MXFP4-DFlash2/README.md#model-weights-and-existing-downloads));
set `PAITON_W3ROT_DIR` to add the [optional 3-bit weights](models/Qwen3.8-MXFP4-DFlash2/README.md#optional-3-bit-w3a4-weights);
use `run-rocm10-200k.sh` for the 200K profile.
² Text-only, 65K, without speculative decoding; see [native presets](#use-your-own-vllm-environment).

`--chat` opens a terminal chat once the server is ready. Each model guide shows its
chat command and API examples.

**Images, video and meeting notes**

| Model | Launcher | When ready |
| --- | --- | --- |
| FLUX.2 klein | `./models/FLUX.2-klein/launch.sh` | Open [ComfyUI · 8188](http://127.0.0.1:8188/?paiton=1) |
| Qwen-Image-2.1 MXFP4 | `./models/Qwen-Image-2.1/serve-docker.sh` | Image API on port 8191 · [examples](models/Qwen-Image-2.1/README.md#generate-an-image) |
| FastWan | `./models/FastWan/launch.sh` | Open [ComfyUI · 8192](http://127.0.0.1:8192/?paiton=1&preset=fast) |
| Wan2.2 | `./models/Wan2.2/launch.sh` | Open [ComfyUI · 8192](http://127.0.0.1:8192/?paiton=1&preset=base) |
| MiniMax H3 | `./models/MiniMax-H3/launch.sh` | Open [ComfyUI · 8190](http://127.0.0.1:8190/?paiton=1&studio=1) |
| Meeting | `./run-docker.sh --paiton /path/to/meeting.mp4 /path/to/new-result` (from `models/Meeting`) | Results in the output folder · [prepare the models first](models/Meeting/README.md) |

The ComfyUI launchers also need Docker Compose. The included workflows expose the
supported prompts, inputs and generation settings.

## Model weights and existing downloads

Cloning this repository gets you the launchers and guides, **not the model weights**.

- **First launch:** most launchers and native presets download their pinned
  weights automatically. Qwen3.8 MXFP4 + DFlash2 and Meeting need a preparation
  step first.
- **Already downloaded?** Every model guide has a *Model weights and existing
  downloads* section for weights in a local folder or in your Hugging Face cache.
  Use the exact model and quantization it lists. Containers only see host caches
  that are mounted as described there.
- **Native CLI:** weights already in your Hugging Face cache are reused and missing
  files are downloaded. For a checkpoint folder elsewhere, run
  `paiton --model-dir /absolute/path serve NAME`.

The [weights and cache guide](docs/MODEL_WEIGHTS.md) explains cache locations
(`HF_HOME`, `HF_HUB_CACHE`), exact revisions and Docker mounts.

<a id="native-cli"></a><a id="native-serving-presets"></a><a id="existing-commands-remain-supported"></a>

## Use your own vLLM environment

The `paiton` CLI serves a supported model by name inside an existing vLLM
environment. It selects the pinned checkpoint, downloads and verifies the matching
native bundle, and starts vLLM with the qualified profile.

- It needs neither the private compiler nor a repository checkout.
- It leaves your vLLM, PyTorch and ROCm installation untouched, and explains the
  mismatch instead of starting when the runtime or payload does not match.
- Installing it does not change plain `vllm serve` commands.

Install the wheel as in the [Quick start](#quick-start), activate the preset's
environment, then:

```bash
paiton doctor           # check the environment
paiton models           # list the presets and their preparation status
paiton serve minicpm5   # launch a preset by name
```

| Preset | Model | Supported environment | Behavior |
| --- | --- | --- | --- |
| `minicpm5` | [MiniCPM5-2B](models/MiniCPM5-2B/README.md#native-serving) | Python 3.14 · ROCm 7.14 · pinned vLLM 0.26.1 build | Thinking disabled by default |
| `qwen38-nvfp4` | [Qwen3.8 NVFP4](models/Qwen3.8-MXFP4-DFlash2/README.md#native-serving) | Python 3.12 · ROCm 10 · vLLM 0.29.0 | Text-only · 65K · FP8 KV · no speculation · lossy NVFP4 → MXFP4 conversion in GPU memory (files on disk unchanged) |
| `qwen38-qronos` | [Qwen3.8 Qronos](models/Qwen3.8/README.md#native-serving) | Python 3.12 · ROCm 7.14 · pinned vLLM 0.28.0 build | Release W4 LM head, quantized while loading (lossy) |
| `qwen38-neo` | [Qwen3.8 NEO](models/Qwen3.8-NEO-CODER-MAX/README.md#native-serving) | Python 3.12 · ROCm 7.14 · pinned vLLM 0.28.0 build | Text + one image |
| `qwen3-coder` | [Qwen3-Coder](models/Qwen3-Coder-30B/README.md#native-serving) | Python 3.12 · ROCm 7.14 · pinned vLLM 0.28.0 build | |
| `gpt-oss-20b` | [GPT-OSS-20B](models/GPT-OSS-20B/README.md#native-serving) | Python 3.14 · ROCm 7.14 · pinned vLLM 0.26.1 build | |
| `ornith` | [Ornith 1.5](models/Ornith-1.5/README.md#native-serving) | Python 3.12 · ROCm 7.14 · pinned vLLM 0.28.0 build | Text-only · 8K · no speculation · cached lossless reshard |

These are separate existing environments; Paiton does not switch between them.
Image, video and meeting models use their [launchers](#launch-commands) instead.

> [!NOTE]
> Native presets are not the published benchmark profiles. `qwen38-nvfp4`, for
> example, does not use DFlash2; the DFlash/DFlash2 results describe the container
> profiles.

<details>
<summary><strong>Existing weights, offline use, custom ports and explicit profiles</strong></summary>

Put Paiton options **before** `serve` or `vllm`, and vLLM arguments after the model:

```bash
# Verify and remember an existing checkpoint without starting the server
paiton --model-dir /models/existing-minicpm5 --prepare-only serve minicpm5

# No downloads and no outbound runtime networking
paiton --offline serve minicpm5

# vLLM arguments go after the model
paiton serve minicpm5 --port 9000 --served-model-name local-mini

# An explicit profile with a local path (repository IDs also work)
paiton --profile minicpm5-awq-text-8k vllm serve /models/existing-minicpm5 --port 9000
```

`paiton vllm serve MODEL` is equivalent to `paiton serve`. The launchers, console
commands, container images, paths, flags, ports and cache variables used in
earlier blog and Reddit instructions still work.

The [native execution guide](docs/NATIVE_EXECUTION.md) covers checkpoint
identities, profiles, preparation, offline use and lockfiles.

</details>

<a id="requirements-and-first-launch"></a>

## Requirements

- **GPU:** tested on one Radeon AI PRO R9700 (32 GB, RDNA4). Smaller GPUs have not
  been qualified.
- **Containers:** Linux, Docker and access to the AMD GPU devices (`/dev/kfd`,
  `/dev/dri`). The ComfyUI launchers also need Docker Compose.
- **Your own vLLM:** the preset's supported environment, listed [above](#use-your-own-vllm-environment).
- **First launch:** downloads the weights and may build or compile runtime
  components; later launches reuse persistent caches. Host RAM, disk space and
  preparation time vary substantially, so check the model guide before downloading.
- **GPU memory:** reported figures apply to the tested settings. Longer context,
  more concurrent requests or higher resolutions can need more.

<a id="reading-the-results"></a>

## Reading the benchmarks

Each result compares Paiton with a baseline **on the same model and workload**; the
library is not a ranking across models. Every report documents the baseline,
sampling settings, repetitions, quality checks and known limitations.

<details>
<summary><strong>Throughput, latency and the baselines behind the numbers</strong></summary>

**Output throughput**

`Improvement = (Paiton tok/s ÷ baseline tok/s − 1) × 100`

MiniCPM5's highlighted comparison is **127.6 → 197.0 tok/s**. Qwen3-Coder's C2
comparison is **101.61 → 172.82 tok/s**. These are aggregate output rates, not
individual-stream decode rates.

**Latency reduction**

`Reduction = (baseline time − Paiton time) ÷ baseline time × 100`

A 50% latency reduction means twice the rate for equivalent fixed work,
not a 50% throughput increase.

**Baseline matters.** NEO is compared with llama.cpp, not stock vLLM. Its longer
128-output text workloads show 5.1% and 0.8% lower request latency; some
prefill-only cases favor llama.cpp. GPT-OSS uses the fastest qualified
4.819-second stock reference because stock timing varied between runs.

**Optimization scope matters.** Ornith's highlighted result includes DFlash
speculative decoding; Qwen3.8's includes lossy 3-bit weights that score about
3 MMLU-Pro points lower. Quantization, activation arithmetic and quality differences
are documented per model. MiniCPM5's repeated timings show unresolved variability;
meeting results also vary substantially.

</details>

## Where to find things

| Looking for | Go to |
| --- | --- |
| Setup, launch, API examples and limits for one model | `models/<model>/README.md` — linked from the [model tables](#models) |
| The full benchmark report for one model | The **Measured result** link in the [model tables](#models) |
| Weights, Hugging Face caches and Docker mounts | [docs/MODEL_WEIGHTS.md](docs/MODEL_WEIGHTS.md) |
| Native CLI details: checkpoints, profiles, offline use, lockfiles | [docs/NATIVE_EXECUTION.md](docs/NATIVE_EXECUTION.md) |
| How native bundles are packaged and qualified | [docs/NATIVE_PACKAGING.md](docs/NATIVE_PACKAGING.md) |
| The older compatibility launcher for an existing vLLM | [docs/EXISTING_VLLM.md](docs/EXISTING_VLLM.md) |
| Plugin and CLI source | [`paiton_vllm_plugin/`](paiton_vllm_plugin) |
| Wheel and native bundle downloads | [GitHub Releases](https://github.com/Eliovp-BV/paiton-vllm-plugin/releases) |
| Container images | [GitHub Container Registry](https://github.com/users/Eliovp/packages/container/package/paiton-vllm-plugin) |
| Weights published by Paiton | [Hugging Face · EliovpAI](https://huggingface.co/EliovpAI) |
| Licenses and third-party notices | [LICENSE](LICENSE) · [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) · each model's notices |
| Questions and bug reports | [Issues](https://github.com/Eliovp-BV/paiton-vllm-plugin/issues) |

## About Paiton

**Public integrations. Compiled runtime artifacts. A proprietary compiler.**

This repository distributes Paiton's public integrations and compiled runtime
artifacts. The Paiton compiler remains proprietary. Model weights are fetched
from their pinned publishers and retain their own licenses.

The vLLM plugin is [Apache-2.0 licensed](LICENSE). Bundled components retain their
applicable licenses, including ComfyUI and separate image tools. See the
[root notices](THIRD_PARTY_NOTICES.md) and each model's notices.

Paiton's broader work also includes AMD Instinct accelerators and multi-GPU
inference. [Explore Paiton or discuss your workload →](https://eliovp.com/products/paiton)

<p align="center">
  <strong>Build locally. Get more from your GPU.</strong><br><br>
  <a href="https://github.com/Eliovp-BV/paiton-vllm-plugin/releases">Release downloads</a> ·
  <a href="https://github.com/users/Eliovp/packages/container/package/paiton-vllm-plugin">Containers</a> ·
  <a href="https://huggingface.co/EliovpAI">Hugging Face</a> ·
  <a href="https://github.com/Eliovp-BV/paiton-studio">Paiton Studio</a> ·
  <a href="https://github.com/Eliovp-BV/paiton-vllm-plugin/issues">Report an issue</a>
</p>
