# Qwen3.8 27B on one R9700

Choose **MXFP4** or **3-bit**. Both use vLLM with DFlash2 on a Radeon AI PRO R9700
(32 GB), with a 65,536-token context and up to eight scheduled requests by default.

| Run | What you get |
| --- | --- |
| [run-mxfp4.sh](run-mxfp4.sh) | Higher weight precision and stronger measured knowledge recall; FP8 KV cache. |
| [run-3bit.sh](run-3bit.sh) | Faster generation and more cache capacity; 3-bit W3A4 weights and a 4-bit KV cache. Requires the extra weights below. |

**Need long context?** `run-3bit.sh --context 262144` serves the model's full
262,144-token context to up to eight requests. [Run it like this](#want-longer-context-run-this).

Both choices use the same NVFP4 target download: the MXFP4 path requantizes it
at load time; the 3-bit path uses our additional weights. The scripts select the
weight mode explicitly, even if you have downloaded both.

[Compare measured speed and cache capacity](#current-benchmark-results).

## Before you start

- Linux x86-64, Python 3, [Hugging Face CLI (`hf`)](https://huggingface.co/docs/huggingface_hub/guides/cli), and Docker usable without `sudo`.
- One R9700 with 32 GB VRAM and AMD device access (`/dev/kfd` and `/dev/dri`).
- About **75 GB free disk**, including the container, target, drafter and optional 3-bit weights.

Run the commands below from the repository root, in the same terminal:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
```

Already cloned it? Use your existing checkout. The launcher pins the
**29 September r2** image, including the DFlash2 draft-head fix.

## Model weights and existing downloads

Both variants need these two downloads. Set the paths again when opening a new
terminal; the downloaded files and runtime cache are reused.

```bash
export PAITON_TARGET_DIR="$PWD/model-cache/qwen38-nvfp4"
export PAITON_DRAFT_DIR="$PWD/model-cache/qwen38-dflash2"
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-20260928"
mkdir -p "$PAITON_TARGET_DIR" "$PAITON_DRAFT_DIR" "$PAITON_CACHE_DIR"

hf download unsloth/Qwen3.8-27B-NVFP4 \
  --revision f0b7c9e722f5565102fff8481c99e4d86ae099c7 \
  --local-dir "$PAITON_TARGET_DIR"
hf download tcclaviger/Qwen3.8-27B-DFlash2-FP8 \
  --revision ee0cb26a8279b7910cc28d82a8a3e15e4728d56f \
  --local-dir "$PAITON_DRAFT_DIR"
```

Already downloaded these revisions? Set the variables to your complete local
folders instead. Each needs its own `config.json` and weights; keep the target's
tokenizer too. For linked Hugging Face cache snapshots, use the
[cache setup instructions](REFERENCE.md#already-in-the-hugging-face-cache).

### Optional: 3-bit W3A4 weights

For `run-3bit.sh`, also download these **9.55 GB** of weights. Keep the target
and drafter above; this is an add-on.

```bash
export PAITON_W3ROT_DIR="$PWD/model-cache/qwen38-w3rot-int3"
mkdir -p "$PAITON_W3ROT_DIR"
hf download EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4 \
  --revision 7a2e3d702144f744c1c5a46eaa42c7a2309d5cf7 \
  --local-dir "$PAITON_W3ROT_DIR"
(cd "$PAITON_W3ROT_DIR" && sha256sum -c SHA256SUMS)
```

## Start the server

Choose **one**:

```bash
# MXFP4
bash models/Qwen3.8-MXFP4-DFlash2/run-mxfp4.sh

# Or 3-bit
bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh
```

The server runs in the foreground. First startup with an empty runtime cache
can take about **seven minutes**; wait for `/health` to succeed. The container
includes the runtime; no private compiler checkout is needed.

**API:** `http://127.0.0.1:18982/v1` · **Model:** `Qwen3.8`

From another terminal:

```bash
curl --fail http://127.0.0.1:18982/health
curl --fail http://127.0.0.1:18982/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8","messages":[{"role":"user","content":"Write a Python function to remove duplicates from a list."}],"max_tokens":256,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

Stop with `docker stop paiton-qwen38` before switching variants or settings.
Add `--detach` to launch in the background; `docker logs -f paiton-qwen38` follows its log.

<a id="images-and-vision"></a><a id="long-context-200k-and-220k"></a>

## Want longer context? Run this

Download the [3-bit weights](#optional-3-bit-w3a4-weights), stop any running server, then:

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --context 262144
```

Same image, same API and model name. Measured on one R9700, 1 October:

| | |
| --- | --- |
| Context per request | 262,144 tokens (input plus output), up to eight requests |
| New 258K-token prompt, time to first token | 126 s |
| Same document again (prefix cache) | 1.7 s |
| Accuracy after a 257K-token document | GSM8K 95.8 %, HumanEval 91.5 %, MMLU-Pro 60.3 %: no measurable loss against short context |
| Short-request speed | 174 tok/s single-stream, 459 tok/s with eight concurrent requests (65K default: 179 / 479) |

[All long-context results](#current-benchmark-results) · [Details and options](REFERENCE.md#long-context-200k-and-220k)

- **Images:** add `--vision` and use `--context 245000`.
- **Short requests beside a long prompt:** add `--long-prefill-threshold 2048`. They are answered within
  seconds; the long prompt takes about 16 % longer (146 s instead of 126 s at 258K).
- **What stays cached:** the cache holds 281,665 tokens: one full-length conversation plus short requests, or
  several shorter ones. A different long document replaces the previous one in the cache.
- **MXFP4 weights:** `run-mxfp4.sh --context 200000` runs one conversation of up to 200,000 tokens (220,000 is the
  largest tested), text only.
- The long-context mode uses prefix caching with an FP8 KV cache and turns thinking off by default.

## Common options

Add these to **either** script:

| Option | What it changes |
| --- | --- |
| `--vision` | Enables image input and reduces the KV allocation to make room: in the 65K mode, and in the long-context mode with the 3-bit weights up to `--context 245000`. |
| `--context 262144` | [Long-context mode](#want-longer-context-run-this). 3-bit weights: the full 262,144-token context, up to eight requests. MXFP4: one conversation, up to 220,000. |
| `--long-prefill-threshold 2048` | Long-context mode: answer short requests within seconds while a long prompt is being processed (the long prompt takes about 16% longer). |
| `--profile desktop` | Smaller starting preset for a shared GPU: 32K context, one request, 2 GiB KV. |
| `--thinking off` | Disables thinking by default; clients can override it. |
| `--kv-cache fp8` | 65K mode: uses FP8 instead of the 3-bit preset's default 4-bit KV cache. |
| `--port 18082` | Changes the API port. |
| `--help` / `--dry-run` | Lists all options / prints the Docker command without starting it. |

For example:

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --vision
# After stopping that server, run the long-context mode with image input on the 3-bit weights:
bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --context 245000 --vision
```

Context includes **input plus output**. The 65K mode has prefix caching off. With MXFP4, use `--vision`
in the 65K mode or the desktop preset: the vision encoder needs memory that the MXFP4 long-context pool does not
have, so that mode refuses it. The launcher reduces the KV budget for vision; custom memory settings replace that
budget. See the [reference](REFERENCE.md#gpu-context-and-memory-controls) for GPU selection, custom memory budgets
and measured capacity.

## Current benchmark results

Measured on **one 300 W R9700**. Speed depends on prompt length, workload and
concurrency; use these as reference measurements for each choice.

**Long context, 3-bit weights: `run-3bit.sh --context 262144`, 1 October.** FP8 KV with prefix caching; the
cache holds 281,665 tokens (one full conversation plus short requests).

| | 198,989-token prompt | 257,992-token prompt |
| --- | ---: | ---: |
| New prompt, time to first token | 85 s | 126 s |
| Identical prompt reused | 1.2 s | 1.7 s |
| Decode at this depth | 73–78 tok/s | 72–76 tok/s |

Accuracy with every question placed after a 257K-token document, against the same weights at short context
(paired per question; all differences are within the 95 % intervals):

| Benchmark | Short context | After 257K tokens |
| --- | ---: | ---: |
| GSM8K 5-shot (1,319) | 95.30 | 95.83 |
| HumanEval pass@1 (164) | 93.90 | 91.46 |
| MMLU-Pro subset, 0-shot (14 × 100) | 59.71 | 60.29 |

With `--vision` the same mode serves images up to `--context 245000` (a 200K-token prompt with a chart: 87 s cold,
1.3 s reused). MXFP4 keeps the one-request long mode up to 220K (199K prompt: 102 s cold, 1.2 s reused, 63–68 tok/s).
[Long-context details and accuracy](REFERENCE.md#long-context-200k-and-220k) ·
[Vision checks](REFERENCE.md#images-and-vision).

**Short-request speed in the 262K mode, 1 October (3-bit weights, full 20-pass BetterBench, thinking off, cold prefix cache).**
The same tool and settings as the September tables, run for 20 passes per category on
`run-3bit.sh --context 262144` and on the 65K default (`run-3bit.sh --thinking off`) in fresh processes:

| BetterBench row | 262K mode | 65K default |
|---|---:|---:|
| Weighted single-stream decode | 174.3 tok/s | 178.9 tok/s |
| Gap between stream updates, p99 | 29.3 ms | 28.0 ms |
| Time to first token, p50 (short prompts) | 85 ms | 86 ms |
| Eight concurrent requests, aggregate output | 458.9 tok/s | 478.8 tok/s |
| Prefill at 47K input tokens (64K depth) | 3,549 tok/s | 3,481 tok/s |
| Prefill at 94K / 184K input tokens | 3,040 / 2,395 tok/s | – |

The 262K mode keeps the FP8 cache with prefix caching; the 65K default uses the 4-bit cache without it.
[Reports and numbers](benchmarks/2026-10-01-262k/README.md).

**65K serving: speed comparison with FP8 KV in both arms, 26 September.**

| Measurement | MXFP4 | 3-bit |
| --- | ---: | ---: |
| Weighted single-stream decode | 156.1 tok/s | 184.4 tok/s |
| Eight concurrent requests, aggregate output | 428.0 tok/s | 492.1 tok/s |
| Prefill, approximately 5.9K input tokens | 3,831 tok/s | 4,165 tok/s |

These runs predate the current image. `run-mxfp4.sh` still uses FP8 KV;
`run-3bit.sh` now defaults to KV4, so this is **not an exact benchmark of today's
two default commands**. Add `--kv-cache fp8` to select the 3-bit FP8 path.
[Full settings, results and subsequent KV4 checks](REFERENCE.md#current-benchmark-results).

In the paired FP8 KV comparison, the 3-bit weights traded some accuracy for
speed and memory: the MMLU-Pro subset fell **2.86 points** versus MXFP4.
Math, coding and retrieval differences
were within the reported uncertainty. [Quality results](REFERENCE.md#faster-decode-and-prefill-3-bit-w3a4-weights-optional).

## Advanced setup and release history

[Release reference](REFERENCE.md) contains image pins, release notes, detailed
benchmarks, long-context checks, KV tuning and older releases. The existing
`run-rocm10*.sh` scripts remain available; `run-rocm10.sh` automatically chooses
3-bit when `PAITON_W3ROT_DIR` is set and MXFP4 otherwise.

### Native serving

Already have the exact supported vLLM environment? `paiton serve qwen38-nvfp4`
uses the separate **text-only, non-speculative** preset: 65K, FP8 KV, no DFlash2
or 3-bit weights. Follow [native setup](REFERENCE.md#native-serving) for installation.
Its bundle is `qwen38-rocm10-native-20260921`, profile
`qwen38-nvfp4-w4a8-text-65k`, with the same pinned target revision above.

[Third-party notices](THIRD_PARTY_NOTICES.md) ·
[3-bit weight license and calibration](https://huggingface.co/EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4)
