# Qwen3.8 MXFP4 + DFlash2 on regular vLLM

## Native serving

Activate the supported environment listed below, then install and serve:

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton serve qwen38-nvfp4
```

Paiton automatically downloads and verifies the native bundle and reuses the
pinned checkpoint in your Hugging Face cache. Missing checkpoint files are
downloaded from the publisher. To reuse an existing local copy, run
`paiton --model-dir /path/to/model serve qwen38-nvfp4`; successful preparation
remembers that path for later launches.

Use `paiton --prepare-only serve qwen38-nvfp4` to prepare without starting the
server, then `paiton --offline serve qwen38-nvfp4` for offline operation.
Paiton options precede `serve`; vLLM options such as `--port` follow the model.
See the [native setup guide](../../docs/NATIVE_EXECUTION.md) for installation,
offline use and troubleshooting. The existing container and Python commands
below remain supported.

- **Native bundle:** `qwen38-rocm10-native-20260921`; downloaded and verified automatically.
- **Checkpoint:** `unsloth/Qwen3.8-27B-NVFP4`, revision `f0b7c9e722f5565102fff8481c99e4d86ae099c7`.
- **Existing runtime:** Python 3.12, vLLM `0.29.0`, Torch `2.12.0+rocm10.0.0`, ROCm SDK 10.0.0; one `gfx1201` R9700. Full ABI/package pins appear in `paiton models`.
- **Preset / profile:** `qwen38-nvfp4` / `qwen38-nvfp4-w4a8-text-65k`.
- **Serving behavior:** Explicit text-only NVFP4→MXFP4 requantization, FP8 activations, FP8 KV, FP16 recurrent cache, 65K context, APC off, no speculation. Original weights stay unchanged; conversion is lossy and occurs only in device memory at model load. Existing upstream warmup/graph compilation remains.
- **API:** `http://127.0.0.1:18982/v1`, model name `Qwen3.8`. Wait for readiness; `curl http://127.0.0.1:18982/health` checks the server.

The shorter command uses the shared native resolver and the same installed
Python. `paiton --profile qwen38-nvfp4-w4a8-text-65k vllm serve /models/existing-qwen38-nvfp4`
also works. The source checkpoint is preserved. See the
[validation and compatibility notes](../../docs/NATIVE_EXECUTION.md#validation-status)
for exactly what was tested.


## Current release

**28 September 2026: one image for 65K multi-request serving and 200K long context.**
The same image runs in two modes on one Radeon AI PRO R9700 (32 GB). Choose the mode when you start it:

| Mode | Start command | Context per request | Requests at once | Best for |
|---|---|---:|---:|---|
| **65K** (default) | `bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10.sh` | 65,536 tokens | up to 8 | several users or agents at once |
| **Long context** | `bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10.sh --context 200000` | 200,000 tokens | 1 | one long conversation, a large codebase or document |

Without options you get the 65K text mode, the configuration we benchmark. Long context, image input
(`--vision`) and prefix caching start only when you ask for them.

- **65K mode.** With the optional [3-bit weights](#optional-3-bit-w3a4-weights), a 4-bit KV cache holds 1.7×
  as many tokens: six 61K-token or eight 32K-token requests now run together (248 and 339 tok/s combined),
  and four 61K-token requests decode 17% faster. Accuracy matches the FP8 cache within noise: GSM8K 96.1%,
  HumanEval 93.9%, MMLU-Pro subset 61.4%, needle at 61K tokens 100%.
- **Long-context mode.** One conversation at a time, with prefix caching. A new 199K-token prompt takes 94 s
  to the first token (102 s with MXFP4); a follow-up turn that reuses it takes 1.2 s. Decode runs at 73 to
  78 tok/s at that length, and every planted fact in our 199K-token test prompts was found. 220,000 tokens
  is tested too.
- **Images.** Add `--vision` in the 65K mode for screenshots, UI captures or charts; see
  [Images and vision](#images-and-vision).
- **Crash fix.** The engine no longer stops under three to five concurrent requests; see the
  [release notes](#release-notes-28-september-2026).

Both modes work with the 3-bit weights (faster, recommended) or with the MXFP4 checkpoint alone; see
[Model weights](#model-weights-and-existing-downloads). The image is
`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20260929-r2`; the launcher pins it by digest, and no registry
login is required.

### Release notes: 28 September 2026 (image r2 of 29 September)

The image `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20260929-r2`
(`sha256:1195f31329966b3dc4e8e2d17327d339827b3d2b09165f9969b053a6fc2db045`) is the 26 September 65K image with the changes below. Weights, drafter and the
other serving settings are unchanged. Update this repository to get the launcher that selects it.

- **Fixed in r2 (29 September): DFlash2 draft head built from an uninitialised tensor.** The
  drafter has no `lm_head` of its own; the target's is shared in after the drafter's weights
  load. The int2 draft head decided at load time whether that tensor was still empty and, if
  the memory it got happened to hold old data, quantised garbage. Text stayed correct (the
  target is untouched), but DFlash2 then accepted 0% of its draft tokens and decode ran at
  about a fifth of the published speed. Our test machines always received zeroed memory and
  never showed it; a user on Unraid did, and reported it with the diagnosis. The draft head is
  now always built on first use from the real head. The 20, 24, 26 and 28 September (r1)
  images all carry the affected file; on them, run the container with
  `-e RADIANCE_FAST_DRAFT=0` (the stock bf16 draft head, about 10% slower decode than the int2
  head) or update.
- **New: one image for both modes.** `run-rocm10.sh` starts the 65K mode. A `--context` above 65,536, for
  example `--context 200000`, starts the long-context mode on the same image: one request, prefix caching
  and an 8 GiB FP8 KV cache (the settings of the earlier 200K `chat` profile). `run-rocm10-65k.sh` and
  `run-rocm10-200k.sh` still work; the 200K script now uses this image instead of the 18 September 200K
  image.
- **New: image input.** `--vision` serves the checkpoint's vision encoder in the 65K mode; see
  [Images and vision](#images-and-vision).
- **New: 4-bit KV cache** with the 3-bit weights in the 65K mode, on by default; see
  [4-bit KV cache](#more-context-capacity-4-bit-kv-cache). New launcher option `--kv-cache auto|kv4|fp8`.
- **Fixed: engine stop under concurrent load.** With three to five requests at once, and rarely with one
  very short prompt, the 26 September image could stop with
  `RuntimeError: Paiton GDN norm nonfinite/arithmetic error: 1`, and requests failed until a restart.
  Unused padding rows in some GPU steps held uninitialized memory, and a safety check in the
  recurrent-layer norm stopped the engine on them. The image now zeroes these rows: a soak test of 10,002
  such steps ran without errors, and outputs are unchanged.
- **If you stay on the 26 September image,** run its container with `-e RADIANCE_DYNAMIC_WIDTH=0`. In our
  runs this avoided more than 99% of these steps. The launcher does not pass it; `--dry-run` prints the full
  Docker command.
- **Unchanged:** prefill and single-request decode speed.

## Run the current release

You need:

- Linux x86-64 with Python 3 and Docker. Your user must be able to run `docker` without `sudo`, for example
  as a member of the `docker` group; `sudo` would drop the exported `PAITON_*` variables.
- One Radeon AI PRO R9700 with 32 GB VRAM and working AMD GPU device access.
- About 75 GB of free disk: about 40 GB for the image (a 9.6 GB download), 23.4 GB for the target, 2.1 GB
  for the draft and 9.55 GB for the optional 3-bit weights.

Run the following commands from the repository root. The image contains the runtime; download the target
and draft weights separately with the Hugging Face CLI (`hf`), or point the variables at existing copies of
these exact snapshots.

The commands below start the **65K mode**, the benchmark configuration. Add `--context 200000` for the
**long-context mode**; see [Long context](#long-context-200k-and-220k).

This release loads the **Unsloth NVFP4 checkpoint through the MXFP4 runtime path** and can also serve the
optional 3-bit W3A4 weights on top of it; see [Optional: 3-bit W3A4 weights](#optional-3-bit-w3a4-weights).
The AMD checkpoint and automatic downloader in the historical release below are for the older images.

## Model weights and existing downloads

Choose **one** of the following paths. The current container needs the target and
the draft; the 3-bit weights are optional:

| Component | Repository | Required revision |
| --- | --- | --- |
| Target | `unsloth/Qwen3.8-27B-NVFP4` | `f0b7c9e722f5565102fff8481c99e4d86ae099c7` |
| DFlash2 draft | `tcclaviger/Qwen3.8-27B-DFlash2-FP8` | `ee0cb26a8279b7910cc28d82a8a3e15e4728d56f` |
| 3-bit W3A4 weights (optional) | `EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4` | `d74ae7d5f5f1b4b45dd12fb1271e3664283a2ec1` |

The native `paiton serve qwen38-nvfp4` preset above is non-speculative and does
not use the 3-bit weights. The instructions here preserve the Docker release's
**DFlash2** profile.

### First download

Run this complete block from the repository root. The `hf download` commands
populate the model directories; the exports and `mkdir` commands alone do not.
Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
if `hf` is not available.

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

After both downloads complete, use [Launch with prepared local folders](#launch-with-prepared-local-folders).
If either download fails, resolve that failure before starting the server.

### Optional: 3-bit W3A4 weights

For faster decode and prefill, also download our rotated 3-bit weights
(9.55 GB). They add to the target and draft above; they do not replace them.

```bash
export PAITON_W3ROT_DIR="$PWD/model-cache/qwen38-w3rot-int3"
mkdir -p "$PAITON_W3ROT_DIR"

hf download EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4 \
  --revision d74ae7d5f5f1b4b45dd12fb1271e3664283a2ec1 \
  --local-dir "$PAITON_W3ROT_DIR"
(cd "$PAITON_W3ROT_DIR" && sha256sum -c SHA256SUMS)
```

When `PAITON_W3ROT_DIR` is set, the launcher serves these weights in both modes. For an
existing copy of the same revision, point `PAITON_W3ROT_DIR` at that folder
instead. See [what the 3-bit weights change](#faster-decode-and-prefill-3-bit-w3a4-weights-optional).

### Already in a local folder

Point to your complete **standalone** target and draft directories. These can
be folders created with `hf download --local-dir` or complete local copies of
the same pinned files. Do not point to the parent directory holding both models.

```bash
export PAITON_TARGET_DIR="/absolute/path/to/qwen38-nvfp4"
export PAITON_DRAFT_DIR="/absolute/path/to/qwen38-dflash2"
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-20260928"
mkdir -p "$PAITON_CACHE_DIR"
```

Each model directory must contain its own `config.json` and complete weight
files; keep the target tokenizer too. The launcher mounts these directories
read-only as `/models/target` and `/models/draft`, and the optional 3-bit
weights from `PAITON_W3ROT_DIR` as `/models/w3rot`. It uses `/cache` for the
separate writable runtime cache. Then [launch below](#launch-with-prepared-local-folders).

### Already in the Hugging Face cache

If you previously ran `hf download` without `--local-dir`, select the Hub cache
that contains both pinned snapshots. The following **65K Docker command** mounts
the entire cache read-only and selects the snapshots inside it, preserving their
links to `blobs/`. It starts the 65K mode with MXFP4 weights and
the same settings as the launcher; only the weight paths differ. The three
`PAITON_W3_*=0` variables switch off the image's 3-bit path and the two
`PAITON_KV4*=0` variables its 4-bit KV cache, as the launcher does without
`PAITON_W3ROT_DIR`. For the 3-bit weights, the long-context mode or `--vision`, use
the launcher instead. It cannot use Hub-cache snapshots: it needs standalone target
and draft folders ([First download](#first-download) or
[Already in a local folder](#already-in-a-local-folder)), plus the
[3-bit folder](#optional-3-bit-w3a4-weights) for the 3-bit weights.

For a cache on another drive, replace the first export with
`export HF_HUB_CACHE="/absolute/path/to/your/hub-cache"`.

```bash
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-20260928"
mkdir -p "$PAITON_CACHE_DIR"

docker run --rm --name paiton-qwen38-65k-cached --network host \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  --mount "type=bind,src=$HF_HUB_CACHE,dst=/hf-hub,readonly" \
  --mount "type=bind,src=$PAITON_CACHE_DIR,dst=/cache" \
  -e HF_HUB_OFFLINE=1 \
  -e PAITON_W3_DECODE=0 -e PAITON_W3_PREFILL=0 -e PAITON_W3_A4=0 \
  -e PAITON_KV4=0 -e PAITON_KV4_CAPACITY=0 \
  -e ROCR_VISIBLE_DEVICES -e HIP_VISIBLE_DEVICES -e CUDA_VISIBLE_DEVICES \
  ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20260929-r2@sha256:1195f31329966b3dc4e8e2d17327d339827b3d2b09165f9969b053a6fc2db045 \
  serve /hf-hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/f0b7c9e722f5565102fff8481c99e4d86ae099c7 \
  --tokenizer /hf-hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/f0b7c9e722f5565102fff8481c99e4d86ae099c7 \
  --served-model-name Qwen3.8 \
  --host 127.0.0.1 \
  --port 18982 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 6535819798 \
  --gpu-memory-utilization 0.98 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --language-model-only \
  --safetensors-load-strategy lazy \
  --attention-backend R4D \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64], "pass_config": {"fuse_norm_quant": true, "fuse_act_quant": true}}' \
  --speculative-config '{"method":"dflash","model":"/hf-hub/models--tcclaviger--Qwen3.8-27B-DFlash2-FP8/snapshots/ee0cb26a8279b7910cc28d82a8a3e15e4728d56f","num_speculative_tokens":7,"draft_tensor_parallel_size":1,"attention_backend":"TRITON_ATTN","max_model_len":65536,"disable_padded_drafter_batch":true,"draft_sample_method":"greedy"}' \
  --mamba-cache-mode align \
  --mamba-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype float16 \
  --no-async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --override-generation-config '{"temperature": 0.7, "top_p": 0.95, "top_k": 20}' \
  --seed 42
```

Both snapshots must be complete; a different cached revision is not selected
implicitly. If one is missing, run the corresponding pinned `hf download`
command from [First download](#first-download) **without `--local-dir`** to fill
your configured Hub cache, then retry this command. Do not mount only a linked
snapshot at `/models/target` or `/models/draft`: that can break its blob links.
[Cache layout and path guide](../../docs/MODEL_WEIGHTS.md).

### Launch with prepared local folders

After completing **First download** or **Already in a local folder**, start one of the two modes:

```bash
# 65K mode: up to 8 requests at once (default)
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10.sh

# Long-context mode: one conversation of up to 200,000 tokens, with prefix caching
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10.sh --context 200000
```

Run one mode at a time on the GPU; stop the running server first with `docker stop paiton-qwen38`.
Without other options, the launcher starts the 65K text mode, the configuration we benchmark. Long context,
images (`--vision`) and prefix caching only start when you ask for them.

With `PAITON_W3ROT_DIR` set, the launcher serves the 3-bit W3A4 weights; without
it, it serves MXFP4 and says how to enable them. `--weights mxfp4` serves MXFP4
even when the variable is set, and `--weights w3a4` stops with an error when it
is missing. With the 3-bit weights, the 65K mode also uses the
[4-bit KV cache](#more-context-capacity-4-bit-kv-cache); `--kv-cache fp8` keeps
the FP8 cache.

The current launcher checks that directories exist, but does not check that
checkpoint files are present. An error about `/models/target` and `config.json`
means the mounted target is missing, incomplete, or has inaccessible links.
Check the selected host directory and both downloads before retrying. The Hub
cache option above launches Docker directly and does not need this extra step.

The server runs in the foreground at `http://127.0.0.1:18982/v1`, with API model
name **`Qwen3.8`**. Add `--detach` to run it in the background instead; follow its
log with `docker logs -f paiton-qwen38` and stop it with `docker stop paiton-qwen38`
(the stopped container and its log are removed). First startup loads/converts
weights and compiles runtime components: about 7 minutes with the 3-bit weights and
an empty `PAITON_CACHE_DIR`. Wait for readiness before sending requests or measuring
throughput.
The persistent cache is reused on subsequent starts. The container writes it as root,
so removing `runtime-cache/` later needs `sudo`. The first image pull is
approximately 9.6 GB, excluding model weights.

From another terminal:

```bash
curl --fail http://127.0.0.1:18982/health
curl --fail http://127.0.0.1:18982/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8","messages":[{"role":"user","content":"Write a Python function that removes duplicate items while preserving order."}],"temperature":0.7,"top_p":0.95,"max_tokens":256,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

For the long-context mode, use the launcher with the **standalone local-folder**
setup above; the direct Hub-cache command is the 65K mode only. See
[Long context](#long-context-200k-and-220k) for what it was tested with. The
performance tables below are for the **65K mode**; they are not measurements of
200K throughput.

No private compiler checkout is needed to run this image. The full qualified
runtime payload is distributed in the image; the public repository alone is
not a complete build context for this release.

## GPU, context, and memory controls

Context is **not compiled into the image**: `--context` sets both the target and
DFlash draft limits at startup, and above 65,536 tokens it selects the long-context
mode. The limit includes input and generated tokens. A larger
limit still requires sufficient cache and VRAM; it does not guarantee useful
model quality at that length.
The checkpoint's configured ceiling is 262,144 tokens; the largest serving
limit tested here is **220,000**, not a claim that 262K fits this GPU.

The launcher exposes `/dev/kfd` and all of `/dev/dri`, adds the `video` group,
and uses host IPC. Select the GPU with your usual ROCm environment variables.
For example, if the R9700 is ROCm GPU 1:

```bash
export ROCR_VISIBLE_DEVICES=1
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10.sh
```

Use the index or GPU UUID appropriate to your system. `--list-gpus` lists physical
render devices and PCI addresses for identification; render-device numbers are
not ROCm visibility indices. The launcher does not choose a GPU automatically.

The launcher forwards your `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` and
`CUDA_VISIBLE_DEVICES` values unchanged. If a variable is unset on the host, it
is also unset in the container, overriding the image's default. The launcher
does not force a GPU index or UUID. An explicitly empty mask remains empty;
it does not mean "show all GPUs".

On Linux, `ROCR_VISIBLE_DEVICES` also filters ROCr tools such as `rocminfo`;
`HIP_VISIBLE_DEVICES` applies at the HIP layer. If you set both, HIP indices refer
to the GPUs remaining after the ROCr filter. Check any existing exports before
launching: setting both variables to `1` does not necessarily select the second
physical card. Leave unused variables unset rather than assigning empty strings.
These controls do not make unsupported GPU architectures compatible with this image.

For a GPU shared with a desktop, start with the smaller preset:

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10.sh \
  --profile desktop
```

This selects 32,768 tokens, one scheduled request, 1,024-token prefill chunks,
smaller graph captures, and a 2 GiB KV allocation. It is a starting point
for sharing VRAM, not a guarantee against memory exhaustion. The unchanged
benchmark presets reserve a fixed KV pool and target a dedicated GPU.

Customize the limits without rebuilding or downloading another image:

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10.sh \
  --profile desktop --context 16384
```

Setting memory utilization switches to automatic KV sizing unless you explicitly
provide `--kv-cache-memory-bytes`. Lowering context alone does **not** reduce a
fixed KV allocation. `--kv-cache-memory-bytes auto` also enables automatic sizing.
Automatic profiling can leave insufficient cache for the requested context in
this runtime; startup reports the required and available cache sizes.
Use `--dry-run` to inspect the complete Docker command, or `--help` for all options.
Customized settings are separate from the benchmark configuration below.

With the 3-bit weights, the 65K mode gives most of the memory they free to the KV
cache: 8,859,648,000 bytes with the default 4-bit cache (393,216 tokens in vLLM's
startup log) or 9,381,235,631 bytes (250,578 tokens) with `--kv-cache fp8`,
against 6,535,819,798 bytes (174,634 tokens) for MXFP4. Under full load, peak VRAM
was 31.8 GiB with the 4-bit cache and 31.6 GiB with the FP8 cache, with no
out-of-memory errors. An explicit `--kv-cache-memory-bytes` or
`--gpu-memory-utilization`, `--vision`, the `desktop` profile and the long-context
mode use their own budgets.

### Long context: 200K and 220K

Any `--context` above 65,536 starts the long-context mode on the **same image**:

```bash
# 200K total context, including generated tokens
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10.sh --context 200000

# Stop the running server before selecting 220K instead
docker stop paiton-qwen38
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10.sh --context 220000
```

The long-context mode uses one scheduled request, an 8 GiB FP8 KV pool, 1,024-token
prefill chunks, prefix caching, and thinking disabled; these are the settings of
the `chat` profile, and `run-rocm10-200k.sh` or `--profile chat` start the same
mode. It also applies the memory-allocation setting needed by this configuration.
It leaves little spare VRAM with MXFP4 weights (3.3 GiB more with the 3-bit
weights): select a dedicated R9700 rather than a card driving a busy desktop.
The checkpoint's limit is 262,144 tokens; 220,000 is the largest tested here.

Measured on one R9700 with a 198,989-token prompt:

| | 3-bit weights | MXFP4 |
|---|---:|---:|
| New prompt, time to first token | 94 s | 102 s |
| Follow-up turn that reuses the prompt | 1.2 s | 1.2 s |
| Decode at this length | 73–78 tok/s | 63–68 tok/s |
| Peak VRAM | 31.0 GiB | 31.8 GiB |

The earlier 200K image needed 115 s for the same prompt. All planted facts (near 5%,
50% and 95% of each prompt) were found. A multi-turn follow-up, a second 199K-token
prompt, plain and streamed tool calls, and an over-limit request (rejected with
HTTP 400) followed by a normal one all passed. With `--context 220000`, a
215,000-token prompt answered after 105 s, and after 1.5 s when reused.

The 65K mode does not use prefix caching (APC), so zero cache hits there are
expected. The persistent disk cache used during startup is separate from the
in-memory conversation prefix cache.

`--prefix-caching on` enables the experimental APC configuration and selects the
compatible recurrent-state settings. This changes memory requirements; do not
assume the 65K mode's fixed cache budget remains sufficient. Use the long-context
mode for the complete configuration. Cache hits require an
unchanged token prefix that is still resident. They reduce repeated prompt work,
not the cost of generating each new token.

The long-context mode reports cached tokens in `usage.prompt_tokens_details.cached_tokens`.
Streaming clients must also request `"stream_options":{"include_usage":true}`
to receive usage in the stream. Server-side cache counters are available at
`http://127.0.0.1:18982/metrics`.

These checks are not a full long-conversation quality evaluation. Repetition
penalties and sampling settings belong in each client's API requests. Report the
prompt, settings and server logs when diagnosing loops; a sampling workaround is
not a general fix.

In the 65K mode, the model's template enables thinking when the client omits that
setting; the long-context mode and our benchmarks disable it. Use `--thinking off` to
set the server default, or send
`"chat_template_kwargs":{"enable_thinking":false}` in each request.

### Images and vision

Add `--vision` to send images, such as screenshots, UI captures or charts, as OpenAI-style `image_url` content:

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10.sh --vision
```

The checkpoint's vision encoder then loads next to the language model, and its
memory comes out of the KV cache: 278,050 tokens instead of 393,216 with the 3-bit
weights and the 4-bit cache (180,416 with `--kv-cache fp8`), and 120,277 instead of
174,634 with MXFP4. An image costs about one token per 32 × 32 pixels: a 1920 × 1080
screenshot is about 2,000 tokens, and larger images are scaled down to at most
16,384 tokens (4096 × 4096 pixels).

On one R9700, with the 3-bit weights, with MXFP4 and in the `desktop` profile, the model
read a code editor, a failing pytest run, a web sign-in form and a bar chart correctly,
and the centre text and corner labels of a 4096 × 4096 image (16,425 prompt tokens,
11 s to the first token). A 58K-token prompt with a chart answered questions about
both, eight concurrent requests with an image each completed, DFlash2 stayed active,
and text answers and decode speed were unchanged.

`--vision` works in the 65K mode, including `--profile desktop`. It is not tested
with prefix caching yet, so the long-context mode refuses it for now. Video input
is not tested. An explicit `--kv-cache-memory-bytes` or `--gpu-memory-utilization`
replaces the vision budget.

## Faster decode and prefill: 3-bit W3A4 weights (optional)

The image can serve our own rotated 3-bit weights for the decoder projections in
place of MXFP4:
[EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4](https://huggingface.co/EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4).
They are an add-on to the same pinned target and DFlash2 drafter, not a
standalone checkpoint. [Download them](#optional-3-bit-w3a4-weights), keep
`PAITON_W3ROT_DIR` set, and start the launcher as usual.

**How they work.** The decoder projections use 3-bit integer weights with one
scale per 128 weights, stored in a block-wise Hadamard-rotated basis and
calibrated with GPTQ on permissively licensed data. During prefill, the rotated
layers also take 4-bit activations using RDNA4 int4 matrix math; decode keeps
8-bit activations. Model memory falls from 19.18 to 15.89 GiB, and the launcher
gives most of the difference to the KV cache.

**What you gain.** +19.9% weighted decode, +15.7% to +22.1% aggregate throughput
at one to eight concurrent requests, and +5.0% to +12.7% prefill with
correspondingly shorter time to first token (see
[Current benchmark results](#current-benchmark-results)). With the FP8 cache, the
larger KV budget holds 3.8 instead of 2.7 concurrent 65K-token requests; the
default [4-bit KV cache](#more-context-capacity-4-bit-kv-cache) raises this to
6.0 by vLLM's count. Four 61K-token requests with 512 output
tokens each, FP8 cache (26 September measurement):

| Configuration | KV cache | Wall time | Requests decoding together |
|---|---:|---:|---:|
| MXFP4 | 174,634 tokens | 105.6 s | 2 |
| W3A4, same KV budget | 174,634 tokens | 92.9 s | 2 |
| W3A4, launcher default with `--kv-cache fp8` | 250,578 tokens | **86.2 s** | **4**, at 199 tok/s combined |

At the same KV budget, two 61K-token requests decode at 117 instead of 69 tok/s
combined. With a warm runtime cache, startup takes about 230 s instead of about 210 s.

**What it costs.** Served model, greedy decoding, thinking off, paired with
MXFP4 on identical items; Δ in points with a 95% interval:

| Benchmark | MXFP4 | W3A4 | Δ [95% CI] |
|---|---:|---:|---:|
| GSM8K 5-shot (1,319) | 95.68 | 95.30 | −0.38 [−1.44, +0.68] |
| HumanEval pass@1 (164) | 95.12 | 93.90 | −1.22 [−4.99, +2.56] |
| MMLU-Pro subset, 0-shot (14 × 100) | 62.57 | 59.71 | −2.86 [−4.81, −0.90] |
| Needle at 61,440 tokens (80) | 100 | 100 | 0 |

Math, code and long-context retrieval stay within noise; knowledge recall drops
by about 3 points. DFlash2 acceptance changes by −0.4% to +2.8%. Outputs differ
from the MXFP4 path, greedy ones included. **For maximum knowledge accuracy, use
MXFP4** (`--weights mxfp4`, or leave `PAITON_W3ROT_DIR` unset); it keeps this
release's other improvements.

The 3-bit weights replace the language model's weights and are tied to the pinned
target revision; with `--vision`, the vision encoder comes from the target
checkpoint. They were benchmarked and evaluated in the 65K mode, and the
long-context mode passed its checks with them ([Long context](#long-context-200k-and-220k)).
The `desktop` profile also uses them when `PAITON_W3ROT_DIR` is set; that
combination has not been measured. Transformers, stock vLLM and llama.cpp cannot
load them.
The weights are Apache-2.0; the
[model card](https://huggingface.co/EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4)
lists the calibration data and its attributions.

## More context capacity: 4-bit KV cache

With the [3-bit W3A4 weights](#faster-decode-and-prefill-3-bit-w3a4-weights-optional),
the 65K mode stores the attention KV cache in 4 bits. It is on by default; launch
as usual.

**How it works.** Keys and values are stored as 4-bit integers in groups of 32
values, each group with its own scale and zero point. Keys are Hadamard-rotated
before quantization, which spreads outlier channels across the group; queries
get the same rotation, so attention scores are unaffected by it. The denser pages
are published to vLLM's KV-cache allocator, so the scheduler admits more tokens
from the same memory; our earlier 4-bit version only reduced the bytes read. The
16 full-attention layers use this cache. The Gated DeltaNet recurrent state and
the DFlash2 drafter's cache are unchanged, and the scales take space, so the
gain is about 1.8× as many attention tokens per byte rather than 2×.

**When it is on.** `--kv-cache auto` (the default) selects the 4-bit cache only
where it was measured end to end: the 65K mode with the 3-bit weights, up to
65,536 tokens. Everything else uses the FP8 cache: MXFP4 weights, the
long-context mode (prefix caching), the `desktop` profile and other contexts.
`--kv-cache fp8` keeps the FP8 cache and the 26 September budget in the 65K mode.
`--kv-cache kv4` selects the 4-bit cache without prefix caching up to 200,000
tokens, beyond what we measured end to end, and stops with an error outside that
range.

**What you gain.** Launcher defaults with the 3-bit weights, requests with 512
output tokens each, greedy decoding:

| | 26 Sept image, FP8 KV | 28 Sept image, 4-bit KV |
|---|---:|---:|
| KV cache size in vLLM's startup log | 250,578 tokens | 393,216 tokens |
| Attention tokens with eight requests running¹ | 211,136 | 358,336 (**1.70×**) |
| Four 61K-token requests, combined decode² | 188 tok/s | **219 tok/s** |
| Six 61K-token requests | not all at once | all six together, 248 tok/s² |
| Eight 32K-token requests | not all at once | all eight together, 339 tok/s² |

¹ vLLM's log figure does not subtract the recurrent-state and draft-cache blocks
that each running request also takes from the same pool.
² While every request is decoding. This rate varies between sessions (the 26
September run of the FP8 configuration measured 199 tok/s at four 61K-token
requests), so compare within this table.

Time to first token is unchanged: prefill from 2K to 64K tokens is within ±0.3%
of the 26 September image. Single-request decode is unchanged too: the time per
decoding step is identical, and over 232 sampled requests per image both caches
decode equally fast.

**VRAM.** The 4-bit cache needs about 0.16 GiB more working memory and runs more
requests at once, so the launcher gives it 8,859,648,000 instead of 9,381,235,631
bytes; at idle, 0.44 GiB more VRAM stays free than with the FP8 cache. Under full
load, peak VRAM was 31.65 to 31.76 GiB (FP8 cache: up to 31.63 GiB), with no
out-of-memory errors; with either cache, PyTorch keeps freed memory reserved. The
capacity figures above use this budget.

**What it costs.** Served model with the 3-bit weights, greedy decoding,
thinking off, paired with the FP8 cache on identical items; Δ in points with a
95% interval:

| Benchmark | FP8 KV | 4-bit KV | Δ [95% CI] |
|---|---:|---:|---:|
| GSM8K 5-shot (1,319) | 95.30 | 96.13 | +0.83 [−0.07, +1.74] |
| HumanEval pass@1 (164) | 93.90 | 93.90 | 0.00 [−2.93, +2.93] |
| MMLU-Pro subset, 0-shot (14 × 100) | 59.71 | 61.43 | +1.71 [+0.21, +3.22] |
| Needle at 61,440 tokens (80) | 100 | 100 | 0 |

We read this as no loss, not as a gain from the 4-bit cache. DFlash2 acceptance
changes by −1.2% to +1.8%.

**Long sessions.** A new test in our suite plants 11 facts in each of 24
synthetic coding-agent sessions of 32K and 61K tokens, written in the model's own
tool-call format: user decisions, values in tool output next to look-alike
distractors, and the assistant's own conclusions. The model then writes a handoff
summary, answers direct questions about the facts, and answers them again from
its summary alone. Over the same 264 facts, the 4-bit cache kept 87.9% of the
facts in its summaries (FP8: 89.0%), recalled 99.6% directly (100%), answered
83.7% from its own summary (86.7%), and wrote lazy references such as "see above"
in 2 sessions (2). All differences are within noise: every 95% interval includes
zero, and an earlier run of the same 4-bit configuration scored 90.5%, 99.6% and
85.6%. On the same test, the 3-bit weights showed no measurable
loss against MXFP4 either (facts kept 89.0 vs 89.4%, answers from the summary
86.7 vs 84.8%).

## Faster agentic coding decode: n-gram co-drafting (opt-in)

The 24 September image introduced an optional second drafter in front of DFlash2;
the 26 September image keeps it.
When the last few generated tokens already occurred earlier in the prompt or the
output, the matcher proposes the tokens that followed last time, so copy-heavy
generations such as file rewrites, code echoed back into an edit, or repeated
structure accept more tokens per step. The proposals enter the existing rejection
sampler as one-hot draft rows, so the target distribution is unchanged; when every
request in a batch has a match, the DFlash2 draft forward is skipped.

It is **off by default**. Enable it per launch with `PAITON_NGRAM_CODRAFT=1`; the
launcher forwards the variable when it is set on the host:

```bash
PAITON_NGRAM_CODRAFT=1 bash models/Qwen3.8-MXFP4-DFlash2/run-rocm10.sh --profile chat
```

Measured with the published adapter on the 24 September image (MXFP4 weights),
A/B/A/B in fresh processes:

| Workload | Off | `PAITON_NGRAM_CODRAFT=1` | Change |
|---|---:|---:|---:|
| 35-turn agentic coding session, 200K chat profile, decode tok/s | 90.8 | 115.4 / 115.7 | **+27%** |
| Accepted tokens per step in that session | 3.0 | 3.6 | |
| 64K-context file rewrite, tok/s¹ | 129 | 167 | +29% |
| 128K-context file rewrite, tok/s¹ | 125 | 158 | +26% |
| BetterBench weighted decode, 65K profile | 153.9 | 153.3 | −0.4% |
| BetterBench file edit / prose / json decode tok/s | 180.1 / 78.3 / 217.7 | 176.2 / 77.4 / 217.5 | −2.2% / −1.2% / −0.1% |
| BetterBench C1 / C2 / C4 / C8 aggregate tok/s | 121.8 / 206.1 / 319.0 / 424.7 | 122.0 / 206.1 / 307.8 / 421.6 | +0.2% / 0.0% / −3.5% / −0.7% |

¹ Measured with an earlier gate version of the adapter; the shipped version was not
re-measured on this workload.

Short prompts with little to copy gain nothing and pay a small bookkeeping cost.
That is why the mode ships off and why the benchmark tables below were measured
with it off. The control's own concurrency-four value ranged 306–319 tok/s across
runs. Session time to first token is unchanged by the mode.

What to expect from the output: sampling still draws from the target distribution,
and greedy decoding still returns the argmax chain in exact arithmetic. A different
acceptance pattern changes which verify row computes a position, so output bits can
differ at near-ties. The two divergences found were word choices inside generated
comments with the top two candidates 0.25 nats apart, and the released image shows
the same class of divergence between its own fresh processes. The twelve greedy
control prompts matched with the mode on, and all 35 session turns produced valid
tool calls with the same tool names. Structured-output workloads that must not change
wording should leave the mode off. `PAITON_NGRAM_CODRAFT_HOT_MATCH` (default 16, the
match length that keeps a request on the drafting path) trades session gain against
the small cost on non-copying traffic.

## Current benchmark results

**28 September image.** The tables below were measured on the 26 September image;
the 28 September image runs at the same speed. In a BetterBench A/B between the two
(3-bit weights, the settings below, two runs each), prefill from 2K to 64K was
within ±0.3% and aggregate throughput at one to eight requests within −2.4% to 0.0%.
Weighted single-stream decode read 181.8 vs 191.0 tok/s (−4.8%), but that is a
sampling effect, not a slowdown: BetterBench's fixed seed replays the same 40
sampled answers in every run, the 4-bit cache sends each answer down a different
path, and the time per decoding step is identical. Over 232 requests with eight
seeds per prompt, both images decode equally fast (within 1%). For long-context
capacity, see [4-bit KV cache](#more-context-capacity-4-bit-kv-cache).

R9700, 300 W; vLLM 0.29 / ROCm 10; 65,536 context; maximum eight sequences; APC off; thinking off; n-gram co-drafting off. Temperature 0.7, top-p 0.95, top-k 20, seed 42. BetterBench 0.6.0 quick. Three arms, each run twice in fresh processes, interleaved: the published 24 September image; this round's runtime with MXFP4 weights; and the 26 September image with the 3-bit W3A4 weights. The tables show the mean of the two runs; changes compare W3A4 with the 24 September release. All arms set `GPU_MAX_HW_QUEUES=1`, so the gains exclude that setting.

**Decode, single stream, tok/s.** The headline of the 26 September release.

| Category | 24 Sept release | 26 Sept, MXFP4 | 26 Sept, W3A4 | Change |
|---|---:|---:|---:|---:|
| chat | 121.2 | 122.9 | **135.8** | +12.0% |
| code | 179.9 | 182.7 | **226.0** | +25.6% |
| file edit | 179.9 | 182.7 | **195.0** | +8.5% |
| json | 217.5 | 220.8 | **269.0** | +23.7% |
| math | 183.8 | 186.5 | **228.9** | +24.5% |
| prose | 78.5 | 79.6 | **94.7** | +20.7% |
| reasoning | 117.8 | 119.4 | **133.5** | +13.3% |
| summarization | 138.4 | 140.5 | **158.4** | +14.4% |

Weighted decode: **153.8 → 156.1 → 184.4 tok/s (+19.9%)**.

**Concurrency, aggregate generated tok/s over each complete 48-request workload.**

| Concurrent requests | 24 Sept release | 26 Sept, MXFP4 | 26 Sept, W3A4 | Change |
|---:|---:|---:|---:|---:|
| 1 | 122.0 | 123.7 | **148.8** | +22.0% |
| 2 | 204.2 | 209.3 | **249.3** | +22.1% |
| 4 | 308.2 | 315.5 | **368.3** | +19.5% |
| 8 | 425.3 | 428.0 | **492.1** | +15.7% |

**Prefill, input tok/s.** Time to first token shortens by the same factors.

| Nominal prefill depth | 24 Sept release | 26 Sept, MXFP4 | 26 Sept, W3A4 | Change |
|---:|---:|---:|---:|---:|
| 2,000 | 3,689 | 3,691 | **4,156** | +12.7% |
| 8,000 | 3,834 | 3,831 | **4,165** | +8.6% |
| 16,000 | 3,871 | 3,871 | **4,103** | +6.0% |
| 32,000 | 3,751 | 3,750 | **3,958** | +5.5% |
| 64,000 | 3,455 | 3,455 | **3,629** | +5.0% |

**What changed in the 26 September image.** The optional [3-bit W3A4 weights](#faster-decode-and-prefill-3-bit-w3a4-weights-optional); a fused Gated DeltaNet speculative-verify kernel, exact against the previous path (the MXFP4 arm returns the same twelve greedy outputs as the 24 September release) and worth +1.5% weighted decode; and `GPU_MAX_HW_QUEUES=1`, which removes a slower decode mode that some fresh server processes on the R9700 started in.

Sampled output content and accepted-token work can differ; these are serving-throughput measurements, not identical-output timing.
Nominal prefill depths correspond to median actual prompt lengths 1516.5, 5894.5, 11802, 23549.5 and 47016.5.

Every run: 40/40 decode, 192/192 concurrency and 40/40 prefill scored requests (plus the fixed warmups), and no serving errors. Each run repeats its twelve greedy outputs after the benchmark. Median C1 decode forward time: 28.3–28.4 ms (24 Sept), 28.0 ms (MXFP4), 22.4–22.5 ms (W3A4). The W3A4 timing runs used an earlier calibration of the same 3-bit format; the tensor layout and runtime are identical, so the timing applies to the published weights.

Decode has five scored requests per category after one warmup. Prefill has eight scored requests per depth after two warmups. Category values are means of two complete runs, not selected across repeats; the two W3A4 runs differ most on file edit (179.0 and 211.1 tok/s) and chat (132.5 and 139.1 tok/s).

[Machine-readable results](benchmarks/2026-09-26-w3a4/numbers.json) · [Benchmark page](benchmarks/2026-09-26-w3a4/README.md).

## Historical releases and comparisons

The sections below describe earlier images, checkpoints, launchers and benchmark
settings. Their numbers are separate from the current release. In particular,
`serve.py`, `runtime.lock.json`, and `checkpoint.lock.json` below describe the
legacy release; use the `run-rocm10-*.sh` launchers above for the current images.

<details>
<summary>Earlier releases, APC investigation, benchmarks and reproduction instructions</summary>

## 26 September 2026 release: optional 3-bit W3A4 weights

The 26 September 65K image (`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-65k-20260926-w3a4-r1`)
introduced the optional 3-bit W3A4 weights with the FP8 KV cache. Its throughput tables remain
[above](#current-benchmark-results) and in [benchmarks/2026-09-26-w3a4](benchmarks/2026-09-26-w3a4/README.md).
Under three to five concurrent requests it can stop the engine; see the
[28 September release notes](#release-notes-28-september-2026) for the fix and a workaround.

## 24 September 2026 release: faster prefill and opt-in n-gram co-drafting

The 24 September 65K image (`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-65k-20260924-r3`)
prefilled 3,457–3,871 input tok/s from 2K to 64K (+3.5–5.4% over the 20 September image) and
measured 154.78 tok/s weighted decode and 422.9 tok/s aggregate at concurrency eight. On the 200K
chat profile, a 35-turn agentic coding session had 10–11% lower session time to first token. Its
full tables are in [benchmarks/2026-09-24-prefill-ngram](benchmarks/2026-09-24-prefill-ngram/README.md);
its opt-in n-gram co-drafting mode is described [above](#faster-agentic-coding-decode-n-gram-co-drafting-opt-in).

## 20 September 2026 release: ROCm 10 and vLLM 0.29 combined runtime

The 20 September 65K image (`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-65k-20260920-r2`)
measured 154.42 tok/s weighted decode, 218.1 tok/s median JSON decode and 421.20 tok/s
aggregate at concurrency eight: +5.16% weighted decode and +4.37–5.61% large-prefill
throughput over the 18 September image. Its full tables, including the GGZ reference
columns, are in [benchmarks/2026-09-20-combined](benchmarks/2026-09-20-combined/README.md).

## Long coding conversations: prefix caching can remove most repeat-turn waiting

**New APC investigation — 17 September 2026.** Automatic prefix caching reuses
computation for an unchanged conversation prefix. It can make a large practical
difference when a coding agent sends the same growing history on every turn.
Our released profiles currently disable it because compact native GDN replay
does not yet support prefix reuse.

On one R9700, the stock-GDN APC path reduced **40K cold-to-repeat first-token
latency from 15.58 s to 1.13 s**. Our experimental native-prefill APC candidate
reduced **150K cold-to-repeat latency from 88.46 s to 2.04 s**, reproduced with a
second distinct prefix at **88.61 s to 2.03 s**. Those are **43–44× faster response
starts on cache hits**, not decode-throughput gains or new competitor results.
The 150K tests used an 8 GiB cache and one active request.

APC is a workload choice: it helps repeated documents and growing conversations,
while the compact native path retains better decode performance and cache
capacity for fresh prompts. The report includes these tradeoffs and the cache
miss, branch, tool-result and concurrency checks.

**Availability:** the existing GHCR images remain unchanged. They do **not**
implement the new `PAITON_PREFIX_CACHING=1` convenience flag yet. Users can try
the measured stock-GDN APC configuration on the published 64K image with the
[explicit profile override](benchmarks/2026-09-17-prefix-caching/REPRODUCE.md).
The native-prefill APC candidate is not yet distributed. Do not apply the
fallback to the 200K image with its existing 8 GiB cache: that budget does not
meet the stock-APC cache requirement at 200K.

[APC results, raw evidence and limitations](benchmarks/2026-09-17-prefix-caching/README.md)
· [Try stock-GDN APC and reproduce the controls](benchmarks/2026-09-17-prefix-caching/REPRODUCE.md).

## 64K and 200K images, tool-call fix, and quick benchmark — 17 September 2026

New v1.1.0 images add Qwen XML tool-call parsing and configurable context limits.
The **64K profile** uses a 5 GiB cache and up to eight scheduled requests; the
**200K profile** uses an 8 GiB cache and one active request, with little spare
VRAM on the 32 GB R9700. The original v1.0.0 image remains available.

The 64K image completed a 52-request BetterBench quick run: **304.1 tok/s aggregate
at concurrency eight**, with **67.8 tok/s median per request**. These are short
prompts with a 128-token output cap and thinking disabled. Separate long-context
retrieval and OpenCode tool tests are documented alongside the benchmark.

[New benchmark page, visuals, evidence, and GHCR package links](benchmarks/2026-09-17-agentic-64k/README.md)
· [64K and 200K launch commands](LAUNCH-agentic-v1.1.0.md)
· [Tool calling and context details](SUPPORT.md).

The comparison sections below retain the measurements from the earlier 8K release.

## Earlier release benchmark suites

Paiton delivers **22% higher weighted decode throughput**, **57% more throughput
at eight concurrent requests**, and **12.5–17.3% faster prefill** than Radiance +
DFlash2 on the same Radeon AI PRO R9700. Paiton leads all eight task categories
and all four concurrency throughput levels in the full 188-request comparison.

At eight concurrent requests, median time to first token falls from **6.59
seconds to 195 ms**, including queueing. The same 5 GiB cache pool provides
**2.89× the estimated token capacity**, with logs recording eight active requests
for Paiton versus three for Radiance.

Paiton combines native HIP kernels, adapted Radiance techniques, and vLLM's
DFlash2 support. The official vLLM installation stays in place: a model-specific
plugin supplies the native implementation. There is no separate Radiance engine
or DFlash package to install.

![Stock vLLM O2, Radiance + DFlash2, and Paiton + DFlash2 throughput and prefill](three-engine-throughput.png)

The common 54-request matrix includes stock vLLM O2: C8 throughput is
**33.7 / 175.6 / 328.5 tok/s** for stock / Radiance / Paiton respectively. Its
128-token generation cap differs from the longer full preset, reported
separately below.

![Full-workload comparison against Radiance + DFlash2](full-confirmation.png)

[Complete three-engine tables, latency, and measurement scope](BENCHMARKS.md).
Radiance retains an approximately 10–11 ms TTFT advantage at one and two
concurrent requests; Paiton's latency advantage appears under concurrent load.

## Run the v1.0.0 benchmark release

[Context overrides, structured tool calls, and coding-agent setup](SUPPORT.md)
cover the reproduced parser mismatch and the local configuration workarounds.
The corrected 64K profile has functional API/client validation separate from the
historical benchmarks.

Release **v1.0.0** is available on GHCR. The launcher pins the qualified image
by digest in [runtime.lock.json](runtime.lock.json).

Use Linux x86-64, Docker, and one Radeon AI PRO R9700 with working AMD GPU device
access. The dedicated image includes the pinned official vLLM 0.28 ROCm runtime,
the Paiton adapter, and all qualified native libraries.

```bash
docker run -d --name paiton-qwen38-mxfp4 \
  --device /dev/kfd --device /dev/dri --group-add video --shm-size 2g \
  -p 127.0.0.1:8000:8000 \
  -v paiton-qwen38-mxfp4-cache:/models/cache \
  ghcr.io/eliovp/paiton-vllm-plugin@sha256:9b2dae214076d35de785e073b31294b033a376b16e6bc1ec1fdada4e54d96c59
```

The first image pull downloads approximately 11.4 GB of runtime layers when
those layers are not already cached. The first start then downloads approximately
21.9 GB of target and draft weights
from their original repositories and verifies the locked file hashes. Later
starts reuse the cache. Follow `docker logs -f paiton-qwen38-mxfp4` until startup
completes, then check `curl --fail http://127.0.0.1:8000/health`.

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-Quark-AWQ-MXFP4","messages":[{"role":"user","content":"Write a Python function that preserves the first occurrence of each item in a list."}],"temperature":0,"max_tokens":256,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

No compiler checkout or build step is required. The image starts the ordinary
vLLM OpenAI API server after checkpoint verification.

## Launcher and cached models

From this repository, [serve.py](serve.py) starts the same image and defaults to
verified downloads with a persistent Docker cache volume:

```bash
python3 models/Qwen3.8-MXFP4-DFlash2/serve.py --detach
```

Use `--cache /path/to/cache` for a host cache directory, `--port` for another
local port, and `--name` for a different container name. `--dry-run` prints the
Docker argument list without downloading files or starting a container.

To download and verify the snapshots without accessing the GPU:

```bash
python3 models/Qwen3.8-MXFP4-DFlash2/serve.py --download-only
```

After downloading, `--offline` requires cached or explicitly mounted files.
Existing snapshot directories can be supplied independently; each is mounted
read-only and verified inside the container:

```bash
python3 models/Qwen3.8-MXFP4-DFlash2/serve.py \
  --target /path/to/pinned-target-snapshot \
  --draft /path/to/pinned-draft-snapshot \
  --offline --detach
```

Use one launch example at a time for the same GPU. `docker stop
paiton-qwen38-mxfp4` stops the default serving container; the cache persists.

## Model and v1.0.0 profile

| Setting | Tested configuration |
|---|---|
| Target | `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` |
| Draft | `tcclaviger/Qwen3.8-27B-DFlash2-FP8` |
| Hardware | One Radeon AI PRO R9700, RDNA 4 / `gfx1201`, 32 GB |
| Runtime | Official vLLM 0.28 ROCm image plus the Paiton plugin |
| Input | Text |
| Maximum context | 8,192 total tokens per request |
| Concurrent requests | Up to eight |
| Cache | FP8 KV, explicit 5 GiB pool, prefix caching disabled |
| Prefill chunk | Up to 4,096 tokens |
| Speculation | DFlash2, seven speculative tokens, unpadded drafting |
| Benchmark sampling | Greedy; no requested log probabilities |

Target and draft revisions, file sizes, and SHA256 hashes are pinned in
[checkpoint.lock.json](checkpoint.lock.json). This checkpoint is distinct from
the Qronos and NEO CODER MAX releases; their speeds are not used as model-matched
baselines here. The token-cache capacity estimate does not extend the tested
8,192-token per-request context limit.

The native overlay and manifests are also distributed through
[Hugging Face](https://huggingface.co/EliovpAI/Qwen3.8-27B-Quark-AWQ-MXFP4-DFlash2-Paiton-RDNA4).
The weight files remain in the original model repositories.

## Integration and provenance

The dedicated image installs the model-specific plugin with `--no-deps` onto
the pinned official vLLM image. It uses vLLM's extension interfaces for the
model, loader, linear kernels, attention backend, and worker. It does not replace
the installed vLLM library. The main repository's general installation has a
different historical vLLM pin; use this model's qualified image and profile.

Paiton's native artifacts use HIP and load independently of PyTorch, Triton,
and Radiance. The serving adapter retains vLLM's existing framework dependencies
and upstream DFlash2 scheduling. The compiler and implementation source stay
private; the runtime package contains only the external adapter, allowlisted
native libraries, and runtime metadata.

[Radiance](https://github.com/magiccodingman/vllm-radiance) and
[StillDeadcode/libr4d](https://codeberg.org/StillDeadcode/libr4d) are credited for
the adapted kernel techniques. [Third-party attribution and terms](THIRD_PARTY_NOTICES.md).

The ordinary CLI deployment check passes streaming, eight concurrent requests,
and generation at the 8K context boundary followed by a fresh request.
[Deployment check](deployment-check.json) · [Unchanged-vLLM audit](runtime-audit.json) · [Published-image audit](release-audit.json).

## Reproduce the image context

The image can be rebuilt from the public adapter and the pinned native overlay.
The context preparer verifies an explicit file allowlist; it does not require
the private compiler. Download the companion release with the Hugging Face CLI,
then prepare a new build directory:

```bash
hf download EliovpAI/Qwen3.8-27B-Quark-AWQ-MXFP4-DFlash2-Paiton-RDNA4 \
  --revision v1.0.0 --local-dir qwen38-paiton-runtime
python3 models/Qwen3.8-MXFP4-DFlash2/prepare_image_context.py \
  --overlay qwen38-paiton-runtime/overlay --output qwen38-image-context
docker build -t paiton-qwen38-mxfp4:local qwen38-image-context
```

Use this model's `runtime-pyproject.toml`, copied automatically by the preparer;
the repository-wide package targets other runtime versions. The published
container digest identifies the tested distribution; a local rebuild creates
its own image identity.

</details>
