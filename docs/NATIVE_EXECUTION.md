# Native execution with an existing vLLM environment

Install the [v0.3.4 wheel](https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/tag/v0.3.4) in an existing supported vLLM
environment. Paiton automatically fetches and verifies each preset's native
bundle. Neither the private compiler nor a repository checkout is needed.

## Start serving

Use `paiton models` to list named presets and their required environments,
then activate the [matching vLLM environment](#qualified-combinations). For MiniCPM5:

```bash
python -m pip install https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/download/v0.3.4/paiton_vllm_plugin-0.3.4-py3-none-any.whl
paiton doctor
paiton serve minicpm5
```

To use an existing checkpoint outside your Hugging Face cache, run
`paiton --model-dir /models/existing-minicpm5 serve minicpm5`.
Preparation verifies and remembers the supplied checkpoint path for this preset.
Later launches revalidate its identity; they do not silently choose another copy.
A new explicit `--model-dir` replaces this local preference after successful
preparation and lock checks. The preference is separate from immutable launch
plans and locks. A missing or corrupt preference fails with an explanation.

Omit `--model-dir` to resolve the preset's pinned repository through your configured
Hugging Face cache. An explicit path or exact repository ID is also supported with
`--profile PROFILE`. Local directories may have any name, but checkpoint bytes
must match the pinned identity. Arbitrary fine-tunes are not accepted.

`paiton serve MODEL` and the original `paiton vllm serve MODEL` use the same
resolver, preparation and launch APIs. Paiton options belong **before `serve`
or `vllm`**. Everything after the model belongs to vLLM. JSON values, spaces and
repeated arguments keep their original tokens. vLLM runs through the same Python
interpreter, retaining standard streams, signals and exit status.

The default wheel has no inference-stack dependencies. It does not install or
replace vLLM, Torch or ROCm. Ordinary `vllm serve MODEL` remains inactive after
installation. Explicit Paiton serving fails when the checkpoint, runtime or
payload is incompatible; there is no automatic stock/container fallback.

## Qualified combinations

The original Qwen NVFP4 and MiniCPM combinations were exercised on Ubuntu, x86-64, one Radeon AI PRO R9700,
32 GB, `gfx1201`. Other devices, operating systems and parallel configurations
have not been qualified by this work. This is a serving smoke qualification,
not an exhaustive quality or maximum-context evaluation.

| Contract | Qwen profile | MiniCPM profile |
| --- | --- | --- |
| Profile | `qwen38-nvfp4-w4a8-text-65k` | `minicpm5-awq-text-8k` |
| Checkpoint | `unsloth/Qwen3.8-27B-NVFP4` | `openbmb/MiniCPM5-2B-GPTQ` |
| Revision | `f0b7c9e722f5565102fff8481c99e4d86ae099c7` | `6c1ee6fa521aa53f47cfb32696e6d8ef5b0db805` |
| Stored representation | Mixed NVFP4 / FP8 / BF16 safetensors | Asymmetric AWQ GEMM, group 128, despite the GPTQ repository name |
| Runtime representation | MXFP4 weights, FP8 activations, BF16 intermediates | W4A16; lossless layout repacking |
| Python tested | 3.12.13 | 3.14.6 |
| vLLM | `0.29.0` | `0.26.1.dev1+g396cd1a43.rocm714` |
| Torch | `2.12.0+rocm10.0.0` | `2.11.0+rocm7.14.0` |
| ROCm SDK | `10.0.0` | `7.14.0` |
| HIP runtime ABI | `71626361` | `71460850` |
| glibc minimum | 2.35 | 2.39 |
| Native bundle | `qwen38-rocm10-native-20260921` | `minicpm5-awq-native-20260921` |
| Bundle manifest digest | `ab5e847e3e29bc5f7454286996970b7deb777b817276e46dbdec81f9bd2415ba` | `f0547ef706a4d8c81e8a412705003b42e46aca1825f7888b9a16185b218d147d` |

The complete executable contract is the shipped
[`catalogue.json`](../paiton_vllm_plugin/execution/data/catalogue.json) together
with its pinned execution inventory. `paiton models` exposes exact Transformers,
safetensors, Triton, AITER and compressed-tensors versions as well. Qwen also
checks original runtime source hashes before applying its process-local import
overlays. Matching a version string alone cannot bypass these checks.

`paiton models` separates runtime compatibility, payload availability and cache
corruption. A prepared payload is still subject to GPU and checkpoint validation
before launch. All seven presets have a pinned HTTPS release archive, size and
SHA-256. Preparation automatically downloads missing bundles and verifies both
the archive and its allowlisted contents. Use `--bundle /path/to/extracted-bundle`
to supply a local copy, or prepare online before using `--offline`.

### Additional native presets

These packages use the same resolver and installed wheel. All artifact inventories
are pinned in the catalogue. Their detailed quality/performance contracts and
legacy commands remain in the model guides.

| Preset | Profile | Checkpoint / behavior | Runtime |
| --- | --- | --- | --- |
| [`qwen38-qronos`](../models/Qwen3.8/README.md#native-serving) | `qwen38-qronos-text-8k` | Original AMD Qronos W4A16; 8K/C1 text; released W4 LM head and graphs | Stack A |
| [`ornith`](../models/Ornith-1.5/README.md#native-serving) | `ornith-text-8k` | Original Quark MXFP4; lossless cached resharding; 8K/C1 text; no DFlash | Stack A |
| [`qwen38-neo`](../models/Qwen3.8-NEO-CODER-MAX/README.md#native-serving) | `qwen38-neo-image-8k` | Original mixed Q4_K_M GGUF and BF16 projector; native FP32 activations; text plus one image; 8K/C1; MTP off | Stack A |
| [`qwen3-coder`](../models/Qwen3-Coder-30B/README.md#native-serving) | `qwen3-coder-text-4k` | Original symmetric AWQ G32; native expert decode; 4K/C2; qwen3_xml tools | Stack A |
| [`gpt-oss-20b`](../models/GPT-OSS-20B/README.md#native-serving) | `gptoss-text-8k` | Original MXFP4; native expert decode; 8K/C2; Harmony reasoning/tools; pinned vocabulary included | Stack B |

**Stack A:** Python 3.12, vLLM
`0.28.0.dev0+eliovp.quark48606.g39bd959b5.rocm714`, Torch
`2.12.0+rocm7.14.0`, Transformers `5.15.1`, ROCm SDK `7.14.0`, glibc ≥2.39.
**Stack B:** the Python 3.14 / vLLM 0.26.1 MiniCPM environment in the table above.
Both require HIP runtime ABI `71460850`, CXX11 ABI and the exact remaining package
versions shown by `paiton models`. A generic installation with the same major
version is insufficient. GPU, checkpoint and artifact checks run before serving.

Qronos explicitly selects the release's W4 LM head, including lossy BF16-to-W4
quantization during model loading. Original checkpoint files stay unchanged.
This CPU preparation can take several minutes; it is part of the released path.

The Ornith native preset is explicitly non-speculative. Its published DFlash
benchmark remains associated with `models/Ornith-1.5/serve-docker.sh`. Likewise,
Qwen's existing DFlash2 65K/200K scripts retain their original settings and results.
Those scripts are preserved; the non-speculative native presets do not claim
their throughput. Every native preset disables prefix caching.

## Profiles and effective settings

Every named preset explicitly selects its documented profile. Raw paths and
repository IDs require `--profile`; they cannot silently impose precision, cache,
modality or serving choices. The legacy model launchers keep their own contracts.

**Qwen:** text only; no vision or draft model; no speculative decoding or prefix
caching. The profile explicitly permits lossy NVFP4→MXFP4 requantization during
model loading, entirely in GPU memory. It retains the released FP8/BF16 layer
exceptions, FP8 KV, FP16 recurrent state, BF16 convolution state, R4D attention,
synchronous scheduling and fixed graph/fusion configuration. It defaults to
65,536 total tokens, eight sequences, 4,096 batched tokens and 6,535,819,798 KV
cache bytes. The native command does not change the checkpoint's chat template,
reasoning or sampling defaults.

**MiniCPM:** text only, unchanged AWQ quantization, FP16 activations, lossless
in-memory packed-nibble transpose, two sequences, 512 batched tokens and 1 GiB
KV cache. It defaults to 8,192 total tokens and seed 1201. It explicitly selects
vLLM generation defaults, the released tool/reasoning parsers, chat-template
`enable_thinking: false`, prefix caching off and no speculation. Upstream
prefill/attention remain in use; the native library handles eligible one- and
two-row decode operations. Thinking and tool-call correctness are not newly
qualified by the smoke test.

The catalogue separates `settings` (defaults), `requirements` (must match) and
`bounds` (permitted numeric overrides). For example, Qwen's context/concurrency
limits can be reduced within the declared bounds; changing its KV dtype or
enabling prefix caching is rejected. Permitted bounds do not imply every
resource combination has been benchmarked or will have sufficient cache space.
Unsupported flags fail before engine startup rather than being guessed safe.

A bounded inspector handles the three pinned vLLM builds. YAML configuration is
merged with command-line overrides using the reviewed vLLM precedence, then
fingerprinted and checked again before launch. Duplicate keys, unknown options,
custom loader overrides, remote code and unreviewed configuration constructs are rejected.
Use `--config FILE.yaml` as two tokens. Default options are injected only when
the user has not supplied them through either source.

## Inspect, prepare, lock and run offline

```bash
paiton --help
paiton vllm serve --help
paiton --profile qwen38-nvfp4-w4a8-text-65k --dry-run \
  vllm serve /models/existing-qwen38-nvfp4 --port 8000

paiton --profile qwen38-nvfp4-w4a8-text-65k \
  --cache-dir /path/to/paiton-cache --bundle /path/to/qwen38-rocm10-native-20260921 \
  --offline --prepare-only --write-lock /path/to/qwen.lock.json \
  vllm serve /models/existing-qwen38-nvfp4 --port 8000

paiton --profile qwen38-nvfp4-w4a8-text-65k \
  --cache-dir /path/to/paiton-cache --offline --lock /path/to/qwen.lock.json \
  vllm serve /models/existing-qwen38-nvfp4 --port 8000
```

Help, doctor and model discovery do not initialize a GPU or download anything.
The installed vLLM help parser is delegated to a help-only process with device
initialization blocked. Dry-run reads metadata and local files, never loads
native libraries, downloads missing weights or starts vLLM. It reports pending
hash, payload and GPU checks; missing inputs produce an unresolved diagnostic.
Prepare-only verifies weights/artifacts and probes the GPU in an isolated child,
but does not load the model or initialize an engine in the launcher parent.

An exact catalogue repository ID can replace a local directory. It resolves
the pinned revision using the existing Hugging Face cache (`HF_HOME` /
`HF_HUB_CACHE`, or an explicit vLLM `--download-dir`). Offline and dry-run
resolution use cached files only. Online
preparation may retrieve missing original checkpoint files into that cache.
Explicit local directories are reused directly; source weights are never
rewritten. Qronos links existing weights into its prepared configuration directory.
Ornith preserves its source and performs the existing lossless resharding once in
the Paiton cache; allow roughly 23 GB extra disk for those required smaller shards.
Verified derived files have private receipts bound to their expected checksums
and file identities. Unchanged shards do not require another complete read on
each launch; changed files lose that cached verification.
NEO reads the original selected GGUF and projector directly. Its released tokenizer
and serving configuration come from the verified execution bundle.

Paiton uses `$XDG_CACHE_HOME/paiton` or `~/.cache/paiton` by default. Native
packages, prepared configurations/reshards, preset preferences, launch plans,
verification receipts and upstream runtime caches are
separate from source weights. Artifacts are staged, verified and atomically
installed under their manifest digest with per-package locks. Live entries are
never replaced. Corruption requires a fresh `--cache-dir`; there is no automatic
destructive cleanup. A later package/schema uses a new entry, while unknown
incompatible schemas fail. Existing single-library manifest version 1 remains
supported by the shared validator.

Locks bind catalogue/package digests, adapter/package version, exact checkpoint
files, effective engine settings and runtime ABI. API keys and transport
credentials are excluded. Transport settings such as the HTTP port and
`--shutdown-timeout 30` may change. The latter asks vLLM to allow time for
in-flight requests and workers to finish when stopping the server.
A conflicting existing lock fails; select a new output path for an intentional
new resolution. An online-prepared lock may be used offline. Plans additionally
bind local source file identities and worker environment; each worker receives
and validates the same plan. Source mutation requires preparation again.

Offline mode blocks outbound networking through Python audit checks and a
prebuilt native libc transport guard inherited by the serving processes. It
allows Unix/loopback IPC and replies to accepted API connections. This is a
control for the qualified runtime, not a sandbox for adversarial native code.
It does not support remote draft/tokenizer/media fetching or multi-node IPC.
Offline cache paths must not contain spaces or colons because of `LD_PRELOAD`.
For direct Python use, the offline audit policy applies to that entire process;
use a dedicated preparation/serving worker in a GUI application.

## Activation, execution evidence and shared API

Structured startup events distinguish `Resolved`, `Prepared`, `Validated`,
`Loaded` and `Executed`. Loading a library is not reported as execution. The
bounded request witness waits for a completed HIP event following a successful
native dispatch outside graph capture, then removes its instrumentation. A
workload that does not hit an eligible native operation within its observation
window is reported as unobserved. No per-token synchronization is added.

Qwen NVFP4, MiniCPM, Qronos, Ornith and NEO produced request-time `Executed`
events in installed-wheel GPU tests. Qwen NVFP4's witness test uses the released
4,089-token prefill specialization; an arbitrary short request may not hit an
eligible operation during the observation window.

Separate ROCm GPU traces confirmed native expert kernels during request-time
graph replay for GPT-OSS and Qwen3-Coder. Their bounded Python observer can
still report `ExecutionNotYetObserved` because replay bypasses Python dispatch.
Qwen NVFP4, MiniCPM and Qronos graph replay was not independently traced.
Ornith and NEO use eager serving in these profiles.

The same typed API serves the CLI and future Studio callers:

```python
from paiton_vllm_plugin.execution import inspect_environment, resolve, prepare, launch

arguments = ["--port", "8000", "--served-model-name", "local-qwen"]
resolution = resolve(
    "/models/existing-qwen38-nvfp4", arguments, inspect_environment(),
    profile="qwen38-nvfp4-w4a8-text-65k",
)
plan = prepare(resolution, bundle="/path/to/qwen38-rocm10-native-20260921")
launch(plan, arguments)  # Replaces this dedicated process with installed vLLM.
```

[`examples/prepare_native.py`](../examples/prepare_native.py) is a tested caller.
Frozen typed results and `PreparationError.code` avoid log scraping. Studio
itself was not changed. Process-wide adapters cannot be hot-switched; activate
in a fresh process before importing vLLM. The installed startup hook is inert
without an explicit prepared plan. Plan startup failures exit with status 78;
CLI compatibility/preparation failures exit with status 2.

An explicit `VLLM_PLUGINS` allowlist must include `register_paiton_models`; the
launcher never rewrites it or removes other plugins. Conflicting disable flags
and inherited execution overrides fail. Qwen NVFP4, MiniCPM, GPT-OSS and Qwen3-Coder retain the upstream ROCm
platform. Qronos, Ornith and NEO explicitly require the Paiton platform and also
require `paiton_platform` in an explicit plugin allowlist. Existing model-specific launchers and `paiton-vllm` remain
available through the central activation gate; their separate model/runtime
contracts still apply. Architecture-only auto-activation is removed.

## Validation status

All seven presets have passed installed-wheel inference checks on one R9700
in their supported runtime environments. Checks cover offline preparation,
checkpoint reuse, normal and streamed responses, chat, unchanged source files,
clean shutdown and request-time native execution.

| Preset | Concurrent requests checked | Additional API coverage |
| --- | ---: | --- |
| `qwen38-nvfp4` | 8 | Text chat and long prefill |
| `minicpm5` | 2 | Function calling |
| `qwen38-qronos` | 1 | Text chat |
| `ornith` | 1 | Non-speculative text chat |
| `qwen38-neo` | 1 | Function calling and one-image chat |
| `gpt-oss-20b` | 2 | Function calling |
| `qwen3-coder` | 2 | Function calling |

These checks cover the pinned runtime combinations. They do not establish
compatibility with other GPUs or arbitrary vLLM installations. The native
presets use the settings documented above; DFlash benchmark results apply to
their separate launchers and profiles.

## Runtime distribution

The wheel contains the public CLI, runtime adapters and dependency notices.
Native bundles contain allowlisted runtime binaries and sanitized metadata.
Compiler source and generated implementation source are not distributed.
See [native packaging](NATIVE_PACKAGING.md) for the public packaging tools.
