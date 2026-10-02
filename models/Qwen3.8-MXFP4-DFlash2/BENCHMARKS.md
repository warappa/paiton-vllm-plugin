# Qwen3.8 MXFP4: Paiton on regular vLLM

> **New, 1 October 2026:** the 262K long-context mode on the 3-bit weights, measured with the full 20-pass
> BetterBench run at the 262,144 setting: 174.3 tok/s weighted decode, 29.3 ms update p99, 85 ms TTFT p50,
> 458.9 tok/s at eight concurrent requests, 3,549 tok/s prefill at 64K, 2,395 tok/s at 184K input tokens.
> [Results](benchmarks/2026-10-01-262k/README.md).

> **New comparison, 16 September 2026:** [BetterBench 0.6.0 against current
> GGZ14 on one R9700](benchmarks/2026-09-16-betterbench/README.md), including
> charts, all four runs, sanitized raw data and reproduction commands.
> The historical suites below use different settings and remain separate.

Paiton combines native HIP kernels, adapted Radiance techniques, and DFlash2
through a plugin on the official vLLM runtime. The checkpoint stays packed in
MXFP4. There is no separate Radiance engine dependency and no replacement of
the installed vLLM library.

The full comparison measures **1.22× weighted serial throughput**
and **1.57× throughput at eight concurrent requests**
against the recommended Radiance + DFlash2 configuration on the same R9700.
Prefill is **12.5–17.3% faster** across the three
tested depths. Median time to first token at eight concurrent requests is
**97.0% lower**, including queueing.
The same 5 GiB cache budget provides **2.89× estimated token capacity**.

## Three-engine comparison

One Radeon AI PRO R9700, the same AMD Qwen3.8-27B-Quark-AWQ-MXFP4 checkpoint,
FP8 KV cache, 8,192-token maximum context, and a 5 GiB cache pool. All three
engines completed the same 54-request matrix: eight task categories,
concurrency 1/2/4/8, and three prompt depths. Generation is capped at 128 tokens
in this shared matrix.

Prefill labels use actual tokenized prompt lengths (approximately 1,556, 3,024,
and 5,226 tokens), rather than the benchmark's nominal 2K/4K/7K depth settings.
Concurrency throughput divides completed output tokens by sweep wall time,
including prefill and queueing. Weighted serial decode combines the preset's
category medians. Prefill divides actual prompt tokens by HTTP time to first
token; it is an end-to-end serving measurement.

| Metric | Stock vLLM O2 | vLLM-Radiance + DFlash2 | Paiton + native adaptations + DFlash2 |
|---|---:|---:|---:|
| Weighted serial decode, tok/s | 4.5 | 99.3 | 113.5 |
| C1 aggregate throughput, tok/s | 4.5 | 78.0 | 90.0 |
| C2 aggregate throughput, tok/s | 8.8 | 151.2 | 160.5 |
| C4 aggregate throughput, tok/s | 17.4 | 189.2 | 231.5 |
| C8 aggregate throughput, tok/s | 33.7 | 175.6 | 328.5 |
| Prefill ~1,556 input tokens, input tok/s | 1,435.9 | 2,818.5 | 3,186.9 |
| Prefill ~3,024 input tokens, input tok/s | 1,385.8 | 2,982.5 | 3,520.2 |
| Prefill ~5,226 input tokens, input tok/s | 1,188.5 | 2,921.2 | 3,377.5 |

## Full-workload confirmation

Both accelerated engines also complete the full 188-request BetterBench preset:
ten measured passes per category, 24 requests at each concurrency level, and
four measurements at each prefill depth. These measurements use the longer
original output budgets; compare them within this table.

| Metric | vLLM-Radiance + DFlash2 | Paiton on regular vLLM + DFlash2 |
|---|---:|---:|
| Weighted serial decode, tok/s | 86.0 | 104.9 |
| C1 aggregate throughput, tok/s | 76.9 | 89.6 |
| C2 aggregate throughput, tok/s | 142.7 | 165.7 |
| C4 aggregate throughput, tok/s | 187.1 | 254.8 |
| C8 aggregate throughput, tok/s | 200.3 | 314.5 |
| Prefill ~1,556 input tokens, input tok/s | 2,828.4 | 3,181.5 |
| Prefill ~3,024 input tokens, input tok/s | 3,002.7 | 3,521.2 |
| Prefill ~5,226 input tokens, input tok/s | 2,925.8 | 3,367.2 |

## Time to first token

Full-workload medians; lower is better.

| Metric | vLLM-Radiance + DFlash2 | Paiton on regular vLLM + DFlash2 |
|---|---:|---:|
| C1 median TTFT, ms | 94.8 | 105.0 |
| C2 median TTFT, ms | 148.6 | 159.9 |
| C4 median TTFT, ms | 1,626.8 | 180.1 |
| C8 median TTFT, ms | 6,585.5 | 194.8 |

## Generation by task

Full-workload median decode throughput, in tokens per second.

| Metric | vLLM-Radiance + DFlash2 | Paiton on regular vLLM + DFlash2 |
|---|---:|---:|
| code | 88.5 | 102.6 |
| reasoning | 63.5 | 77.9 |
| prose | 69.0 | 80.0 |
| json | 110.4 | 147.3 |
| file_edit | 99.2 | 116.7 |
| summarization | 99.1 | 127.1 |
| math | 111.6 | 138.9 |
| chat | 60.8 | 71.6 |

## More concurrent work from the same cache budget

The serving engines report 25,746 token slots for Radiance
and 74,430 for Paiton: **2.89× capacity** from the same
5 GiB pool. Logs record up to 3 and
8 active requests respectively. The concurrent
completion and latency measurements above show the practical effect.
The token-slot figure is a cache-capacity estimate. The tested server is configured
for an 8,192-token per-request limit and up to eight concurrent requests.

## Reproduction and scope

- Target: `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, revision
  `5233554c5fa56afda40150556b95573c2d7d29c0`.
- Draft: `tcclaviger/Qwen3.8-27B-DFlash2-FP8`, revision
  `ee0cb26a8279b7910cc28d82a8a3e15e4728d56f`; seven speculative tokens,
  unpadded drafting, synchronous scheduling for both accelerated engines.
- Stock retains the best tested O2 compiler, fusion, graph, and async settings.
  Its checkpoint-native W4A4 emulation differs from the accelerated W4A8 paths.
  This comparison measures complete engine configurations on the same weights.
- Prefix caching is disabled. Both accelerated engines use greedy sampling,
  the same FP8 KV settings, and the same original benchmark corpus and limits.
- Benchmark: [BetterBench](https://github.com/GGZ14/BetterBench/tree/575cc3925bac922d6ad4a39e62502673799979d9),
  pinned commit `575cc3925bac922d6ad4a39e62502673799979d9`, corpus v1.
- Paiton uses the official vLLM 0.28 ROCm image through its extension APIs.
  The native artifacts use HIP; the external vLLM adapter retains standard
  vLLM's framework dependencies.
- The public Radiance README and X screenshots used two GPUs and other run
  conditions. They are context, not a controlled comparator for these one-GPU
  measurements. No per-GPU extrapolation is presented as a measurement.
- Stock completed the common 54-request matrix. Its earlier full preset was
  interrupted for runtime cost, so it has no full-preset aggregate result.

Radiance and StillDeadcode/libr4d are credited for the adapted kernel techniques.
Implementation research and correctness diagnostics remain in the private
engineering records. Commands, model/artifact hashes, raw timing samples,
request counts, engine audits, and memory evidence are retained for review.

Evidence runs: stock403; fresh Radiance493 and Paiton492 (shared matrix);
Radiance495 and Paiton494 (full preset). These measurements qualify the native payload in the v1.0.0 release.

## Ordinary vLLM deployment check

The pinned candidate also passes 12 requests through the ordinary
`vllm.entrypoints.openai.api_server` entry point: streaming chat, eight concurrent
requests, an 8,190-input/2-output-token context-boundary request, and a successful
fresh request afterward. The private benchmark server and audit worker are absent.
The target and draft files are verified against their locked hashes before launch.

All 2,893 installed vLLM files match the official base image. All 16 native
libraries match the allowlist and load without importing serving frameworks;
there is no separate Radiance or DFlash distribution. Standard vLLM's own
framework dependencies remain in the serving image.
[Runtime audit](runtime-audit.json) · [Deployment check](deployment-check.json).

[Summary metrics](metrics.json) · [Timing samples](raw-timings/manifest.json) ·
[Three-engine chart](three-engine-throughput.svg) · [Full confirmation chart](full-confirmation.svg).
