# Qwen3.8 3-bit W3A4 weights — 2026-09-26

R9700, 300 W; vLLM 0.29 / ROCm 10; 65,536 context; maximum eight sequences; APC off; thinking off; n-gram co-drafting off. Temperature 0.7, top-p 0.95, top-k 20, seed 42. BetterBench 0.6.0 quick. Three arms, each run twice in fresh processes, interleaved (24 Sept, 26 Sept MXFP4, 26 Sept W3A4, then again); the tables show the mean of the two runs. Changes compare the W3A4 release with the 24 September release and are computed from unrounded values.

| Arm | Image | Weights |
|---|---|---|
| 24 Sept release | `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-65k-20260924-r3` (`sha256:c2511888b76a…`) | MXFP4 |
| 26 Sept, MXFP4 | This round's runtime, measured on an MXFP4-only build | MXFP4 |
| 26 Sept, W3A4 | **`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-65k-20260926-w3a4-r1`** (`sha256:c4134aba665f6dd3b89354a43be2b5b814f7078db456351647a3f1b106a0da49`) | [3-bit W3A4](https://huggingface.co/EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4) |

All arms ran with `GPU_MAX_HW_QUEUES=1` (now set in the image), so every process was in the same decode timing mode and the tables do not include the gain from that setting. The W3A4 timing runs used an earlier calibration of the same 3-bit format; the tensor layout and runtime are identical, so the timing applies to the published weights. The accuracy results and the four-request run at the launcher's default KV budget used the published weights.

**Decode, single stream, tok/s.**

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

Weighted decode: **153.8 → 156.1 → 184.4 tok/s (+19.9%)**. Median C1 decode forward time: 28.3–28.4 ms → 28.0 ms → 22.4–22.5 ms.

The two W3A4 runs differ most on file edit (179.0 and 211.1 tok/s) and chat (132.5 and 139.1 tok/s); every other category agrees within 3 tok/s.

**Concurrency, aggregate generated tok/s over each complete 48-request workload.**

| Concurrent requests | 24 Sept release | 26 Sept, MXFP4 | 26 Sept, W3A4 | Change |
|---:|---:|---:|---:|---:|
| 1 | 122.0 | 123.7 | **148.8** | +22.0% |
| 2 | 204.2 | 209.3 | **249.3** | +22.1% |
| 4 | 308.2 | 315.5 | **368.3** | +19.5% |
| 8 | 425.3 | 428.0 | **492.1** | +15.7% |

Run-to-run spread: the 24 September release read 201.3 / 207.0 at C2 and 304.0 / 312.4 at C4; W3A4 read 252.0 / 246.7 at C2, 364.3 / 372.3 at C4 and 496.2 / 488.1 at C8.

**Prefill, input tok/s.**

| Nominal prefill depth | 24 Sept release | 26 Sept, MXFP4 | 26 Sept, W3A4 | Change |
|---:|---:|---:|---:|---:|
| 2,000 | 3,689 | 3,691 | **4,156** | +12.7% |
| 8,000 | 3,834 | 3,831 | **4,165** | +8.6% |
| 16,000 | 3,871 | 3,871 | **4,103** | +6.0% |
| 32,000 | 3,751 | 3,750 | **3,958** | +5.5% |
| 64,000 | 3,455 | 3,455 | **3,629** | +5.0% |

Time to first token at these depths shortens by the same factors: 407 → 363 ms at 2K, 3.06 → 2.88 s at 16K and 13.6 → 13.0 s at 64K.

**Long context: four 61K-token requests.** Four requests of about 61,400 prompt tokens and 512 output tokens each, sent together, on the 65K release profile:

| Configuration | KV cache | Wall time | Requests decoding together | Decode tok/s, all active |
|---|---:|---:|---:|---:|
| MXFP4, release KV budget | 174,634 tokens | 105.6 s | 2 | 69¹ |
| W3A4, same KV budget | 174,634 tokens | 92.9 s | 2 | 117¹ |
| W3A4, launcher default (freed memory as KV) | 250,578 tokens | **86.2 s** | **4** | **199** |

¹ Two 61K requests decoding together. The MXFP4 row used this round's MXFP4 build. With the launcher default, peak VRAM is 31.39 GiB, level with MXFP4's 31.37 GiB. Model memory falls from 19.18 to 15.89 GiB.

**Accuracy.** Served model, greedy decoding, thinking off, W3A4 paired against MXFP4 on identical items. Δ is in points with a 95% interval.

| Benchmark | MXFP4 | W3A4 | Δ [95% CI] |
|---|---:|---:|---:|
| GSM8K 5-shot (1,319) | 95.68 | 95.30 | −0.38 [−1.44, +0.68] |
| HumanEval pass@1 (164) | 95.12 | 93.90 | −1.22 [−4.99, +2.56] |
| MMLU-Pro subset, 0-shot (14 × 100) | 62.57 | 59.71 | −2.86 [−4.81, −0.90] |
| Needle at 61,440 tokens (80) | 100 | 100 | 0 |

Math, code and long-context retrieval stay within noise. Knowledge recall drops by about 3 MMLU-Pro points, the trade-off of 3-bit weights. DFlash2 mean acceptance length changes by −0.4% to +2.8% relative to MXFP4. MXFP4 (`--weights mxfp4`) remains the choice for maximum knowledge accuracy.

**What changed in the image.**

- **Optional 3-bit W3A4 weights.** The decoder projections use 3-bit integer weights with one scale per 128 weights, stored in a block-wise Hadamard-rotated basis and calibrated with GPTQ on permissively licensed data. During prefill, the rotated layers also take 4-bit activations on RDNA4's int4 matrix instructions; decode keeps 8-bit activations. All other tensors come from the same pinned target checkpoint, and the DFlash2 drafter is unchanged.
- **Fused GDN speculative verify.** The Gated DeltaNet layers verify drafted tokens in one fused kernel. It is exact: the 26 September MXFP4 arm returned the same twelve greedy outputs as the 24 September release, with +1.5% weighted decode.
- **`GPU_MAX_HW_QUEUES=1`.** Some fresh server processes on the R9700 started in a slower decode mode. The image now sets this variable, which keeps every process in the fast mode.

Every run: 40/40 decode, 192/192 concurrency and 40/40 prefill scored requests (plus the fixed warmups), and no serving errors. Each run repeated its twelve greedy outputs after the benchmark. W3A4 changes 7 of the 12 greedy outputs relative to MXFP4, as expected from lossy weights, and reproduced its own outputs in both processes. Server startup took about 210 s for the MXFP4 arms and 230 s for W3A4.

Sampled output content and accepted-token work can differ; these are serving-throughput measurements, not identical-output timing.
Nominal prefill depths correspond to median actual prompt lengths 1516.5, 5894.5, 11802, 23549.5 and 47016.5.
Decode has five scored requests per category after one warmup. Prefill has eight scored requests per depth after two warmups. Weighted decode uses the BetterBench category weights (code 0.30, reasoning 0.20, prose 0.15, json 0.15, file edit 0.10, summarization 0.10); chat and math carry no weight.

[Machine-readable results](numbers.json) · [Model page](../../README.md)
