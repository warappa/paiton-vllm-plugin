# 1 October 2026: the 262K long-context mode on the 3-bit weights, full BetterBench

One R9700 (300 W), the 29 September r2 image, 3-bit W3A4 weights, BetterBench 0.6.0 **full** settings (20 passes per
category, 48 requests per concurrency level, 2 + 8 prefill runs per depth), temperature 0.7, thinking off, cold
prefix cache. Two fresh server processes: `run-3bit.sh --context 262144` (262K mode: FP8 KV, prefix caching, eight
requests) and `run-3bit.sh --thinking off` (65K default: 4-bit KV, no prefix caching). Reports:
[262K mode](report-262k-mode.md), [262K mode deep prefill](report-262k-deep-prefill.md),
[65K default](report-65k-default.md); [numbers.json](numbers.json).

| BetterBench row | 262K mode | 65K default |
|---|---:|---:|
| combined single-stream decode (weighted) | 174.3 tok/s | 178.9 tok/s |
| update p99 (gap between stream updates) | 29.3 ms | 28.0 ms |
| TTFT p50, batch 1 | 85 ms | 86 ms |
| aggregate @ 1 / 2 / 4 / 8 concurrent | 144.9 / 249.3 / 345.3 / 458.9 tok/s | 151.3 / 250.0 / 366.4 / 478.8 tok/s |
| prefill @ 2K / 8K / 16K / 32K / 64K | 4,179 / 4,087 / 3,915 / 3,836 / 3,549 tok/s | 4,177 / 4,158 / 4,094 / 3,939 / 3,481 tok/s |
| prefill @ 94K / 184K tokens (BetterBench depths 128K / 250K) | 3,040 / 2,395 tok/s (31.0 s / 76.7 s) | – |

Single-stream decode by category, tok/s (262K mode / 65K default): chat 148.9 / 129.4, code 202.9 / 203.5, file edit 221.3 / 228.1, json 246.4 / 258.7, math 211.6 / 220.1, prose 92.1 / 93.7, reasoning 131.5 / 134.1, summarization 142.2 / 153.2.
Every request completed (160/160 decode, 192/192 concurrency, 40/40 + 20 prefill) in both arms.

Reading the two columns: at eight concurrent requests the 262K mode is -4.2% against the 65K default and at
the 64K prefill depth +1.9% (prefix-caching bookkeeping and 3,520-row chunks); its weighted single-stream
decode is -2.6%. The two modes differ in cache type (FP8 with prefix caching versus 4-bit without), so the
single-stream figures are not a kernel-for-kernel comparison.

For reference, a BetterBench screenshot posted for another R9700 stack at the same 262,144 setting read 118.8 tok/s
combined decode, 38.9 ms update p99, 92 ms TTFT p50, 249.1 tok/s at eight concurrent and 2,853 tok/s prefill at 64K.
