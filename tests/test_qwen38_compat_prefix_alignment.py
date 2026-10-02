"""Prefix-cache alignment of the vendored vLLM overlays (run inside the serving image: needs vllm and torch).

KV4 long-context mode: 1,600-token target attention / GDN blocks next to the DFlash2 drafter's sliding-window layers,
whose 16-token pages are grown to fill the 1,792,000-byte page. Grown to 864 tokens, the hit alignment (lcm of the
group blocks) became 43,200 tokens and a repeated 258K prompt hit only 172,800 tokens; with prefix caching the drafter
block must divide the target block (800), and align-mode prefill chunks must end on the GDN state-slot block.
"""
import importlib.util
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)

OVERLAYS = Path(__file__).resolve().parents[1] / "paiton_vllm_plugin/_vendor/qwen38/compat/overlays/vllm/v1/core"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module   # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


kcu = _load("paiton_overlay_kv_cache_utils", OVERLAYS / "kv_cache_utils.py")
sched = _load("paiton_overlay_scheduler", OVERLAYS / "sched/scheduler.py")

PAGE = 1_792_000   # one KV4 target layer: 1,600 tokens x 1,120 B


def _kv4_specs():
    return {
        "target.0": FullAttentionSpec(block_size=1600, num_kv_heads=1, head_size=560, dtype=torch.int8),
        "drafter.0": SlidingWindowSpec(block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float8_e4m3fn,
                                       sliding_window=2048),
        "gdn.0": MambaSpec(block_size=1600, shapes=((4,),), dtypes=(torch.float16,), page_size_padded=PAGE,
                           mamba_cache_mode="align"),
    }


def test_kv4_specs_have_the_measured_page_sizes():
    specs = _kv4_specs()
    assert specs["target.0"].page_size_bytes == PAGE
    assert specs["drafter.0"].page_size_bytes == 32_768
    assert specs["gdn.0"].page_size_bytes == PAGE


def test_prefix_caching_grows_the_drafter_block_to_a_divisor_of_the_target_block():
    out = kcu.unify_kv_cache_spec_page_size(_kv4_specs(), align_padded_blocks=True)
    assert out["drafter.0"].block_size == 800
    assert out["drafter.0"].page_size_bytes == PAGE
    assert math.lcm(*(s.block_size for s in out.values())) == 1600


def test_without_prefix_caching_the_drafter_block_still_fills_the_page():
    out = kcu.unify_kv_cache_spec_page_size(_kv4_specs())
    assert out["drafter.0"].block_size == 864
    assert out["drafter.0"].page_size_bytes == PAGE


def test_exact_page_ratio_is_unchanged_with_prefix_caching():
    # fp8 long mode: 880-token target page = 55 drafter pages, no padding either way
    specs = {
        "target.0": FullAttentionSpec(block_size=880, num_kv_heads=8, head_size=128, dtype=torch.float8_e4m3fn),
        "drafter.0": SlidingWindowSpec(block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float8_e4m3fn,
                                       sliding_window=2048),
    }
    for align in (False, True):
        out = kcu.unify_kv_cache_spec_page_size(dict(specs), align_padded_blocks=align)
        assert out["drafter.0"].block_size == 880


def _groups(*specs):
    return [KVCacheGroupSpec([f"layer.{i}"], s) for i, s in enumerate(specs)]


def test_align_mode_splits_on_the_gdn_state_block():
    s = _kv4_specs()
    groups = _groups(s["target.0"], SlidingWindowSpec(block_size=800, num_kv_heads=8, head_size=128,
                                                       dtype=torch.float8_e4m3fn, sliding_window=2048,
                                                       page_size_padded=PAGE), s["gdn.0"])
    assert sched._paiton_mamba_align_block_size(groups, fallback=800) == 1600


def test_align_block_is_unchanged_when_all_groups_share_one_block():
    groups = _groups(FullAttentionSpec(block_size=880, num_kv_heads=8, head_size=128, dtype=torch.float8_e4m3fn),
                     MambaSpec(block_size=880, shapes=((4,),), dtypes=(torch.float16,), mamba_cache_mode="align"))
    assert sched._paiton_mamba_align_block_size(groups, fallback=880) == 880


def test_align_block_falls_back_without_mamba_groups():
    groups = _groups(FullAttentionSpec(block_size=1600, num_kv_heads=1, head_size=560, dtype=torch.int8))
    assert sched._paiton_mamba_align_block_size(groups, fallback=16) == 16
