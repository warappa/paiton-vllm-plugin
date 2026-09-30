"""The int2 draft head must never be quantised from a DFlash2 drafter's lm_head at load time.

That tensor is only allocated when load_weights returns; vLLM shares the target's head in afterwards.
Whether the allocated memory reads as zero depends on what the allocator hands out: a user saw
recycled, non-zero memory, the head was quantised from garbage, and DFlash2 acceptance fell to 0 %
with the text still coherent (the target is untouched)."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch

SRC = Path(__file__).parents[1] / 'paiton_vllm_plugin/_vendor/qwen38/radiance_drafthead.py'
spec = importlib.util.spec_from_file_location('radiance_drafthead_under_test', SRC)
dh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dh)


class FakeLogitsProcessor:
    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        return ('stock', lm_head)


def drafter(fill):
    weight = torch.full((64, 5120), fill, dtype=torch.bfloat16)
    return types.SimpleNamespace(lm_head=types.SimpleNamespace(weight=weight),
                                 candidate_logits_processor=FakeLogitsProcessor(),
                                 logits_processor=FakeLogitsProcessor())


class DraftHeadDeferTests(unittest.TestCase):
    def test_dflash2_candidate_head_is_never_quantised_at_load(self):
        for fill in (0.0, 3.0):   # zero memory (our machines) and recycled non-zero memory (the report)
            with self.subTest(fill=fill):
                mtp = drafter(fill)
                with patch.object(dh, '_quantize_head_now', side_effect=AssertionError('quantised at load')) as now:
                    status = dh._quantize_draft_head(mtp, 'candidate_logits_processor')
                self.assertEqual(now.call_count, 0)
                self.assertIn('first use', status)
                lp = mtp.candidate_logits_processor
                self.assertIs(lp._apply_head.__func__, dh._apply_head_lazy)
                self.assertTrue(lp._radiance_topk_only)

    def test_first_use_quantises_the_head_actually_passed_in(self):
        mtp = drafter(3.0)
        with patch.object(dh, '_quantize_head_now', side_effect=AssertionError('quantised at load')):
            dh._quantize_draft_head(mtp, 'candidate_logits_processor')
        lp = mtp.candidate_logits_processor
        shared = types.SimpleNamespace(weight=torch.full((64, 5120), 1.0, dtype=torch.bfloat16))
        seen = []

        def now(lp_, lm_head):
            seen.append(lm_head)
            lp_._apply_head = types.MethodType(lambda self, h, x, b: ('int2', h), lp_)
            return 'bound'
        with patch.object(dh, '_quantize_head_now', side_effect=now):
            out = lp._apply_head(shared, torch.zeros(1, 5120), None)
        self.assertEqual(seen, [shared])
        self.assertEqual(out, ('int2', shared))

    def test_mtp_head_present_at_load_still_quantises_at_load(self):
        mtp = drafter(3.0)
        with patch.object(dh, '_quantize_head_now', return_value='bound') as now:
            status = dh._quantize_draft_head(mtp, 'logits_processor')
        self.assertEqual(now.call_count, 1)
        self.assertEqual(status, 'bound')


if __name__ == '__main__':
    unittest.main()
