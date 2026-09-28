import ctypes
import importlib
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

# Importing the hook installs its import finder unless switched off; keep this test process side-effect free.
os.environ.setdefault('PAITON_ATTN_PAD_ZERO', '0')
from paiton_runtime_compat import attn_pad_rows  # noqa: E402

ENV = 'PAITON_ATTN_PAD_ZERO'
REAL = 7.0


class FakeHip:
    """hipMemsetAsync stand-in: records the call and clears the host bytes it was given."""

    def __init__(self, rc=0):
        self.rc = rc
        self.calls = []

    def __call__(self, ptr, value, nbytes, stream):
        self.calls.append((ptr, value, nbytes, stream))
        if self.rc == 0:
            ctypes.memset(ptr, value, nbytes)
        return self.rc


class R4DAttentionImpl:
    """Like the R4D backend: writes the rows of the scheduled requests only."""

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return output.fill_(0)
        output[:getattr(attn_metadata, 'num_actual_tokens', output.shape[0])] = REAL
        return output


class TritonAttentionImpl:
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                output_scale=None, output_block_scale=None):
        return output


class Attention(torch.nn.Module):
    def __init__(self, impl):
        super().__init__()
        self.impl = impl


def radiance_module(cls=R4DAttentionImpl):
    module = ModuleType('radiance_r4d_attn')
    module.R4DAttentionImpl = cls
    return module


def target_model(r4d=3, triton=1):
    layers = [Attention(R4DAttentionImpl()) for _ in range(r4d)]
    layers += [Attention(TritonAttentionImpl()) for _ in range(triton)]
    return torch.nn.Sequential(*layers)


def stale(rows, *shape):
    # Pad rows hold whatever the graph pool had there: model it as NaN.
    return torch.full((rows, *shape), math.nan, dtype=torch.bfloat16)


def call(layer, metadata, output):
    # vLLM's unified_attention_with_output calling convention.
    return layer.impl.forward(layer, None, None, None, None, metadata, output=output,
                              output_scale=None, output_block_scale=None)


class HookCase(unittest.TestCase):
    def setUp(self):
        self.hip = FakeHip()
        self.stream = 0x5151
        patches = [patch.object(attn_pad_rows, '_MEMSET', self.hip),
                   patch.object(attn_pad_rows, 'current_stream', lambda: self.stream),
                   patch.dict(sys.modules, {'radiance_r4d_attn': radiance_module()}),
                   patch.dict(os.environ, {ENV: '1'})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def installed(self, **kw):
        model = target_model(**kw)
        attn_pad_rows.install(model)
        return model


class ZeroingTests(HookCase):
    def test_padded_step_zeroes_only_pad_rows(self):
        layer = self.installed()[0]
        output = stale(32, 48)
        call(layer, SimpleNamespace(num_actual_tokens=26), output)
        self.assertTrue(torch.equal(output[:26], torch.full((26, 48), REAL, dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(output[26:], torch.zeros(6, 48, dtype=torch.bfloat16)))
        row = 48 * 2
        self.assertEqual(self.hip.calls, [(output.data_ptr() + 26 * row, 0, 6 * row, self.stream)])

    def test_unpadded_step_is_untouched(self):
        layer = self.installed()[0]
        output = stale(32, 48)
        call(layer, SimpleNamespace(num_actual_tokens=32), output)
        self.assertEqual(self.hip.calls, [])
        self.assertTrue(torch.equal(output, torch.full((32, 48), REAL, dtype=torch.bfloat16)))

    def test_profile_run_without_metadata_is_untouched(self):
        layer = self.installed()[0]
        output = stale(16, 8)
        call(layer, None, output)
        self.assertEqual(self.hip.calls, [])

    def test_three_dimensional_output_row_bytes(self):
        layer = self.installed()[0]
        output = stale(16, 4, 8)
        call(layer, SimpleNamespace(num_actual_tokens=9), output)
        row = 4 * 8 * 2
        self.assertEqual(self.hip.calls, [(output.data_ptr() + 9 * row, 0, 7 * row, self.stream)])
        self.assertTrue(torch.equal(output[9:], torch.zeros(7, 4, 8, dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(output[:9], torch.full((9, 4, 8), REAL, dtype=torch.bfloat16)))

    def test_positional_output_is_found(self):
        layer = self.installed()[0]
        output = stale(8, 4)
        layer.impl.forward(layer, None, None, None, None, SimpleNamespace(num_actual_tokens=5), output)
        self.assertEqual(len(self.hip.calls), 1)
        self.assertTrue(torch.equal(output[5:], torch.zeros(3, 4, dtype=torch.bfloat16)))

    def test_stream_is_read_at_call_time(self):
        layer = self.installed()[0]
        seen = []
        for stream in (11, 22):
            self.stream = stream
            output = stale(16, 4)
            call(layer, SimpleNamespace(num_actual_tokens=10), output)
            seen.append(self.hip.calls[-1][3])
        self.assertEqual(seen, [11, 22])

    def test_non_contiguous_output_fails_closed(self):
        layer = self.installed()[0]
        output = stale(32, 96)[:, ::2]
        with self.assertRaisesRegex(RuntimeError, 'contiguous'):
            call(layer, SimpleNamespace(num_actual_tokens=26), output)
        self.assertEqual(self.hip.calls, [])

    def test_memset_error_fails_closed(self):
        self.hip.rc = 1
        layer = self.installed()[0]
        with self.assertRaisesRegex(RuntimeError, 'hipMemsetAsync'):
            call(layer, SimpleNamespace(num_actual_tokens=26), stale(32, 4))

    def test_metadata_without_token_count_fails_closed(self):
        layer = self.installed()[0]
        with self.assertRaisesRegex(RuntimeError, 'num_actual_tokens'):
            call(layer, SimpleNamespace(), stale(32, 4))


class InstallTests(HookCase):
    def test_wraps_only_the_targets_r4d_layers(self):
        model = target_model(r4d=3, triton=2)
        triton_forward = [m.impl.forward for m in model[3:]]
        self.assertEqual(attn_pad_rows.install(model), 3)
        for m in model[3:]:
            self.assertNotIn('forward', vars(m.impl))
        self.assertEqual([m.impl.forward for m in model[3:]], triton_forward)
        output = stale(32, 4)
        call(model[4], SimpleNamespace(num_actual_tokens=26), output)
        self.assertEqual(self.hip.calls, [])

    def test_install_is_idempotent(self):
        model = target_model()
        self.assertEqual(attn_pad_rows.install(model), 3)
        self.assertEqual(attn_pad_rows.install(model), 0)
        call(model[0], SimpleNamespace(num_actual_tokens=26), stale(32, 4))
        self.assertEqual(len(self.hip.calls), 1)

    def test_wraps_the_forward_current_at_install(self):
        # An adapter (KV4) replaced the class forward before serving; its forward must stay in the chain.
        routed = []

        def kv4_forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                        output_scale=None, output_block_scale=None):
            routed.append(attn_metadata.num_actual_tokens)
            output[:attn_metadata.num_actual_tokens] = REAL
            return output

        with patch.object(R4DAttentionImpl, 'forward', kv4_forward):
            layer = self.installed()[0]
            output = stale(32, 4)
            result = call(layer, SimpleNamespace(num_actual_tokens=30), output)
        self.assertIs(result, output)
        self.assertEqual(routed, [30])
        self.assertTrue(torch.equal(output[30:], torch.zeros(2, 4, dtype=torch.bfloat16)))

    def test_off_switch_leaves_the_forward_alone(self):
        with patch.dict(os.environ, {ENV: '0'}):
            model = target_model()
            self.assertEqual(attn_pad_rows.install(model), 0)
        self.assertNotIn('forward', vars(model[0].impl))
        output = stale(32, 4)
        call(model[0], SimpleNamespace(num_actual_tokens=26), output)
        self.assertEqual(self.hip.calls, [])
        self.assertTrue(torch.isnan(output[26:].float()).all())

    def test_default_is_on(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)
            self.assertEqual(attn_pad_rows.install(target_model()), 3)

    def test_invalid_switch_fails_closed(self):
        with patch.dict(os.environ, {ENV: 'off'}):
            with self.assertRaisesRegex(RuntimeError, ENV):
                attn_pad_rows.install(target_model())

    def test_without_r4d_backend_nothing_is_wrapped(self):
        with patch.dict(sys.modules, {'radiance_r4d_attn': None}):
            model = target_model()
            self.assertEqual(attn_pad_rows.install(model), 0)
        self.assertNotIn('forward', vars(model[0].impl))


def worker_module(model, layer_rows):
    """Worker whose warmup runs padded forwards, like vLLM's capture/profile warmups."""
    module = ModuleType('fake_gpu_worker')

    class Worker:
        def __init__(self):
            self.model_runner = SimpleNamespace(get_model=lambda: model)

        def compile_or_warm_up_model(self):
            layer_rows.append(call(model[0], SimpleNamespace(num_actual_tokens=20), stale(24, 4)))
            return 'compiled'

    module.Worker = Worker
    return module


class WorkerHookTests(HookCase):
    def test_installs_after_warmup_so_warmup_and_capture_are_unchanged(self):
        model = target_model()
        warm = []
        module = worker_module(model, warm)
        attn_pad_rows.patch(module)
        attn_pad_rows.patch(module)  # idempotent
        self.assertEqual(module.Worker().compile_or_warm_up_model(), 'compiled')
        self.assertTrue(torch.isnan(warm[0][20:].float()).all())  # warmup/capture ran the original forward
        self.assertEqual(self.hip.calls, [])
        output = stale(32, 4)
        call(model[0], SimpleNamespace(num_actual_tokens=26), output)
        self.assertEqual(len(self.hip.calls), 1)  # one wrapper, installed once

    def test_does_not_copy_other_hooks_markers(self):
        module = worker_module(target_model(), [])
        module.Worker.compile_or_warm_up_model._paiton_gdn_warmup_reset = True
        attn_pad_rows.patch(module)
        method = module.Worker.compile_or_warm_up_model
        self.assertFalse(getattr(method, '_paiton_gdn_warmup_reset', False))
        self.assertTrue(getattr(method, attn_pad_rows.MARK))

    def test_finder_patches_on_import_and_keeps_loader_path(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'fake_attn_worker_target.py'
            source.write_text('class Worker:\n'
                              '    def compile_or_warm_up_model(self):\n'
                              '        return "compiled"\n')
            finder = attn_pad_rows._Finder()
            with patch.object(attn_pad_rows, 'TARGET', 'fake_attn_worker_target'), \
                    patch.object(sys, 'path', [root, *sys.path]), \
                    patch.object(sys, 'meta_path', [finder, *sys.meta_path]), \
                    patch.dict(sys.modules):
                sys.modules.pop('fake_attn_worker_target', None)
                module = importlib.import_module('fake_attn_worker_target')
            loader = module.__spec__.loader
            self.assertIsInstance(loader, attn_pad_rows._Loader)
            self.assertEqual(loader.path, str(source))
            self.assertTrue(getattr(module.Worker.compile_or_warm_up_model, attn_pad_rows.MARK))


class HipRuntimeTests(unittest.TestCase):
    LINE = '7f00-7f10 r-xp 00000000 00:1a 42 {}'

    def maps(self, *paths):
        return '\n'.join(self.LINE.format(p) for p in paths)

    def test_single_mapped_runtime(self):
        path = '/x/_rocm_sdk_devel/lib/libamdhip64.so.7.16.26361-25e14349f2'
        text = self.maps(path, path, '/usr/lib/libc.so.6')
        self.assertEqual(attn_pad_rows.hip_runtime_path(text), path)

    def test_missing_or_ambiguous_runtime_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'libamdhip64'):
            attn_pad_rows.hip_runtime_path(self.maps('/usr/lib/libc.so.6'))
        with self.assertRaisesRegex(RuntimeError, 'libamdhip64'):
            attn_pad_rows.hip_runtime_path(self.maps('/a/libamdhip64.so.7', '/b/libamdhip64.so.6'))


if __name__ == '__main__':
    unittest.main()
