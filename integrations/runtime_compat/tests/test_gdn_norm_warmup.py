import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from paiton_runtime_compat import gdn_norm_warmup


class Layer(torch.nn.Module):
    def __init__(self, status):
        super().__init__()
        self.register_buffer('_paiton_gdn_norm_error', status, persistent=False)


def model_with_shared_status():
    status = torch.zeros(1, dtype=torch.int32)
    model = torch.nn.Sequential(Layer(status), Layer(status), torch.nn.Linear(2, 2))
    return model, status


def worker_module(model, writes):
    module = ModuleType('fake_gpu_worker')

    class Worker:
        def __init__(self):
            self.model_runner = SimpleNamespace(get_model=lambda: model)

        def compile_or_warm_up_model(self):
            # A synthetic warmup forward leaves the sticky status set.
            model[0]._paiton_gdn_norm_error.fill_(writes)
            return 'compiled'

    module.Worker = Worker
    return module


class GdnNormWarmupTests(unittest.TestCase):
    def test_clears_warmup_status_in_place_once(self):
        model, status = model_with_shared_status()
        address = status.data_ptr()
        module = worker_module(model, 1)
        gdn_norm_warmup.patch(module)
        gdn_norm_warmup.patch(module)  # idempotent
        self.assertEqual(module.Worker().compile_or_warm_up_model(), 'compiled')
        self.assertEqual(int(status.item()), 0)
        self.assertEqual(status.data_ptr(), address)
        self.assertEqual(len(gdn_norm_warmup.status_buffers(model)), 1)

    def test_status_set_after_warmup_is_kept(self):
        model, status = model_with_shared_status()
        module = worker_module(model, 1)
        gdn_norm_warmup.patch(module)
        module.Worker().compile_or_warm_up_model()
        status.fill_(1)  # a served step's nonfinite row
        self.assertEqual(int(status.item()), 1)

    def test_finder_patches_on_import_and_keeps_loader_path(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'fake_worker_target.py'
            source.write_text('class Worker:\n'
                              '    def compile_or_warm_up_model(self):\n'
                              '        return "compiled"\n')
            finder = gdn_norm_warmup._Finder()
            with patch.object(gdn_norm_warmup, 'TARGET', 'fake_worker_target'), \
                    patch.object(sys, 'path', [root, *sys.path]), \
                    patch.object(sys, 'meta_path', [finder, *sys.meta_path]), \
                    patch.dict(sys.modules):
                sys.modules.pop('fake_worker_target', None)
                module = importlib.import_module('fake_worker_target')
            loader = module.__spec__.loader
            self.assertIsInstance(loader, gdn_norm_warmup._Loader)
            self.assertEqual(loader.path, str(source))
            self.assertTrue(module.Worker.compile_or_warm_up_model._paiton_gdn_warmup_reset)


if __name__ == '__main__':
    unittest.main()
