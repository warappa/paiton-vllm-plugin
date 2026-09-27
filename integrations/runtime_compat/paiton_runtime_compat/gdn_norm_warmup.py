"""Clear the native GDN norm status once, after vLLM's synthetic warmup.

paiton_vllm_plugin.runtime_native_gdn_norm binds one int32 status buffer to the
GDN layers. Its small-row norm sets the status on any nonfinite input row, and
the output guard fails every later step while it is set. vLLM's profile,
compile and CUDA-graph warmup forwards read dummy rows that can come from
uninitialized buffers and own no user request history. This hook reports the
status they leave and clears it in place at the end of
Worker.compile_or_warm_up_model, before the first request; captured graphs keep
writing the same buffer and served steps still fail on any nonfinite row.

It stays outside runtime_native_gdn_norm, whose reviewed source is pinned by
the GDN zero-copy consumer check. Loaded by zzzzzz_paiton_gdn_norm_warmup.pth;
opt-in with PAITON_GDN_NORM_WARMUP_RESET=1 (set by the W3A4 image).
"""
import importlib.abc
import os
import sys

TARGET = 'vllm.v1.worker.gpu_worker'
BUFFER = '_paiton_gdn_norm_error'


def status_buffers(model):
    """The distinct status tensors bound to the model's GDN layers."""
    seen = {}
    for module in model.modules():
        buffer = getattr(module, BUFFER, None)
        if buffer is not None and hasattr(buffer, 'data_ptr'):
            seen.setdefault(buffer.data_ptr(), buffer)
    return list(seen.values())


def clear(model):
    import torch
    statuses = []
    for buffer in status_buffers(model):
        if buffer.is_cuda:
            torch.cuda.synchronize(buffer.device)
        statuses.append(int(buffer.reshape(-1)[0].item()))
        buffer.zero_()
        if buffer.is_cuda:
            torch.cuda.synchronize(buffer.device)
    return statuses


def _model(runner):
    get = getattr(runner, 'get_model', None)
    model = get() if callable(get) else None
    return model if model is not None else getattr(runner, 'model', None)


def patch(module):
    worker = getattr(module, 'Worker', None)
    if worker is None or getattr(worker.compile_or_warm_up_model, '_paiton_gdn_warmup_reset', False):
        return
    original = worker.compile_or_warm_up_model

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        model = _model(self.model_runner)
        statuses = clear(model) if model is not None else []
        print(f'[paiton.gdn_norm_warmup] buffers={len(statuses)} warmup_status={statuses} cleared', flush=True)
        return result

    compile_or_warm_up_model._paiton_gdn_warmup_reset = True
    compile_or_warm_up_model.__wrapped__ = original
    worker.compile_or_warm_up_model = compile_or_warm_up_model


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        # Source pinning reads loader.path; every other attribute delegates too.
        return getattr(self.original, name)

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        after = False
        for finder in list(sys.meta_path):
            if finder is self:
                after = True
                continue
            find = getattr(finder, 'find_spec', None) if after else None
            spec = find(fullname, path, target) if find else None
            if spec is not None:
                if spec.loader:
                    spec.loader = _Loader(spec.loader)
                return spec
        return None


def install():
    if TARGET in sys.modules:
        patch(sys.modules[TARGET])
    elif not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


if os.environ.get('PAITON_GDN_NORM_WARMUP_RESET', '0') == '1':
    install()
