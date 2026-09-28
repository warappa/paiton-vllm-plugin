"""Zero the CUDA-graph pad rows of the target's full-attention output on padded steps.

A PIECEWISE step whose token count is not a capture size replays the graph of the next
capture size. The attention op between the graph pieces runs eagerly and the R4D backend
writes only the rows of the scheduled requests, [0, num_actual_tokens). The rows behind
them belong to an output buffer that vLLM allocates with torch.empty in the shared graph
pool, so they carry whatever an earlier graph left there. Through o_proj that data reaches
the pad rows of every later GDN layer, where paiton_vllm_plugin.runtime_native_gdn_norm
checks all rows of its M 8/16/24/32 launches and stops the engine on a non-finite value.

After each R4D attention call this hook clears rows [num_actual_tokens:] of the output
with hipMemsetAsync on the current stream, the stream the backend launched on. Real rows
are not touched and every op on their path is row-independent, so served tokens do not
change. Steps without pad rows (uniform FULL-graph replays, eager steps) are not affected.

The hook wraps each R4D impl's forward as it is at the end of
Worker.compile_or_warm_up_model: every load-time adapter (KV4, prefill attention) has
patched the class by then and stays in the call chain, and the FULL graphs were captured
without the wrapper. Warmup dummy rows are gdn_norm_warmup's concern. The file stays
outside runtime_native_gdn_norm, whose source the GDN zero-copy consumer check pins.

Default on; PAITON_ATTN_PAD_ZERO=0 turns it off. Loaded by zzzzzz_paiton_attn_pad_rows.pth.
"""
import ctypes
import importlib.abc
import os
import sys

TARGET = 'vllm.v1.worker.gpu_worker'
ENV = 'PAITON_ATTN_PAD_ZERO'
MARK = '_paiton_attn_pad_rows'
STATS = {'calls': 0, 'rows': 0}
_MEMSET = None


def enabled():
    value = os.environ.get(ENV, '1')
    if value not in ('0', '1'):
        raise RuntimeError(f'{ENV} must be 0 or 1, not {value!r}')
    return value == '1'


def _log(message):
    print(f'[paiton.attn_pad_rows] {message}', flush=True)


def hip_runtime_path(maps):
    """The HIP runtime file already mapped into this process (the one torch launches with)."""
    paths = set()
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and os.path.basename(fields[5]).startswith('libamdhip64.so'):
            paths.add(fields[5])
    if len(paths) != 1:
        raise RuntimeError(f'expected one mapped libamdhip64 runtime, found {sorted(paths)}')
    return paths.pop()


def memset_async():
    """hipMemsetAsync of the loaded HIP runtime; never loads a second runtime."""
    global _MEMSET
    if _MEMSET is None:
        with open('/proc/self/maps') as maps:
            path = hip_runtime_path(maps.read())
        runtime = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
        function = runtime.hipMemsetAsync
        function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
        function.restype = ctypes.c_int
        _MEMSET = function
    return _MEMSET


def current_stream():
    import torch  # Existing external stream boundary only.
    return torch.cuda.current_stream().cuda_stream


def zero_pad_rows(output, num_actual_tokens):
    rows = output.shape[0]
    if num_actual_tokens >= rows:
        return 0
    if not output.is_contiguous():
        raise RuntimeError('attn_pad_rows: attention output is not contiguous')
    row_bytes = output.stride(0) * output.element_size()
    pad = rows - num_actual_tokens
    status = memset_async()(output.data_ptr() + num_actual_tokens * row_bytes, 0, pad * row_bytes,
                            current_stream())
    if status != 0:
        raise RuntimeError(f'attn_pad_rows: hipMemsetAsync failed ({status})')
    STATS['calls'] += 1
    STATS['rows'] += pad
    calls = STATS['calls']
    if calls == 1 or (calls >= 1024 and calls & (calls - 1) == 0):
        _log(f'zeroed calls={calls} rows={STATS["rows"]}')
    return pad


def wrap(impl):
    """Wrap impl.forward as it is now; False if this impl is already wrapped."""
    if getattr(impl, MARK, False):
        return False
    inner = impl.forward

    def forward(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs):
        result = inner(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)
        if attn_metadata is not None:  # None: profile run, the backend fills the whole output itself
            output = kwargs['output'] if 'output' in kwargs else (args[0] if args else None)
            if output is not None:
                count = getattr(attn_metadata, 'num_actual_tokens', None)
                if type(count) is not int:
                    raise RuntimeError('attn_pad_rows: attention metadata has no int num_actual_tokens')
                zero_pad_rows(output, count)
        return result

    forward.__wrapped__ = inner
    impl.forward = forward
    setattr(impl, MARK, True)
    return True


def install(model):
    """Wrap every R4D attention impl of the target model; returns the number newly wrapped."""
    if not enabled():
        _log(f'off ({ENV}=0)')
        return 0
    try:
        from radiance_r4d_attn import R4DAttentionImpl
    except ImportError:
        _log('no R4D attention backend; nothing wrapped')
        return 0
    impls = {}
    for module in model.modules():
        impl = getattr(module, 'impl', None)
        if isinstance(impl, R4DAttentionImpl):
            impls.setdefault(id(impl), impl)
    if impls:
        memset_async()  # bind the runtime now, not on the first padded step
    wrapped = sum(wrap(impl) for impl in impls.values())
    _log(f'wrapped={wrapped} r4d_layers={len(impls)}; rows [num_actual_tokens:] zeroed on padded steps')
    return wrapped


def _model(runner):
    get = getattr(runner, 'get_model', None)
    model = get() if callable(get) else None
    return model if model is not None else getattr(runner, 'model', None)


def patch(module):
    worker = getattr(module, 'Worker', None)
    if worker is None or getattr(worker.compile_or_warm_up_model, MARK, False):
        return
    original = worker.compile_or_warm_up_model

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        model = _model(self.model_runner)
        if model is not None:
            install(model)
        return result

    # No functools.wraps: copying __dict__ would copy other hooks' markers onto this wrapper.
    compile_or_warm_up_model.__name__ = original.__name__
    compile_or_warm_up_model.__qualname__ = getattr(original, '__qualname__', original.__name__)
    compile_or_warm_up_model.__doc__ = original.__doc__
    compile_or_warm_up_model.__wrapped__ = original
    setattr(compile_or_warm_up_model, MARK, True)
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


def install_finder():
    if TARGET in sys.modules:
        patch(sys.modules[TARGET])
    elif not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


# Default on. Only an explicit 0 skips the import hook; any other value is checked at install,
# inside the engine, where a bad value stops startup instead of being dropped by site.py.
if os.environ.get(ENV, '1') != '0':
    install_finder()
