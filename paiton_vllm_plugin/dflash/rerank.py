"""Draft-instance-only proposal pool; preserve original target head and globals.

The installed head function's code object is reused unchanged. Its RERANK and
optional KCAND lookups use a private globals dictionary, and only explicitly
identified DFlash2 candidate-logits processors receive that function. This is
external runtime configuration, not new kernel arithmetic or a compiler feature.
"""
import functools
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import types

FLAG='PAITON_DFLASH_RERANK'
BLOCK_FLAG='PAITON_DFLASH_BLOCK_CANDIDATES'
TARGET='radiance_drafthead'
SOURCE_SHA='46a56931148a13b1cb5e8c8816f0309821eab3c5e5df93b712a7384552bf36d2'
DRAFT_CLASS=('vllm.model_executor.models.qwen3_dflash2','DFlash2Qwen3ForCausalLM')


def configuration(env):
    value=env.get(FLAG,'0')
    if value in ('','0'):return None
    if value not in ('128','256'):raise ValueError(FLAG+' supports only 0, 128 or 256 in this scoped round')
    if env.get('RADIANCE_DRAFT_RERANK')!='80' or env.get('RADIANCE_FAST_DRAFT')!='1':
        raise ValueError('Requires originalR80 fast-head release configuration')
    return int(value)


def block_configuration(env,pool):
    value=env.get(BLOCK_FLAG,'0')
    if value not in ('0','16'):raise ValueError(BLOCK_FLAG+' supports only 0 or 16')
    if value=='16' and (pool!=128 or env.get('PAITON_DFLASH_DRAFT_SAMPLE_METHOD')!='probabilistic'):
        raise ValueError('Sixteen block candidates require scoped R128 probabilistic proposals')
    return 8 if value=='0' else 16


def bind(module,pool=128,block_candidates=8):
    if pool not in (128,256):raise ValueError('Only predeclared R128 and R256 draft pools are supported')
    if block_candidates not in (8,16) or (block_candidates==16 and pool!=128):
        raise ValueError('Block-candidate experiment is scoped to R128 only')
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ImportError('Unreviewed draft head source')
    if getattr(module,'_paiton_rerank_installed',False):raise RuntimeError('Duplicate draft pool override')
    if module.RERANK!=80 or not module.FAST or module.KCAND!=8 or module.BITS!=2 or module._HEAD_CACHE:
        raise ValueError('Install before weight binding under the unchanged released policy')
    original_head=module._apply_head_int2
    # All other values, including helper functions/JIT objects, are the original
    # objects. Never mutate shared globals: the target verifier uses them too.
    scope=dict(module.__dict__,RERANK=pool,KCAND=block_candidates)
    head=types.FunctionType(original_head.__code__,scope,original_head.__name__,
                            original_head.__defaults__,original_head.__closure__)
    head.__kwdefaults__=original_head.__kwdefaults__
    head.__module__=original_head.__module__
    head.__qualname__=original_head.__qualname__
    quantize=module._quantize_head_now
    identify=module._quantize_draft_head

    @functools.wraps(original_head)
    def checked(self,lm_head,hidden_states,embedding_bias=None):
        if module.RERANK!=80 or module.KCAND!=8 or module.BITS!=2 or not module.FAST or module._apply_head_int2 is not original_head:
            raise RuntimeError('Shared target-head policy changed')
        if getattr(self,'_paiton_draft_rerank',None)!=pool or getattr(self,'_paiton_draft_block_candidates',None)!=block_candidates or not getattr(self,'_radiance_topk_only',False):
            raise RuntimeError('Scoped rerank called on an unmarked logits processor')
        if getattr(self,'_radiance_n',None)!=248320 or hidden_states.shape[-1]!=5120:
            raise RuntimeError('Unexpected Qwen draft head shape')
        if not getattr(self,'_paiton_scoped_rerank_reported',False):
            print('[PAITON_DFLASH_RERANK_EXECUTED] '+json.dumps({'draft_rerank':pool,'target_rerank':module.RERANK,
                'draft_block_candidates':block_candidates,'target_block_candidates':module.KCAND,
                'scope':'explicit DFlash2 candidate_logits_processor','native_kernel_gain_claim':False}),flush=True)
            self._paiton_scoped_rerank_reported=True
        return head(self,lm_head,hidden_states,embedding_bias)

    @functools.wraps(quantize)
    def quantize_scoped(lp,lm_head):
        result=quantize(lp,lm_head)
        if getattr(lp,'_paiton_draft_rerank',None)==pool and hasattr(lp,'_radiance_wq'):
            if not getattr(lp,'_radiance_topk_only',False) or getattr(lp,'_radiance_n',None)!=248320:
                raise RuntimeError('Unexpected marked draft head after quantization')
            lp._apply_head=types.MethodType(checked,lp)
        return result

    @functools.wraps(identify)
    def identify_scoped(mtp,lp_attr='logits_processor'):
        cls=type(mtp)
        if (cls.__module__,cls.__name__)==DRAFT_CLASS and lp_attr=='candidate_logits_processor':
            lp=getattr(mtp,lp_attr,None)
            if lp is None or lp is getattr(mtp,'logits_processor',None):
                raise RuntimeError('Require a separate DFlash candidate logits processor')
            if hasattr(lp,'_paiton_draft_rerank'):raise RuntimeError('Draft processor already marked')
            lp._paiton_draft_rerank=pool
            lp._paiton_draft_block_candidates=block_candidates
        return identify(mtp,lp_attr)

    module._quantize_head_now=quantize_scoped
    module._quantize_draft_head=identify_scoped
    module._paiton_rerank_installed=True
    print('[PAITON_DFLASH_RERANK] '+json.dumps({'draft_rerank':pool,'target_rerank':80,
        'draft_block_candidates':block_candidates,'target_block_candidates':8,
        'target_global_unchanged':True,'code_object_reused':True,'policy_version':'draft-scoped-r5'}),flush=True)


class Loader:
    def __init__(self,original,pool,block_candidates=8):self.original,self.pool,self.block_candidates=original,pool,block_candidates
    def __getattr__(self,name):return getattr(self.original,name)
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):self.original.exec_module(module);bind(module,self.pool,self.block_candidates)
class Finder(importlib.abc.MetaPathFinder):
    def __init__(self,pool,block_candidates=8):self.pool,self.block_candidates=pool,block_candidates
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=TARGET:return None
        after=False
        for finder in list(sys.meta_path):
            if finder is self:after=True;continue
            if not after:continue
            spec=finder.find_spec(fullname,path,target)
            if spec is not None:
                if not spec.loader:raise ImportError('Missing draft head loader')
                spec.loader=Loader(spec.loader,self.pool,self.block_candidates);return spec
        raise ImportError('Missing released draft head')

def install():
    pool=configuration(os.environ)
    block_candidates=block_configuration(os.environ,pool)
    if pool is not None:
        if TARGET in sys.modules:raise RuntimeError('Install draft pool override before imports')
        sys.meta_path.insert(0,Finder(pool,block_candidates))
