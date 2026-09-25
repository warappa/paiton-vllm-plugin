#!/usr/bin/env python3
"""Configure the pinned ROCm 10 runtime without rebuilding its image."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys


IMAGES = {
    '65k': 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-65k-20260924-r3@sha256:c2511888b76abf463c7f5c51f0c70bf66c593f3c2fca8d910765d85318d399ce',
    '200k': 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-200k-20260918-r2@sha256:32dab97330ea84b86967537d25f91878c30f21ff844f71369508c5a049b89178',
}
SYS_DRM = Path('/sys/class/drm')
SYS_KFD = Path('/sys/class/kfd/kfd/topology/nodes')


def release_command(release):
    """Exact Config.Cmd of each immutable image; Docker overrides replace CMD."""
    context, sequences, cache, capture, mamba = {
        '65k': (65536, 8, 6535819798, [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64], 'align'),
        '200k': (200000, 1, 6979321856, [1, 2, 4, 8], 'none'),
    }[release]
    compilation = {'cudagraph_capture_sizes': capture,
                   'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True}}
    speculative = {'method': 'dflash', 'model': '/models/draft',
                   'num_speculative_tokens': 7, 'draft_tensor_parallel_size': 1,
                   'attention_backend': 'TRITON_ATTN', 'max_model_len': context,
                   'disable_padded_drafter_batch': True, 'draft_sample_method': 'greedy'}
    return [
        'serve', '/models/target', '--tokenizer', '/models/target',
        '--served-model-name', 'Qwen3.8', '--host', '127.0.0.1', '--port', '18982',
        '--tensor-parallel-size', '1', '--dtype', 'bfloat16',
        '--max-model-len', str(context), '--max-num-seqs', str(sequences),
        #'--max-num-batched-tokens', '4096',
        '--max-num-batched-tokens', '8192',
         '--kv-cache-dtype', 'fp8',
         #'--backend', 'triton',
        '--language-model-only',
        '--kv-cache-memory-bytes', str(cache), '--gpu-memory-utilization', '0.98',
        '--no-enable-prefix-caching', '--enable-chunked-prefill',
        '--safetensors-load-strategy', 'lazy', '--attention-backend', 'R4D',
        '--compilation-config', json.dumps(compilation),
        '--speculative-config', json.dumps(speculative), '--mamba-cache-mode', mamba,
        '--mamba-cache-dtype', 'bfloat16', '--mamba-ssm-cache-dtype', 'float16',
        '--no-async-scheduling', '--enable-auto-tool-choice',
        '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3',
        '--override-generation-config', json.dumps({'temperature': 0.7, 'top_p': 0.95, 'top_k': 20}),
        '--seed', '42',
    ]


def positive_integer(value):
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a positive integer') from None
    if result <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def utilization(value):
    try:
        result = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a number greater than 0 and less than 1') from None
    if not math.isfinite(result) or not 0 < result < 1:
        raise argparse.ArgumentTypeError('must be greater than 0 and less than 1')
    return result


def cache_bytes(value):
    return 'auto' if value == 'auto' else positive_integer(value)


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, epilog=(
        'No overrides preserves the selected release settings. --profile desktop reserves '
        'more VRAM for other applications; available VRAM still determines whether startup succeeds. '
        '--profile chat is the measured long-context APC configuration for a dedicated 32 GiB '
        'R9700; it does not establish model quality for every workload. '
        'Context includes prompt and generated tokens. Requests should set '
        'chat_template_kwargs.enable_thinking=false to match the reported benchmarks.'))
    result.add_argument('--release', choices=IMAGES, default='65k', help=argparse.SUPPRESS)
    result.add_argument('--image', help='compatible runtime image override; preserves the selected release settings')
    result.add_argument('--profile', choices=('release', 'desktop', 'chat'), default='release',
                        help='chat: 200000 context, APC on, thinking off, 8 GiB KV; '
                             'desktop: 32768 context, 2 GiB KV; both use one request and 1024 prefill chunks')
    result.add_argument('--list-gpus', action='store_true', help='list physical render devices without starting Docker')
    result.add_argument('--context', type=positive_integer, metavar='TOKENS', help='set both target and draft context limits')
    result.add_argument('--max-num-seqs', type=positive_integer, metavar='COUNT', help='maximum concurrent requests (1 to 8)')
    result.add_argument('--gpu-memory-utilization', type=utilization, metavar='FRACTION',
                        help='automatic memory budget; implies automatic KV sizing unless explicit bytes are supplied')
    result.add_argument('--kv-cache-memory-bytes', type=cache_bytes, metavar='BYTES|auto',
                        help='fixed KV budget in bytes, or automatic sizing from GPU memory utilization')
    result.add_argument('--prefix-caching', choices=('on', 'off'),
                        help='experimental prefix reuse with materialized recurrent state; off in both releases')
    result.add_argument('--thinking', choices=('on', 'off'),
                        help='server default for enable_thinking; individual requests may override it')
    result.add_argument('--max-num-batched-tokens', type=positive_integer, metavar='TOKENS')
    result.add_argument('--port', type=positive_integer, help='localhost API port (default: 18982)')
    result.add_argument('--name', help='Docker container name')
    result.add_argument('--detach', action='store_true', help='run Docker in the background')
    result.add_argument('--dry-run', action='store_true', help='print Docker argv as JSON; do not pull or start the image')
    return result


def read_text(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ''


def read_number(path):
    try:
        return int(read_text(path), 0)
    except ValueError:
        return 0


def discover_gpus():
    """Match DRM and KFD by render minor, without initializing a GPU runtime."""
    architectures = {}
    for path in SYS_KFD.glob('*/properties'):
        properties = dict(line.split(maxsplit=1) for line in read_text(path).splitlines()
                          if len(line.split(maxsplit=1)) == 2)
        try:
            minor = int(properties.get('drm_render_minor', 0))
            architecture = int(properties.get('gfx_target_version', 0))
        except ValueError:
            continue
        architectures[minor] = architecture
    devices = []
    for path in sorted(SYS_DRM.glob('renderD*')):
        if not re.fullmatch(r'renderD\d+', path.name):
            continue
        device = path / 'device'
        vendor, identifier = read_number(device / 'vendor'), read_number(device / 'device')
        properties = dict(line.split('=', 1) for line in read_text(device / 'uevent').splitlines() if '=' in line)
        architecture = architectures.get(int(path.name[7:]), 0)
        vram = read_number(device / 'mem_info_vram_total')
        # The qualified card is a 32 GiB gfx1201 device. The same architecture's
        # smaller consumer cards cannot hold this release's model and cache.
        supported = vendor == 0x1002 and architecture == 120001 and vram >= 30 * 1024**3
        devices.append({'path': '/dev/dri/' + path.name,
                        'pci': properties.get('PCI_SLOT_NAME', device.resolve().name),
                        'vendor': vendor, 'device': identifier, 'vram': vram,
                        'gfx': architecture, 'supported': supported})
    return devices


def describe_gpu(gpu):
    label = 'R9700 / gfx1201' if gpu['supported'] else {
        0x1002: 'AMD (not supported by this release)',
        0x8086: 'Intel (not supported by this release)',
        0x10de: 'NVIDIA (not supported by this release)',
    }.get(gpu['vendor'], 'not supported by this release')
    vram = f"{gpu['vram'] / 1024**3:.1f} GiB" if gpu['vram'] else 'unknown VRAM'
    return (f"{gpu['path']}  PCI {gpu['pci']}  {gpu['vendor']:04x}:{gpu['device']:04x}  "
            f"{vram}  {label}")


def replace_value(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def prefix_caching_enabled(args):
    return args.prefix_caching == 'on' or (args.prefix_caching is None and args.profile == 'chat')


def engine_command(args):
    if args.context is not None and args.context > 262144:
        raise ValueError('--context exceeds this checkpoint\'s 262144-token model limit')
    if args.max_num_seqs is not None and args.max_num_seqs > 8:
        raise ValueError('--max-num-seqs must be between 1 and 8 for this release')
    if args.port is not None and args.port > 65535:
        raise ValueError('--port must be between 1 and 65535')
    command = release_command(args.release)
    desktop = args.profile == 'desktop'
    chat = args.profile == 'chat'
    compact_graphs = desktop or chat
    default_context = 200000 if chat else (32768 if desktop else None)
    context = args.context if args.context is not None else default_context
    sequences = args.max_num_seqs if args.max_num_seqs is not None else (1 if compact_graphs else None)
    batched_tokens = args.max_num_batched_tokens if args.max_num_batched_tokens is not None else (1024 if compact_graphs else None)
    default_budget = 0.98 if chat else (0.90 if desktop else None)
    budget = args.gpu_memory_utilization if args.gpu_memory_utilization is not None else default_budget
    cache = args.kv_cache_memory_bytes
    if cache is None:
        if args.gpu_memory_utilization is not None:
            cache = 'auto'
        elif chat:
            cache = 8 * 1024**3
        elif desktop:
            cache = 2 * 1024**3
    for flag, value in (('--max-model-len', context), ('--max-num-seqs', sequences),
                        ('--gpu-memory-utilization', budget), ('--port', args.port),
                        ('--max-num-batched-tokens', batched_tokens)):
        if value is not None:
            replace_value(command, flag, value)
    if context is not None:
        index = command.index('--speculative-config') + 1
        speculative = json.loads(command[index])
        speculative['max_model_len'] = context
        command[index] = json.dumps(speculative)
    if compact_graphs:
        index = command.index('--compilation-config') + 1
        compilation = json.loads(command[index])
        compilation['cudagraph_capture_sizes'] = [1, 2, 4, 8]
        command[index] = json.dumps(compilation)
    if cache == 'auto':
        index = command.index('--kv-cache-memory-bytes')
        del command[index:index + 2]
    elif cache is not None:
        replace_value(command, '--kv-cache-memory-bytes', cache)
    if prefix_caching_enabled(args):
        command[command.index('--no-enable-prefix-caching')] = '--enable-prefix-caching'
        replace_value(command, '--mamba-cache-mode', 'align')
    thinking = args.thinking if args.thinking is not None else ('off' if chat else None)
    if thinking is not None:
        command += ['--default-chat-template-kwargs',
                    json.dumps({'enable_thinking': thinking == 'on'})]
    if chat:
        command.append('--enable-prompt-tokens-details')
    return command


def model_mounts(environment):
    mounts = []
    for variable, destination, writable in (
        ('PAITON_TARGET_DIR', '/models/target', False),
        ('PAITON_DRAFT_DIR', '/models/draft', False),
        ('PAITON_CACHE_DIR', '/cache', True),
    ):
        setting = environment.get(variable)
        if not setting:
            raise ValueError(f'Set {variable} to an existing {"writable cache" if writable else "checkpoint"} directory')
        path = Path(setting).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f'{variable} is not an existing directory: {path}')
        if ':' in str(path) or '\n' in str(path):
            raise ValueError(f'{variable} must not contain colons or newlines (Docker volume syntax)')
        if not os.access(path, os.R_OK | os.X_OK | (os.W_OK if writable else 0)):
            raise ValueError(f'{variable} is not {"writable" if writable else "readable"}: {path}')
        mounts += ['-v', f'{path}:{destination}:{"rw" if writable else "ro"}']
    return mounts


def docker_command(args, environment):
    name = args.name or f'paiton-qwen38-{args.release}'
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name):
        raise ValueError('--name must be a valid Docker container name')
    image = args.image or IMAGES[args.release]
    if image.startswith('-') or any(c.isspace() for c in image):
        raise ValueError('--image must be a Docker image reference')
    engine = engine_command(args)
    command = ['docker', 'run', '--rm', '--name', name, '--network', 'host',
               '--device', '/dev/kfd', '--device', '/dev/dri',
               '--group-add', 'video', '--ipc', 'host']
    if not args.detach and sys.stdin.isatty() and sys.stdout.isatty():
        command.append('-it')
    for variable in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        # A bare name removes an image default when absent from Docker's host
        # environment. Explicit values, including empty strings, stay unchanged.
        setting = variable + '=' + environment[variable] if variable in environment else variable
        command += ['-e', setting]
    for variable in ('PAITON_NGRAM_CODRAFT', 'PAITON_NGRAM_CODRAFT_HOT_MATCH'):
        # Opt-in n-gram co-drafting. Forwarded only when set on the host, so the
        # image default (off) applies otherwise.
        if variable in environment:
            command += ['-e', variable + '=' + environment[variable]]
    if prefix_caching_enabled(args):
        command += ['-e', 'RADIANCE_GDN_LAZY=0']
    if args.profile == 'chat':
        command += ['-e', 'PYTORCH_ALLOC_CONF=max_split_size_mb:64']
    if args.detach:
        command.append('--detach')
    return command + model_mounts(environment) + [image] + engine


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.list_gpus:
        devices = discover_gpus()
        print('\n'.join(describe_gpu(gpu) for gpu in devices) or 'No DRM render devices found.')
        return 0
    try:
        command = docker_command(args, os.environ)
    except ValueError as error:
        arguments.error(str(error))
    print('Exposing /dev/dri; GPU selection follows your visibility environment and runtime.', file=sys.stderr)
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    try:
        os.execvp(command[0], command)
    except FileNotFoundError:
        arguments.error('Docker is not installed or is not on PATH')


if __name__ == '__main__':
    sys.exit(main())
