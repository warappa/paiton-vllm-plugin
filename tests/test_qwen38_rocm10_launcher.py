"""Exercise device isolation and Docker argument boundaries without a GPU runtime."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


MODEL_DIR = Path(__file__).resolve().parents[1] / 'models/Qwen3.8-MXFP4-DFlash2'
SCRIPT = MODEL_DIR / 'launch-rocm10.py'
SPEC = importlib.util.spec_from_file_location('rocm10_launcher', SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def value(command, flag):
    return command[command.index(flag) + 1]


class Rocm10LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.drm = self.root / 'drm'
        self.kfd = self.root / 'kfd'
        self.drm.mkdir()
        self.kfd.mkdir()
        self.record = self.root / 'docker-argv.json'
        binary = self.root / 'bin'
        binary.mkdir()
        # A real exec into a harmless Docker stand-in catches quoting, argument
        # positioning and accidental host-shell interpretation at the boundary.
        docker = binary / 'docker'
        docker.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                          'open(os.environ["DOCKER_ARGV_RECORD"],"w").write(json.dumps(sys.argv[1:]))\n')
        docker.chmod(0o755)
        self.environment = dict(os.environ)
        for name in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
            self.environment.pop(name, None)
        self.environment['PATH'] = str(binary) + os.pathsep + os.environ.get('PATH', '')
        self.environment['DOCKER_ARGV_RECORD'] = str(self.record)
        for name in ('TARGET', 'DRAFT', 'CACHE'):
            directory = self.root / (name.lower() + ' directory $(echo not-a-shell)')
            directory.mkdir()
            self.environment[f'PAITON_{name}_DIR'] = str(directory)
        self.device(128, 0x1002, 0x7551, 120001, 32 * 1024**3)
        self.device(129, 0x8086, 0x3e92, 0, 0)

    def device(self, minor, vendor, identifier, architecture, vram):
        path = self.drm / f'renderD{minor}' / 'device'
        path.mkdir(parents=True)
        for name, setting in (('vendor', hex(vendor)), ('device', hex(identifier)),
                              ('mem_info_vram_total', str(vram)),
                              ('uevent', f'PCI_SLOT_NAME=0000:{minor - 128:02x}:00.0')):
            (path / name).write_text(setting)
        node = self.kfd / str(minor)
        node.mkdir()
        (node / 'properties').write_text(f'drm_render_minor {minor}\ngfx_target_version {architecture}\n')

    def run_launcher(self, *args):
        harness = ('import importlib.util, pathlib, sys\n'
                   f's = importlib.util.spec_from_file_location("launch", {str(SCRIPT)!r})\n'
                   'm = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n'
                   f'm.SYS_DRM = pathlib.Path({str(self.drm)!r})\n'
                   f'm.SYS_KFD = pathlib.Path({str(self.kfd)!r})\n'
                   'sys.exit(m.main(sys.argv[1:]))\n')
        return subprocess.run([sys.executable, '-c', harness, *args],
                              env=self.environment, capture_output=True, text=True)

    def command(self, *args):
        result = self.run_launcher(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return ['docker', *json.loads(self.record.read_text())]

    def engine(self, command):
        index = next(i for i, item in enumerate(command) if item.startswith('ghcr.io/'))
        return command[index + 1:]

    def test_default_exec_preserves_release_limits_and_leaves_gpu_choice_to_user(self):
        command = self.command()
        self.assertEqual([command[i + 1] for i, item in enumerate(command) if item == '--device'],
                         ['/dev/kfd', '/dev/dri'])
        self.assertIn('ROCR_VISIBLE_DEVICES', command)
        self.assertIn('HIP_VISIBLE_DEVICES', command)
        self.assertIn('CUDA_VISIBLE_DEVICES', command)
        self.assertNotIn('ROCR_VISIBLE_DEVICES=0', command)
        self.assertNotIn('HIP_VISIBLE_DEVICES=0', command)
        self.assertEqual(value(command, '--group-add'), 'video')
        self.assertEqual(value(command, '--ipc'), 'host')
        self.assertNotIn('--shm-size', command)
        self.assertNotIn('-it', command)
        self.assertIn(launcher.IMAGES['65k'], command)
        engine = self.engine(command)
        self.assertEqual(engine[:2], ['serve', '/models/target'])
        self.assertEqual(value(engine, '--max-model-len'), '65536')
        self.assertEqual(value(engine, '--max-num-seqs'), '8')
        self.assertEqual(value(engine, '--max-num-batched-tokens'), '4096')
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '6535819798')
        self.assertIn('--no-enable-prefix-caching', engine)
        self.assertEqual(value(engine, '--tool-call-parser'), 'qwen3_coder')
        self.assertNotIn('--default-chat-template-kwargs', engine)
        self.assertIn(self.environment['PAITON_CACHE_DIR'] + ':/cache:rw', command)

    def test_200k_context_override_synchronizes_draft_and_preserves_sampling(self):
        command = self.command('--release', '200k', '--context', '220000', '--port', '19000', '--name', 'custom-qwen')
        self.assertIn(launcher.IMAGES['200k'], command)
        self.assertEqual(value(command, '--name'), 'custom-qwen')
        engine = self.engine(command)
        self.assertEqual(value(engine, '--max-model-len'), '220000')
        self.assertEqual(json.loads(value(engine, '--speculative-config'))['max_model_len'], 220000)
        self.assertEqual(value(engine, '--max-num-seqs'), '1')
        self.assertEqual(value(engine, '--port'), '19000')
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '6979321856')
        self.assertEqual(json.loads(value(engine, '--override-generation-config')),
                         {'temperature': 0.7, 'top_p': 0.95, 'top_k': 20})

    def test_desktop_uses_small_fixed_kv_and_explicit_overrides_take_precedence(self):
        engine = self.engine(self.command('--profile', 'desktop'))
        self.assertEqual(value(engine, '--max-model-len'), '32768')
        self.assertEqual(value(engine, '--max-num-seqs'), '1')
        self.assertEqual(value(engine, '--gpu-memory-utilization'), '0.9')
        self.assertEqual(value(engine, '--max-num-batched-tokens'), '1024')
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '2147483648')
        self.assertEqual(json.loads(value(engine, '--compilation-config'))['cudagraph_capture_sizes'], [1, 2, 4, 8])
        engine = self.engine(self.command('--profile', 'desktop', '--context', '16384',
                                          '--max-num-seqs', '2', '--gpu-memory-utilization', '0.85',
                                          '--max-num-batched-tokens', '2048',
                                          '--kv-cache-memory-bytes', '4294967296'))
        self.assertEqual(value(engine, '--max-model-len'), '16384')
        self.assertEqual(value(engine, '--max-num-seqs'), '2')
        self.assertEqual(value(engine, '--gpu-memory-utilization'), '0.85')
        self.assertEqual(value(engine, '--max-num-batched-tokens'), '2048')
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '4294967296')

    def test_desktop_automatic_kv_is_an_explicit_option(self):
        for options, expected_utilization in ((['--kv-cache-memory-bytes', 'auto'], '0.9'),
                                               (['--gpu-memory-utilization', '0.85'], '0.85')):
            with self.subTest(options=options):
                engine = self.engine(self.command('--profile', 'desktop', *options))
                self.assertNotIn('--kv-cache-memory-bytes', engine)
                self.assertEqual(value(engine, '--gpu-memory-utilization'), expected_utilization)
                self.assertEqual(value(engine, '--max-model-len'), '32768')

    def test_chat_defaults_and_220k_override_match_measured_runtime_configuration(self):
        for context in (200000, 220000):
            with self.subTest(context=context):
                options = [] if context == 200000 else ['--context', str(context)]
                command = self.command('--release', '200k', '--profile', 'chat', *options)
                image_index = command.index(launcher.IMAGES['200k'])
                environment = [command[i + 1] for i in range(image_index) if command[i] == '-e']
                self.assertCountEqual(environment, ['ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                                                   'CUDA_VISIBLE_DEVICES',
                                                   'RADIANCE_GDN_LAZY=0',
                                                   'PYTORCH_ALLOC_CONF=max_split_size_mb:64'])
                engine = self.engine(command)
                self.assertEqual(value(engine, '--max-model-len'), str(context))
                self.assertEqual(value(engine, '--max-num-seqs'), '1')
                self.assertEqual(value(engine, '--max-num-batched-tokens'), '1024')
                self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '8589934592')
                self.assertEqual(value(engine, '--gpu-memory-utilization'), '0.98')
                self.assertEqual(value(engine, '--mamba-cache-mode'), 'align')
                self.assertIn('--enable-prefix-caching', engine)
                self.assertNotIn('--no-enable-prefix-caching', engine)
                self.assertEqual(json.loads(value(engine, '--default-chat-template-kwargs')),
                                 {'enable_thinking': False})
                self.assertEqual(engine.count('--default-chat-template-kwargs'), 1)
                self.assertIn('--enable-prompt-tokens-details', engine)
                self.assertEqual(json.loads(value(engine, '--compilation-config')),
                                 {'cudagraph_capture_sizes': [1, 2, 4, 8],
                                  'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True}})
                self.assertEqual(json.loads(value(engine, '--speculative-config')),
                                 {'method': 'dflash', 'model': '/models/draft',
                                  'num_speculative_tokens': 7, 'draft_tensor_parallel_size': 1,
                                  'attention_backend': 'TRITON_ATTN', 'max_model_len': context,
                                  'disable_padded_drafter_batch': True, 'draft_sample_method': 'greedy'})
                self.assertEqual(value(engine, '--tool-call-parser'), 'qwen3_coder')
                self.assertEqual(json.loads(value(engine, '--override-generation-config')),
                                 {'temperature': 0.7, 'top_p': 0.95, 'top_k': 20})

    def test_explicit_flags_override_chat_defaults(self):
        command = self.command('--release', '200k', '--profile', 'chat',
                               '--context', '65536', '--max-num-seqs', '2',
                               '--max-num-batched-tokens', '2048', '--kv-cache-memory-bytes', '4294967296',
                               '--gpu-memory-utilization', '0.95', '--prefix-caching', 'off', '--thinking', 'on')
        engine = self.engine(command)
        self.assertEqual(value(engine, '--max-model-len'), '65536')
        self.assertEqual(value(engine, '--max-num-seqs'), '2')
        self.assertEqual(value(engine, '--max-num-batched-tokens'), '2048')
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '4294967296')
        self.assertEqual(value(engine, '--gpu-memory-utilization'), '0.95')
        self.assertIn('--no-enable-prefix-caching', engine)
        self.assertNotIn('--enable-prefix-caching', engine)
        self.assertNotIn('RADIANCE_GDN_LAZY=0', command)
        self.assertEqual(json.loads(value(engine, '--default-chat-template-kwargs')), {'enable_thinking': True})
        self.assertEqual(engine.count('--default-chat-template-kwargs'), 1)

    def test_chat_allocator_setting_does_not_leak_into_other_profiles(self):
        for profile in ('release', 'desktop'):
            command = self.command('--profile', profile)
            self.assertNotIn('PYTORCH_ALLOC_CONF=max_split_size_mb:64', command)
            self.assertNotIn('--enable-prompt-tokens-details', command)

    def test_memory_fraction_override_is_effective_without_desktop_preset(self):
        engine = self.engine(self.command('--gpu-memory-utilization', '0.85'))
        self.assertNotIn('--kv-cache-memory-bytes', engine)
        self.assertEqual(value(engine, '--gpu-memory-utilization'), '0.85')
        self.assertEqual(value(engine, '--max-model-len'), '65536')

    def test_prefix_caching_materializes_recurrent_state_before_vllm_entrypoint(self):
        command = self.command('--release', '200k', '--prefix-caching', 'on')
        image_index = command.index(launcher.IMAGES['200k'])
        self.assertIn('RADIANCE_GDN_LAZY=0', command[:image_index])
        engine = self.engine(command)
        self.assertIn('--enable-prefix-caching', engine)
        self.assertNotIn('--no-enable-prefix-caching', engine)
        self.assertEqual(value(engine, '--mamba-cache-mode'), 'align')

    def test_explicit_thinking_sets_server_default_without_changing_sampling_or_limits(self):
        for release in ('65k', '200k'):
            baseline = self.engine(self.command('--release', release))
            self.assertNotIn('--default-chat-template-kwargs', baseline)
            for mode, expected in (('on', True), ('off', False)):
                with self.subTest(release=release, thinking=mode):
                    command = self.command('--release', release, '--thinking', mode)
                    engine = self.engine(command)
                    self.assertEqual(engine[:-2], baseline)
                    self.assertEqual(engine[-2], '--default-chat-template-kwargs')
                    self.assertEqual(json.loads(engine[-1]), {'enable_thinking': expected})
                    self.assertEqual(engine.count('--default-chat-template-kwargs'), 1)

    def test_two_compatible_cards_are_allowed_without_automatic_selection(self):
        self.device(130, 0x1002, 0x7551, 120001, 32 * 1024**3)
        command = self.command()
        self.assertIn('/dev/dri', command)
        self.assertNotIn('/dev/dri/renderD130', command)
        self.assertNotIn('/dev/dri/renderD128', command)
        self.assertNotIn('/dev/dri/renderD129', command)
        self.assertIn('ROCR_VISIBLE_DEVICES', command)
        self.assertNotIn('ROCR_VISIBLE_DEVICES=0', command)

    def test_mixed_amd_generations_leave_selection_to_user_by_default(self):
        self.device(130, 0x1002, 0x73bf, 100300, 16 * 1024**3)
        command = self.command()
        self.assertIn('/dev/dri', command)
        self.assertNotIn('/dev/dri/renderD128', command)
        self.assertNotIn('/dev/dri/renderD130', command)

    def test_launch_does_not_require_kfd_architecture_metadata(self):
        properties = self.kfd / '128' / 'properties'
        for contents in ('drm_render_minor 128\n',
                         'drm_render_minor invalid\ngfx_target_version 120001\n',
                         'drm_render_minor 128\ngfx_target_version invalid\n', ''):
            with self.subTest(contents=contents):
                properties.write_text(contents)
                self.assertIn('/dev/dri', self.command())

    def test_list_and_dry_run_do_not_invoke_docker(self):
        result = self.run_launcher('--list-gpus')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('R9700 / gfx1201', result.stdout)
        self.assertIn('Intel (not supported', result.stdout)
        self.assertFalse(self.record.exists())
        result = self.run_launcher('--dry-run', '--context', '8192')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(value(json.loads(result.stdout), '--max-model-len'), '8192')
        self.assertFalse(self.record.exists())

    def test_inherited_host_masks_are_forwarded_exactly_including_empty_values(self):
        for name in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
            for setting in ('1', '0', '', 'GPU-1234567890abcdef', '1,0'):
                with self.subTest(name=name, setting=setting):
                    self.environment[name] = setting
                    command = self.command()
                    self.assertIn(name + '=' + setting, command)
                    self.assertNotIn(name, command)
                    del self.environment[name]

    def test_second_amd_render_device_preserves_user_rocr_and_hip_masks_without_uuid(self):
        # Replace the Intel fixture with another AMD GPU; KFD properties contain
        # no UUID, and their node names need not match GPU ordinals.
        path = self.drm / 'renderD129' / 'device'
        (path / 'vendor').write_text('0x1002')
        (path / 'device').write_text('0x7551')
        (path / 'mem_info_vram_total').write_text(str(32 * 1024**3))
        (self.kfd / '129' / 'properties').write_text('drm_render_minor 129\ngfx_target_version 120001\n')
        (self.kfd / '129').rename(self.kfd / '7')
        self.environment.update(ROCR_VISIBLE_DEVICES='1', HIP_VISIBLE_DEVICES='0')
        command = self.command()
        self.assertEqual([command[i + 1] for i, item in enumerate(command) if item == '--device'],
                         ['/dev/kfd', '/dev/dri'])
        self.assertIn('ROCR_VISIBLE_DEVICES=1', command)
        self.assertIn('HIP_VISIBLE_DEVICES=0', command)
        self.assertIn('CUDA_VISIBLE_DEVICES', command)
        self.assertFalse(any('GPU-' in item for item in command))

    def test_interactive_docker_flags_require_foreground_stdin_and_stdout_ttys(self):
        for detached in (False, True):
            for stdin_tty, stdout_tty in ((False, False), (False, True), (True, False), (True, True)):
                with self.subTest(detached=detached, stdin_tty=stdin_tty, stdout_tty=stdout_tty):
                    args = launcher.parser().parse_args(['--detach'] if detached else [])
                    with patch.object(launcher.sys.stdin, 'isatty', return_value=stdin_tty), \
                         patch.object(launcher.sys.stdout, 'isatty', return_value=stdout_tty):
                        command = launcher.docker_command(args, self.environment)
                    self.assertEqual('-it' in command, not detached and stdin_tty and stdout_tty)
                    self.assertEqual('--detach' in command, detached)

    def test_bad_inputs_fail_before_any_docker_execution(self):
        cases = [('--context', '0'), ('--context', '262145'), ('--max-num-seqs', '9'),
                 ('--gpu-memory-utilization', 'nan'), ('--gpu-memory-utilization', '1'),
                 ('--gpu-memory-utilization', '-0.1'), ('--kv-cache-memory-bytes', '4GiB'),
                 ('--kv-cache-memory-bytes', '-1'), ('--port', '65536'),
                 ('--name', 'a b'), ('--gpu', '0.9'), ('--thinking', 'false'), ('--unknown',)]
        for case in cases:
            with self.subTest(case=case):
                result = self.run_launcher(*case)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(self.record.exists())

    def test_missing_mount_fails_before_docker_and_help_needs_no_mounts(self):
        self.environment['PAITON_TARGET_DIR'] = str(self.root / 'missing')
        result = self.run_launcher()
        self.assertEqual(result.returncode, 2)
        self.assertIn('not an existing directory', result.stderr)
        self.assertFalse(self.record.exists())
        for release in ('65k', '200k'):
            result = subprocess.run(['bash', str(MODEL_DIR / f'run-rocm10-{release}.sh'), '--help'],
                                    env=self.environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--profile', result.stdout)
            self.assertIn('--list-gpus', result.stdout)
            self.assertNotIn('--gpu ', result.stdout)

    def test_bash_wrappers_forward_arguments_without_shell_reinterpretation(self):
        python = self.root / 'bin' / 'python3'
        python.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                          'open(os.environ["DOCKER_ARGV_RECORD"],"w").write(json.dumps(sys.argv[1:]))\n')
        python.chmod(0o755)
        options = ['--profile', 'desktop', '--context', '16384', '--name', 'literal $value']
        # Both wrappers start the current image; the 200K one selects its long-context profile and container
        # name first, so every option the user passes still overrides them.
        for script, preset in (('run-rocm10.sh', ['--name', 'paiton-qwen38']), ('run-rocm10-65k.sh', []),
                               ('run-rocm10-200k.sh', ['--profile', 'chat', '--name', 'paiton-qwen38-200k'])):
            result = subprocess.run(['bash', str(MODEL_DIR / script), *options],
                                    env=self.environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(self.record.read_text()),
                             [str(SCRIPT), '--release', '65k', *preset, *options])

    def test_long_context_selects_the_chat_profile_on_the_current_image(self):
        image = launcher.IMAGES['65k']
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        for weights in ('mxfp4', 'w3a4'):
            if weights == 'w3a4':
                self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
            for context in ('200000', '220000'):
                with self.subTest(weights=weights, context=context):
                    command = self.command('--context', context)
                    self.assertIn(image, command)
                    self.assertIn('RADIANCE_GDN_LAZY=0', command)
                    # prefix caching is not qualified with the 4-bit cache: FP8 cache, the chat profile's budget
                    self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0'])
                    engine = self.engine(command)
                    self.assertEqual(value(engine, '--max-model-len'), context)
                    self.assertEqual(json.loads(value(engine, '--speculative-config'))['max_model_len'], int(context))
                    self.assertEqual(value(engine, '--max-num-seqs'), '1')
                    self.assertEqual(value(engine, '--max-num-batched-tokens'), '1024')
                    self.assertEqual(value(engine, '--kv-cache-memory-bytes'), '8589934592')
                    self.assertEqual(value(engine, '--mamba-cache-mode'), 'align')
                    self.assertIn('--enable-prefix-caching', engine)
                    self.assertEqual(json.loads(value(engine, '--default-chat-template-kwargs')),
                                     {'enable_thinking': False})
        # up to the 65K preset's own limit, and with an explicit profile, nothing changes
        engine = self.engine(self.command('--context', '65536'))
        self.assertEqual(value(engine, '--max-num-seqs'), '8')
        self.assertIn('--no-enable-prefix-caching', engine)
        engine = self.engine(self.command('--profile', 'release', '--context', '200000'))
        self.assertEqual(value(engine, '--max-num-seqs'), '8')
        self.assertIn('--no-enable-prefix-caching', engine)
        engine = self.engine(self.command('--profile', 'desktop', '--context', '100000'))
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), str(2 * 1024**3))
        self.assertEqual(value(engine, '--max-model-len'), '100000')

    def test_ngram_codraft_forwarded_only_when_set_on_host(self):
        command = self.command()
        self.assertFalse(any(item.startswith('PAITON_NGRAM_CODRAFT') for item in command))
        self.environment.update(PAITON_NGRAM_CODRAFT='1', PAITON_NGRAM_CODRAFT_HOT_MATCH='32')
        command = self.command()
        image_index = command.index(launcher.IMAGES['65k'])
        self.assertIn('PAITON_NGRAM_CODRAFT=1', command[:image_index])
        self.assertIn('PAITON_NGRAM_CODRAFT_HOT_MATCH=32', command[:image_index])
        self.assertEqual(command[command.index('PAITON_NGRAM_CODRAFT=1') - 1], '-e')

    def test_w3a4_weights_follow_the_mounted_directory(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        flags = [name + '=0' for name in launcher.W3_FLAGS]
        # Without the 3-bit weights the image serves MXFP4 and says how to enable them.
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PAITON_W3ROT_DIR', result.stderr)
        command = ['docker', *json.loads(self.record.read_text())]
        image_index = command.index(launcher.IMAGES['65k'])
        for flag in flags:
            self.assertIn(flag, command[:image_index])
        self.assertFalse(any(item.endswith(':/models/w3rot:ro') for item in command))
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        command = self.command()
        image_index = command.index(launcher.IMAGES['65k'])
        self.assertIn(f'{w3rot}:/models/w3rot:ro', command[:image_index])
        self.assertFalse(any(flag in command for flag in flags))
        # The memory the 3-bit weights free goes to the KV cache unless a budget is given,
        # with the allocator setting that budget was measured with (4-bit KV cache budget by default).
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.W3_KV4_CACHE_BYTES))
        self.assertIn('PYTORCH_ALLOC_CONF=max_split_size_mb:64', command[:image_index])
        command = self.command('--kv-cache-memory-bytes', '7000000000')
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '7000000000')
        command = self.command('--profile', 'desktop')
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(2 * 1024**3))
        command = self.command('--weights', 'mxfp4')
        self.assertFalse(any(item.endswith(':/models/w3rot:ro') for item in command))
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '6535819798')
        for flag in flags:
            self.assertIn(flag, command)

    def test_w3a4_weights_require_their_directory_and_the_65k_image(self):
        result = self.run_launcher('--weights', 'w3a4')
        self.assertEqual(result.returncode, 2)
        self.assertIn('PAITON_W3ROT_DIR', result.stderr)
        self.environment['PAITON_W3ROT_DIR'] = str(self.root / 'missing')
        result = self.run_launcher('--weights', 'w3a4')
        self.assertEqual(result.returncode, 2)
        self.assertIn('not an existing directory', result.stderr)
        result = self.run_launcher('--release', '200k', '--weights', 'w3a4')
        self.assertEqual(result.returncode, 2)
        self.assertIn('not available for the 200k release', result.stderr)
        self.assertFalse(self.record.exists())
        command = self.command('--release', '200k')
        self.assertFalse(any(item.startswith('PAITON_W3_') for item in command))
        self.assertFalse(any(item.endswith(':/models/w3rot:ro') for item in command))


    def kv4_flags(self, command, image):
        index = command.index(image)
        return [item for item in command[:index] if item.startswith('PAITON_KV4')]

    def test_w3a4_serves_the_capacity_kv4_cache_with_its_budget(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        image = launcher.IMAGES['65k']
        command = self.command()
        self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=1', 'PAITON_KV4_CAPACITY=1'])
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.W3_KV4_CACHE_BYTES))
        self.assertLess(launcher.W3_KV4_CACHE_BYTES, launcher.W3_KV_CACHE_BYTES)
        # explicit fp8 keeps the previous release budget and cache
        command = self.command('--kv-cache', 'fp8')
        self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0'])
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.W3_KV_CACHE_BYTES))
        # a user budget is respected either way
        command = self.command('--kv-cache-memory-bytes', '7000000000')
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '7000000000')
        self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=1', 'PAITON_KV4_CAPACITY=1'])

    def test_kv4_stays_off_where_it_was_not_qualified(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        image = launcher.IMAGES['65k']
        # MXFP4 weights (no rotated weights mounted): fp8 KV as before
        command = self.command()
        self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0'])
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '6535819798')
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        # prefix caching (and the chat profile that uses it): fp8 KV
        for options in (('--prefix-caching', 'on'), ('--profile', 'chat')):
            with self.subTest(options=options):
                command = self.command(*options)
                self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0'])
        # an explicit kv4 request where it is not qualified is refused before Docker runs
        for options in (('--kv-cache', 'kv4', '--weights', 'mxfp4'), ('--kv-cache', 'kv4', '--prefix-caching', 'on'),
                        ('--kv-cache', 'kv4', '--release', '200k')):
            with self.subTest(options=options):
                self.record.unlink(missing_ok=True)
                result = self.run_launcher(*options)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('--kv-cache kv4', result.stderr)
                self.assertFalse(self.record.exists())
        # the 200k release is not a KV4 image: no KV4 flags at all
        command = self.command('--release', '200k')
        self.assertEqual(self.kv4_flags(command, launcher.IMAGES['200k']), [])


    def test_kv4_stays_within_its_context_limit(self):
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        image = launcher.IMAGES['65k']
        on, off = ['PAITON_KV4=1', 'PAITON_KV4_CAPACITY=1'], ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0']
        # automatic selection only in the measured configuration: the 65K preset at up to 65,536 tokens
        self.assertEqual(self.kv4_flags(self.command('--context', '65536'), image), on)
        for options in (('--profile', 'release', '--context', '65537'), ('--profile', 'desktop')):
            with self.subTest(options=options):
                self.assertEqual(self.kv4_flags(self.command(*options), image), off)
        command = self.command('--profile', 'release', '--context', str(launcher.KV4_MAX_CONTEXT))
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.W3_KV_CACHE_BYTES))
        # an explicit request covers the kernels' range, with that mode's own budget
        command = self.command('--profile', 'release', '--kv-cache', 'kv4', '--context', str(launcher.KV4_MAX_CONTEXT))
        self.assertEqual(self.kv4_flags(command, image), on)
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.W3_KV4_CACHE_BYTES))
        command = self.command('--profile', 'desktop', '--kv-cache', 'kv4')
        self.assertEqual(self.kv4_flags(command, image), on)
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(2 * 1024**3))
        self.record.unlink(missing_ok=True)
        result = self.run_launcher('--profile', 'release', '--kv-cache', 'kv4',
                                   '--context', str(launcher.KV4_MAX_CONTEXT + 1))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('--kv-cache kv4', result.stderr)
        self.assertFalse(self.record.exists())


    def test_vision_loads_the_encoder_with_a_smaller_kv_budget(self):
        image = launcher.IMAGES['65k']
        self.assertIn('--language-model-only', self.engine(self.command()))
        # MXFP4 weights: image input on, KV budget reduced for the vision encoder, and the allocator setting the
        # vision budgets were measured with (without it the encoder's startup profile fragments the cache)
        command = self.command('--vision')
        self.assertIn('PYTORCH_ALLOC_CONF=max_split_size_mb:64', command[:command.index(image)])
        engine = self.engine(command)
        self.assertNotIn('--language-model-only', engine)
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), str(launcher.VISION_KV_CACHE_BYTES['mxfp4', 'fp8']))
        self.assertLess(launcher.VISION_KV_CACHE_BYTES['mxfp4', 'fp8'], 6535819798)
        # 3-bit weights: the 4-bit KV cache stays the default, the FP8 cache on request, each with its own budget
        w3rot = self.root / 'w3rot directory'
        w3rot.mkdir()
        self.environment['PAITON_W3ROT_DIR'] = str(w3rot)
        command = self.command('--vision')
        self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=1', 'PAITON_KV4_CAPACITY=1'])
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.VISION_KV_CACHE_BYTES['w3a4', 'kv4']))
        self.assertLess(launcher.VISION_KV_CACHE_BYTES['w3a4', 'kv4'], launcher.W3_KV4_CACHE_BYTES)
        command = self.command('--vision', '--kv-cache', 'fp8')
        self.assertEqual(self.kv4_flags(command, image), ['PAITON_KV4=0', 'PAITON_KV4_CAPACITY=0'])
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), str(launcher.VISION_KV_CACHE_BYTES['w3a4', 'fp8']))
        self.assertLess(launcher.VISION_KV_CACHE_BYTES['w3a4', 'fp8'], launcher.W3_KV_CACHE_BYTES)
        # explicit budgets still win
        command = self.command('--vision', '--kv-cache-memory-bytes', '3000000000')
        self.assertEqual(value(command, '--kv-cache-memory-bytes'), '3000000000')
        engine = self.engine(self.command('--vision', '--gpu-memory-utilization', '0.9'))
        self.assertNotIn('--kv-cache-memory-bytes', engine)
        self.assertNotIn('--language-model-only', engine)

    def test_vision_with_the_desktop_profile_keeps_its_budget(self):
        engine = self.engine(self.command('--profile', 'desktop', '--vision'))
        self.assertNotIn('--language-model-only', engine)
        self.assertEqual(value(engine, '--kv-cache-memory-bytes'), str(2 * 1024**3))
        self.assertEqual(value(engine, '--max-model-len'), '32768')

    def test_vision_is_refused_where_it_was_not_qualified(self):
        for options, reason in ((('--context', '200000'), 'long-context mode'),
                                (('--profile', 'chat'), 'long-context mode'),
                                (('--prefix-caching', 'on'), 'long-context mode'),
                                (('--release', '200k'), 'not available for the 200k release')):
            with self.subTest(options=options):
                self.record.unlink(missing_ok=True)
                result = self.run_launcher('--vision', *options)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('--vision', result.stderr)
                self.assertIn(reason, result.stderr)
                self.assertFalse(self.record.exists())


if __name__ == '__main__':
    unittest.main()
