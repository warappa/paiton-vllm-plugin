"""Published entry points remain usable beside the shorter serving command."""

import importlib.util
import os
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from paiton_vllm_plugin.activation import activate_model_launcher
from paiton_vllm_plugin.execution.presets import presets
from paiton_vllm_plugin.cli import parser

ROOT = Path(__file__).resolve().parents[1]


def test_short_help_is_forwarded_without_a_model():
    for flag in ("--help", "-h", "--help=ModelConfig"):
        args = parser().parse_args(["serve", flag])
        assert args.arguments == []
        assert args.serve_help == ("ModelConfig" if "=" in flag else "")


def test_every_documented_native_preset_has_its_exact_package_and_model_guide():
    main = (ROOT / "README.md").read_text()
    guides = {
        "qwen38-nvfp4": "Qwen3.8-MXFP4-DFlash2",
        "minicpm5": "MiniCPM5-2B",
        "qwen38-qronos": "Qwen3.8",
        "ornith": "Ornith-1.5",
        "qwen38-neo": "Qwen3.8-NEO-CODER-MAX",
        "qwen3-coder": "Qwen3-Coder-30B",
        "gpt-oss-20b": "GPT-OSS-20B",
    }
    assert set(guides) == set(presets())
    for name, (package, profile) in presets().items():
        guide = (ROOT / "models" / guides[name] / "README.md").read_text()
        assert f"](models/{guides[name]}/README.md)" in main
        assert "paiton serve " + name in guide
        assert package["id"] in guide
        assert package["model"]["revision"] in guide
        assert profile in guide
        manifest = json.loads(
            (
                ROOT / "paiton_vllm_plugin/execution/data" / package["manifest"]
            ).read_text()
        )
        assert "support/offline_guard.so" in manifest["files"]
        assert any(
            name.endswith(".so") and not name.startswith("support/")
            for name in manifest["files"]
        )
        environment = package["profiles"][profile].get("environment", {})
        for key, value in environment.items():
            if key.endswith("_SHA256"):
                library = environment.get(key.removesuffix("_SHA256") + "_SO")
                if library and library.startswith("{bundle}/"):
                    relative = library.removeprefix("{bundle}/")
                    assert manifest["files"][relative]["sha256"] == value


def test_launch_guides_have_no_missing_local_link_targets():
    paths = [
        ROOT / "README.md",
        ROOT / "docs/NATIVE_EXECUTION.md",
        ROOT / "docs/NATIVE_PACKAGING.md",
        ROOT / "models/Qwen3.8-MXFP4-DFlash2/REFERENCE.md",
        *sorted((ROOT / "models").glob("*/README.md")),
    ]
    for path in paths:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            target = target.split("#")[0].split(' "')[0]
            if target and ":" not in target and not target.startswith("//"):
                assert (path.parent / target).exists(), (path, target)


def test_legacy_console_script_targets_are_retained():
    tomllib = pytest.importorskip("tomllib")

    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert {
        name: scripts[name]
        for name in (
            "paiton-vllm",
            "paiton-chat",
            "paiton-qwen38-serve",
            "paiton-ornith-serve",
            "paiton-dflash-serve",
            "paiton-offline-benchmark",
        )
    } == {
        "paiton-vllm": "paiton_vllm_plugin.existing_env:main",
        "paiton-chat": "paiton_vllm_plugin.chat:main",
        "paiton-qwen38-serve": "paiton_vllm_plugin.qwen38_release_server:main",
        "paiton-ornith-serve": "paiton_vllm_plugin.ornith_release_server:main",
        "paiton-dflash-serve": "paiton_vllm_plugin.dflash.serve:main",
        "paiton-offline-benchmark": "paiton_vllm_plugin.benchmarks.offline_benchmark:main",
    }


def test_legacy_neo_flags_keep_the_same_python_and_plugin_allowlist(tmp_path):
    from paiton_vllm_plugin import gguf_release_server as server

    model = tmp_path / "model"
    model.mkdir()
    (model / "vision-runtime.json").write_text('{"checkpoint":{"file":"mmproj.gguf"}}')
    checkpoint, projector = tmp_path / "model.gguf", tmp_path / "projector.gguf"
    checkpoint.write_bytes(b"fixture")
    projector.write_bytes(b"fixture")
    allowlist = "another,paiton_platform,register_paiton_models"
    with (
        patch.dict(os.environ, {"VLLM_PLUGINS": allowlist}, clear=True),
        patch.object(
            sys,
            "argv",
            [
                "gguf_release_server",
                "--model-dir",
                str(model),
                "--checkpoint",
                str(checkpoint),
                "--projector",
                str(projector),
                "--prefill-chunk-tokens",
                "1024",
                "--port",
                "8123",
            ],
        ),
        patch.object(
            server,
            "verify_payload",
            return_value={"multimodal": True, "max_num_batched_tokens": 2048},
        ),
        patch.object(server, "_validate_runtime_environment"),
        patch.object(os, "execv") as execute,
    ):
        server.main()
        executable, command = execute.call_args.args
        assert executable == sys.executable
        assert command[:4] == [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
        ]
        assert command[command.index("--max-num-batched-tokens") + 1] == "1024"
        assert command[command.index("--port") + 1] == "8123"
        assert os.environ["VLLM_PLUGINS"] == allowlist
        assert os.environ["PAITON_PLUGIN_MODE"] == "legacy"


@pytest.mark.parametrize("stock", [False, True])
def test_legacy_model_activation_preserves_other_plugins(stock):
    env = {"VLLM_PLUGINS": "another,register_paiton_models"}
    activate_model_launcher(stock, env)
    assert env["VLLM_PLUGINS"] == "another,register_paiton_models"
    assert env["PAITON_PLUGIN_MODE"] == ("off" if stock else "models")


def test_legacy_launcher_respects_explicit_restrictions():
    for env in ({"VLLM_PLUGINS": "another"}, {"PAITON_PLUGIN_MODE": "off"}):
        with pytest.raises(ValueError):
            activate_model_launcher(False, env)


@pytest.mark.parametrize("directory", ["GPT-OSS-20B", "Qwen3-Coder-30B", "MiniCPM5-2B"])
@pytest.mark.parametrize("stock", [False, True])
def test_published_python_server_command_still_launches(directory, stock):
    # Verify the real launcher, including its activation and same-Python exec;
    # checkpoint downloads and inference are covered by separate GPU tests.
    spec = importlib.util.spec_from_file_location(
        "legacy_server", ROOT / "models" / directory / "server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hub = SimpleNamespace(snapshot_download=lambda *a, **kw: "/existing/checkpoint")
    argv = ["server.py", "--offline", *(["--stock"] if stock else [])]
    with (
        patch.dict(
            os.environ, {"VLLM_PLUGINS": "another,register_paiton_models"}, clear=True
        ),
        patch.dict(sys.modules, {"huggingface_hub": hub}),
        patch.object(sys, "argv", argv),
        patch.object(
            module, "prepare", return_value=Path("/existing/checkpoint"), create=True
        ),
        patch.object(os, "execv") as execute,
    ):
        module.main()
        executable, command = execute.call_args.args
        assert executable == sys.executable == command[0]
        assert "/existing/checkpoint" in command
        assert os.environ["PAITON_PLUGIN_MODE"] == ("off" if stock else "models")
        assert os.environ["VLLM_PLUGINS"] == "another,register_paiton_models"
        assert ("--hf-overrides" in command) != stock
