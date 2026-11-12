"""Unit tests and verification suite for huggingface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_ecosystem_snapshots.frameworks.huggingface import (
    collect_diffusers,
    collect_tokenizers,
    collect_transformers,
)


def test_collect_transformers() -> None:
    """Verify collect transformers behavior."""
    mock_mod = MagicMock()

    class DummyConfig:
        """Stub Hugging Face `PreTrainedConfig` subclass with typed fields."""

        __annotations__ = {"vocab_size": int, "hidden_size": int}

    class AutoModelForCausalLM:
        """Stub Hugging Face `AutoModelForCausalLM` factory class."""

        pass

    class DummyModel:
        """Stub Hugging Face causal model exposing a `generate` method."""

        def generate(self, input_ids: str, max_length: int = 20) -> str:
            """Generate decoded text from token identifiers.

            Args:
                input_ids: Input token ID tensor.
                max_length: Maximum generation sequence length.

            Returns:
                Empty placeholder string for signature extraction tests.
            """
            return ""

    mock_mod.DummyConfig = DummyConfig
    mock_mod.AutoModelForCausalLM = AutoModelForCausalLM
    mock_mod.DummyModel = DummyModel
    pytest.MonkeyPatch().setattr(
        mock_mod,
        "__dir__",
        lambda self: [
            "DummyConfig",
            "AutoModelForCausalLM",
            "DummyModel",
        ],
        raising=False,
    )

    with (
        patch("importlib.import_module", return_value=mock_mod),
        patch("ml_ecosystem_snapshots.frameworks.huggingface.GhostInspector") as MockInspector,
    ):
        from ml_switcheroo_ir.schema.ghost import GhostRef

        MockInspector.return_value.inspect.side_effect = lambda obj, path: GhostRef(
            name=path.split(".")[-1],
            api_path=path,
            kind="function",
            params=[],
            docstring="",
        )

        res = collect_transformers(SemanticTier.MODEL)
        models = {r.name: r for r in res}

        assert "AutoModelForCausalLM" in models
        config_params = {p.name: p for p in models["DummyConfig"].params}
        assert "vocab_size" in config_params
        assert config_params["vocab_size"].annotation == "int"

        assert "AutoModelForCausalLM" in models
        auto_params = {p.name: p for p in models["AutoModelForCausalLM"].params}
        assert "config" in auto_params
        assert auto_params["config"].annotation == "PreTrainedConfig"

        assert "DummyModel" in models
        dummy_params = {p.name: p for p in models["DummyModel"].params}
        assert "max_length" in dummy_params
        assert dummy_params["max_length"].annotation == "int"


def test_collect_diffusers() -> None:
    """Verify collect diffusers behavior."""
    mock_mod = MagicMock()

    class DummyScheduler:
        """Stub Diffusers noise scheduler class."""

        pass

    mock_mod.DummyScheduler = DummyScheduler
    pytest.MonkeyPatch().setattr(mock_mod, "__dir__", lambda self: ["DummyScheduler"], raising=False)

    with (
        patch("importlib.import_module", return_value=mock_mod),
        patch("ml_ecosystem_snapshots.frameworks.huggingface.GhostInspector") as MockInspector,
    ):
        from ml_switcheroo_ir.schema.ghost import GhostRef

        MockInspector.return_value.inspect.side_effect = lambda obj, path: GhostRef(
            name=path.split(".")[-1],
            api_path=path,
            kind="function",
            params=[],
            docstring="",
        )

        res = collect_diffusers(SemanticTier.OPTIMIZER)
        assert len(res) == 1
        assert res[0].name == "DummyScheduler"


def test_collect_tokenizers() -> None:
    """Verify collect tokenizers behavior."""
    mock_mod = MagicMock()

    class DummyTokenizer:
        """Stub Hugging Face fast tokenizer class."""

        pass

    mock_mod.DummyTokenizer = DummyTokenizer
    pytest.MonkeyPatch().setattr(mock_mod, "__dir__", lambda self: ["DummyTokenizer"], raising=False)

    with (
        patch("importlib.import_module", return_value=mock_mod),
        patch("ml_ecosystem_snapshots.frameworks.huggingface.GhostInspector") as MockInspector,
    ):
        from ml_switcheroo_ir.schema.ghost import GhostRef

        MockInspector.return_value.inspect.side_effect = lambda obj, path: GhostRef(
            name=path.split(".")[-1],
            api_path=path,
            kind="function",
            params=[],
            docstring="",
        )

        res = collect_tokenizers(SemanticTier.MODEL)
        assert len(res) == 1
        assert res[0].name == "DummyTokenizer"


def test_parse_pretrained_config_with_empty_annotations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `collect_transformers` leaves config params empty when `__annotations__` is `{}`.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import sys
    import types

    class EmptyConfig:
        """Stub configuration class with empty `__annotations__`."""

        def __init__(self) -> None:
            """Initialize empty configuration."""
            pass

    monkeypatch.setattr(EmptyConfig, "__annotations__", {}, raising=False)
    fake_tf = types.ModuleType("transformers")
    monkeypatch.setattr(fake_tf, "EmptyConfig", EmptyConfig, raising=False)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    res = collect_transformers(SemanticTier.MODEL)
    cfg_ref = next(r for r in res if r.name == "EmptyConfig")
    assert len(cfg_ref.params) == 0
