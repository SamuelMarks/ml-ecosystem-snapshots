"""Unit tests and verification suite for huggingface."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from ml_switcheroo_ir.schema.ghost import SemanticTier, GhostRef
from ml_ecosystem_snapshots.frameworks.huggingface import (
    collect_transformers,
    collect_diffusers,
    collect_tokenizers,
    _parse_pretrained_config,
)


def test_collect_transformers() -> None:
    """Verify collect transformers behavior."""
    mock_mod = MagicMock()

    class DummyConfig:
        """Mock or helper class DummyConfig for unit testing."""

        __annotations__ = {"vocab_size": int, "hidden_size": int}

    class AutoModelForCausalLM:
        """Mock or helper class AutoModelForCausalLM for unit testing."""

        pass

    class DummyModel:
        """Mock or helper class DummyModel for unit testing."""

        def generate(self, input_ids: str, max_length: int = 20) -> str:
            """Execute helper generate.

            Args:
                input_ids: Input token ID tensor.
                max_length: Maximum generation sequence length.


            Returns:
                Computed result.
            """
            return ""

    mock_mod.DummyConfig = DummyConfig
    mock_mod.AutoModelForCausalLM = AutoModelForCausalLM
    mock_mod.DummyModel = DummyModel
    setattr(
        mock_mod,
        "__dir__",
        lambda self: [
            "DummyConfig",
            "AutoModelForCausalLM",
            "DummyModel",
        ],
    )

    with (
        patch("importlib.import_module", return_value=mock_mod),
        patch(
            "ml_ecosystem_snapshots.frameworks.huggingface.GhostInspector"
        ) as MockInspector,
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
        """Mock or helper class DummyScheduler for unit testing."""

        pass

    mock_mod.DummyScheduler = DummyScheduler
    setattr(mock_mod, "__dir__", lambda self: ["DummyScheduler"])

    with (
        patch("importlib.import_module", return_value=mock_mod),
        patch(
            "ml_ecosystem_snapshots.frameworks.huggingface.GhostInspector"
        ) as MockInspector,
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
        """Mock or helper class DummyTokenizer for unit testing."""

        pass

    mock_mod.DummyTokenizer = DummyTokenizer
    setattr(mock_mod, "__dir__", lambda self: ["DummyTokenizer"])

    with (
        patch("importlib.import_module", return_value=mock_mod),
        patch(
            "ml_ecosystem_snapshots.frameworks.huggingface.GhostInspector"
        ) as MockInspector,
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


def test_parse_pretrained_config_with_empty_annotations() -> None:
    """Verify parse pretrained config with empty annotations behavior."""
    ref = GhostRef(name="MyConfig", api_path="pkg.MyConfig", kind="class")

    class DummyConfig:
        """Mock or helper class DummyConfig for unit testing."""

        def __init__(self, **kwargs: object) -> None:
            """Init docstring.

            Args:
                **kwargs: kwargs
            """
            pass

    setattr(DummyConfig, "__annotations__", {})

    _parse_pretrained_config(DummyConfig, ref)
    assert len(ref.params) == 0
