"""Unit tests and verification suite for delegation."""

from __future__ import annotations

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector


class BaseLayer:
    """Mock or helper class BaseLayer for unit testing."""

    def __init__(self, hidden_dim: int = 256, dropout: int = 0) -> None:
        """Execute helper init  .

        Args:
            hidden_dim: Hidden layer dimension size.
            dropout: Dropout probability rate.
        """
        pass


class SubLayer(BaseLayer):
    """Mock or helper class SubLayer for unit testing."""

    def __init__(self, name: str, **kwargs: int) -> None:
        """Execute helper init  .

        Args:
            name: Submodule layer identifier.
            kwargs: Forwarded keyword arguments to parent initializer.
        """
        super().__init__(**kwargs)


def test_delegation_tracing() -> None:
    """Verify delegation tracing behavior."""
    ref = GhostInspector.inspect(SubLayer, "SubLayer")

    assert ref.name == "SubLayer"

    param_names = [p.name for p in ref.params]
    assert "name" in param_names
    assert "hidden_dim" in param_names
    assert "dropout" in param_names

    # kwargs might still be there or might be removed, but hidden_dim should be explicitly extracted

    hidden_param = next(p for p in ref.params if p.name == "hidden_dim")
    assert hidden_param.annotation == "int"
    assert hidden_param.default == "256"
