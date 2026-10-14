"""Unit tests and verification suite for delegation."""

from typing import Any


from ml_ecosystem_snapshots.models import GhostInspector


class BaseLayer:
    """Mock or helper class BaseLayer for unit testing."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        """Execute helper init  .

        Args:
            hidden_dim: Input parameter value.
            dropout: Input parameter value.
        """
        pass


class SubLayer(BaseLayer):
    """Mock or helper class SubLayer for unit testing."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Execute helper init  .

        Args:
            name: Input parameter value.
            kwargs: Input parameter value.
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
