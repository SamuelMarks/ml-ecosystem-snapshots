"""Tests for unwrapping torch.compile decorated callables in GhostInspector."""

from __future__ import annotations

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import InspectableTarget
from ml_ecosystem_snapshots.models import JsonValue


def torch_compile_mock(func: InspectableTarget) -> InspectableTarget:
    """Mock decorator simulating torch.compile wrapper.

    Args:
        func: The original function to wrap.

    Returns:
        Compiled mock wrapper instance.
    """

    class Compiled:
        """Mock container for compiled function."""

        def __init__(self, f: InspectableTarget) -> None:
            """Initialize with original function.

            Args:
                f: The original function.
            """
            self._orig_mod = f

    return Compiled(func)


def test_unwrap_torch_compile() -> None:
    """Test GhostInspector correctly unwraps and inspects torch.compile decorated functions."""

    @torch_compile_mock
    def my_compiled(x: JsonValue, y: int = 10) -> None:
        """Sample function for compilation unwrapping.

        Args:
            x: Input parameter.
            y: Optional integer parameter.
        """
        pass

    ref = GhostInspector.inspect(my_compiled, "my_compiled")
    assert ref.has_arg("x")
    assert ref.has_arg("y")
