"""Unit tests and verification suite for models forwardref."""

from __future__ import annotations

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector


class TensorRef:
    """Placeholder tensor type referenced via postponed string annotations."""

    pass


def forward(x: TensorRef) -> TensorRef:
    """Return `x` unchanged to exercise `ForwardRef` annotation resolution.

    Args:
        x: Forward-referenced parameter operand.

    Returns:
        Unmodified `TensorRef` input operand.
    """
    return x


def test_resolve_forward_ref() -> None:
    """Verify resolve forward ref behavior."""
    ref = GhostInspector.inspect(forward, "forward")
    assert ref.has_arg("x")
    assert ref.params[0].annotation is not None
    assert "TensorRef" in ref.params[0].annotation
    assert ref.returns_type is not None
    assert "TensorRef" in ref.returns_type
