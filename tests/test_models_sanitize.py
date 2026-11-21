"""Unit tests and verification suite for models sanitize."""

from __future__ import annotations

import inspect

import numpy as np

from ml_ecosystem_snapshots.introspection.sanitize import (
    TYPE_STR_SANITIZE_ENDO,
    FactoryDefault,
    LiteralDefault,
    MandatoryDefault,
    UnrepresentableDefault,
    default_value_or_none,
    sanitize_param_default,
    sanitize_type_str,
)


def test_sanitize_type_str() -> None:
    """Verify sanitize_type_str and TYPE_STR_SANITIZE_ENDO behavior."""
    assert sanitize_type_str(None) is None
    assert sanitize_type_str("") == ""
    assert sanitize_type_str(":py:class:`int`") == "int"
    assert sanitize_type_str("<class 'list'>") == "list"
    assert sanitize_type_str("CustomType at 0x7fff1234") == "CustomType"
    assert sanitize_type_str("typing.List[int]") == "list[int]"
    assert sanitize_type_str("typing.Dict[str, typing.Tuple[int, int]]") == "dict[str, tuple[int, int]]"
    assert sanitize_type_str("Union[list, dict]") == "list | dict"
    assert sanitize_type_str("typing.Union[str, int]") == "str | int"
    assert sanitize_type_str("Optional[int]") == "int | None"
    assert sanitize_type_str("typing.Optional[float]") == "float | None"
    assert sanitize_type_str("typing.Set[int]") == "set[int]"
    assert sanitize_type_str("Type[int]") == "type[int]"
    assert sanitize_type_str("int") == "int"
    assert sanitize_type_str("some_invalid_ast!") == "some_invalid_ast!"
    assert sanitize_type_str("Union[int]") == "Union[int]"
    assert sanitize_type_str("typing.Any") == "Any"
    assert TYPE_STR_SANITIZE_ENDO.app("") == ""


def test_sanitize_param_default_sum_type_variants() -> None:
    """Test sanitize_param_default closed SanitizedDefault variants and default_value_or_none folding."""
    assert sanitize_param_default(inspect.Parameter.empty) == MandatoryDefault()
    assert default_value_or_none(MandatoryDefault()) is None

    assert sanitize_param_default(Ellipsis) == LiteralDefault(value_repr="...")
    assert sanitize_param_default(42) == LiteralDefault(value_repr="42")
    assert sanitize_param_default("abc") == LiteralDefault(value_repr="'abc'")
    assert default_value_or_none(LiteralDefault(value_repr="42")) == "42"

    def sample_factory() -> int:
        """Return sample default integer.

        Returns:
            Constant integer 1.
        """
        return 1

    fac = sanitize_param_default(sample_factory)
    assert fac == FactoryDefault(placeholder="<factory_default>", factory_repr="<function sample_factory>")
    assert default_value_or_none(fac) == "<factory_default>"
    assert default_value_or_none(UnrepresentableDefault()) == "<unrepresentable>"
    assert default_value_or_none(UnrepresentableDefault(), unrepresentable_token="...") == "..."


def test_sanitize_param_default_edge_cases() -> None:
    """Test sanitize_param_default for non-bracket scrubbed repr and string-only address leaks."""

    class CustomReprWithAngleBrackets:
        """Helper class with angle-bracket repr containing address."""

        def __repr__(self) -> str:
            """Return angle-bracket repr with address.

            Returns:
                String with bracketed address.
            """
            return "<MyObject at 0x7fff1234>"

    class CustomReprWithoutAngleBrackets:
        """Helper class with non-angle-bracket repr containing memory address."""

        def __repr__(self) -> str:
            """Return custom repr with address.

            Returns:
                String with custom address.
            """
            return "custom_object at 0x7fff1234"

    class CustomStrWithAddress:
        """Helper class with clean repr but address in str()."""

        def __repr__(self) -> str:
            """Return clean repr.

            Returns:
                Clean repr string.
            """
            return "CleanRepr"

        def __str__(self) -> str:
            """Return str with address.

            Returns:
                String containing address leak.
            """
            return "address_leak at 0x7fff1234"

    class NormalCustomObject:
        """Helper class with standard custom repr and str."""

        def __repr__(self) -> str:
            """Return standard repr.

            Returns:
                Standard repr string.
            """
            return "NormalCustomObject()"

        def __str__(self) -> str:
            """Return standard str.

            Returns:
                Standard str string.
            """
            return "NormalCustomObject()"

    class BuggyRepr:
        """Helper class that raises on repr."""

        def __repr__(self) -> str:
            """Raise error on repr.

            Raises:
                RuntimeError: Always raised for testing.

            Returns:
                Never returns.
            """
            raise RuntimeError("broken repr")

    assert sanitize_param_default(CustomReprWithAngleBrackets()) == FactoryDefault(
        placeholder="<factory_default>",
        factory_repr="<MyObject>",
    )
    assert sanitize_param_default(CustomReprWithoutAngleBrackets()) == LiteralDefault(value_repr="custom_object")
    assert sanitize_param_default(CustomStrWithAddress()) == FactoryDefault(
        placeholder="<factory_default>",
        factory_repr="<factory_default>",
    )
    assert sanitize_param_default(NormalCustomObject()) == LiteralDefault(value_repr="NormalCustomObject()")
    assert sanitize_param_default(BuggyRepr()) == UnrepresentableDefault()


def test_sanitize_param_default_pointer_scrubbing() -> None:
    """Test scrubbing raw hexadecimal pointer addresses from default representations."""

    class PointerDefault:
        """Class with repr containing hex address without at prefix."""

        def __repr__(self) -> str:
            """Return repr with embedded hex address.

            Returns:
                String with hex pointer.
            """
            return "Device(0x7fffbeef)"

    class PointerStrOnly:
        """Class with str containing hex address without at prefix."""

        def __repr__(self) -> str:
            """Return clean repr.

            Returns:
                Clean string.
            """
            return "CleanDevice"

        def __str__(self) -> str:
            """Return str with embedded hex address.

            Returns:
                String with hex pointer.
            """
            return "Device(0x7fffbeef)"

    assert sanitize_param_default(PointerDefault()) == LiteralDefault(value_repr="Device(<addr>)")
    assert sanitize_param_default(PointerStrOnly()) == FactoryDefault(
        placeholder="<factory_default>",
        factory_repr="<factory_default>",
    )


def test_sanitize_framework_constants() -> None:
    """Test preservation of framework dtypes, classes, and device constants."""
    assert sanitize_param_default(np.float32) == LiteralDefault(value_repr="numpy.float32")

    class MockTFDtype:
        """Mock TF DType with angle-bracket repr."""

        def __repr__(self) -> str:
            """Return TF dtype representation.

            Returns:
                Dtype string.
            """
            return "<dtype: 'float32'>"

    assert sanitize_param_default(MockTFDtype()) == LiteralDefault(value_repr="tf.float32")

    class MockTorchDevice:
        """Mock Torch Device."""

        def __repr__(self) -> str:
            """Return device representation.

            Returns:
                Device string.
            """
            return "device(type='cpu')"

    assert sanitize_param_default(MockTorchDevice()) == FactoryDefault(
        placeholder="'cpu'",
        factory_repr="device(type='cpu')",
    )
