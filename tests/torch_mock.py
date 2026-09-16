"""Mock PyTorch module for isolated testing and offline environments.

Provides in-memory simulations of PyTorch runtime structures including
ATen namespaces, JIT operator schemas, overloads, and C-extension functions
when the heavy binary dependency is unavailable.
"""

import sys
import types
from typing import Any, Dict, List


class MockArgument:
    """Mock JIT/ATen schema parameter argument."""

    def __init__(
        self,
        name: str,
        type_str: str,
        kwarg_only: bool = False,
        default_value: Any = None,
        is_out: bool = False,
    ) -> None:
        """Initialize mock argument.

        Args:
            name: Argument identifier.
            type_str: Argument type representation.
            kwarg_only: Whether argument is keyword-only.
            default_value: Default value if provided.
            is_out: Whether argument represents an out-tensor.
        """
        self.name = name
        self.type = type_str
        self.kwarg_only = kwarg_only
        self.default_value = default_value
        self.is_out = is_out

    def has_default_value(self) -> bool:
        """Check if argument has a default value.

        Returns:
            True if default_value is present.
        """
        return self.default_value is not None


class MockReturn:
    """Mock return type container."""

    def __init__(self, type_str: str) -> None:
        """Initialize mock return type.

        Args:
            type_str: Return type string representation.
        """
        self.type = type_str


class MockSchema:
    """Mock ATen/JIT operator schema container."""

    def __init__(
        self,
        name: str,
        arguments: List[MockArgument],
        returns: List[MockReturn],
        overload_name: str = "",
    ) -> None:
        """Initialize mock schema.

        Args:
            name: Fully qualified operator name.
            arguments: List of schema arguments.
            returns: List of return specifications.
            overload_name: Variant overload identifier.
        """
        self.name = name
        self.arguments = arguments
        self.returns = returns
        self.overload_name = overload_name


class MockOverload:
    """Mock ATen overload variant."""

    def __init__(self, schema: MockSchema) -> None:
        """Initialize mock overload with schema.

        Args:
            schema: Associated schema instance.
        """
        self._schema = schema


class MockAtenOp:
    """Mock ATen operator container with overload dispatch."""

    def __init__(self, overloads_dict: Dict[str, MockOverload]) -> None:
        """Initialize mock ATen operator.

        Args:
            overloads_dict: Mapping of overload variant names to MockOverload instances.
        """
        self._overloads = overloads_dict
        for k, v in overloads_dict.items():
            setattr(self, k, v)

    def overloads(self) -> List[str]:
        """List registered overload variants.

        Returns:
            List of overload variant name strings.
        """
        return list(self._overloads.keys())


class MockTorchOp:
    """Mock PyTorch C-extension callable operator without Python signature."""

    def __init__(self, name: str, doc: str) -> None:
        """Initialize mock callable operator.

        Args:
            name: Operator function identifier.
            doc: C-extension docstring containing signature.
        """
        self.__name__ = name
        self.__doc__ = doc
        self.__module__ = "torch"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute mock operator.

        Args:
            *args: Positional call arguments.
            **kwargs: Keyword call arguments.

        Returns:
            None.
        """
        return None

    @property
    def __signature__(self) -> Any:
        """Simulate C-extension lacking inspectable Python signature.

        Raises:
            ValueError: Always raised to match PyTorch C-extension behavior.
        """
        raise ValueError(f"no signature found for builtin {self.__name__}")


class MockTensor:
    """Mock PyTorch Tensor class."""

    def __init__(self) -> None:
        """Initialize mock tensor."""
        pass

    def add(self, other: Any, *, alpha: int = 1) -> "MockTensor":
        """Add tensor.

        Args:
            other: Operand.
            alpha: Multiplier.

        Returns:
            Self instance.
        """
        return self

    def add_(self, other: Any, *, alpha: int = 1) -> "MockTensor":
        """In-place add tensor.

        Args:
            other: Operand.
            alpha: Multiplier.

        Returns:
            Self instance.
        """
        return self

    def relu(self) -> "MockTensor":
        """Apply ReLU.

        Returns:
            Self instance.
        """
        return self

    def matmul(self, other: Any) -> "MockTensor":
        """Matrix multiplication.

        Args:
            other: Operand.

        Returns:
            Self instance.
        """
        return self

    def sum(self) -> "MockTensor":
        """Sum elements.

        Returns:
            Self instance.
        """
        return self

    def cat(self, tensors: Any, dim: int = 0) -> "MockTensor":
        """Concatenate tensors.

        Args:
            tensors: Sequence of tensors.
            dim: Dimension along which to concatenate.

        Returns:
            Self instance.
        """
        return self

    def mm(self, mat2: Any) -> "MockTensor":
        """Matrix multiply.

        Args:
            mat2: Second matrix operand.

        Returns:
            Self instance.
        """
        return self

    def bmm(self, mat2: Any) -> "MockTensor":
        """Batch matrix multiply.

        Args:
            mat2: Second batch matrix operand.

        Returns:
            Self instance.
        """
        return self

    def detach(self) -> "MockTensor":
        """Detach tensor.

        Returns:
            Self instance.
        """
        return self


def create_mock_torch() -> types.ModuleType:
    """Construct a full mock PyTorch module hierarchy.

    Returns:
        The configured mock torch module.
    """
    torch_mod = types.ModuleType("torch")
    setattr(torch_mod, "Tensor", MockTensor)

    ops_mod = types.ModuleType("torch.ops")
    aten_mod = types.ModuleType("torch.ops.aten")
    c_mod = types.ModuleType("torch._C")

    # ATen add overloads
    add_schemas = {
        "Tensor": MockOverload(
            MockSchema(
                "aten::add.Tensor",
                [
                    MockArgument("self", "Tensor"),
                    MockArgument("other", "Tensor"),
                    MockArgument("alpha", "Scalar", default_value="1"),
                ],
                [MockReturn("Tensor")],
                "Tensor",
            )
        ),
        "Scalar": MockOverload(
            MockSchema(
                "aten::add.Scalar",
                [
                    MockArgument("self", "Tensor"),
                    MockArgument("other", "Scalar"),
                    MockArgument("alpha", "Scalar", default_value="1"),
                ],
                [MockReturn("Tensor")],
                "Scalar",
            )
        ),
        "out": MockOverload(
            MockSchema(
                "aten::add.out",
                [
                    MockArgument("self", "Tensor"),
                    MockArgument("other", "Tensor"),
                    MockArgument("alpha", "Scalar", default_value="1"),
                    MockArgument("out", "Tensor", kwarg_only=True, is_out=True),
                ],
                [MockReturn("Tensor")],
                "out",
            )
        ),
    }

    add_inplace_schemas = {
        "Tensor": MockOverload(
            MockSchema(
                "aten::add_.Tensor",
                [
                    MockArgument("self", "Tensor"),
                    MockArgument("other", "Tensor"),
                    MockArgument("alpha", "Scalar", default_value="1"),
                ],
                [MockReturn("Tensor")],
                "Tensor",
            )
        ),
        "Scalar": MockOverload(
            MockSchema(
                "aten::add_.Scalar",
                [
                    MockArgument("self", "Tensor"),
                    MockArgument("other", "Scalar"),
                    MockArgument("alpha", "Scalar", default_value="1"),
                ],
                [MockReturn("Tensor")],
                "Scalar",
            )
        ),
    }

    relu_schemas = {
        "default": MockOverload(
            MockSchema(
                "aten::relu",
                [MockArgument("self", "Tensor")],
                [MockReturn("Tensor")],
                "default",
            )
        ),
    }

    matmul_schemas = {
        "default": MockOverload(
            MockSchema(
                "aten::matmul",
                [
                    MockArgument("self", "Tensor"),
                    MockArgument("other", "Tensor"),
                ],
                [MockReturn("Tensor")],
                "default",
            )
        ),
    }

    sum_schemas = {
        "default": MockOverload(
            MockSchema(
                "aten::sum",
                [MockArgument("self", "Tensor")],
                [MockReturn("Tensor")],
                "default",
            )
        ),
    }

    cat_schemas = {
        "default": MockOverload(
            MockSchema(
                "aten::cat",
                [
                    MockArgument("tensors", "TensorList"),
                    MockArgument("dim", "int", default_value="0"),
                ],
                [MockReturn("Tensor")],
                "default",
            )
        ),
    }

    mm_schemas = {
        "default": MockOverload(
            MockSchema(
                "aten::mm",
                [
                    MockArgument("self", "Tensor"),
                    MockArgument("mat2", "Tensor"),
                ],
                [MockReturn("Tensor")],
                "default",
            )
        ),
    }

    bmm_schemas = {
        "default": MockOverload(
            MockSchema(
                "aten::bmm",
                [
                    MockArgument("self", "Tensor"),
                    MockArgument("mat2", "Tensor"),
                ],
                [MockReturn("Tensor")],
                "default",
            )
        ),
    }

    setattr(aten_mod, "add", MockAtenOp(add_schemas))
    setattr(aten_mod, "add_", MockAtenOp(add_inplace_schemas))
    setattr(aten_mod, "relu", MockAtenOp(relu_schemas))
    setattr(aten_mod, "matmul", MockAtenOp(matmul_schemas))
    setattr(aten_mod, "sum", MockAtenOp(sum_schemas))
    setattr(aten_mod, "cat", MockAtenOp(cat_schemas))
    setattr(aten_mod, "mm", MockAtenOp(mm_schemas))
    setattr(aten_mod, "bmm", MockAtenOp(bmm_schemas))
    # Non-callable member to test branch where member is skipped
    setattr(aten_mod, "name", "aten")

    setattr(ops_mod, "aten", aten_mod)
    setattr(torch_mod, "ops", ops_mod)

    # JIT registry
    jit_schemas = [
        MockSchema(
            "aten::add",
            [
                MockArgument("self", "Tensor"),
                MockArgument("other", "Tensor"),
                MockArgument("alpha", "Scalar", default_value="1"),
            ],
            [MockReturn("Tensor")],
            "Tensor",
        ),
        MockSchema(
            "aten::relu",
            [MockArgument("self", "Tensor")],
            [MockReturn("Tensor")],
            "default",
        ),
        MockSchema(
            "aten::matmul",
            [
                MockArgument("self", "Tensor"),
                MockArgument("other", "Tensor"),
            ],
            [MockReturn("Tensor")],
            "default",
        ),
    ]
    setattr(c_mod, "_jit_get_all_schemas", lambda: jit_schemas)
    setattr(torch_mod, "_C", c_mod)

    # Callable operators
    setattr(
        torch_mod,
        "add",
        MockTorchOp("add", "add(input, other, *, alpha=1, out=None) -> Tensor\n"),
    )
    setattr(torch_mod, "relu", MockTorchOp("relu", "relu(input) -> Tensor\n"))
    setattr(
        torch_mod,
        "matmul",
        MockTorchOp("matmul", "matmul(input, other) -> Tensor\n"),
    )
    setattr(torch_mod, "sum", MockTorchOp("sum", "sum(input) -> Tensor\n"))
    setattr(
        torch_mod,
        "cat",
        MockTorchOp("cat", "cat(tensors, dim=0) -> Tensor\n"),
    )
    setattr(torch_mod, "mm", MockTorchOp("mm", "mm(input, mat2) -> Tensor\n"))
    setattr(
        torch_mod,
        "bmm",
        MockTorchOp("bmm", "bmm(input, mat2) -> Tensor\n"),
    )

    # Submodules
    nn_mod = types.ModuleType("torch.nn")
    optim_mod = types.ModuleType("torch.optim")
    data_mod = types.ModuleType("torch.utils.data")
    utils_mod = types.ModuleType("torch.utils")
    utils_mod.data = data_mod  # type: ignore[attr-defined]

    setattr(torch_mod, "nn", nn_mod)
    setattr(torch_mod, "optim", optim_mod)
    setattr(torch_mod, "utils", utils_mod)

    return torch_mod


def install_mock_torch() -> types.ModuleType:
    """Install the mock PyTorch module into sys.modules.

    Returns:
        The installed mock torch module.
    """
    torch_mod = create_mock_torch()
    sys.modules["torch"] = torch_mod
    sys.modules["torch.ops"] = getattr(torch_mod, "ops")
    sys.modules["torch.ops.aten"] = getattr(torch_mod.ops, "aten")
    sys.modules["torch._C"] = getattr(torch_mod, "_C")
    sys.modules["torch.nn"] = getattr(torch_mod, "nn")
    sys.modules["torch.optim"] = getattr(torch_mod, "optim")
    sys.modules["torch.utils"] = getattr(torch_mod, "utils")
    sys.modules["torch.utils.data"] = getattr(torch_mod.utils, "data")
    return torch_mod


def ensure_torch() -> Any:
    """Ensure PyTorch is available, falling back to an in-memory mock if uninstalled.

    Returns:
        The real torch module if available, or a mock torch module.
    """
    try:
        import torch

        if hasattr(torch, "ops") and hasattr(torch.ops, "aten"):
            return torch
    except (ImportError, Exception):
        pass

    return install_mock_torch()
