"""PyTorch API Snapshot Extractor.

Provides functions to dynamically introspect the PyTorch library and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import re
import types
import typing
import warnings

import pydantic
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.c_extension_sig import CExtensionSignature
from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    get_member_dict,
)
from ml_ecosystem_snapshots.models import ExtendedGhostRef, JsonValue, ensure_present
from ml_ecosystem_snapshots.utils import get_all_members

_LOGGER = logging.getLogger(__name__)


@typing.runtime_checkable
class TorchOptimWithSchedulerProtocol(typing.Protocol):
    """Protocol for `torch.optim` exposing `.lr_scheduler`."""

    lr_scheduler: types.ModuleType


@typing.runtime_checkable
class TorchNnWithInitProtocol(typing.Protocol):
    """Protocol for `torch.nn` exposing `.init`."""

    init: types.ModuleType


@typing.runtime_checkable
class TorchNnWithFunctionalProtocol(typing.Protocol):
    """Protocol for `torch.nn` exposing `.functional`."""

    functional: types.ModuleType


@typing.runtime_checkable
class TorchModuleWithNnProtocol(typing.Protocol):
    """Protocol for `torch` exposing `.nn`."""

    nn: TorchNnWithFunctionalProtocol


@typing.runtime_checkable
class TorchModuleWithTensorProtocol(typing.Protocol):
    """Protocol for `torch` exposing `.Tensor`."""

    Tensor: type


@typing.runtime_checkable
class TorchOpsContainerProtocol(typing.Protocol):
    """Protocol for `torch.ops` exposing `.aten`."""

    aten: types.ModuleType


@typing.runtime_checkable
class TorchModuleWithOpsProtocol(typing.Protocol):
    """Protocol for `torch` exposing `.ops`."""

    ops: TorchOpsContainerProtocol


warnings.filterwarnings(
    "ignore",
    message=".*Failed to initialize NumPy.*",
    category=UserWarning,
)

try:
    importlib.import_module("torch")
    import torch.nn as _nn
    import torch.optim as _optim
    import torch.utils.data as _data

    nn: types.ModuleType | None = _nn
    optim: types.ModuleType | None = _optim
    data: types.ModuleType | None = _data
except (ImportError, RuntimeError):
    nn = None
    optim = None
    data = None


def _scan_losses(include_nonpublic: bool) -> list[GhostRef]:
    """Scan `torch.nn` for loss classes.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found losses.

    """
    if not nn:
        return []
    found: list[GhostRef] = []
    for name, obj in get_all_members(nn):
        if inspect.isclass(obj) and name.endswith("Loss") and name != "_Loss":
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, nn.Module):
                    found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
            except TypeError:
                pass
    return found


def _scan_optimizers(include_nonpublic: bool) -> list[GhostRef]:
    """Scan `torch.optim` for optimizer classes.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found optimizers.

    """
    if not optim:
        return []
    found: list[GhostRef] = []
    for name, obj in get_all_members(optim):
        if inspect.isclass(obj) and name != "Optimizer":
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, optim.Optimizer):
                    found.append(GhostInspector.inspect(obj, f"torch.optim.{name}"))
            except TypeError:
                pass
    return found


def _get_activation_names() -> set[str]:
    """Retrieve all activation module names from torch.nn.modules.activation with fallback.

    Returns:
        Set of activation module class names.
    """
    names: set[str] = set()
    try:
        import torch.nn.modules.activation as act_mod

        nn_mod = ensure_present(nn, "torch.nn")
        for name, obj in get_all_members(act_mod):
            if inspect.isclass(obj) and issubclass(obj, nn_mod.Module):
                names.add(name)
    except (
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        LookupError,
        SyntaxError,
        AssertionError,
        SnapshotError,
    ):
        _LOGGER.debug("Failed to inspect torch.nn.modules.activation", exc_info=True)
    if not names:
        names = {
            "ReLU",
            "Sigmoid",
            "Tanh",
            "GELU",
            "SiLU",
            "Softmax",
            "LeakyReLU",
        }
    return names


def _scan_activations(include_nonpublic: bool) -> list[GhostRef]:
    """Scan `torch.nn` for activation functions.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found activations.

    """
    if not nn:
        return []
    found: list[GhostRef] = []
    target_activations = _get_activation_names()

    for name, obj in get_all_members(nn):
        if inspect.isclass(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, nn.Module) and name in target_activations:
                    found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
            except TypeError:
                pass
    return found


def _scan_layers(include_nonpublic: bool) -> list[GhostRef]:
    """Scan `torch.nn` for layer classes (excluding losses and activations).

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found layers.

    """
    if not nn:
        return []
    found: list[GhostRef] = []
    target_activations = _get_activation_names()

    for name, obj in get_all_members(nn):
        if inspect.isclass(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            try:
                if issubclass(obj, nn.Module):
                    if not name.endswith("Loss") and name not in target_activations:
                        found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
            except TypeError:
                pass
    return found


def _scan_schedulers(include_nonpublic: bool) -> list[GhostRef]:
    """Scan `torch.optim.lr_scheduler` for schedulers.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found schedulers.

    """
    if not isinstance(optim, TorchOptimWithSchedulerProtocol):
        return []
    found: list[GhostRef] = []
    for name, obj in get_all_members(optim.lr_scheduler):
        if inspect.isclass(obj) and name != "LRScheduler":
            if not include_nonpublic and name.startswith("_"):
                continue
            found.append(GhostInspector.inspect(obj, f"torch.optim.lr_scheduler.{name}"))
    return found


def _scan_initializers(include_nonpublic: bool) -> list[GhostRef]:
    """Scan `torch.nn.init` for initializers.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found initializers.

    """
    if not isinstance(nn, TorchNnWithInitProtocol):
        return []
    found: list[GhostRef] = []
    for name, obj in get_all_members(nn.init):
        if inspect.isfunction(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            found.append(GhostInspector.inspect(obj, f"torch.nn.init.{name}"))
    return found


def _scan_metrics(include_nonpublic: bool) -> list[GhostRef]:
    """Scan for metrics (delegates to torchmetrics if installed, otherwise returns empty).

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found metrics.

    """
    found: list[GhostRef] = []
    try:
        import torchmetrics

        for name, obj in get_all_members(torchmetrics):
            if inspect.isclass(obj) and (include_nonpublic or not name.startswith("_")):
                try:
                    found.append(GhostInspector.inspect(obj, f"torchmetrics.{name}"))
                except (
                    ImportError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    OSError,
                    RuntimeError,
                    LookupError,
                    SyntaxError,
                    AssertionError,
                ):
                    _LOGGER.debug("Failed to inspect torchmetrics.%s", name, exc_info=True)
    except ImportError:
        pass
    return found


def _scan_dataloaders(include_nonpublic: bool) -> list[GhostRef]:
    """Scan `torch.utils.data` for dataloader-related classes.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found dataloaders.

    """
    if not data:
        return []
    found: list[GhostRef] = []
    for name, obj in get_all_members(data):
        if inspect.isclass(obj):
            if not include_nonpublic and name.startswith("_"):
                continue
            found.append(GhostInspector.inspect(obj, f"torch.utils.data.{name}"))
    return found


def _collect_torch_namespace(
    ns_obj: types.ModuleType | type,
    api_prefix: str,
    include_nonpublic: bool,
) -> list[GhostRef]:
    """Collect callable non-class members from a PyTorch submodule namespace.

    Args:
        ns_obj: Target PyTorch submodule object.
        api_prefix: Qualified API path prefix (e.g. 'torch.linalg').
        include_nonpublic: Whether to include private underscore-prefixed symbols.

    Returns:
        List of inspected `GhostRef` objects.
    """
    collected: list[GhostRef] = []
    for name, obj in get_all_members(ns_obj):
        if (include_nonpublic or not name.startswith("_")) and callable(obj) and not inspect.isclass(obj):
            try:
                collected.append(GhostInspector.inspect(obj, f"{api_prefix}.{name}"))
            except (
                ImportError,
                AttributeError,
                TypeError,
                ValueError,
                OSError,
                RuntimeError,
                LookupError,
                SyntaxError,
                AssertionError,
            ):
                _LOGGER.debug("Failed to inspect %s.%s", api_prefix, name, exc_info=True)
    return collected


def _tag_in_place_tensor_method(ref: ExtendedGhostRef, name: str) -> None:
    """Attach in-place mutation domain metadata and environment tags to a `torch.Tensor` method ref.

    Args:
        ref: Inspected `ExtendedGhostRef` for the tensor method.
        name: Method name on `torch.Tensor`.
    """
    is_in_place = name.endswith("_") and not name.startswith("__")
    if ref.domain_metadata is None:
        ref.domain_metadata = {}
    ref.domain_metadata["is_in_place"] = is_in_place
    if is_in_place:
        if ref.environment_tags is None:
            ref.environment_tags = []
        if "in_place_mutation" not in ref.environment_tags:
            ref.environment_tags.append("in_place_mutation")


def _collect_torch_tensor_methods(tensor_cls: type, include_nonpublic: bool) -> list[GhostRef]:
    """Collect instance methods from `torch.Tensor` with in-place mutation metadata.

    Args:
        tensor_cls: The `torch.Tensor` class to introspect.
        include_nonpublic: Whether to include private methods starting with an underscore.

    Returns:
        List of `GhostRef` method descriptors for `torch.Tensor`.
    """
    collected: list[GhostRef] = []
    for name, obj in get_member_dict(tensor_cls).items():
        if (include_nonpublic or not name.startswith("_")) and callable(obj) and not inspect.isclass(obj):
            try:
                ref = GhostInspector.inspect(obj, f"torch.Tensor.{name}", kind="method")
                _tag_in_place_tensor_method(ref, name)
                collected.append(ref)
            except (
                AttributeError,
                TypeError,
                ValueError,
                KeyError,
                IndexError,
                SyntaxError,
                OSError,
                ImportError,
                RuntimeError,
            ):
                pass
    return collected


def _scan_array_api(include_nonpublic: bool) -> list[GhostRef]:
    """Scan top-level PyTorch module for array API functions and submodules.

    Exhaustively discovers:
        - All public operations in `torch.*`
        - `torch.linalg.*`
        - `torch.special.*`
        - `torch.fft.*`
        - `torch.nn.functional.*` (convolutions, pooling, linear, activations, attention, loss)

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found array API functions.
    """
    found: list[GhostRef] = []
    try:
        import torch

        # 1. Top-level array API functions
        found.extend(_collect_torch_namespace(torch, "torch", include_nonpublic))

        # 2-7. Submodule namespaces
        torch_members = get_member_dict(torch)
        for attr_name in ("linalg", "special", "fft", "autograd", "distributed"):
            submod = torch_members.get(attr_name)
            if isinstance(submod, (types.ModuleType, type)):
                found.extend(
                    _collect_torch_namespace(
                        submod,
                        f"torch.{attr_name}",
                        include_nonpublic,
                    )
                )

        if isinstance(torch, TorchModuleWithNnProtocol) and isinstance(torch.nn, TorchNnWithFunctionalProtocol):
            found.extend(
                _collect_torch_namespace(
                    torch.nn.functional,
                    "torch.nn.functional",
                    include_nonpublic,
                )
            )

        # 8. torch.Tensor instance methods
        if isinstance(torch, TorchModuleWithTensorProtocol):
            found.extend(_collect_torch_tensor_methods(torch.Tensor, include_nonpublic))

        # 9. torch.ops.aten operations
        if isinstance(torch, TorchModuleWithOpsProtocol) and isinstance(torch.ops, TorchOpsContainerProtocol):
            for name, obj in get_member_dict(torch.ops.aten).items():
                if not name.startswith("_") and callable(obj) and not inspect.isclass(obj):
                    try:
                        found.append(GhostInspector.inspect(obj, f"torch.ops.aten.{name}"))
                    except (
                        AttributeError,
                        TypeError,
                        ValueError,
                        KeyError,
                        IndexError,
                        SyntaxError,
                        OSError,
                        ImportError,
                        RuntimeError,
                    ):
                        pass

    except ImportError:
        pass

    return found


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Entrypoint to collect the PyTorch API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.

    """
    scanners: dict[SemanticTier, typing.Callable[[bool], list[GhostRef]]] = {
        SemanticTier.LOSS: _scan_losses,
        SemanticTier.OPTIMIZER: _scan_optimizers,
        SemanticTier.ACTIVATION: _scan_activations,
        SemanticTier.LAYER: _scan_layers,
        SemanticTier.SCHEDULER: _scan_schedulers,
        SemanticTier.INITIALIZER: _scan_initializers,
        SemanticTier.METRIC: _scan_metrics,
        SemanticTier.DATALOADER: _scan_dataloaders,
        SemanticTier.ARRAY_API: _scan_array_api,
    }
    scanner = scanners.get(category)
    if scanner is not None:
        return scanner(include_nonpublic)
    return []


FLOAT_COMPLEX_OPS: set[str] = {
    "cholesky",
    "inverse",
    "linalg_inv",
    "linalg_cholesky",
    "linalg_det",
    "linalg_slogdet",
    "linalg_eig",
    "linalg_eigh",
    "linalg_eigvals",
    "linalg_eigvalsh",
    "linalg_svd",
    "linalg_svdvals",
    "linalg_solve",
    "linalg_qr",
    "linalg_lu",
    "linalg_matrix_power",
    "linalg_pinv",
}

FLOAT_OPS: set[str] = {
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "log2",
    "log10",
    "sqrt",
    "rsqrt",
    "erf",
    "erfc",
    "sigmoid",
    "softmax",
    "log_softmax",
    "gelu",
    "silu",
}

RANK_RULES: dict[str, dict[str, int | str]] = {
    "mm": {"input": 2, "mat2": 2},
    "bmm": {"input": 3, "mat2": 3},
    "mv": {"input": 2, "mat": 2, "vec": 1},
    "dot": {"input": 1, "other": 1},
    "vdot": {"input": 1, "other": 1},
    "linalg_inv": {"input": ">=2"},
    "linalg_cholesky": {"input": ">=2"},
    "cholesky": {"input": ">=2"},
}


class AtenSchemaArg(pydantic.BaseModel):
    """Pydantic model for a single argument in an ATen or JIT operator schema."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    kind: str = "POSITIONAL_OR_KEYWORD"
    default: str | None = None
    annotation: str = "Tensor"
    is_out: bool = False
    allowed_dtypes: list[str] | None = None
    rank_constraint: str | None = None
    dtypes: list[str] | None = None
    rank: str | None = None


class AtenOpSchema(pydantic.BaseModel):
    """Pydantic model for an ATen or JIT operator overload schema."""

    model_config = pydantic.ConfigDict(extra="forbid")

    func: str | None = None
    name: str | None = None
    overload_name: str = "default"
    params: list[AtenSchemaArg] = pydantic.Field(default_factory=list)
    returns_type: str | None = "Tensor"
    is_inplace: bool = False
    is_out: bool = False


def infer_torch_dtype_and_rank(
    op_name: str, param_name: str, type_str: str = ""
) -> tuple[list[str] | None, str | None]:
    """Infer allowed PyTorch tensor dtypes and rank constraints for an operator parameter.

    Args:
        op_name: Name of the operator (e.g. 'linalg_inv', 'mm', 'add').
        param_name: Parameter name (e.g. 'A', 'input', 'mat2').
        type_str: Optional type annotation string.

    Returns:
        A tuple of (allowed_dtypes_list_or_None, rank_constraint_or_None).
    """
    clean_op = op_name.split(".")[-1].lower()
    clean_param = param_name.lower()

    dtypes: list[str] | None = None
    rank: str | None = None

    normalized_op = clean_op.replace("linalg.", "linalg_")
    if clean_op in FLOAT_COMPLEX_OPS or normalized_op in FLOAT_COMPLEX_OPS:
        dtypes = ["float32", "float64", "complex64", "complex128"]
    elif clean_op in FLOAT_OPS or normalized_op in FLOAT_OPS:
        dtypes = ["float16", "bfloat16", "float32", "float64"]
    elif "float" in type_str.lower():
        dtypes = ["float16", "bfloat16", "float32", "float64"]
    elif "int" in type_str.lower() and "tensor" not in type_str.lower():
        dtypes = ["int8", "int16", "int32", "int64"]

    rank_dict = RANK_RULES.get(clean_op) or RANK_RULES.get(normalized_op)
    if rank_dict:
        raw_rank: int | str | None = None
        if clean_param in rank_dict:
            raw_rank = rank_dict[clean_param]
        elif "input" in rank_dict and clean_param in ("self", "a"):
            raw_rank = rank_dict["input"]

        if raw_rank is not None:
            if isinstance(raw_rank, int):
                rank = f"=={raw_rank}"
            else:
                rank = str(raw_rank)

    return dtypes, rank


def parse_native_functions_yaml(
    yaml_content_or_path: str,
) -> dict[str, list[AtenOpSchema]]:
    """Parse PyTorch native_functions.yaml into structured operator definitions and overloads.

    Args:
        yaml_content_or_path: File path to native_functions.yaml or raw YAML string.

    Returns:
        Dictionary mapping base operator names to a list of `AtenOpSchema` objects.
    """
    import importlib

    yaml = importlib.import_module("yaml")

    content = yaml_content_or_path
    if os.path.exists(yaml_content_or_path):
        with open(yaml_content_or_path, encoding="utf-8") as f:
            content = f.read()

    data = yaml.safe_load(content)
    if not isinstance(data, list):
        return {}

    ops: dict[str, list[AtenOpSchema]] = {}
    sig_pattern = re.compile(r"^([a-zA-Z0-9_]+)(?:\.([a-zA-Z0-9_]+))?\((.*)\)\s*->\s*(.*)$")

    for item in data:
        if not isinstance(item, dict) or "func" not in item:
            continue
        func_str = str(item["func"]).strip()
        m = sig_pattern.match(func_str)
        if not m:
            continue

        base_op, overload_name, args_str, returns_type = m.groups()
        overload_name = overload_name or "default"
        returns_type = returns_type.strip()

        params: list[AtenSchemaArg] = []
        is_kwarg_only = False
        raw_args = [a.strip() for a in args_str.split(",") if a.strip()]

        for raw_arg in raw_args:
            if raw_arg == "*":
                is_kwarg_only = True
                continue

            parts = raw_arg.split("=")
            default_val = parts[1].strip() if len(parts) > 1 else None
            decl = parts[0].strip()

            decl_parts = decl.split()
            if len(decl_parts) >= 2:
                arg_type = decl_parts[0]
                arg_name = decl_parts[1]
            else:
                arg_type = "Tensor"
                arg_name = decl_parts[0]

            is_out = arg_name == "out" or "(a!)" in arg_type
            clean_type = re.sub(r"\(.*?\)", "", arg_type).strip()

            dtypes, rank = infer_torch_dtype_and_rank(base_op, arg_name, clean_type)

            kind = "KEYWORD_ONLY" if is_kwarg_only else "POSITIONAL_OR_KEYWORD"
            norm_name = "input" if arg_name == "self" else arg_name

            params.append(
                AtenSchemaArg(
                    name=norm_name,
                    kind=kind,
                    default=default_val,
                    annotation=clean_type,
                    is_out=is_out,
                    allowed_dtypes=dtypes,
                    rank_constraint=rank,
                    dtypes=dtypes,
                    rank=rank,
                )
            )

        overload_info = AtenOpSchema(
            func=func_str,
            name=base_op,
            overload_name=overload_name,
            params=params,
            returns_type=returns_type,
            is_inplace=base_op.endswith("_"),
            is_out=any(p.is_out for p in params),
        )

        if base_op not in ops:
            ops[base_op] = []
        ops[base_op].append(overload_info)

    return ops


class _JitTypeProto(typing.Protocol):
    """Protocol representing a TorchScript JIT type object."""

    def __str__(self) -> str:
        """Return string representation of the JIT type.

        Returns:
            String representation.
        """
        ...


@typing.runtime_checkable
class TorchJitArgProtocol(typing.Protocol):
    """Protocol representing a TorchScript JIT schema argument."""

    @property
    def name(self) -> str:
        """Return argument name.

        Returns:
            Argument name string.
        """
        ...

    @property
    def kwarg_only(self) -> bool:
        """Return whether argument is keyword-only.

        Returns:
            True if keyword-only, False otherwise.
        """
        ...

    @property
    def type(self) -> _JitTypeProto:
        """Return argument JIT type object.

        Returns:
            JIT type object.
        """
        ...

    @property
    def default_value(self) -> str | int | float | bool | None:
        """Return argument default value.

        Returns:
            Default value or None.
        """
        ...

    @property
    def is_out(self) -> bool:
        """Return whether argument is an output tensor.

        Returns:
            True if output tensor, False otherwise.
        """
        ...

    def has_default_value(self) -> bool:
        """Check whether the argument defines a default value.

        Returns:
            True if a default value exists, False otherwise.
        """
        ...


@typing.runtime_checkable
class TorchJitReturnProtocol(typing.Protocol):
    """Protocol representing a TorchScript JIT schema return entry."""

    @property
    def type(self) -> _JitTypeProto:
        """Return return value JIT type object.

        Returns:
            JIT type object.
        """
        ...


@typing.runtime_checkable
class TorchJitSchemaProtocol(typing.Protocol):
    """Protocol representing a TorchScript JIT operator schema."""

    @property
    def name(self) -> str:
        """Return operator name.

        Returns:
            Operator name string.
        """
        ...

    @property
    def overload_name(self) -> str:
        """Return operator overload name.

        Returns:
            Overload name string.
        """
        ...

    @property
    def arguments(self) -> typing.Sequence[TorchJitArgProtocol]:
        """Return operator arguments.

        Returns:
            Sequence of argument objects.
        """
        ...

    @property
    def returns(self) -> typing.Sequence[TorchJitReturnProtocol]:
        """Return operator return specifications.

        Returns:
            Sequence of return objects.
        """
        ...


@typing.runtime_checkable
class TorchAtenOverloadVariantProtocol(typing.Protocol):
    """Protocol representing a resolved ATen overload object carrying `._schema`."""

    _schema: TorchJitSchemaProtocol | None


@typing.runtime_checkable
class TorchAtenOverloadPacketProtocol(typing.Protocol):
    """Protocol representing a `torch.ops.aten.<op>` overload packet."""

    def overloads(self) -> typing.Sequence[str]:
        """Return overload variant names.

        Returns:
            Sequence of overload name strings.
        """
        ...


@typing.runtime_checkable
class TorchCModuleProtocol(typing.Protocol):
    """Protocol representing `torch._C` with JIT schema lookup."""

    def _jit_get_all_schemas(self) -> typing.Sequence[TorchJitSchemaProtocol]:
        """Return all registered JIT operator schemas.

        Returns:
            Sequence of JIT schema objects.
        """
        ...


@typing.runtime_checkable
class TorchModuleWithCProtocol(typing.Protocol):
    """Protocol representing the `torch` module exposing `._C`."""

    _C: TorchCModuleProtocol


_JIT_SCHEMAS_CACHE: dict[str, list[TorchJitSchemaProtocol]] | None = None


def get_jit_schemas_for_op(op_name: str) -> list[AtenOpSchema]:
    """Retrieve structured schemas from torch._C._jit_get_all_schemas() for an operator.

    Args:
        op_name: Operator name (e.g. 'add', 'relu', 'matmul', 'aten::add').

    Returns:
        List of `AtenOpSchema` objects.
    """
    global _JIT_SCHEMAS_CACHE
    try:
        import torch

        if not isinstance(torch, TorchModuleWithCProtocol) or not isinstance(torch._C, TorchCModuleProtocol):
            return []
    except (
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        LookupError,
        SyntaxError,
        AssertionError,
    ):
        _LOGGER.debug("Failed to resolve torch._C for JIT schemas", exc_info=True)
        return []

    clean_op = op_name.split(".")[-1]

    if _JIT_SCHEMAS_CACHE is None:
        cache: dict[str, list[TorchJitSchemaProtocol]] = {}
        for s in torch._C._jit_get_all_schemas():
            s_name = str(s.name)
            s_short = s_name.split("::")[-1]
            cache.setdefault(s_name, []).append(s)
            cache.setdefault(s_short, []).append(s)
        _JIT_SCHEMAS_CACHE = cache

    schemas = _JIT_SCHEMAS_CACHE.get(clean_op) or _JIT_SCHEMAS_CACHE.get(f"aten::{clean_op}") or []
    if not schemas:
        return []

    results: list[AtenOpSchema] = []
    for schema in schemas:
        params: list[AtenSchemaArg] = []
        raw_arguments: list[TorchJitArgProtocol] = list(schema.arguments)
        for arg in raw_arguments:
            p_kind = "KEYWORD_ONLY" if bool(arg.kwarg_only) else "POSITIONAL_OR_KEYWORD"
            raw_arg_name = str(arg.name)
            p_name = "input" if raw_arg_name == "self" else raw_arg_name
            def_val = arg.default_value
            p_default = str(def_val) if arg.has_default_value() else None
            p_anno = str(arg.type)
            is_out = bool(arg.is_out)

            dtypes, rank = infer_torch_dtype_and_rank(clean_op, p_name, p_anno)

            params.append(
                AtenSchemaArg(
                    name=p_name,
                    kind=p_kind,
                    default=p_default,
                    annotation=p_anno,
                    is_out=is_out,
                    allowed_dtypes=dtypes,
                    rank_constraint=rank,
                    dtypes=dtypes,
                    rank=rank,
                )
            )

        raw_returns: list[TorchJitReturnProtocol] = list(schema.returns)
        ret_type = str(raw_returns[0].type) if raw_returns else "Tensor"
        ov_name = str(schema.overload_name or "default")
        results.append(
            AtenOpSchema(
                overload_name=ov_name,
                params=params,
                returns_type=ret_type,
                is_inplace=clean_op.endswith("_"),
                is_out=any(p.is_out for p in params),
            )
        )

    return results


def get_aten_op_schema(op_name: str) -> list[AtenOpSchema] | None:
    """Retrieve structured schema and overloads for an ATen operator from torch.ops.aten or JIT registry.

    Args:
        op_name: Operator name (e.g. 'add', 'relu', 'matmul', 'add_').

    Returns:
        List of `AtenOpSchema` objects if found, otherwise None.
    """
    try:
        import torch
    except ImportError:
        return None

    clean_op = op_name.split(".")[-1]
    aten_members = get_member_dict(torch.ops.aten)
    aten_op = aten_members.get(clean_op)
    results: list[AtenOpSchema] = []

    if aten_op is not None and isinstance(aten_op, TorchAtenOverloadPacketProtocol):
        ov_members: typing.Mapping[str, TorchAtenOverloadVariantProtocol | LivePythonTarget] = get_member_dict(aten_op)
        for ov_name in aten_op.overloads():
            ov = ov_members.get(ov_name)
            schema = ov._schema if isinstance(ov, TorchAtenOverloadVariantProtocol) else None
            if not schema:
                continue

            params: list[AtenSchemaArg] = []
            for arg in schema.arguments:
                p_kind = "KEYWORD_ONLY" if arg.kwarg_only else "POSITIONAL_OR_KEYWORD"
                p_name = "input" if arg.name == "self" else arg.name
                p_default = str(arg.default_value) if arg.has_default_value() else None
                p_anno = str(arg.type)
                is_out = bool(arg.is_out)

                dtypes, rank = infer_torch_dtype_and_rank(clean_op, p_name, p_anno)

                params.append(
                    AtenSchemaArg(
                        name=p_name,
                        kind=p_kind,
                        default=p_default,
                        annotation=p_anno,
                        is_out=is_out,
                        allowed_dtypes=dtypes,
                        rank_constraint=rank,
                        dtypes=dtypes,
                        rank=rank,
                    )
                )

            ret_type = str(schema.returns[0].type) if schema.returns else "Tensor"
            results.append(
                AtenOpSchema(
                    overload_name=str(ov_name),
                    params=params,
                    returns_type=ret_type,
                    is_inplace=clean_op.endswith("_"),
                    is_out=any(p.is_out for p in params),
                )
            )

    if results:
        return results

    jit_results = get_jit_schemas_for_op(clean_op)
    return jit_results if jit_results else None


def _coerce_aten_schema(
    ov: AtenOpSchema | typing.Mapping[str, JsonValue],
) -> AtenOpSchema:
    """Coerce an `AtenOpSchema` or dictionary into `AtenOpSchema`.

    Args:
        ov: Overload schema model or mapping.

    Returns:
        Validated `AtenOpSchema`.
    """
    if isinstance(ov, AtenOpSchema):
        return ov
    return AtenOpSchema.model_validate(dict(ov))


def extract_aten_c_extension_signature(
    target: LivePythonTarget, name: str | None = None, is_method: bool = False
) -> CExtensionSignature | None:
    """Extract a CExtensionSignature with exhaustive overloads from torch.ops.aten.

    Args:
        target: Target callable or operator.
        name: Optional qualified name or attribute path.
        is_method: Whether the target is an instance method (stripping receiver 'self').

    Returns:
        CExtensionSignature with primary parameters and overloads if found, else None.
    """
    raw_t_name = get_member_dict(target).get("__name__")
    target_name = str(raw_t_name) if raw_t_name is not None else ""
    op_name = str(name or target_name or "")
    raw_schemas = get_aten_op_schema(op_name)
    if not raw_schemas:
        return None

    schemas = [_coerce_aten_schema(ov) for ov in raw_schemas]
    is_method_call = is_method or (name is not None and ".Tensor." in name)

    primary_idx = 0
    for idx, ov in enumerate(schemas):
        if ov.overload_name in ("Tensor", "default"):
            primary_idx = idx
            break

    primary = schemas[primary_idx]
    secondary = [ov for idx, ov in enumerate(schemas) if idx != primary_idx]

    def to_param_tuples(
        p_list: list[AtenSchemaArg],
    ) -> list[tuple[str, str, str | None, str | None]]:
        """Convert parameter model list to CExtensionSignature tuple format.

        Args:
            p_list: List of `AtenSchemaArg` objects.

        Returns:
            List of (name, kind, default, annotation) tuples.
        """
        filtered = p_list
        if is_method_call and filtered and filtered[0].name in ("input", "self"):
            filtered = filtered[1:]
        return [(p.name, p.kind, p.default, p.annotation) for p in filtered]

    overload_objs = [
        CExtensionSignature(
            params=to_param_tuples(ov.params),
            returns_type=ov.returns_type,
            overloads=[],
        )
        for ov in secondary
    ]

    return CExtensionSignature(
        params=to_param_tuples(primary.params),
        returns_type=primary.returns_type,
        overloads=overload_objs,
    )
