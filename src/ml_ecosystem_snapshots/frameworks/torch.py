"""PyTorch API Snapshot Extractor.

Provides functions to dynamically introspect the PyTorch library and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

from __future__ import annotations

import inspect
import os
import re
import types
import typing
import warnings

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier
import pydantic

from ml_ecosystem_snapshots.c_extension_sig import CExtensionSignature
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.models import JsonValue
from ml_ecosystem_snapshots.utils import get_all_members

warnings.filterwarnings(
    "ignore",
    message=".*Failed to initialize NumPy.*",
    category=UserWarning,
)

try:
    import torch  # noqa: F401
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

        assert nn is not None
        for name, obj in get_all_members(act_mod):
            if inspect.isclass(obj) and issubclass(obj, nn.Module):
                names.add(name)
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
        Exception,
    ):
        pass
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
    if not optim or not hasattr(optim, "lr_scheduler"):
        return []
    found: list[GhostRef] = []
    for name, obj in get_all_members(optim.lr_scheduler):
        if inspect.isclass(obj) and name != "LRScheduler":
            if not include_nonpublic and name.startswith("_"):
                continue
            found.append(
                GhostInspector.inspect(obj, f"torch.optim.lr_scheduler.{name}")
            )
    return found


def _scan_initializers(include_nonpublic: bool) -> list[GhostRef]:
    """Scan `torch.nn.init` for initializers.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found initializers.

    """
    if not nn or not hasattr(nn, "init"):
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
                    AttributeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    IndexError,
                    SyntaxError,
                    OSError,
                    ImportError,
                    RuntimeError,
                    Exception,
                ):
                    pass
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
        if (
            (include_nonpublic or not name.startswith("_"))
            and callable(obj)
            and not inspect.isclass(obj)
        ):
            try:
                collected.append(GhostInspector.inspect(obj, f"{api_prefix}.{name}"))
            except Exception:
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
        for name, obj in get_all_members(torch):
            if not include_nonpublic and name.startswith("_"):
                continue
            if callable(obj) and not inspect.isclass(obj):
                try:
                    found.append(GhostInspector.inspect(obj, f"torch.{name}"))
                except Exception:
                    pass

        # 2-7. Submodule namespaces
        for attr_name in ("linalg", "special", "fft", "autograd", "distributed"):
            if hasattr(torch, attr_name):
                found.extend(
                    _collect_torch_namespace(
                        getattr(torch, attr_name),
                        f"torch.{attr_name}",
                        include_nonpublic,
                    )
                )

        if hasattr(torch, "nn") and hasattr(torch.nn, "functional"):
            found.extend(
                _collect_torch_namespace(
                    torch.nn.functional,
                    "torch.nn.functional",
                    include_nonpublic,
                )
            )

        # 8. torch.Tensor instance methods
        if hasattr(torch, "Tensor"):
            for name, obj in inspect.getmembers(torch.Tensor):
                if not include_nonpublic and name.startswith("_"):
                    continue
                if callable(obj) and not inspect.isclass(obj):
                    try:
                        ref = GhostInspector.inspect(
                            obj, f"torch.Tensor.{name}", kind="method"
                        )
                        is_in_place = name.endswith("_") and not name.startswith("__")
                        if ref.domain_metadata is None:
                            ref.domain_metadata = {}
                        ref.domain_metadata["is_in_place"] = is_in_place
                        if is_in_place:
                            if ref.environment_tags is None:
                                ref.environment_tags = []
                            if "in_place_mutation" not in ref.environment_tags:
                                ref.environment_tags.append("in_place_mutation")
                        found.append(ref)
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
                        Exception,
                    ):
                        pass

        # 9. torch.ops.aten operations
        if hasattr(torch, "ops") and hasattr(torch.ops, "aten"):
            for name, obj in inspect.getmembers(torch.ops.aten):
                if not name.startswith("_"):
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            found.append(
                                GhostInspector.inspect(obj, f"torch.ops.aten.{name}")
                            )
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
                            Exception,
                        ):
                            pass

    except ImportError:
        pass

    return found


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Entrypoint to collect the PyTorch API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.

    """
    if category == SemanticTier.LOSS:
        return _scan_losses(include_nonpublic)
    elif category == SemanticTier.OPTIMIZER:
        return _scan_optimizers(include_nonpublic)
    elif category == SemanticTier.ACTIVATION:
        return _scan_activations(include_nonpublic)
    elif category == SemanticTier.LAYER:
        return _scan_layers(include_nonpublic)
    elif category == SemanticTier.SCHEDULER:
        return _scan_schedulers(include_nonpublic)
    elif category == SemanticTier.INITIALIZER:
        return _scan_initializers(include_nonpublic)
    elif category == SemanticTier.METRIC:
        return _scan_metrics(include_nonpublic)
    elif category == SemanticTier.DATALOADER:
        return _scan_dataloaders(include_nonpublic)
    elif category == SemanticTier.ARRAY_API:
        return _scan_array_api(include_nonpublic)
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

    @typing.overload
    def __getitem__(self, key: typing.Literal["name", "kind", "annotation"]) -> str: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["is_out"]) -> bool: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["default", "rank_constraint", "rank"]
    ) -> str | None: ...

    @typing.overload
    def __getitem__(self, key: str) -> JsonValue: ...

    def __getitem__(self, key: str) -> str | bool | None | JsonValue:
        """Retrieve an argument field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        d: dict[str, JsonValue] = self.model_dump()
        return d[key]


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

    @typing.overload
    def __getitem__(self, key: typing.Literal["overload_name"]) -> str: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["is_inplace", "is_out"]) -> bool: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["params"]) -> list[AtenSchemaArg]: ...

    @typing.overload
    def __getitem__(self, key: str) -> str | bool | list[AtenSchemaArg] | JsonValue: ...

    def __getitem__(self, key: str) -> str | bool | list[AtenSchemaArg] | JsonValue:
        """Retrieve an overload schema field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        if key == "params":
            return self.params
        d: dict[str, JsonValue] = self.model_dump()
        return d[key]


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
        with open(yaml_content_or_path, "r", encoding="utf-8") as f:
            content = f.read()

    data = yaml.safe_load(content)
    if not isinstance(data, list):
        return {}

    ops: dict[str, list[AtenOpSchema]] = {}
    sig_pattern = re.compile(
        r"^([a-zA-Z0-9_]+)(?:\.([a-zA-Z0-9_]+))?\((.*)\)\s*->\s*(.*)$"
    )

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
                arg_type = "Any"
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


_JIT_SCHEMAS_CACHE: dict[str, list[object]] | None = None


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

        if not hasattr(torch, "_C") or not hasattr(torch._C, "_jit_get_all_schemas"):
            return []
    except Exception:
        return []

    clean_op = op_name.split(".")[-1]

    if _JIT_SCHEMAS_CACHE is None:
        cache: dict[str, list[object]] = {}
        for s in torch._C._jit_get_all_schemas():
            s_name = str(getattr(s, "name", ""))
            s_short = s_name.split("::")[-1]
            cache.setdefault(s_name, []).append(s)
            cache.setdefault(s_short, []).append(s)
        _JIT_SCHEMAS_CACHE = cache

    schemas = (
        _JIT_SCHEMAS_CACHE.get(clean_op)
        or _JIT_SCHEMAS_CACHE.get(f"aten::{clean_op}")
        or []
    )
    if not schemas:
        return []

    results: list[AtenOpSchema] = []
    for schema in schemas:
        params: list[AtenSchemaArg] = []
        raw_arguments: list[object] = list(getattr(schema, "arguments", []))
        for arg in raw_arguments:
            p_kind = (
                "KEYWORD_ONLY"
                if bool(getattr(arg, "kwarg_only", False))
                else "POSITIONAL_OR_KEYWORD"
            )
            raw_arg_name = str(getattr(arg, "name", ""))
            p_name = "input" if raw_arg_name == "self" else raw_arg_name
            has_def_fn = getattr(arg, "has_default_value", None)
            def_val = getattr(arg, "default_value", None)
            p_default = (
                str(def_val)
                if callable(has_def_fn) and bool(has_def_fn())
                else (str(def_val) if def_val is not None else None)
            )
            p_anno = str(getattr(arg, "type", "Tensor"))
            is_out = bool(getattr(arg, "is_out", False))

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

        raw_returns: list[object] = list(getattr(schema, "returns", []))
        ret_type = (
            str(getattr(raw_returns[0], "type", "Tensor")) if raw_returns else "Tensor"
        )
        ov_name = str(getattr(schema, "overload_name", "default") or "default")
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
    aten_op = getattr(torch.ops.aten, clean_op, None)
    results: list[AtenOpSchema] = []

    if aten_op and hasattr(aten_op, "overloads"):
        for ov_name in aten_op.overloads():
            ov = getattr(aten_op, ov_name)
            schema = getattr(ov, "_schema", None)
            if not schema:
                continue

            params: list[AtenSchemaArg] = []
            for arg in schema.arguments:
                p_kind = "KEYWORD_ONLY" if arg.kwarg_only else "POSITIONAL_OR_KEYWORD"
                p_name = "input" if arg.name == "self" else arg.name
                p_default = str(arg.default_value) if arg.has_default_value() else None
                p_anno = str(arg.type)
                is_out = bool(getattr(arg, "is_out", False))

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
    target: object, name: str | None = None, is_method: bool = False
) -> CExtensionSignature | None:
    """Extract a CExtensionSignature with exhaustive overloads from torch.ops.aten.

    Args:
        target: Target callable or operator.
        name: Optional qualified name or attribute path.
        is_method: Whether the target is an instance method (stripping receiver 'self').

    Returns:
        CExtensionSignature with primary parameters and overloads if found, else None.
    """
    op_name = str(name or getattr(target, "__name__", "") or "")
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
