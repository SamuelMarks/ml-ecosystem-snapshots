"""Declarative domain operation CLI command handlers (`check-onnx`, `check-metal`, `check-wasm`, etc.)."""

from __future__ import annotations

import argparse
import inspect
import sys
import typing

import typing_extensions

from ml_ecosystem_snapshots.mcp.models import DomainOpCheckResult
from ml_ecosystem_snapshots.models import JsonValue


def _read_namespace_attr(args: argparse.Namespace, attr_name: str, default: JsonValue = None) -> JsonValue:
    """Read an optional attribute from an `argparse.Namespace` via member lookup.

    Args:
        args: Parsed CLI `Namespace` instance.
        attr_name: Attribute name to look up.
        default: Default value when absent.

    Returns:
        Attribute value or `default`.
    """
    raw = dict(inspect.getmembers(args)).get(attr_name, default)
    assert isinstance(raw, (str, int, float, bool, list, dict)) or raw is None
    return raw


class _DomainOpNameCheckProto(typing.Protocol):
    """Protocol for MCP domain op check functions keyed by `op_name`."""

    def __call__(
        self,
        op_name: str,
        inputs_count: int | None = None,
        attributes: list[str] | None = None,
    ) -> DomainOpCheckResult | typing.Mapping[str, JsonValue]:
        """Validate a domain operation by `op_name`.

        Args:
            op_name: Operation name.
            inputs_count: Optional expected input count.
            attributes: Optional list of attribute names.

        Returns:
            Check result model or dictionary.
        """
        ...


class _DomainMnemonicCheckProto(typing.Protocol):
    """Protocol for MCP domain op check functions keyed by `mnemonic`."""

    def __call__(
        self,
        mnemonic: str,
        operands_count: int | None = None,
        attributes: list[str] | None = None,
    ) -> DomainOpCheckResult | typing.Mapping[str, JsonValue]:
        """Validate a domain instruction by `mnemonic`.

        Args:
            mnemonic: Instruction mnemonic.
            operands_count: Optional expected operand count.
            attributes: Optional list of attribute names.

        Returns:
            Check result model or dictionary.
        """
        ...


class _DomainOnnxCheckProto(typing.Protocol):
    """Protocol for ONNX domain op check function."""

    def __call__(
        self,
        op_name: str,
        domain: str = "",
        inputs_count: int | None = None,
        attributes: list[str] | None = None,
    ) -> DomainOpCheckResult | typing.Mapping[str, JsonValue]:
        """Validate an ONNX operation.

        Args:
            op_name: Operation name.
            domain: Operator domain.
            inputs_count: Optional expected input count.
            attributes: Optional list of attribute names.

        Returns:
            Check result model or dictionary.
        """
        ...


class _DomainMetalCheckProto(typing.Protocol):
    """Protocol for Metal domain op check function."""

    def __call__(
        self,
        op_name: str,
        address_space: str | None = None,
        inputs_count: int | None = None,
        attributes: list[str] | None = None,
    ) -> DomainOpCheckResult | typing.Mapping[str, JsonValue]:
        """Validate a Metal operation.

        Args:
            op_name: Operation name.
            address_space: Optional address space.
            inputs_count: Optional expected input count.
            attributes: Optional list of attribute names.

        Returns:
            Check result model or dictionary.
        """
        ...


DomainCheckCallable: typing_extensions.TypeAlias = "_DomainOpNameCheckProto | _DomainMnemonicCheckProto | _DomainOnnxCheckProto | _DomainMetalCheckProto"


def _is_onnx_check_fn(
    check_fn: DomainCheckCallable,
    extra_kwargs: dict[str, JsonValue] | None,
) -> typing_extensions.TypeGuard[_DomainOnnxCheckProto]:
    """Narrow `check_fn` to `_DomainOnnxCheckProto` when `domain` extra kwargs are supplied.

    Args:
        check_fn: Domain check callable.
        extra_kwargs: Optional extra keyword arguments dictionary.

    Returns:
        True if `extra_kwargs` contains a `'domain'` key.
    """
    return bool(extra_kwargs is not None and "domain" in extra_kwargs)


def _is_metal_check_fn(
    check_fn: DomainCheckCallable,
    extra_kwargs: dict[str, JsonValue] | None,
) -> typing_extensions.TypeGuard[_DomainMetalCheckProto]:
    """Narrow `check_fn` to `_DomainMetalCheckProto` when `address_space` extra kwargs are supplied.

    Args:
        check_fn: Domain check callable.
        extra_kwargs: Optional extra keyword arguments dictionary.

    Returns:
        True if `extra_kwargs` contains an `'address_space'` key.
    """
    return bool(extra_kwargs is not None and "address_space" in extra_kwargs)


def _is_mnemonic_check_fn(
    check_fn: DomainCheckCallable,
    op_attr: str,
) -> typing_extensions.TypeGuard[_DomainMnemonicCheckProto]:
    """Narrow `check_fn` to `_DomainMnemonicCheckProto` when `op_attr` is `'mnemonic'`.

    Args:
        check_fn: Domain check callable.
        op_attr: Target operation attribute name.

    Returns:
        True if `op_attr == 'mnemonic'`.
    """
    return op_attr == "mnemonic"


def _is_op_name_check_fn(
    check_fn: DomainCheckCallable,
    op_attr: str,
) -> typing_extensions.TypeGuard[_DomainOpNameCheckProto]:
    """Narrow `check_fn` to `_DomainOpNameCheckProto` for standard domain operations.

    Args:
        check_fn: Domain check callable.
        op_attr: Target operation attribute name.

    Returns:
        True if `op_attr != 'mnemonic'`.
    """
    return op_attr != "mnemonic"


def _run_domain_op_check_cli(
    args: argparse.Namespace,
    check_fn: DomainCheckCallable,
    label: str,
    op_attr: str = "op_name",
    count_attr: str = "inputs_count",
    count_param: str = "inputs_count",
    invalid_format: str = "Invalid:",
    extra_kwargs: dict[str, JsonValue] | None = None,
) -> None:
    """Execute a domain op verification check and format CLI stdout/exit codes.

    Args:
        args: Parsed CLI `Namespace` object.
        check_fn: Underlying MCP verification callable.
        label: Human-readable domain operation prefix (e.g. 'ONNX Operation', 'WASM Instruction').
        op_attr: Attribute name on `args` holding the operation or mnemonic string.
        count_attr: Attribute name on `args` holding the expected operand/input count.
        count_param: Keyword parameter name accepted by `check_fn` for operand/input count.
        invalid_format: Suffix string when reporting an invalid operation ('Invalid:' or 'is invalid:').
        extra_kwargs: Optional additional keyword arguments to forward to `check_fn`.
    """
    del count_param
    raw_attrs = _read_namespace_attr(args, "attributes")
    attrs: list[str] | None = [a.strip() for a in str(raw_attrs).split(",") if a.strip()] if raw_attrs else None
    target = str(_read_namespace_attr(args, op_attr, ""))
    raw_count = _read_namespace_attr(args, count_attr)
    count_val: int | None = int(raw_count) if isinstance(raw_count, int) else None

    res: DomainOpCheckResult | typing.Mapping[str, JsonValue]
    if extra_kwargs and _is_onnx_check_fn(check_fn, extra_kwargs):
        res = check_fn(
            op_name=target,
            domain=str(extra_kwargs["domain"] or ""),
            inputs_count=count_val,
            attributes=attrs,
        )
    elif extra_kwargs and _is_metal_check_fn(check_fn, extra_kwargs):
        addr_val = extra_kwargs["address_space"]
        res = check_fn(
            op_name=target,
            address_space=str(addr_val) if addr_val is not None else None,
            inputs_count=count_val,
            attributes=attrs,
        )
    elif _is_mnemonic_check_fn(check_fn, op_attr):
        res = check_fn(
            mnemonic=target,
            operands_count=count_val,
            attributes=attrs,
        )
    else:
        assert _is_op_name_check_fn(check_fn, op_attr)
        res = check_fn(
            op_name=target,
            inputs_count=count_val,
            attributes=attrs,
        )
    is_valid = res.is_valid if isinstance(res, DomainOpCheckResult) else bool(res.get("is_valid"))
    if not is_valid:
        print(f"{label} '{target}' {invalid_format}")
        errs = res.errors if isinstance(res, DomainOpCheckResult) else res.get("errors", [])
        assert isinstance(errs, list)
        for err in errs:
            print(f"  - {err}")
        sys.exit(1)

    exp_inputs = res.expected_inputs if isinstance(res, DomainOpCheckResult) else res.get("expected_inputs", [])
    exp_attrs = res.expected_attributes if isinstance(res, DomainOpCheckResult) else res.get("expected_attributes", [])
    inp_len = len(exp_inputs) if isinstance(exp_inputs, list) else 0
    attr_strs = [str(a) for a in exp_attrs] if isinstance(exp_attrs, list) else []
    print(f"{label} '{target}' is valid.")
    print(f"  Inputs:     {inp_len}")
    print(f"  Attributes: {', '.join(attr_strs)}")


def cmd_check_onnx(args: argparse.Namespace) -> None:
    """Validate ONNX operation against specification definitions.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_onnx_op

    _run_domain_op_check_cli(
        args,
        check_onnx_op,
        "ONNX Operation",
        extra_kwargs={"domain": str(_read_namespace_attr(args, "domain", "") or "")},
    )


def cmd_check_metal(args: argparse.Namespace) -> None:
    """Validate Apple Metal MSL compute operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_metal_op

    raw_addr = _read_namespace_attr(args, "address_space")
    _run_domain_op_check_cli(
        args,
        check_metal_op,
        "Metal Operation",
        extra_kwargs={"address_space": str(raw_addr) if raw_addr is not None else None},
    )


def cmd_check_wasm(args: argparse.Namespace) -> None:
    """Validate WebAssembly 2.0 / SIMD instruction against specification.

    Args:
        args: Parsed command line arguments.
    """
    from ml_ecosystem_snapshots.mcp_server import check_wasm_instruction

    _run_domain_op_check_cli(
        args,
        check_wasm_instruction,
        "WASM Instruction",
        op_attr="mnemonic",
        count_attr="operands_count",
        count_param="operands_count",
    )


def _resolve_simple_domain_check_fn(fn_name: str) -> _DomainOpNameCheckProto:
    """Resolve an MCP domain verifier callable by name without reflection.

    Args:
        fn_name: Name of the verifier function exported by `ml_ecosystem_snapshots.mcp_server`.

    Returns:
        Typed domain op check callable.
    """
    import ml_ecosystem_snapshots.mcp_server as _mcp_srv

    dispatch: dict[str, _DomainOpNameCheckProto] = {
        "check_webgl_op": _mcp_srv.check_webgl_op,
        "check_cpp_op": _mcp_srv.check_cpp_op,
        "check_array_api_op": _mcp_srv.check_array_api_op,
        "check_scipy_op": _mcp_srv.check_scipy_op,
        "check_torchvision_op": _mcp_srv.check_torchvision_op,
        "check_torchaudio_op": _mcp_srv.check_torchaudio_op,
        "check_safetensors_op": _mcp_srv.check_safetensors_op,
        "check_aten_op": _mcp_srv.check_aten_op,
        "check_nccl_op": _mcp_srv.check_nccl_op,
        "check_flash_attention_op": _mcp_srv.check_flash_attention_op,
        "check_chex_op": _mcp_srv.check_chex_op,
        "check_grain_op": _mcp_srv.check_grain_op,
        "check_orbax_op": _mcp_srv.check_orbax_op,
        "check_optax_op": _mcp_srv.check_optax_op,
    }
    return dispatch[fn_name]


def _make_simple_domain_check_cmd(
    fn_name: str,
    label: str,
    invalid_format: str = "is invalid:",
) -> typing.Callable[[argparse.Namespace], None]:
    """Create a CLI command handler for a standard MCP domain operation verifier.

    Args:
        fn_name: Name of the verifier function exported by `ml_ecosystem_snapshots.mcp_server`.
        label: Human-readable domain operation prefix.
        invalid_format: Suffix string when reporting an invalid operation.

    Returns:
        CLI handler callable accepting `argparse.Namespace`.
    """

    def _handler(args: argparse.Namespace) -> None:
        """Validate domain operation against specification.

        Args:
            args: Parsed command line arguments.
        """
        check_fn = _resolve_simple_domain_check_fn(fn_name)
        _run_domain_op_check_cli(
            args,
            check_fn,
            label,
            invalid_format=invalid_format,
        )

    return _handler


cmd_check_webgl = _make_simple_domain_check_cmd("check_webgl_op", "WebGL Operation", invalid_format="Invalid:")
cmd_check_cpp = _make_simple_domain_check_cmd("check_cpp_op", "C++ Operation", invalid_format="Invalid:")
cmd_check_array_api = _make_simple_domain_check_cmd("check_array_api_op", "Array API Operation")
cmd_check_scipy = _make_simple_domain_check_cmd("check_scipy_op", "SciPy Operation")
cmd_check_torchvision = _make_simple_domain_check_cmd("check_torchvision_op", "TorchVision Operation")
cmd_check_torchaudio = _make_simple_domain_check_cmd("check_torchaudio_op", "TorchAudio Operation")
cmd_check_safetensors = _make_simple_domain_check_cmd("check_safetensors_op", "SafeTensors Operation")
cmd_check_aten = _make_simple_domain_check_cmd("check_aten_op", "ATen Operation")
cmd_check_nccl = _make_simple_domain_check_cmd("check_nccl_op", "NCCL Operation")
cmd_check_flash_attention = _make_simple_domain_check_cmd("check_flash_attention_op", "Attention Operation")
cmd_check_chex = _make_simple_domain_check_cmd("check_chex_op", "Chex Operation")
cmd_check_grain = _make_simple_domain_check_cmd("check_grain_op", "Grain Operation")
cmd_check_orbax = _make_simple_domain_check_cmd("check_orbax_op", "Orbax Operation")
cmd_check_optax = _make_simple_domain_check_cmd("check_optax_op", "Optax Operation")


__all__ = [
    "DomainCheckCallable",
    "cmd_check_array_api",
    "cmd_check_aten",
    "cmd_check_chex",
    "cmd_check_cpp",
    "cmd_check_flash_attention",
    "cmd_check_grain",
    "cmd_check_metal",
    "cmd_check_nccl",
    "cmd_check_onnx",
    "cmd_check_optax",
    "cmd_check_orbax",
    "cmd_check_safetensors",
    "cmd_check_scipy",
    "cmd_check_torchaudio",
    "cmd_check_torchvision",
    "cmd_check_wasm",
    "cmd_check_webgl",
]
