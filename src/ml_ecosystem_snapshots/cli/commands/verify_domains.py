"""Declarative domain operation CLI command handlers (`check-onnx`, `check-metal`, `check-wasm`, etc.)."""

from __future__ import annotations

import argparse
import sys
import typing

import typing_extensions

from ml_ecosystem_snapshots.grounding.domain_ops import check_domain_op
from ml_ecosystem_snapshots.grounding.models import DomainOpCheckResult
from ml_ecosystem_snapshots.introspection.runtime_boundary import get_member_dict
from ml_ecosystem_snapshots.models import JsonValue


def _parse_cli_attributes(args: argparse.Namespace) -> list[str] | None:
    """Extract comma-separated CLI attributes from `args` if provided.

    Args:
        args: Parsed CLI `Namespace` instance.

    Returns:
        List of non-empty stripped attribute names, or None if unspecified.
    """
    raw_attrs = get_member_dict(args).get("attributes")
    if not raw_attrs:
        return None
    return [item.strip() for item in str(raw_attrs).split(",") if item.strip()]


def _parse_inputs_count(args: argparse.Namespace) -> int | None:
    """Extract optional integer `inputs_count` from `args`.

    Args:
        args: Parsed CLI `Namespace` instance.

    Returns:
        Integer input count or None.
    """
    raw = get_member_dict(args).get("inputs_count")
    return int(str(raw)) if raw is not None else None


def _parse_operands_count(args: argparse.Namespace) -> int | None:
    """Extract optional integer `operands_count` from `args`.

    Args:
        args: Parsed CLI `Namespace` instance.

    Returns:
        Integer operand count or None.
    """
    raw = get_member_dict(args).get("operands_count")
    return int(str(raw)) if raw is not None else None


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


DomainCheckCallable: typing_extensions.TypeAlias = (
    "_DomainOpNameCheckProto | _DomainMnemonicCheckProto | _DomainOnnxCheckProto | _DomainMetalCheckProto"
)


def _report_domain_check_result(
    label: str,
    target: str,
    res: DomainOpCheckResult | typing.Mapping[str, JsonValue],
    invalid_format: str = "is invalid:",
) -> None:
    """Format and print a domain operation verification result, exiting with code 1 when invalid.

    Args:
        label: Human-readable domain label (e.g. `'ONNX Operation'`).
        target: Verified operation or mnemonic name.
        res: Verification result model or mapping.
        invalid_format: Suffix text rendered when `res` is invalid.

    Raises:
        SystemExit: Exits with status code `1` when verification fails.
    """
    is_valid = res.is_valid if isinstance(res, DomainOpCheckResult) else bool(res.get("is_valid"))
    if not is_valid:
        print(f"{label} '{target}' {invalid_format}")
        errs = res.errors if isinstance(res, DomainOpCheckResult) else res.get("errors", [])
        err_list = errs if isinstance(errs, list) else []
        for err in err_list:
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
    target = str(args.op_name)
    domain_val = str(get_member_dict(args).get("domain") or "")
    res = check_domain_op(
        "onnx",
        op_name=target,
        domain=domain_val,
        inputs_count=_parse_inputs_count(args),
        attributes=_parse_cli_attributes(args),
    )
    _report_domain_check_result("ONNX Operation", target, res, invalid_format="Invalid:")


def cmd_check_metal(args: argparse.Namespace) -> None:
    """Validate Apple Metal MSL compute operation against specification.

    Args:
        args: Parsed command line arguments.
    """
    target = str(args.op_name)
    raw_addr = get_member_dict(args).get("address_space")
    addr_str = str(raw_addr) if raw_addr is not None else None
    res = check_domain_op(
        "metal",
        op_name=target,
        address_space=addr_str,
        inputs_count=_parse_inputs_count(args),
        attributes=_parse_cli_attributes(args),
    )
    _report_domain_check_result("Metal Operation", target, res, invalid_format="Invalid:")


def cmd_check_wasm(args: argparse.Namespace) -> None:
    """Validate WebAssembly 2.0 / SIMD instruction against specification.

    Args:
        args: Parsed command line arguments.
    """
    target = str(args.mnemonic)
    res = check_domain_op(
        "wasm",
        op_name=target,
        inputs_count=_parse_operands_count(args),
        attributes=_parse_cli_attributes(args),
    )
    _report_domain_check_result("WASM Instruction", target, res, invalid_format="Invalid:")


def _make_simple_domain_check_cmd(
    domain_key: str,
    label: str,
    invalid_format: str = "is invalid:",
) -> typing.Callable[[argparse.Namespace], None]:
    """Create a CLI command handler for a `grounding.domain_ops` domain verifier.

    Args:
        domain_key: Key into `DOMAIN_OP_CONFIGS` (e.g. `'webgl'`, `'array_api'`).
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
        target = str(args.op_name)
        res = check_domain_op(
            domain_key,
            op_name=target,
            inputs_count=_parse_inputs_count(args),
            attributes=_parse_cli_attributes(args),
        )
        _report_domain_check_result(label, target, res, invalid_format=invalid_format)

    return _handler


cmd_check_webgl = _make_simple_domain_check_cmd("webgl", "WebGL Operation", invalid_format="Invalid:")
cmd_check_cpp = _make_simple_domain_check_cmd("cpp", "C++ Operation", invalid_format="Invalid:")
cmd_check_array_api = _make_simple_domain_check_cmd("array_api", "Array API Operation")
cmd_check_scipy = _make_simple_domain_check_cmd("scipy", "SciPy Operation")
cmd_check_torchvision = _make_simple_domain_check_cmd("torchvision", "TorchVision Operation")
cmd_check_torchaudio = _make_simple_domain_check_cmd("torchaudio", "TorchAudio Operation")
cmd_check_safetensors = _make_simple_domain_check_cmd("safetensors", "SafeTensors Operation")
cmd_check_aten = _make_simple_domain_check_cmd("aten", "ATen Operation")
cmd_check_nccl = _make_simple_domain_check_cmd("nccl", "NCCL Operation")
cmd_check_flash_attention = _make_simple_domain_check_cmd("flash_attention", "Attention Operation")
cmd_check_chex = _make_simple_domain_check_cmd("chex", "Chex Operation")
cmd_check_grain = _make_simple_domain_check_cmd("grain", "Grain Operation")
cmd_check_orbax = _make_simple_domain_check_cmd("orbax", "Orbax Operation")
cmd_check_optax = _make_simple_domain_check_cmd("optax", "Optax Operation")


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
