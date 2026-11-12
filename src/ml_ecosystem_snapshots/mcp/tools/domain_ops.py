"""Thin MCP adapter for domain-operation verifiers."""

from __future__ import annotations

import typing

from ml_ecosystem_snapshots.grounding.domain_ops import (
    DOMAIN_OP_CONFIGS,
    DomainOpCheckConfig,
    DomainOpRecord,
    build_domain_op_tool_schemas,
    check_domain_op,
    verify_domain_op_generic,
)
from ml_ecosystem_snapshots.mcp.models import DomainOpCheckResult

check_domain_op_generic = verify_domain_op_generic


class SimpleDomainCheckFn(typing.Protocol):
    """Protocol for standard `(op_name, inputs_count=None, attributes=None)` MCP domain adapters."""

    def __call__(
        self,
        op_name: str,
        inputs_count: int | None = None,
        attributes: list[str] | None = None,
    ) -> DomainOpCheckResult:
        """Validate a domain operation against canonical specifications.

        Args:
            op_name: Operation identifier.
            inputs_count: Optional count of positional inputs.
            attributes: Optional list of attributes or keyword arguments.

        Returns:
            `DomainOpCheckResult` validation report model.
        """
        ...


def _make_op_adapter(domain_key: str) -> SimpleDomainCheckFn:
    """Create a standard `(op_name, inputs_count=None, attributes=None)` MCP domain adapter.

    Args:
        domain_key: Key in `DOMAIN_OP_CONFIGS`.

    Returns:
        Typed domain operation check callable.
    """

    def _check(
        op_name: str,
        inputs_count: int | None = None,
        attributes: list[str] | None = None,
    ) -> DomainOpCheckResult:
        """Validate domain operation inputs and attributes against canonical specification.

        Args:
            op_name: Operation identifier.
            inputs_count: Optional count of positional inputs.
            attributes: Optional list of attributes or keyword arguments.

        Returns:
            `DomainOpCheckResult` validation report model.
        """
        return check_domain_op(domain_key, op_name, inputs_count=inputs_count, attributes=attributes)

    return _check


def check_onnx_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
    domain: str = "",
) -> DomainOpCheckResult:
    """Verify whether an ONNX operator exists and check its input count and attributes.

    Args:
        op_name: Name of the ONNX operator (e.g. Add, Conv, onnx.MatMul).
        inputs_count: Optional number of positional inputs passed.
        attributes: Optional list of attribute names passed.
        domain: Operator domain (default `""` for standard ai.onnx).

    Returns:
        `DomainOpCheckResult` validation report model.
    """
    return check_domain_op("onnx", op_name, inputs_count=inputs_count, attributes=attributes, domain=domain)


def check_metal_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
    address_space: str | None = None,
) -> DomainOpCheckResult:
    """Verify whether an Apple Metal Shading Language operation exists and check address space/inputs.

    Args:
        op_name: Name of the MSL operation (e.g. simdgroup_multiply_accumulate).
        inputs_count: Optional number of positional inputs passed.
        attributes: Optional list of attribute names passed.
        address_space: Optional address space qualifier (e.g. device, threadgroup, thread).

    Returns:
        `DomainOpCheckResult` validation report model.
    """
    return check_domain_op(
        "metal", op_name, inputs_count=inputs_count, attributes=attributes, address_space=address_space
    )


def check_wasm_instruction(
    mnemonic: str, operands_count: int | None = None, attributes: list[str] | None = None
) -> DomainOpCheckResult:
    """Verify whether a WebAssembly SIMD instruction mnemonic exists and check operand count/immediates.

    Args:
        mnemonic: WASM instruction mnemonic (e.g. f32x4.add, v128.load).
        operands_count: Optional number of stack operands passed.
        attributes: Optional list of immediate attribute names (e.g. `[offset, align, lanes]`).

    Returns:
        `DomainOpCheckResult` validation report model.
    """
    return check_domain_op("wasm", mnemonic, inputs_count=operands_count, attributes=attributes)


check_wgsl_op = _make_op_adapter("wgsl")
check_webgl_op = _make_op_adapter("webgl")
check_cpp_op = _make_op_adapter("cpp")
check_array_api_op = _make_op_adapter("array_api")
check_scipy_op = _make_op_adapter("scipy")
check_torchvision_op = _make_op_adapter("torchvision")
check_torchaudio_op = _make_op_adapter("torchaudio")
check_safetensors_op = _make_op_adapter("safetensors")
check_aten_op = _make_op_adapter("aten")
check_nccl_op = _make_op_adapter("nccl")
check_flash_attention_op = _make_op_adapter("flash_attention")
check_chex_op = _make_op_adapter("chex")
check_grain_op = _make_op_adapter("grain")
check_orbax_op = _make_op_adapter("orbax")
check_optax_op = _make_op_adapter("optax")
__all__ = ["DOMAIN_OP_CONFIGS", "DomainOpCheckConfig", "DomainOpRecord", "build_domain_op_tool_schemas"]
__all__ += ["check_array_api_op", "check_aten_op", "check_chex_op", "check_cpp_op", "check_domain_op_generic"]
__all__ += ["check_flash_attention_op", "check_grain_op", "check_metal_op", "check_nccl_op", "check_onnx_op"]
__all__ += ["check_optax_op", "check_orbax_op", "check_safetensors_op", "check_scipy_op", "check_torchaudio_op"]
__all__ += ["check_torchvision_op", "check_wasm_instruction", "check_webgl_op", "check_wgsl_op"]
