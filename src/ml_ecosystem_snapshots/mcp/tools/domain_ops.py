"""Unified domain operation verifiers for WGSL, ONNX, Metal, WASM, WebGL, C++, Array API, SciPy, TorchVision, TorchAudio, SafeTensors, ATen, NCCL, and FlashAttention."""

from __future__ import annotations

import typing
from typing_extensions import TypeAlias

import pydantic

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalPythonOpSpec,
    CanonicalSsaOpSpec,
    CanonicalSubmoduleOpSpec,
)
from ml_ecosystem_snapshots.mcp.models import DomainOpCheckResult
from ml_ecosystem_snapshots.models import JsonValue

DomainOpRecord: TypeAlias = typing.Union[
    CanonicalSsaOpSpec,
    CanonicalPythonOpSpec,
    CanonicalSubmoduleOpSpec,
    dict[str, JsonValue],
]


class DomainOpCheckConfig(pydantic.BaseModel):
    """Declarative configuration for a domain-specific operation verifier."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    tool_name: str
    display_name: str
    op_label: str
    unknown_label: str
    key_field: str
    prefixes: tuple[str, ...]
    split_last_dot: bool
    lowercase: bool
    is_python_params: bool
    loader_fn: typing.Callable[[], typing.Sequence[DomainOpRecord]]
    tool_description: str
    op_arg_name: str = "op_name"
    op_arg_description: str = "Operation name."
    count_arg_name: str = "inputs_count"
    count_arg_description: str = "Optional inputs count."
    attr_arg_description: str = "Optional attributes."


def check_domain_op_generic(
    config: DomainOpCheckConfig,
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
    domain: str | None = None,
    address_space: str | None = None,
) -> DomainOpCheckResult:
    """Execute unified validation against a domain operation specification.

    Args:
        config: Declarative `DomainOpCheckConfig` instance.
        op_name: Name or mnemonic of the target operation.
        inputs_count: Optional expected count of positional inputs or operands.
        attributes: Optional list of attribute or keyword parameter names to verify.
        domain: Optional operator domain (used by ONNX).
        address_space: Optional memory address space (used by Apple Metal).

    Returns:
        Validation report model with `is_valid`, `op_exists`, `expected_inputs`,
        `expected_attributes`, and `errors`.
    """
    clean_name = op_name.strip()
    if config.lowercase:
        clean_name = clean_name.lower()

    if config.split_last_dot and (
        not config.prefixes or any(clean_name.startswith(p) for p in config.prefixes)
    ):
        clean_name = clean_name.split(".")[-1]
    else:
        for prefix in config.prefixes:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix) :]
                break

    ops = config.loader_fn()

    matched: DomainOpRecord | None = None
    if domain is not None:
        op_map_dom = {
            (str(op.get("domain") or ""), str(op[config.key_field])): op for op in ops
        }
        matched = op_map_dom.get((domain, clean_name))
        if matched is None and domain == "":
            for (_d, n), op in op_map_dom.items():
                if n == clean_name:
                    matched = op
                    break
        if matched is None:
            return DomainOpCheckResult(
                is_valid=False,
                op_exists=False,
                expected_inputs=[],
                expected_attributes=[],
                errors=[
                    f"Unknown {config.unknown_label}: '{op_name}' in domain '{domain}'"
                ],
            )
    else:
        op_map = {str(op[config.key_field]): op for op in ops}
        if clean_name not in op_map:
            return DomainOpCheckResult(
                is_valid=False,
                op_exists=False,
                expected_inputs=[],
                expected_attributes=[],
                errors=[f"Unknown {config.unknown_label}: '{op_name}'"],
            )
        matched = op_map[clean_name]

    exp_inputs: list[str] = []
    exp_attrs: list[str] = []
    if config.is_python_params:
        assert isinstance(matched, CanonicalPythonOpSpec)
        exp_inputs = [
            p.name
            for p in matched.params
            if p.kind in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
        ]
        exp_attrs = [
            p.name
            for p in matched.params
            if p.kind in ("KEYWORD_ONLY", "POSITIONAL_OR_KEYWORD")
        ]
    else:
        if isinstance(matched, CanonicalSsaOpSpec):
            exp_inputs = list(matched.inputs)
            exp_attrs = [a.name for a in matched.attributes if a.name]
        else:
            assert isinstance(matched, dict)
            raw_inputs = matched.get("inputs")
            assert isinstance(raw_inputs, list)
            exp_inputs = [str(i) for i in raw_inputs]
            raw_attrs = matched.get("attributes")
            assert isinstance(raw_attrs, list)
            exp_attrs = [
                str(a.get("name"))
                for a in raw_attrs
                if isinstance(a, dict) and a.get("name")
            ]

    errors: list[str] = []
    if inputs_count is not None and inputs_count != len(exp_inputs):
        if config.is_python_params:
            errors.append(
                f"Operation '{clean_name}' expects {len(exp_inputs)} positional inputs, but got {inputs_count}."
            )
        else:
            errors.append(
                f"{config.op_label} '{clean_name}' expects {len(exp_inputs)} inputs, but got {inputs_count}."
            )

    matched_addr = matched.get("address_space")
    if address_space is not None and matched_addr != address_space:
        errors.append(
            f"Operation '{clean_name}' operates in '{matched_addr}' address space, expected '{address_space}'."
        )

    if attributes:
        for attr in attributes:
            if attr not in exp_attrs:
                if config.is_python_params:
                    errors.append(
                        f"Operation '{clean_name}' does not accept keyword argument '{attr}'."
                    )
                elif config.display_name:
                    errors.append(
                        f"Attribute '{attr}' is not valid for {config.display_name} {config.op_label.lower()} '{clean_name}'."
                    )
                else:
                    errors.append(
                        f"Attribute '{attr}' is not valid for {config.op_label.lower()} '{clean_name}'."
                    )

    return DomainOpCheckResult(
        is_valid=len(errors) == 0,
        op_exists=True,
        expected_inputs=exp_inputs,
        expected_attributes=exp_attrs,
        errors=errors,
    )


def _load_wgsl() -> typing.Sequence[DomainOpRecord]:
    """Load canonical WGSL operations.

    Returns:
        List of canonical WGSL operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.wgsl import _load_wgsl_ops

    return _load_wgsl_ops()


def _load_onnx() -> typing.Sequence[DomainOpRecord]:
    """Load canonical ONNX operations.

    Returns:
        List of canonical ONNX operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.onnx_spec import _load_onnx_ops

    return _load_onnx_ops()


def _load_metal() -> typing.Sequence[DomainOpRecord]:
    """Load canonical Apple Metal operations.

    Returns:
        List of canonical Apple Metal operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.metal import _load_metal_ops

    return _load_metal_ops()


def _load_wasm() -> typing.Sequence[DomainOpRecord]:
    """Load canonical WebAssembly SIMD operations.

    Returns:
        List of canonical WebAssembly SIMD operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.wasm_simd import _load_wasm_simd_ops

    return _load_wasm_simd_ops()


def _load_webgl() -> typing.Sequence[DomainOpRecord]:
    """Load canonical WebGL operations.

    Returns:
        List of canonical WebGL operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.webgl import _load_webgl_ops

    return _load_webgl_ops()


def _load_cpp() -> typing.Sequence[DomainOpRecord]:
    """Load canonical C++ runtime operations.

    Returns:
        List of canonical C++ runtime operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.cpp_runtime import _load_cpp_runtime_ops

    return _load_cpp_runtime_ops()


def _load_array_api() -> typing.Sequence[DomainOpRecord]:
    """Load canonical Array API operations.

    Returns:
        List of canonical Array API operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.array_api import CANONICAL_ARRAY_API_OPS

    return list(CANONICAL_ARRAY_API_OPS)


def _load_scipy() -> typing.Sequence[DomainOpRecord]:
    """Load canonical SciPy operations.

    Returns:
        List of canonical SciPy operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.scipy import CANONICAL_SCIPY_OPS

    return list(CANONICAL_SCIPY_OPS)


def _load_torchvision() -> typing.Sequence[DomainOpRecord]:
    """Load canonical TorchVision operations.

    Returns:
        List of canonical TorchVision operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.torchvision import CANONICAL_TORCHVISION_OPS

    return list(CANONICAL_TORCHVISION_OPS)


def _load_torchaudio() -> typing.Sequence[DomainOpRecord]:
    """Load canonical TorchAudio operations.

    Returns:
        List of canonical TorchAudio operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.torchaudio import CANONICAL_TORCHAUDIO_OPS

    return list(CANONICAL_TORCHAUDIO_OPS)


def _load_safetensors() -> typing.Sequence[DomainOpRecord]:
    """Load canonical SafeTensors operations.

    Returns:
        List of canonical SafeTensors operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.safetensors import CANONICAL_SAFETENSORS_OPS

    return list(CANONICAL_SAFETENSORS_OPS)


def _load_aten() -> typing.Sequence[DomainOpRecord]:
    """Load canonical ATen operations.

    Returns:
        List of canonical ATen operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.aten import CANONICAL_ATEN_OPS

    return list(CANONICAL_ATEN_OPS)


def _load_nccl() -> typing.Sequence[DomainOpRecord]:
    """Load canonical NCCL operations.

    Returns:
        List of canonical NCCL operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.nccl import CANONICAL_NCCL_OPS

    return list(CANONICAL_NCCL_OPS)


def _load_flash_attn() -> typing.Sequence[DomainOpRecord]:
    """Load canonical FlashAttention operations.

    Returns:
        List of canonical FlashAttention operation dictionaries.
    """
    from ml_ecosystem_snapshots.frameworks.flash_attention import (
        CANONICAL_FLASH_ATTN_OPS,
    )

    return list(CANONICAL_FLASH_ATTN_OPS)


DOMAIN_OP_CONFIGS: dict[str, DomainOpCheckConfig] = {
    "wgsl": DomainOpCheckConfig(
        tool_name="check_wgsl_op",
        display_name="WGSL",
        op_label="Operation",
        unknown_label="WGSL operation",
        key_field="name",
        prefixes=(),
        split_last_dot=True,
        lowercase=False,
        is_python_params=False,
        loader_fn=_load_wgsl,
        tool_description="Validate a WebGPU WGSL built-in function, memory barrier, or address space.",
        op_arg_description="WGSL function name (e.g. 'workgroupBarrier', 'storageStore').",
        count_arg_description="Optional expected input operands count.",
        attr_arg_description="Optional operation attributes.",
    ),
    "onnx": DomainOpCheckConfig(
        tool_name="check_onnx_op",
        display_name="ONNX",
        op_label="Operation",
        unknown_label="ONNX operation",
        key_field="name",
        prefixes=(),
        split_last_dot=True,
        lowercase=False,
        is_python_params=False,
        loader_fn=_load_onnx,
        tool_description="Validate an ONNX operator specification.",
        op_arg_description="ONNX operator name (e.g. 'MatMul', 'Conv', 'Relu').",
        count_arg_description="Optional input count.",
        attr_arg_description="Optional operator attributes.",
    ),
    "metal": DomainOpCheckConfig(
        tool_name="check_metal_op",
        display_name="Metal",
        op_label="Operation",
        unknown_label="Metal MSL operation",
        key_field="name",
        prefixes=(),
        split_last_dot=True,
        lowercase=False,
        is_python_params=False,
        loader_fn=_load_metal,
        tool_description="Validate Apple Metal Shading Language (MSL) compute builtins.",
        op_arg_description="Metal builtin name (e.g. 'simdgroup_matrix', 'threadgroup_barrier').",
        count_arg_description="Optional input count.",
        attr_arg_description="Optional attributes.",
    ),
    "wasm": DomainOpCheckConfig(
        tool_name="check_wasm_instruction",
        display_name="",
        op_label="Instruction",
        unknown_label="WebAssembly SIMD instruction",
        key_field="mnemonic",
        prefixes=("wasm.",),
        split_last_dot=False,
        lowercase=True,
        is_python_params=False,
        loader_fn=_load_wasm,
        tool_description="Validate W3C WebAssembly 128-bit SIMD vector instruction opcodes.",
        op_arg_name="mnemonic",
        op_arg_description="WASM opcode mnemonic (e.g. 'f32x4.add', 'v128.load').",
        count_arg_name="operands_count",
        count_arg_description="Optional operands count.",
        attr_arg_description="Optional instruction attributes.",
    ),
    "webgl": DomainOpCheckConfig(
        tool_name="check_webgl_op",
        display_name="WebGL",
        op_label="Operation",
        unknown_label="WebGL operation",
        key_field="name",
        prefixes=("webgl.",),
        split_last_dot=False,
        lowercase=False,
        is_python_params=False,
        loader_fn=_load_webgl,
        tool_description="Validate WebGL 2.0 / GLSL ES 3.00 shader builtins and uniforms.",
        op_arg_description="GLSL builtin name (e.g. 'texture', 'dFdx', 'fma').",
        count_arg_description="Optional inputs count.",
        attr_arg_description="Optional attributes.",
    ),
    "cpp": DomainOpCheckConfig(
        tool_name="check_cpp_op",
        display_name="C++",
        op_label="Operation",
        unknown_label="C++ operation",
        key_field="name",
        prefixes=("cpp.",),
        split_last_dot=False,
        lowercase=False,
        is_python_params=False,
        loader_fn=_load_cpp,
        tool_description="Validate C++17 STL mathematical kernel runtime functions.",
        op_arg_description="C++ function name (e.g. 'clamp', 'from_blob').",
        count_arg_description="Optional inputs count.",
        attr_arg_description="Optional attribute names.",
    ),
    "array_api": DomainOpCheckConfig(
        tool_name="check_array_api_op",
        display_name="Array API",
        op_label="Operation",
        unknown_label="Array API operation",
        key_field="name",
        prefixes=("array_api.",),
        split_last_dot=False,
        lowercase=False,
        is_python_params=True,
        loader_fn=_load_array_api,
        tool_description="Validate Python Array API standard consortium specification functions.",
        op_arg_description="Array API function name (e.g. 'matmul', 'astype', 'reshape').",
        count_arg_description="Optional positional arguments count.",
        attr_arg_description="Optional keyword arguments.",
    ),
    "scipy": DomainOpCheckConfig(
        tool_name="check_scipy_op",
        display_name="SciPy",
        op_label="Operation",
        unknown_label="SciPy operation",
        key_field="name",
        prefixes=("scipy.",),
        split_last_dot=True,
        lowercase=False,
        is_python_params=True,
        loader_fn=_load_scipy,
        tool_description="Validate SciPy special functions, signal, or linalg APIs.",
        op_arg_description="SciPy API name (e.g. 'erf', 'inv', 'convolve').",
        count_arg_description="Optional positional arguments count.",
        attr_arg_description="Optional keyword arguments.",
    ),
    "torchvision": DomainOpCheckConfig(
        tool_name="check_torchvision_op",
        display_name="TorchVision",
        op_label="Operation",
        unknown_label="TorchVision operation",
        key_field="name",
        prefixes=("torchvision.",),
        split_last_dot=True,
        lowercase=False,
        is_python_params=True,
        loader_fn=_load_torchvision,
        tool_description="Validate TorchVision vision transforms and operators.",
        op_arg_description="TorchVision API name (e.g. 'nms', 'roi_align').",
        count_arg_description="Optional positional arguments count.",
        attr_arg_description="Optional keyword arguments.",
    ),
    "torchaudio": DomainOpCheckConfig(
        tool_name="check_torchaudio_op",
        display_name="TorchAudio",
        op_label="Operation",
        unknown_label="TorchAudio operation",
        key_field="name",
        prefixes=("torchaudio.",),
        split_last_dot=True,
        lowercase=False,
        is_python_params=True,
        loader_fn=_load_torchaudio,
        tool_description="Validate TorchAudio audio processing transforms and functionals.",
        op_arg_description="TorchAudio API name (e.g. 'spectrogram', 'melscale_fbanks').",
        count_arg_description="Optional positional arguments count.",
        attr_arg_description="Optional keyword arguments.",
    ),
    "safetensors": DomainOpCheckConfig(
        tool_name="check_safetensors_op",
        display_name="SafeTensors",
        op_label="Operation",
        unknown_label="SafeTensors operation",
        key_field="name",
        prefixes=("safetensors.",),
        split_last_dot=True,
        lowercase=False,
        is_python_params=True,
        loader_fn=_load_safetensors,
        tool_description="Validate SafeTensors weight loading and serialization routines.",
        op_arg_description="SafeTensors API name (e.g. 'save_file', 'load_file', 'safe_open').",
        count_arg_description="Optional positional arguments count.",
        attr_arg_description="Optional keyword arguments.",
    ),
    "aten": DomainOpCheckConfig(
        tool_name="check_aten_op",
        display_name="ATen",
        op_label="Operation",
        unknown_label="ATen operation",
        key_field="name",
        prefixes=("aten.",),
        split_last_dot=False,
        lowercase=False,
        is_python_params=True,
        loader_fn=_load_aten,
        tool_description="Validate PyTorch ATen C++ kernel dispatch operation.",
        op_arg_description="ATen operator name (e.g. 'add', 'matmul', 'relu').",
        count_arg_description="Optional inputs count.",
        attr_arg_description="Optional keyword argument names.",
    ),
    "nccl": DomainOpCheckConfig(
        tool_name="check_nccl_op",
        display_name="NCCL",
        op_label="Operation",
        unknown_label="NCCL collective operation",
        key_field="name",
        prefixes=("nccl.", "rccl."),
        split_last_dot=False,
        lowercase=False,
        is_python_params=True,
        loader_fn=_load_nccl,
        tool_description="Validate NVIDIA NCCL / AMD RCCL GPU collective communications.",
        op_arg_description="NCCL operation name (e.g. 'all_reduce', 'all_gather', 'reduce_scatter').",
        count_arg_description="Optional inputs count.",
        attr_arg_description="Optional attributes.",
    ),
    "flash_attention": DomainOpCheckConfig(
        tool_name="check_flash_attention_op",
        display_name="Attention",
        op_label="Operation",
        unknown_label="Attention operation",
        key_field="name",
        prefixes=("flash_attn.", "vllm.", "ml.switcheroo.custom."),
        split_last_dot=False,
        lowercase=False,
        is_python_params=True,
        loader_fn=_load_flash_attn,
        tool_description="Validate FlashAttention-2/3, PagedAttention, and custom kernel parameters.",
        op_arg_description="Attention kernel API name (e.g. 'flash_attn_func', 'flash_attn_varlen_func').",
        count_arg_description="Optional positional arguments count.",
        attr_arg_description="Optional keyword arguments.",
    ),
}


def check_wgsl_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate a WebGPU WGSL operation against grounded schema specifications.

    Args:
        op_name: Name of the WGSL operation (e.g. 'storageStore', 'wgsl.atomicAdd').
        inputs_count: Optional expected number of input operands.
        attributes: Optional list of attribute names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["wgsl"],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )


def check_onnx_op(
    op_name: str,
    domain: str = "",
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate an ONNX operation against grounded operator specification.

    Args:
        op_name: Name of the ONNX operation (e.g. 'Conv', 'onnx.Add').
        domain: Optional operator domain (e.g. '', 'ai.onnx.ml').
        inputs_count: Optional expected number of input operands.
        attributes: Optional list of attribute names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["onnx"],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
        domain=domain,
    )


def check_metal_op(
    op_name: str,
    address_space: str | None = None,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate an Apple Metal MSL compute operation against grounded specification.

    Args:
        op_name: Name of the MSL operation (e.g. 'simdgroup_multiply_accumulate', 'metal.fma').
        address_space: Optional expected address space (e.g. 'device', 'threadgroup').
        inputs_count: Optional expected number of input operands.
        attributes: Optional list of attribute names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["metal"],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
        address_space=address_space,
    )


def check_wasm_instruction(
    mnemonic: str,
    operands_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate a WebAssembly 2.0 / SIMD instruction against grounded specification.

    Args:
        mnemonic: Name of the instruction (e.g. 'f32x4.add', 'v128.load').
        operands_count: Optional expected number of input operands.
        attributes: Optional list of attribute names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["wasm"],
        op_name=mnemonic,
        inputs_count=operands_count,
        attributes=attributes,
    )


def check_webgl_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate a WebGL 2.0 / GLSL ES 3.00 operation against grounded specification.

    Args:
        op_name: Name of the WebGL operation (e.g. 'texelFetch', 'webgl.fma').
        inputs_count: Optional expected number of input operands.
        attributes: Optional list of attribute names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["webgl"],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )


def check_cpp_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate a C++17 / PyBind11 operation against grounded specification.

    Args:
        op_name: Name of the C++ operation (e.g. 'clamp', 'from_blob', 'cpp.sqrt').
        inputs_count: Optional expected number of input operands.
        attributes: Optional list of attribute names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["cpp"],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )


def check_array_api_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate a Python Array API Standard operation against grounded specification.

    Args:
        op_name: Name of the Array API operation (e.g. 'add', 'matmul', 'array_api.sum').
        inputs_count: Optional expected number of input operands.
        attributes: Optional list of keyword parameter names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["array_api"],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )


def check_scipy_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate a SciPy operation against grounded specification.

    Args:
        op_name: Name of the SciPy operation (e.g. 'erf', 'inv', 'convolve').
        inputs_count: Optional expected number of input operands.
        attributes: Optional list of keyword parameter names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["scipy"],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )


def check_torchvision_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate a TorchVision operation against grounded specification.

    Args:
        op_name: Name of the TorchVision operation (e.g. 'nms', 'roi_align', 'gaussian_blur').
        inputs_count: Optional expected number of input operands.
        attributes: Optional list of keyword parameter names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["torchvision"],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )


def check_torchaudio_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate a TorchAudio operation against grounded specification.

    Args:
        op_name: Name of the TorchAudio operation (e.g. 'melscale_fbanks', 'spectrogram', 'MFCC').
        inputs_count: Optional expected number of input operands.
        attributes: Optional list of keyword parameter names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["torchaudio"],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )


def check_safetensors_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate a SafeTensors operation against grounded specification.

    Args:
        op_name: Name of the SafeTensors operation (e.g. 'save_file', 'load_file', 'safe_open').
        inputs_count: Optional expected number of input operands.
        attributes: Optional list of keyword parameter names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["safetensors"],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )


def check_aten_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate an ATen operator against grounded specification.

    Args:
        op_name: Name of the ATen operation (e.g. 'add', 'matmul', 'relu').
        inputs_count: Optional expected number of input operands.
        attributes: Optional list of keyword parameter names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["aten"],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )


def check_nccl_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate an NCCL / RCCL collective operation against grounded specification.

    Args:
        op_name: Name of the collective operation (e.g. 'all_reduce', 'all_gather', 'reduce_scatter').
        inputs_count: Optional expected number of input operands.
        attributes: Optional list of keyword parameter names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["nccl"],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )


def check_flash_attention_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
) -> DomainOpCheckResult:
    """Validate a FlashAttention / PagedAttention operation against grounded specification.

    Args:
        op_name: Name of the attention operation (e.g. 'flash_attn_func', 'paged_attention_v1', 'FlashAttention').
        inputs_count: Optional expected number of input operands.
        attributes: Optional list of keyword parameter names to verify.

    Returns:
        Validation report dictionary with is_valid, expected_inputs, expected_attributes, and errors.
    """
    return check_domain_op_generic(
        DOMAIN_OP_CONFIGS["flash_attention"],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
    )


def build_domain_op_tool_schemas() -> list[dict[str, JsonValue]]:
    """Generate MCP JSON Schema tool definitions for all registered domain op verifiers.

    Returns:
        List of MCP tool schema dictionaries.
    """
    schemas: list[dict[str, JsonValue]] = []
    for key, cfg in DOMAIN_OP_CONFIGS.items():
        props: dict[str, JsonValue] = {
            cfg.op_arg_name: {
                "type": "string",
                "description": cfg.op_arg_description,
            }
        }
        if key == "onnx":
            props["domain"] = {
                "type": "string",
                "description": "Optional ONNX domain (e.g. 'ai.onnx').",
            }
        if key == "metal":
            props["address_space"] = {
                "type": "string",
                "description": "Optional address space (e.g. 'device', 'threadgroup').",
            }
        props[cfg.count_arg_name] = {
            "type": "integer",
            "description": cfg.count_arg_description,
        }
        props["attributes"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": cfg.attr_arg_description,
        }
        schemas.append(
            {
                "name": cfg.tool_name,
                "description": cfg.tool_description,
                "inputSchema": {
                    "type": "object",
                    "properties": props,
                    "required": [cfg.op_arg_name],
                },
            }
        )
    return schemas


__all__ = [
    "DOMAIN_OP_CONFIGS",
    "DomainOpCheckConfig",
    "DomainOpRecord",
    "build_domain_op_tool_schemas",
    "check_array_api_op",
    "check_aten_op",
    "check_cpp_op",
    "check_domain_op_generic",
    "check_flash_attention_op",
    "check_metal_op",
    "check_nccl_op",
    "check_onnx_op",
    "check_safetensors_op",
    "check_scipy_op",
    "check_torchaudio_op",
    "check_torchvision_op",
    "check_wasm_instruction",
    "check_webgl_op",
    "check_wgsl_op",
]
