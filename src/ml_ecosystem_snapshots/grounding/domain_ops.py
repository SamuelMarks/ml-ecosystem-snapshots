"""Domain-operation grounding verifiers honoring `GroundingEngine.snapshots_dir`."""

from __future__ import annotations

import typing

from ml_switcheroo_ir.schema.ghost import ExtendedGhostParam, ExtendedGhostRef

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalPythonOpSpec,
    CanonicalSsaOpSpec,
)
from ml_ecosystem_snapshots.grounding._domain_configs import (
    DOMAIN_OP_CONFIGS,
    DomainOpCheckConfig,
    DomainOpRecord,
    build_domain_op_tool_schemas,
)
from ml_ecosystem_snapshots.grounding.engine import GroundingEngine
from ml_ecosystem_snapshots.grounding.models import (
    DiagnosticSeverity,
    DomainOpCheckResult,
    GroundingReport,
)
from ml_ecosystem_snapshots.models import JsonValue, ensure_json_list


def _normalize_domain_op_name(
    config: DomainOpCheckConfig,
    op_name: str,
) -> str:
    """Normalize an operation name according to the domain configuration's prefix and casing rules.

    Args:
        config: Declarative `DomainOpCheckConfig` instance.
        op_name: Raw operation name or mnemonic.

    Returns:
        Normalized operation name string.
    """
    clean_name = op_name.strip()
    if config.lowercase:
        clean_name = clean_name.lower()

    if config.split_last_dot and (not config.prefixes or any(clean_name.startswith(p) for p in config.prefixes)):
        clean_name = clean_name.split(".")[-1]
    else:
        for prefix in config.prefixes:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix) :]
                break
    return clean_name


def _read_domain_record_key(op: DomainOpRecord, key_field: str) -> str:
    """Read `name`, `mnemonic`, or `api_path` from a `DomainOpRecord` without dict cosplay.

    Args:
        op: Canonical domain operation record.
        key_field: Identifier field selector (`'name'`, `'mnemonic'`, or `'api_path'`).

    Returns:
        String identifier for `op`.
    """
    if isinstance(op, CanonicalSsaOpSpec):
        return op.mnemonic if key_field == "mnemonic" else op.name
    if isinstance(op, CanonicalPythonOpSpec):
        return op.api_path if key_field == "api_path" else op.name
    return str(op.get(key_field) or "")


def _read_domain_record_domain(op: DomainOpRecord) -> str:
    """Read operator domain string from a `DomainOpRecord`.

    Args:
        op: Canonical domain operation record.

    Returns:
        Domain string (empty string if unset).
    """
    if isinstance(op, CanonicalSsaOpSpec):
        return op.domain
    if isinstance(op, CanonicalPythonOpSpec):
        return ""
    return str(op.get("domain") or "")


def _read_domain_record_address_space(op: DomainOpRecord) -> str | None:
    """Read optional memory address space from a `DomainOpRecord`.

    Args:
        op: Canonical domain operation record.

    Returns:
        Address space string or `None`.
    """
    if isinstance(op, CanonicalSsaOpSpec):
        return op.address_space or None
    if isinstance(op, CanonicalPythonOpSpec):
        return None
    val = op.get("address_space")
    return str(val) if val is not None else None


def _find_domain_op_record(
    config: DomainOpCheckConfig,
    ops: typing.Sequence[DomainOpRecord],
    clean_name: str,
    domain: str | None,
) -> DomainOpRecord | None:
    """Locate a matching domain operation record by key and optional domain.

    Args:
        config: Declarative `DomainOpCheckConfig` instance.
        ops: Sequence of canonical domain operation records.
        clean_name: Normalized operation name.
        domain: Optional operator domain (e.g., for ONNX).

    Returns:
        Matched `DomainOpRecord`, or None if no matching operation exists.
    """
    if domain is not None:
        op_map_dom = {(_read_domain_record_domain(op), _read_domain_record_key(op, config.key_field)): op for op in ops}
        matched = op_map_dom.get((domain, clean_name))
        if matched is None and domain == "":
            for (_d, n), op in op_map_dom.items():
                if n == clean_name:
                    matched = op
                    break
        return matched
    op_map = {_read_domain_record_key(op, config.key_field): op for op in ops}
    return op_map.get(clean_name)


def _extract_domain_op_signature(
    config: DomainOpCheckConfig,
    matched: DomainOpRecord,
) -> tuple[list[str], list[str]]:
    """Extract expected positional inputs and keyword/attribute names from a matched record.

    Args:
        config: Declarative `DomainOpCheckConfig` instance.
        matched: Matched `DomainOpRecord` specification.

    Returns:
        Tuple of `(expected_inputs, expected_attributes)`.
    """
    if isinstance(matched, CanonicalPythonOpSpec):
        exp_inputs = [p.name for p in matched.params if p.kind in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")]
        exp_attrs = [p.name for p in matched.params if p.kind in ("KEYWORD_ONLY", "POSITIONAL_OR_KEYWORD")]
        return exp_inputs, exp_attrs
    if isinstance(matched, CanonicalSsaOpSpec):
        exp_inputs = list(matched.inputs)
        exp_attrs = [a.name for a in matched.attributes if a.name]
        return exp_inputs, exp_attrs
    raw_inputs = ensure_json_list(matched.get("inputs"))
    exp_inputs = [str(i) for i in raw_inputs]
    raw_attrs = ensure_json_list(matched.get("attributes"))
    exp_attrs = [str(a.get("name")) for a in raw_attrs if isinstance(a, dict) and a.get("name")]
    return exp_inputs, exp_attrs


def _extract_engine_ref_signature(
    config: DomainOpCheckConfig,
    ref: ExtendedGhostRef,
) -> tuple[list[str], list[str], str | None]:
    """Extract expected inputs, attributes, and optional address space from a `GroundingEngine` `ExtendedGhostRef`.

    Args:
        config: Declarative `DomainOpCheckConfig`.
        ref: Matched `ExtendedGhostRef` from `GroundingEngine`.

    Returns:
        Tuple of `(expected_inputs, expected_attributes, address_space_or_none)`.
    """
    exp_inputs: list[str] = []
    exp_attrs: list[str] = []
    for p in ref.params:
        role_str = str(p.role if isinstance(p, ExtendedGhostParam) else "")
        kind_str = str(p.kind)
        if "ATTRIBUTE" in role_str or "KEYWORD_ONLY" in kind_str:
            exp_attrs.append(p.name)
        elif config.is_python_params:
            if "POSITIONAL_ONLY" in kind_str or ("POSITIONAL_OR_KEYWORD" in kind_str and p.default is None):
                exp_inputs.append(p.name)
            if "POSITIONAL_OR_KEYWORD" in kind_str:
                exp_attrs.append(p.name)
        elif "VAR_" not in kind_str:
            exp_inputs.append(p.name)
    meta = ref.domain_metadata or {}
    raw_addr = meta.get("address_space")
    addr = str(raw_addr) if raw_addr is not None else None
    return exp_inputs, exp_attrs, addr


def _format_domain_attr_mismatch(config: DomainOpCheckConfig, clean_name: str, attr: str) -> str:
    """Format an unrecognized attribute or keyword argument error message.

    Args:
        config: Declarative `DomainOpCheckConfig`.
        clean_name: Normalized operation name.
        attr: Unrecognized attribute or kwarg name.

    Returns:
        Formatted error message string.
    """
    if config.is_python_params:
        return f"Operation '{clean_name}' does not accept keyword argument '{attr}'."
    if config.display_name:
        return f"Attribute '{attr}' is not valid for {config.display_name} {config.op_label.lower()} '{clean_name}'."
    return f"Attribute '{attr}' is not valid for {config.op_label.lower()} '{clean_name}'."


def _check_domain_op_constraints(
    config: DomainOpCheckConfig,
    clean_name: str,
    signature: tuple[list[str], list[str], str | None],
    inputs_count: int | None,
    attributes: list[str] | None,
    address_space: str | None,
) -> list[str]:
    """Check arity, address-space, and attribute constraints for a matched domain operation.

    Args:
        config: Declarative `DomainOpCheckConfig`.
        clean_name: Normalized operation name.
        signature: Tuple of `(exp_inputs, exp_attrs, matched_addr)`.
        inputs_count: Optional passed positional input count.
        attributes: Optional passed attribute/kwarg names.
        address_space: Optional expected address space.

    Returns:
        List of formatted error messages.
    """
    exp_inputs, exp_attrs, matched_addr = signature
    errors: list[str] = []
    if inputs_count is not None and inputs_count != len(exp_inputs):
        if config.is_python_params:
            errors.append(
                f"Operation '{clean_name}' expects {len(exp_inputs)} positional inputs, but got {inputs_count}."
            )
        else:
            errors.append(f"{config.op_label} '{clean_name}' expects {len(exp_inputs)} inputs, but got {inputs_count}.")

    if address_space is not None and matched_addr != address_space:
        errors.append(
            f"Operation '{clean_name}' operates in '{matched_addr}' address space, expected '{address_space}'."
        )

    for attr in attributes or []:
        if attr not in exp_attrs:
            errors.append(_format_domain_attr_mismatch(config, clean_name, attr))
    return errors


def verify_domain_op_generic(
    config: DomainOpCheckConfig,
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
    domain: str | None = None,
    address_space: str | None = None,
    engine: GroundingEngine | None = None,
) -> DomainOpCheckResult:
    """Validate a domain operation against an injected `GroundingEngine` (honoring `snapshots_dir`) or canonical specs.

    Args:
        config: Declarative `DomainOpCheckConfig` instance.
        op_name: Operation or instruction name to verify.
        inputs_count: Optional count of positional inputs or operands.
        attributes: Optional list of attribute or keyword parameter names to verify.
        domain: Optional operator domain (used by ONNX).
        address_space: Optional memory address space (used by Apple Metal).
        engine: Optional `GroundingEngine` instance queried first when provided.

    Returns:
        `DomainOpCheckResult` validation report model.
    """
    clean_name = _normalize_domain_op_name(config, op_name)
    exp_inputs: list[str] = []
    exp_attrs: list[str] = []
    matched_addr: str | None = None
    found = False

    if engine is not None and engine.snapshots_dir is not None:
        target_key = config.tool_name.replace("check_", "").replace("_op", "").replace("_instruction", "")
        target_map = {"wasm": "wasm_simd", "cpp": "cpp_runtime", "flash_attention": "flash_attention"}
        fw_key = target_map.get(target_key, target_key)
        eng_ref = engine.get_symbol(fw_key, clean_name)
        if eng_ref is not None:
            exp_inputs, exp_attrs, matched_addr = _extract_engine_ref_signature(config, eng_ref)
            found = True

    if not found:
        ops = config.loader_fn()
        matched = _find_domain_op_record(config, ops, clean_name, domain)
        if matched is None:
            not_found_msg = (
                f"Unknown {config.unknown_label}: '{op_name}' in domain '{domain}'"
                if domain is not None
                else f"Unknown {config.unknown_label}: '{op_name}'"
            )
            return DomainOpCheckResult(is_valid=False, op_exists=False, errors=[not_found_msg])
        exp_inputs, exp_attrs = _extract_domain_op_signature(config, matched)
        matched_addr = _read_domain_record_address_space(matched)

    errors = _check_domain_op_constraints(
        config,
        clean_name,
        (exp_inputs, exp_attrs, matched_addr),
        inputs_count,
        attributes,
        address_space,
    )

    return DomainOpCheckResult(
        is_valid=len(errors) == 0,
        op_exists=True,
        expected_inputs=exp_inputs,
        expected_attributes=exp_attrs,
        errors=errors,
    )


def _domain_result_to_grounding_report(
    target_framework: str,
    symbol_name: str,
    res: DomainOpCheckResult | typing.Mapping[str, JsonValue],
    unrecognized_label: str,
    missing_field: str = "operation",
    err_field: str = "operation",
    engine: GroundingEngine | None = None,
) -> GroundingReport:
    """Convert a `DomainOpCheckResult` into a `GroundingReport` with Levenshtein suggestions.

    Args:
        target_framework: Target framework identifier.
        symbol_name: Checked operation or mnemonic name.
        res: `DomainOpCheckResult` or dictionary.
        unrecognized_label: Label used in missing-symbol diagnostics.
        missing_field: Diagnostic field for missing symbols.
        err_field: Diagnostic field for validation errors.
        engine: Optional `GroundingEngine` instance.

    Returns:
        Populated `GroundingReport`.
    """
    eng = engine or GroundingEngine()
    report = GroundingReport(is_grounded=True, target=target_framework, symbol=symbol_name)
    op_exists = res.op_exists if isinstance(res, DomainOpCheckResult) else bool(res.get("op_exists"))
    if not op_exists:
        suggested = eng.suggest_closest_symbol(target_framework, symbol_name)
        report.add_diagnostic(
            field=missing_field,
            message=f"Unrecognized {unrecognized_label} '{symbol_name}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report
    raw_errs = res.errors if isinstance(res, DomainOpCheckResult) else ensure_json_list(res.get("errors", []))
    for err in raw_errs:
        report.add_diagnostic(field=err_field, message=str(err), severity=DiagnosticSeverity.ERROR)
    return report


def check_domain_op(
    domain_key: str,
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
    domain: str | None = None,
    address_space: str | None = None,
    engine: GroundingEngine | None = None,
) -> DomainOpCheckResult:
    """Verify a domain operation by `DOMAIN_OP_CONFIGS` key.

    Args:
        domain_key: Key into `DOMAIN_OP_CONFIGS` (e.g. `wgsl`, `onnx`, `array_api`).
        op_name: Operation or instruction name.
        inputs_count: Optional count of positional inputs/operands.
        attributes: Optional list of attributes or keyword arguments.
        domain: Optional ONNX operator domain.
        address_space: Optional Metal address space.
        engine: Optional `GroundingEngine` instance.

    Returns:
        `DomainOpCheckResult` validation report model.
    """
    return verify_domain_op_generic(
        DOMAIN_OP_CONFIGS[domain_key],
        op_name=op_name,
        inputs_count=inputs_count,
        attributes=attributes,
        domain=domain,
        address_space=address_space,
        engine=engine,
    )


def validate_wgsl_op(
    op_name: str,
    operands_count: int | None = None,
    attributes: list[str] | None = None,
    engine: GroundingEngine | None = None,
) -> GroundingReport:
    """Verify a WebGPU WGSL builtin function or memory barrier against ground truth.

    Args:
        op_name: WGSL operation name.
        operands_count: Optional number of passed arguments.
        attributes: Optional list of operation attributes.
        engine: Optional `GroundingEngine` instance.

    Returns:
        `GroundingReport` containing diagnostic results.
    """
    res = check_domain_op("wgsl", op_name, inputs_count=operands_count, attributes=attributes, engine=engine)
    return _domain_result_to_grounding_report("wgsl", op_name, res, "WGSL builtin", engine=engine)


def validate_onnx_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
    engine: GroundingEngine | None = None,
) -> GroundingReport:
    """Verify an ONNX operator specification against the standard schema.

    Args:
        op_name: ONNX operator name.
        inputs_count: Optional number of passed inputs.
        attributes: Optional list of attributes.
        engine: Optional `GroundingEngine` instance.

    Returns:
        `GroundingReport` containing diagnostic results.
    """
    res = check_domain_op("onnx", op_name, inputs_count=inputs_count, attributes=attributes, engine=engine)
    return _domain_result_to_grounding_report("onnx", op_name, res, "ONNX operator", engine=engine)


def validate_metal_op(
    op_name: str,
    address_space: str | None = None,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
    engine: GroundingEngine | None = None,
) -> GroundingReport:
    """Verify an Apple Metal Shading Language compute builtin against specifications.

    Args:
        op_name: Metal builtin name.
        address_space: Optional expected address space.
        inputs_count: Optional number of passed arguments.
        attributes: Optional list of attributes.
        engine: Optional `GroundingEngine` instance.

    Returns:
        `GroundingReport` containing diagnostic results.
    """
    res = check_domain_op(
        "metal", op_name, inputs_count=inputs_count, attributes=attributes, address_space=address_space, engine=engine
    )
    return _domain_result_to_grounding_report("metal", op_name, res, "Metal builtin", engine=engine)


def validate_wasm_op(
    mnemonic: str,
    operands_count: int | None = None,
    attributes: list[str] | None = None,
    engine: GroundingEngine | None = None,
) -> GroundingReport:
    """Verify a W3C WebAssembly SIMD opcode mnemonic against standard specifications.

    Args:
        mnemonic: WASM opcode mnemonic.
        operands_count: Optional number of passed operands.
        attributes: Optional list of attributes.
        engine: Optional `GroundingEngine` instance.

    Returns:
        `GroundingReport` containing diagnostic results.
    """
    res = check_domain_op("wasm", mnemonic, inputs_count=operands_count, attributes=attributes, engine=engine)
    return _domain_result_to_grounding_report(
        "wasm_simd",
        mnemonic,
        res,
        "WebAssembly SIMD opcode",
        missing_field="mnemonic",
        err_field="instruction",
        engine=engine,
    )


def validate_webgl_op(
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
    engine: GroundingEngine | None = None,
) -> GroundingReport:
    """Verify a WebGL 2.0 / GLSL ES 3.00 shader builtin against standard specifications.

    Args:
        op_name: GLSL builtin function name.
        inputs_count: Optional number of passed arguments.
        attributes: Optional list of attributes.
        engine: Optional `GroundingEngine` instance.

    Returns:
        `GroundingReport` containing diagnostic results.
    """
    res = check_domain_op("webgl", op_name, inputs_count=inputs_count, attributes=attributes, engine=engine)
    return _domain_result_to_grounding_report("webgl", op_name, res, "WebGL / GLSL builtin", engine=engine)


def validate_domain_op(
    domain_key: str,
    op_name: str,
    inputs_count: int | None = None,
    attributes: list[str] | None = None,
    engine: GroundingEngine | None = None,
) -> GroundingReport:
    """Verify any registered domain operation (`cpp`, `array_api`, `scipy`, `torchvision`, etc.) via `GroundingEngine`.

    Args:
        domain_key: Key into `DOMAIN_OP_CONFIGS`.
        op_name: Operation name.
        inputs_count: Optional count of positional inputs.
        attributes: Optional list of attributes or keyword arguments.
        engine: Optional `GroundingEngine` instance.

    Returns:
        `GroundingReport` containing diagnostic results.
    """
    cfg = DOMAIN_OP_CONFIGS[domain_key]
    res = verify_domain_op_generic(cfg, op_name, inputs_count=inputs_count, attributes=attributes, engine=engine)
    return _domain_result_to_grounding_report(domain_key, op_name, res, cfg.op_label, engine=engine)


class SimpleDomainValidateFn(typing.Protocol):
    """Protocol for a standard `(op_name, inputs_count=None, attributes=None, engine=None)` domain validator."""

    def __call__(
        self,
        op_name: str,
        inputs_count: int | None = None,
        attributes: list[str] | None = None,
        engine: GroundingEngine | None = None,
    ) -> GroundingReport:
        """Validate a domain operation against `GroundingEngine` or canonical specs.

        Args:
            op_name: Operation name.
            inputs_count: Optional count of positional inputs.
            attributes: Optional list of attributes or keyword arguments.
            engine: Optional `GroundingEngine` instance.

        Returns:
            `GroundingReport` containing diagnostic results.
        """
        ...


def _make_domain_validator(domain_key: str) -> SimpleDomainValidateFn:
    """Build a typed domain validator delegating to `validate_domain_op(domain_key, ..., engine=engine)`.

    Args:
        domain_key: Key into `DOMAIN_OP_CONFIGS`.

    Returns:
        Typed `SimpleDomainValidateFn` validator callable.
    """

    def _validate(
        op_name: str,
        inputs_count: int | None = None,
        attributes: list[str] | None = None,
        engine: GroundingEngine | None = None,
    ) -> GroundingReport:
        """Validate a domain operation via `validate_domain_op`.

        Args:
            op_name: Operation name.
            inputs_count: Optional count of positional inputs.
            attributes: Optional list of attributes or keyword arguments.
            engine: Optional `GroundingEngine` instance.

        Returns:
            `GroundingReport` containing diagnostic results.
        """
        return validate_domain_op(
            domain_key,
            op_name,
            inputs_count=inputs_count,
            attributes=attributes,
            engine=engine,
        )

    return _validate


validate_cpp_op = _make_domain_validator("cpp")
validate_array_api_op = _make_domain_validator("array_api")
validate_scipy_op = _make_domain_validator("scipy")
validate_torchvision_op = _make_domain_validator("torchvision")
validate_torchaudio_op = _make_domain_validator("torchaudio")
validate_safetensors_op = _make_domain_validator("safetensors")
validate_aten_op = _make_domain_validator("aten")
validate_nccl_op = _make_domain_validator("nccl")
validate_flash_attention_op = _make_domain_validator("flash_attention")
validate_flash_attn_op = validate_flash_attention_op
validate_chex_op = _make_domain_validator("chex")
validate_grain_op = _make_domain_validator("grain")
validate_orbax_op = _make_domain_validator("orbax")
validate_optax_op = _make_domain_validator("optax")


__all__ = [
    "DOMAIN_OP_CONFIGS",
    "DomainOpCheckConfig",
    "DomainOpRecord",
    "SimpleDomainValidateFn",
    "build_domain_op_tool_schemas",
    "check_domain_op",
    "validate_array_api_op",
    "validate_aten_op",
    "validate_chex_op",
    "validate_cpp_op",
    "validate_domain_op",
    "validate_flash_attention_op",
    "validate_flash_attn_op",
    "validate_grain_op",
    "validate_metal_op",
    "validate_nccl_op",
    "validate_onnx_op",
    "validate_optax_op",
    "validate_orbax_op",
    "validate_safetensors_op",
    "validate_scipy_op",
    "validate_torchaudio_op",
    "validate_torchvision_op",
    "validate_wasm_op",
    "validate_webgl_op",
    "validate_wgsl_op",
    "verify_domain_op_generic",
]
