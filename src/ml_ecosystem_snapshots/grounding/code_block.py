"""AST and assembly/IR snippet call-site verifier (`verify_code_block`)."""

from __future__ import annotations

import ast
import re
import typing

from ml_ecosystem_snapshots.grounding.models import (
    CodeBlockCheckResult,
    CodeBlockFinding,
    HallucinationCheckResult,
    IrCheckResult,
    IsaCheckResult,
)
from ml_ecosystem_snapshots.models import JsonValue


class _CallVisitor(ast.NodeVisitor):
    """AST visitor for tracking module imports and collecting qualified function calls."""

    def __init__(self) -> None:
        """Initialize empty import table and call records list."""
        self.imports: dict[str, str] = {}
        self.calls: list[tuple[int, str, list[str], int, dict[str, JsonValue]]] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Track 'import foo' and 'import foo as bar' statements.

        Args:
            node: The AST Import node.
        """
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            self.imports[asname] = name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track 'from foo.bar import baz' statements.

        Args:
            node: The AST ImportFrom node.
        """
        mod = node.module or ""
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            full = f"{mod}.{name}" if mod else name
            self.imports[asname] = full
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Inspect Call AST nodes and record resolved target path and constant kwargs.

        Args:
            node: The AST Call node.
        """
        curr: ast.expr = node.func
        attr_parts: list[str] = []
        while isinstance(curr, ast.Attribute):
            attr_parts.append(curr.attr)
            curr = curr.value

        root_id = curr.id if isinstance(curr, ast.Name) else None
        if root_id is not None:
            resolved_base = self.imports.get(root_id, root_id)
            if attr_parts:
                full_path = f"{resolved_base}.{'.'.join(reversed(attr_parts))}"
            else:
                full_path = resolved_base

            kw_names = [kw.arg for kw in node.keywords if kw.arg]
            kw_vals: dict[str, JsonValue] = {}
            for kw in node.keywords:
                if kw.arg and isinstance(kw.value, ast.Constant):
                    val = kw.value.value
                    kw_vals[kw.arg] = val if isinstance(val, (str, int, float, bool)) else None
            self.calls.append((node.lineno, full_path, kw_names, len(node.args), kw_vals))
        self.generic_visit(node)


_CALL_PREFIX_FRAMEWORK_MAP: tuple[tuple[str, str, str | None], ...] = (
    ("torch.", "torch", None),
    ("torch", "torch", None),
    ("jax.", "jax", None),
    ("jax", "jax", None),
    ("tf.", "tensorflow", None),
    ("tensorflow.", "tensorflow", None),
    ("tensorflow", "tensorflow", None),
    ("np.", "numpy", None),
    ("numpy.", "numpy", None),
    ("numpy", "numpy", None),
    ("nn.", "torch", "torch.nn."),
    ("F.", "torch", "torch.F."),
    ("stablehlo.", "stablehlo", None),
    ("chex.", "chex", None),
    ("grain.", "grain", None),
    ("orbax.", "orbax", None),
    ("ocp.", "orbax", "orbax.checkpoint."),
    ("optax.", "optax", None),
)


def _infer_call_framework(
    api_name: str,
    framework_hint: str | None,
) -> tuple[str | None, str]:
    """Infer target framework and normalized API path from a Python call target.

    Args:
        api_name: Dotted API path extracted from the AST Call node.
        framework_hint: Optional caller-supplied framework name.

    Returns:
        Tuple of `(resolved_framework, normalized_api_name)`.
    """
    if framework_hint:
        return framework_hint, api_name
    for prefix, fw, replacement in _CALL_PREFIX_FRAMEWORK_MAP:
        if api_name.startswith(prefix):
            norm_api = f"{replacement}{api_name[len(prefix) :]}" if replacement else api_name
            return fw, norm_api
    return None, api_name


_MLIR_DIALECT_PREFIXES: tuple[str, ...] = (
    "arith.",
    "math.",
    "linalg.",
    "stablehlo.",
    "memref.",
    "func.",
    "scf.",
    "gpu.",
    "vector.",
    "llvm.",
    "nvvm.",
    "rocdl.",
)


class PythonCallCheckerProto(typing.Protocol):
    """Protocol for Python API call hallucination checker callbacks."""

    def __call__(
        self,
        framework: str,
        api_path: str,
        *,
        kwargs: list[str] | None = None,
        args_count: int | None = None,
        kwarg_values: typing.Mapping[str, JsonValue] | None = None,
    ) -> HallucinationCheckResult:
        """Verify a Python call site against framework snapshots.

        Args:
            framework: Framework name.
            api_path: Qualified API path.
            kwargs: Passed keyword argument names.
            args_count: Passed positional argument count.
            kwarg_values: Passed literal keyword argument values.

        Returns:
            `HallucinationCheckResult` report.
        """
        ...


def _check_python_calls(
    tree: ast.AST,
    framework: str | None,
    call_checker: PythonCallCheckerProto,
) -> tuple[int, list[CodeBlockFinding]]:
    """Extract and verify all Python call sites within a parsed AST module.

    Args:
        tree: Parsed Python `ast.AST` root node.
        framework: Optional explicit framework override.
        call_checker: Callable verifying `(fw, api_path, kwargs=..., args_count=..., kwarg_values=...)`.

    Returns:
        Tuple of `(total_analyzed, findings)`.
    """
    visitor = _CallVisitor()
    visitor.visit(tree)

    findings: list[CodeBlockFinding] = []
    total_analyzed = 0

    for lineno, parts, kw_names, arg_cnt, kw_vals in visitor.calls:
        fw, api_name = _infer_call_framework(parts, framework)
        if not fw:
            continue

        total_analyzed += 1
        res = call_checker(
            fw,
            api_name,
            kwargs=kw_names,
            args_count=arg_cnt,
            kwarg_values=kw_vals,
        )
        if res.is_hallucinated:
            findings.append(
                CodeBlockFinding(
                    line=lineno,
                    type="api_hallucination",
                    framework=fw,
                    target=api_name,
                    reason=res.reason or "Invalid API call",
                    invalid_kwargs=list(res.invalid_kwargs),
                )
            )
    return total_analyzed, findings


def _check_assembly_or_ir_line(
    lineno: int,
    line: str,
    framework: str | None,
    sm_arch: str | None,
    gfx_arch: str | None,
    mlir_checker: typing.Callable[[str], IrCheckResult],
    sass_checker: typing.Callable[[str, list[str], list[str], str | None], IsaCheckResult],
    rdna_checker: typing.Callable[[str, list[str], list[str], str | None], IsaCheckResult],
) -> tuple[int, CodeBlockFinding | None]:
    """Validate a single non-Python assembly or MLIR/StableHLO instruction line.

    Args:
        lineno: 1-based source line number.
        line: Stripped non-empty, non-comment instruction line.
        framework: Optional framework hint (mlir, stablehlo, nvidia_sass, amd_rdna).
        sm_arch: Optional NVIDIA SM architecture target.
        gfx_arch: Optional AMD RDNA GFX architecture target.
        mlir_checker: MLIR/StableHLO operation verifier callback.
        sass_checker: NVIDIA SASS instruction verifier callback.
        rdna_checker: AMD RDNA instruction verifier callback.

    Returns:
        Tuple of `(analyzed_increment, optional_finding)`.
    """
    mlir_match = re.search(r"(?:%[a-zA-Z0-9_]+\s*=\s*)?([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)", line)
    sass_match = re.match(r"^(?:@!?U?P\d+\s+)?([A-Z0-9_\.]+)(?:\s+(.*))?;?$", line)
    rdna_match = re.match(r"^([vsa]_[a-zA-Z0-9_\.]+)(?:\s+(.*))?$", line)

    if mlir_match and (framework in ("mlir", "stablehlo") or any(d in line for d in _MLIR_DIALECT_PREFIXES)):
        op_name = mlir_match.group(1)
        mlir_res = mlir_checker(op_name)
        finding = (
            None
            if mlir_res.is_valid
            else CodeBlockFinding(
                line=lineno,
                type="mlir_error",
                target=op_name,
                reason="; ".join(mlir_res.errors),
            )
        )
        return 1, finding

    if sass_match and (
        framework == "nvidia_sass" or any(s in line for s in ("R0", "UR0", "P0", "sm_")) or line.endswith(";")
    ):
        from ml_ecosystem_snapshots.frameworks.nvidia_sass import tokenize_sass_line

        _pred, mnem, mods, ops = tokenize_sass_line(line)
        sass_res = sass_checker(mnem, ops, mods, sm_arch)
        finding = (
            None
            if sass_res.is_valid
            else CodeBlockFinding(
                line=lineno,
                type="sass_error",
                target=mnem,
                reason="; ".join(sass_res.errors),
            )
        )
        return 1, finding

    if rdna_match:
        from ml_ecosystem_snapshots.frameworks.amd_rdna import tokenize_rdna_line

        base_mnem, ops, _enc_suf, mods = tokenize_rdna_line(line)
        rdna_res = rdna_checker(base_mnem, ops, mods, gfx_arch)
        finding = (
            None
            if rdna_res.is_valid
            else CodeBlockFinding(
                line=lineno,
                type="rdna_error",
                target=base_mnem,
                reason="; ".join(rdna_res.errors),
            )
        )
        return 1, finding

    return 0, None


def verify_code_block(
    code: str,
    call_checker: PythonCallCheckerProto,
    mlir_checker: typing.Callable[[str], IrCheckResult],
    sass_checker: typing.Callable[[str, list[str], list[str], str | None], IsaCheckResult],
    rdna_checker: typing.Callable[[str, list[str], list[str], str | None], IsaCheckResult],
    framework: str | None = None,
    sm_arch: str | None = None,
    gfx_arch: str | None = None,
) -> CodeBlockCheckResult:
    """Analyze a multi-line code snippet (Python, MLIR, SASS, or RDNA) for API/ISA hallucinations.

    Args:
        code: Raw source code block string.
        call_checker: Python call verifier callback.
        mlir_checker: MLIR/StableHLO op verifier callback.
        sass_checker: NVIDIA SASS instruction verifier callback.
        rdna_checker: AMD RDNA instruction verifier callback.
        framework: Optional explicit framework override.
        sm_arch: Optional NVIDIA architecture target (e.g. sm_80, sm_90).
        gfx_arch: Optional AMD RDNA architecture target (e.g. GFX11/RDNA3).

    Returns:
        Structured `CodeBlockCheckResult` report.
    """
    findings: list[CodeBlockFinding] = []
    total_analyzed = 0

    is_explicit_non_python = framework in ("nvidia_sass", "amd_rdna", "mlir", "stablehlo")
    parsed_as_python = False

    if not is_explicit_non_python:
        try:
            tree = ast.parse(code)
            parsed_as_python = True
            total_analyzed, findings = _check_python_calls(tree, framework, call_checker)
        except SyntaxError:
            parsed_as_python = False

    if not parsed_as_python or (total_analyzed == 0 and any(p in code for p in ("%", "R", "v_", "s_"))):
        for lineno, raw_line in enumerate(code.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith(("//", "#", ";", "/*")):
                continue
            inc, finding = _check_assembly_or_ir_line(
                lineno=lineno,
                line=line,
                framework=framework,
                sm_arch=sm_arch,
                gfx_arch=gfx_arch,
                mlir_checker=mlir_checker,
                sass_checker=sass_checker,
                rdna_checker=rdna_checker,
            )
            total_analyzed += inc
            if finding is not None:
                findings.append(finding)

    return CodeBlockCheckResult(
        total_analyzed=total_analyzed,
        hallucinations_detected=len(findings),
        is_valid=len(findings) == 0,
        findings=findings,
    )


__all__ = ["verify_code_block"]
