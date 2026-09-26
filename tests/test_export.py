"""Module docstring."""

from typing import Any
from unittest.mock import patch

import pytest

from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind
from ml_ecosystem_snapshots.export import (
    to_json_schema,
    to_openapi,
    _ghost_to_cdd_ir,
    to_pydantic,
    to_protobuf,
    export_llm_prompt_context,
    export_sass_prompt_context,
    export_mlir_prompt_context,
    export_scoped_prompt_context,
)


@pytest.fixture
def sample_ghost_ref() -> GhostRef:
    """Function docstring.

    Returns:
        Return value.
    """
    return GhostRef(
        name="Linear",
        api_path="torch.nn.Linear",
        kind="class",
        docstring="Applies a linear transformation to the incoming data.",
        params=[
            GhostParam(
                name="in_features",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation="int",
                description="size of each input sample",
            ),
            GhostParam(
                name="out_features",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation="int",
                description="size of each output sample",
            ),
            GhostParam(
                name="bias",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation="bool",
                default="True",
                description='If set to "False", the layer will not learn an additive bias.',
            ),
            GhostParam(
                name="args",
                kind=ParameterKind.VAR_POSITIONAL,
                annotation="Any",
            ),
        ],
        returns_type="torch.Tensor",
        returns_description="A tensor of shape",
    )


def test_ghost_to_cdd_ir(sample_ghost_ref: Any) -> None:
    """Function docstring.

    Args:
        sample_ghost_ref: description
    """
    ir = _ghost_to_cdd_ir(sample_ghost_ref)

    assert ir["name"] == "Linear"
    assert ir["type"] == "class"
    assert ir["doc"] == "Applies a linear transformation to the incoming data."

    params = ir["params"]
    assert "in_features" in params
    assert params["in_features"]["typ"] == "int"
    assert params["in_features"]["doc"] == "size of each input sample"

    assert "out_features" in params
    assert params["out_features"]["typ"] == "int"

    assert "bias" in params
    assert params["bias"]["typ"] == "bool"
    assert params["bias"]["default"] == "True"

    returns = ir["returns"]
    assert returns is not None
    assert "return_type" in returns
    assert returns["return_type"]["typ"] == "torch.Tensor"
    assert returns["return_type"]["doc"] == "A tensor of shape"


def test_to_json_schema(sample_ghost_ref: Any) -> None:
    """Function docstring.

    Args:
        sample_ghost_ref: description
    """
    schema = to_json_schema(sample_ghost_ref)

    assert schema["$id"] == "torch.nn.Linear"
    assert schema["description"].startswith(
        "Applies a linear transformation to the incoming data."
    )
    assert schema["type"] == "object"

    properties = schema.get("properties", {})
    assert "in_features" in properties
    assert "out_features" in properties
    assert "bias" in properties


def test_to_openapi(sample_ghost_ref: Any) -> None:
    """Function docstring.

    Args:
        sample_ghost_ref: description
    """
    openapi_dict = to_openapi([sample_ghost_ref])

    assert openapi_dict["openapi"] == "3.0.0"

    paths = openapi_dict.get("paths", {})
    assert "/torch/nn/Linear" in paths

    components = openapi_dict.get("components", {})
    schemas = components.get("schemas", {})
    assert "Linear" in schemas


def test_to_pydantic(sample_ghost_ref: Any) -> None:
    """Function docstring.

    Args:
        sample_ghost_ref: description
    """
    code = to_pydantic(sample_ghost_ref)
    assert "class Linear(BaseModel):" in code
    assert (
        'in_features: int = Field(..., description="size of each input sample")' in code
    )
    assert (
        'out_features: int = Field(..., description="size of each output sample")'
        in code
    )
    assert (
        "bias: bool = Field(default=True, description=\"If set to 'False', the layer will not learn an additive bias.\")"
        in code
    )
    assert "args" not in code  # VAR_POSITIONAL skipped


def test_to_pydantic_empty() -> None:
    """Function docstring."""
    ref = GhostRef(name="Empty", api_path="a.Empty", kind="class", params=[])
    code = to_pydantic(ref)
    assert "class Empty(BaseModel):" in code
    assert "pass" in code


def test_to_protobuf(sample_ghost_ref: Any) -> None:
    """Function docstring.

    Args:
        sample_ghost_ref: description
    """
    code = to_protobuf(sample_ghost_ref)
    assert 'syntax = "proto3";' in code
    assert "message Linear {" in code
    assert "int64 in_features = 1;" in code
    assert "int64 out_features = 2;" in code
    assert "optional bool bias = 3;" in code
    assert "args" not in code


def test_to_protobuf_types() -> None:
    """Function docstring."""
    ref = GhostRef(
        name="Types",
        api_path="a.Types",
        kind="class",
        params=[
            GhostParam(name="s", kind=ParameterKind.POSITIONAL_ONLY, annotation="str"),
            GhostParam(
                name="l",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="list[int]",
            ),
            GhostParam(
                name="d",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="dict[str, int]",
            ),
            GhostParam(
                name="f", kind=ParameterKind.POSITIONAL_ONLY, annotation="float"
            ),
            GhostParam(name="u", kind=ParameterKind.POSITIONAL_ONLY, annotation=None),
            GhostParam(
                name="mode",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="InterpolationMode",
            ),
            GhostParam(
                name="unknown",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="weird_type",
            ),
        ],
    )
    code = to_protobuf(ref)
    assert "string s = 1;" in code
    assert "repeated string l = 2;" in code
    assert "map<string, string> d = 3;" in code
    assert "double f = 4;" in code
    assert "string u = 5;" in code
    assert "InterpolationMode mode = 6;" in code
    assert "string unknown = 7;" in code


def test_export_branches() -> None:
    """Function docstring."""
    from ml_ecosystem_snapshots.export import (
        _py_type_to_proto,
        to_pydantic,
        to_json_schema,
        _ghost_to_cdd_ir,
    )

    assert _py_type_to_proto(None) == "string"
    assert _py_type_to_proto("") == "string"

    from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind

    r = GhostRef(
        name="X",
        api_path="X",
        kind="function",
        params=[
            GhostParam(
                name="p1",
                kind=ParameterKind.KEYWORD_ONLY,
                default="None",
                annotation="int",
            ),
            GhostParam(
                name="p2",
                kind=ParameterKind.VAR_POSITIONAL,
                default="None",
                annotation="int",
            ),
        ],
    )
    to_pydantic(r)
    to_json_schema(r)

    r_empty_anno = GhostRef(
        name="Y",
        api_path="Y",
        kind="function",
        params=[GhostParam(name="p_empty", kind=ParameterKind.KEYWORD_ONLY)],
        returns_type="str",
    )  # No returns_description
    _ghost_to_cdd_ir(r_empty_anno)

    r_ret_desc = GhostRef(
        name="Z", api_path="Z", kind="function", params=[], returns_description="desc"
    )
    _ghost_to_cdd_ir(r_ret_desc)


def test_export_branches_more() -> None:
    """Function docstring."""
    from ml_ecosystem_snapshots.export import (
        to_pydantic,
        to_openapi,
        to_json_schema,
    )
    from ml_switcheroo_ir.schema.ghost import GhostParam
    from ml_switcheroo_ir.schema.ghost import GhostRef

    r = GhostRef(
        name="X",
        api_path="X",
        kind="class",
        docstring="doc",
        params=[
            GhostParam(
                name="p1",
                kind=ParameterKind.KEYWORD_ONLY,
                default="None",
                annotation="int",
            ),
            GhostParam(
                name="p2",
                kind=ParameterKind.VAR_POSITIONAL,
                default="None",
                annotation="int",
            ),
        ],
    )
    to_pydantic(r)
    to_openapi([r])
    to_json_schema(r)

    r2 = GhostRef(
        name="X",
        api_path="X",
        kind="function",
        returns_type="int",
        returns_description="desc",
        params=[
            GhostParam(
                name="p1",
                kind=ParameterKind.KEYWORD_ONLY,
                default="None",
                annotation="int",
                description="desc",
            ),
        ],
    )
    to_pydantic(r2)


def test_export_llm_prompt_context_full(sample_ghost_ref: GhostRef) -> None:
    """Test export_llm_prompt_context with complete GhostRef containing raises, env tags, returns, docstring.

    Args:
        sample_ghost_ref: Fixture providing a sample GhostRef.
    """
    sample_ghost_ref.raises = ["ValueError"]
    sample_ghost_ref.environment_tags = ["cpu", "cuda"]
    context = export_llm_prompt_context([sample_ghost_ref])
    assert "### `torch.nn.Linear`" in context
    assert "- **Signature**:" in context
    assert "- **Parameters**:" in context
    assert "-> torch.Tensor" in context
    assert "- **Raises**:" in context
    assert "ValueError" in context
    assert "- **Environments**: cpu, cuda" in context
    assert "- **Summary**:" in context

    # Test minimal GhostRef covering all False branches in export_llm_prompt_context
    bare_ref = GhostRef(name="bare", api_path="pkg.bare", kind="function", params=[])
    context_bare = export_llm_prompt_context([bare_ref])
    assert "### `pkg.bare`" in context_bare
    assert "- **Parameters**:" not in context_bare


def test_export_sass_prompt_context() -> None:
    """Test exporting SASS instruction prompt context with assembly syntax templates."""
    from ml_ecosystem_snapshots.models import (
        ExtendedGhostRef,
        ExtendedGhostParam,
        OperandDirection,
        IRParameterRole,
    )

    sass_ref = ExtendedGhostRef(
        name="FADD",
        api_path="nvidia_sass.inst.FADD",
        kind="function",
        docstring="Floating point add instruction.",
        params=[
            ExtendedGhostParam(
                name="op0",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="R",
                standardized_name="dst",
                direction=OperandDirection.WRITE,
                role=IRParameterRole.OPERAND,
            ),
            ExtendedGhostParam(
                name="op1",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="R",
                standardized_name="src0",
                direction=OperandDirection.READ,
                role=IRParameterRole.OPERAND,
            ),
        ],
        domain_metadata={
            "modifiers": [".FTZ", ".SAT"],
            "valid_architectures": ["sm_70", "sm_80", "sm_90"],
            "operand_signatures": [["R", "R", "R"]],
        },
    )

    context = export_sass_prompt_context([sass_ref])
    assert "### `FADD`" in context
    assert "[@P0] FADD" in context
    assert "sm_70" in context
    assert ".FTZ" in context
    assert "WRITE" in context

    # Test minimal SASS ref with no operands, no modifiers, and no metadata
    sass_minimal = ExtendedGhostRef(
        name="NOP",
        api_path="nvidia_sass.inst.NOP",
        kind="function",
        params=[
            ExtendedGhostParam(
                name="raw_op",
                kind=ParameterKind.POSITIONAL_ONLY,
            )
        ],
    )
    context_min = export_sass_prompt_context([sass_minimal])
    assert "### `NOP`" in context_min
    assert "[@P0] NOP;" in context_min
    assert "raw_op" in context_min

    # Test SASS ref with only "cuda" in environment_tags and no params/modifiers
    sass_cuda_only = ExtendedGhostRef(
        name="SYNC",
        api_path="nvidia_sass.inst.SYNC",
        kind="function",
        environment_tags=["cuda"],
        params=[],
    )
    context_cuda = export_sass_prompt_context([sass_cuda_only])
    assert "### `SYNC`" in context_cuda
    assert "Architectures" not in context_cuda
    assert "Modifiers" not in context_cuda


def test_export_mlir_prompt_context() -> None:
    """Test exporting MLIR/StableHLO prompt context with SSA syntax templates."""
    from ml_ecosystem_snapshots.models import (
        ExtendedGhostRef,
        ExtendedGhostParam,
        GhostResult,
        IRParameterRole,
    )

    mlir_ref = ExtendedGhostRef(
        name="AddFOp",
        api_path="arith.addf",
        kind="function",
        docstring="Floating point add operation.",
        returns_type="FloatLike",
        returns=[GhostResult(name="result", type="FloatLike")],
        params=[
            ExtendedGhostParam(
                name="lhs",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation="FloatLike",
                role=IRParameterRole.OPERAND,
            ),
            ExtendedGhostParam(
                name="fastmath",
                kind=ParameterKind.KEYWORD_ONLY,
                annotation="FastMathFlagsAttr",
                role=IRParameterRole.ATTRIBUTE,
            ),
            ExtendedGhostParam(
                name="body_reg",
                kind=ParameterKind.KEYWORD_ONLY,
                annotation="Region",
                role=IRParameterRole.REGION,
            ),
            GhostParam(
                name="region_untyped",
                kind=ParameterKind.KEYWORD_ONLY,
                annotation="Region",
            ),
        ],
        domain_metadata={
            "traits": ["Commutative", "Pure"],
            "regions": ["body"],
        },
    )

    context = export_mlir_prompt_context([mlir_ref])
    assert "### `arith.addf`" in context
    assert "%res = arith.addf(%lhs)" in context
    assert "{fastmath = ...}" in context
    assert "%lhs" in context
    assert "FastMathFlagsAttr" in context
    assert "%result" in context
    assert "Commutative" in context
    assert "**Regions**: body" in context

    # Test minimal MLIR ref without operands or attributes or returns
    mlir_minimal = ExtendedGhostRef(
        name="BarrierOp",
        api_path="gpu.barrier",
        kind="function",
    )
    context_min = export_mlir_prompt_context([mlir_minimal])
    assert "### `gpu.barrier`" in context_min
    assert "%res = gpu.barrier() : none" in context_min


def test_export_scoped_prompt_context(mocker: Any) -> None:
    """Test export_scoped_prompt_context with hierarchical indexing and scoping across domains.

    Args:
        mocker: Pytest mocker fixture.
    """
    mock_torch_snap = {
        "categories": {
            "nn": [
                {
                    "name": f"Module{i}",
                    "api_path": f"torch.nn.Module{i}",
                    "kind": "class",
                    "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}],
                }
                for i in range(10)
            ]
        }
    }
    from ml_ecosystem_snapshots import mcp_server

    orig_get_snap = mcp_server.get_framework_snapshot

    def mock_get_snap(framework: str, version: Any = None) -> Any:
        """Mock framework snapshot getter returning mock torch snapshot.

        Args:
            framework: Framework identifier.
            version: Optional version identifier.

        Returns:
            Snapshot dictionary or original snapshot result.
        """
        if framework == "torch":
            return mock_torch_snap
        return orig_get_snap(framework, version=version)

    mocker.patch(
        "ml_ecosystem_snapshots.mcp_server.get_framework_snapshot",
        side_effect=mock_get_snap,
    )

    # 1. Scoped torch context with module prefix
    torch_ctx = export_scoped_prompt_context(
        "torch", module_prefix="torch.nn", max_symbols=3
    )
    assert "# Framework Grounding Context: `torch`" in torch_ctx
    assert "## Index of Available Operations" in torch_ctx
    assert "## Detailed Operation Signatures" in torch_ctx
    assert "torch.nn" in torch_ctx
    assert "*(Showing 3 of" in torch_ctx

    # 2. Scoped SASS context
    sass_ctx = export_scoped_prompt_context("nvidia_sass", max_symbols=5)
    assert "# Framework Grounding Context: `nvidia_sass`" in sass_ctx
    assert "- **Operands**:" in sass_ctx or "- **Modifiers**:" in sass_ctx

    # 3. Scoped StableHLO context
    stablehlo_ctx = export_scoped_prompt_context("stablehlo", max_symbols=5)
    assert "# Framework Grounding Context: `stablehlo`" in stablehlo_ctx
    assert "- **Syntax**:" in stablehlo_ctx

    # 4. Scoped context when total_matching <= max_symbols and with corrupt item
    mock_snap = {
        "categories": {
            "test": [
                {"name": "op1", "api_path": "test.op1", "kind": "function"},
                "corrupt_non_dict_item",
                {"api_path": "test.invalid_fields", "kind": 12345},
            ]
        }
    }
    with patch(
        "ml_ecosystem_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    ):
        ctx = export_scoped_prompt_context("test_fw", max_symbols=50)
        assert "# Framework Grounding Context: `test_fw`" in ctx
        assert "*(Showing" not in ctx


def test_export_extended_prompt_contexts_and_headers() -> None:
    """Test export_rdna_prompt_context, export_ptx_prompt_context, export_stablehlo_prompt_context, to_cpp_header, and to_typescript_interface."""
    from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind
    from ml_ecosystem_snapshots.export import (
        export_ptx_prompt_context,
        export_rdna_prompt_context,
        export_stablehlo_prompt_context,
        to_cpp_header,
        to_typescript_interface,
    )

    rdna_ref = GhostRef(
        name="v_add_f32",
        api_path="v_add_f32",
        kind="instruction",
        params=[
            GhostParam(
                name="vdst",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="vgpr",
            ),
            GhostParam(
                name="vsrc0",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="vgpr",
            ),
            GhostParam(
                name="vsrc1",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="vgpr",
            ),
        ],
        environment_tags=["VOP2", "gfx1100"],
        docstring="Vector floating point addition.",
    )
    rdna_ctx = export_rdna_prompt_context([rdna_ref])
    assert "### `v_add_f32`" in rdna_ctx
    assert "**Encoding**: `VOP2`" in rdna_ctx
    assert "gfx1100" in rdna_ctx

    ptx_ref = GhostRef(
        name="add",
        api_path="add",
        kind="instruction",
        params=[
            GhostParam(name="d", kind=ParameterKind.POSITIONAL_ONLY, annotation="reg"),
            GhostParam(name="a", kind=ParameterKind.POSITIONAL_ONLY, annotation="reg"),
            GhostParam(name="b", kind=ParameterKind.POSITIONAL_ONLY, annotation="reg"),
        ],
        environment_tags=["sm_50", ".s32", ".f32"],
        docstring="PTX integer addition.",
    )
    ptx_ctx = export_ptx_prompt_context([ptx_ref])
    assert "### `add`" in ptx_ctx
    assert "**Minimum SM**: `sm_50`" in ptx_ctx
    assert ".s32, .f32" in ptx_ctx

    hlo_ref = GhostRef(
        name="stablehlo.add",
        api_path="stablehlo.add",
        kind="operation",
        params=[
            GhostParam(
                name="lhs",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="tensor",
            ),
            GhostParam(
                name="rhs",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="tensor",
            ),
        ],
    )
    hlo_ctx = export_stablehlo_prompt_context([hlo_ref])
    assert "stablehlo.add" in hlo_ctx

    # C++ and TypeScript exports with various types
    typed_ref = GhostRef(
        name="AllTypes",
        api_path="AllTypes",
        kind="function",
        params=[
            GhostParam(
                name="f_val",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation="float",
            ),
            GhostParam(
                name="i_val",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation="int",
            ),
            GhostParam(
                name="b_val",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation="bool",
            ),
            GhostParam(
                name="s_val",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation="str",
            ),
            GhostParam(
                name="l_val",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation="list[int]",
                default="None",
            ),
        ],
        docstring="Docstring for AllTypes.",
    )
    cpp_code = to_cpp_header(typed_ref, namespace="test_ns")
    assert "namespace test_ns {" in cpp_code
    assert "struct AllTypes {" in cpp_code
    assert "int64_t i_val;" in cpp_code
    assert "bool b_val;" in cpp_code
    assert "std::string s_val;" in cpp_code
    assert "std::vector<float> l_val;" in cpp_code

    ts_code = to_typescript_interface(typed_ref)
    assert "export interface AllTypes {" in ts_code
    assert "i_val: number;" in ts_code
    assert "b_val: boolean;" in ts_code
    assert "s_val: string;" in ts_code
    assert "l_val?: unknown[];" in ts_code


def test_export_coverage_hardening() -> None:
    """Test edge cases across to_pydantic, to_protobuf, export_rdna_prompt_context, export_ptx_prompt_context, and export_scoped_prompt_context."""
    from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind
    from ml_ecosystem_snapshots.export import (
        _py_type_to_proto,
        export_ptx_prompt_context,
        export_rdna_prompt_context,
        export_scoped_prompt_context,
        to_protobuf,
        to_pydantic,
    )

    # 1. to_pydantic with VAR_POSITIONAL, VAR_KEYWORD, defaults with description, and overloads with GhostRefs
    ov_ref = GhostRef(
        name="FnOverload",
        api_path="fn.overload",
        kind="function",
        params=[
            GhostParam(name="x", kind=ParameterKind.POSITIONAL_ONLY, annotation="int")
        ],
        docstring="Variant docstring",
    )
    main_ref = GhostRef(
        name="FnWithVars",
        api_path="fn.vars",
        kind="function",
        params=[
            GhostParam(
                name="args",
                kind=ParameterKind.VAR_POSITIONAL,
                annotation="str",
                description="varargs",
            ),
            GhostParam(
                name="kwargs",
                kind=ParameterKind.VAR_KEYWORD,
                annotation="Any",
                description="kwargs",
            ),
            GhostParam(
                name="opt",
                kind=ParameterKind.KEYWORD_ONLY,
                annotation="int",
                default="10",
                description="opt val",
            ),
            GhostParam(
                name="no_desc_opt",
                kind=ParameterKind.KEYWORD_ONLY,
                annotation="int",
                default="20",
            ),
        ],
        overloads=[ov_ref],
        docstring="Main function docstring",
    )
    pyd_code = to_pydantic(main_ref, validate_varargs=True)
    assert "FnWithVarsVariant0" in pyd_code
    assert "FnWithVarsVariant1" in pyd_code
    assert "FnWithVars = Union[FnWithVarsVariant0, FnWithVarsVariant1]" in pyd_code

    # 2. _py_type_to_proto edge cases
    assert _py_type_to_proto(None) == "string"
    assert _py_type_to_proto("tensor") == "TensorProto"
    assert _py_type_to_proto("int", param_name="shape") == "repeated int64"
    assert _py_type_to_proto("tuple[int, ...]", param_name="other") == "repeated int64"
    assert _py_type_to_proto("int", param_name="reduction") == "ReductionType"
    assert _py_type_to_proto("int", param_name="padding") == "PaddingMode"
    assert _py_type_to_proto("int", param_name="layout") == "LayoutMode"
    assert (
        _py_type_to_proto("interpolation_mode", param_name="mode")
        == "InterpolationMode"
    )
    assert _py_type_to_proto("dict[str, int]") == "map<string, string>"
    assert _py_type_to_proto("bool") == "bool"
    assert _py_type_to_proto("str") == "string"
    assert _py_type_to_proto("bytes") == "string"
    assert _py_type_to_proto("complex") == "string"

    # to_protobuf with package
    proto_def = to_protobuf(main_ref, package="custom_pkg")
    assert 'syntax = "proto3";' in proto_def
    assert "package custom_pkg;" in proto_def
    assert "message FnWithVars {" in proto_def

    # 3. export_rdna_prompt_context with various tags (DS, FLAT, SOP, SMEM) and minimal ref
    ds_ref = GhostRef(
        name="ds_add_rtn_u32",
        api_path="ds_add_rtn_u32",
        kind="instruction",
        params=[
            GhostParam(
                name="vdst", kind=ParameterKind.POSITIONAL_ONLY, annotation="vgpr"
            )
        ],
        environment_tags=["DS_ADD", "gfx900"],
    )
    flat_ref = GhostRef(
        name="flat_load_dword",
        api_path="flat_load_dword",
        kind="instruction",
        params=[],
        environment_tags=["FLAT_LOAD", "gfx1030"],
    )
    minimal_rdna = GhostRef(
        name="v_nop",
        api_path="v_nop",
        kind="instruction",
        params=[],
        environment_tags=["VOP1"],
    )
    rdna_out = export_rdna_prompt_context([ds_ref, flat_ref, minimal_rdna])
    assert "### `ds_add_rtn_u32`" in rdna_out
    assert "**Encoding**: `DS_ADD`" in rdna_out
    assert "### `flat_load_dword`" in rdna_out
    assert "**Encoding**: `FLAT_LOAD`" in rdna_out
    assert "### `v_nop`" in rdna_out

    # 4. export_ptx_prompt_context with various tags and minimal ref
    ptx_mod_ref = GhostRef(
        name="mma_op",
        api_path="mma_op",
        kind="instruction",
        params=[],
        environment_tags=["sm_90", ".b32", ".cta", "non_modifier_tag"],
    )
    minimal_ptx = GhostRef(
        name="nop",
        api_path="nop",
        kind="instruction",
        params=[],
        environment_tags=["sm_70"],
    )
    ptx_out = export_ptx_prompt_context([ptx_mod_ref, minimal_ptx])
    assert "### `mma_op`" in ptx_out
    assert "**Minimum SM**: `sm_90`" in ptx_out
    assert ".b32, .cta" in ptx_out
    assert "### `nop`" in ptx_out

    # 5. export_scoped_prompt_context with module_prefix and non-dict items
    mock_mixed_snap = {
        "categories": {
            "nn": [
                {
                    "name": "Linear",
                    "api_path": "torch.nn.Linear",
                    "kind": "class",
                    "params": [],
                },
                {
                    "name": "Conv2d",
                    "api_path": "torch.nn.Conv2d",
                    "kind": "class",
                    "params": [],
                },
                "not_a_dict",
            ],
            "optim": [
                {
                    "name": "SGD",
                    "api_path": "torch.optim.SGD",
                    "kind": "class",
                    "params": [],
                },
            ],
        }
    }
    with patch(
        "ml_ecosystem_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_mixed_snap,
    ):
        scoped_nn = export_scoped_prompt_context("torch", module_prefix="torch.nn")
        assert "Linear" in scoped_nn
        assert "Conv2d" in scoped_nn
        assert "SGD" not in scoped_nn

    # 6. export_llm_prompt_context constraints branch and no-docstring headers
    from ml_ecosystem_snapshots.models import ExtendedGhostParam
    from ml_ecosystem_snapshots.export import (
        export_llm_prompt_context,
        to_cpp_header,
        to_typescript_interface,
    )

    constrained_param = ExtendedGhostParam(
        name="input_t",
        kind=ParameterKind.POSITIONAL_ONLY,
        annotation="Tensor",
        allowed_dtypes=["float32", "bfloat16"],
        rank_constraint="4",
    )
    constrained_ref = GhostRef(
        name="CustomNorm",
        api_path="torch.nn.CustomNorm",
        kind="class",
        params=[constrained_param],
    )
    llm_out = export_llm_prompt_context([constrained_ref])
    assert "dtypes: ['float32', 'bfloat16']" in llm_out
    assert "rank: 4" in llm_out

    # no docstring and unannotated parameter header / interface
    no_doc_ref = GhostRef(
        name="SimpleStruct",
        api_path="SimpleStruct",
        kind="class",
        params=[
            GhostParam(name="val", kind=ParameterKind.POSITIONAL_ONLY, annotation=None),
            GhostParam(
                name="custom_t",
                kind=ParameterKind.POSITIONAL_ONLY,
                annotation="MyCustomClass",
            ),
        ],
    )
    cpp_no_doc = to_cpp_header(no_doc_ref)
    assert "struct SimpleStruct {" in cpp_no_doc
    assert "float val;" in cpp_no_doc
    ts_no_doc = to_typescript_interface(no_doc_ref)
    assert "export interface SimpleStruct {" in ts_no_doc
    assert "val: unknown;" in ts_no_doc
    assert "custom_t: unknown;" in ts_no_doc
