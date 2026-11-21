# Architecture

The `ml_ecosystem_snapshots` library is engineered to solve a critical foundational problem: **Introspecting dynamic, C-extension heavy Machine Learning frameworks, compiler intermediate representations (IR), and hardware instruction set architectures (ISAs) safely, deterministically, and offline**, then serializing that structural data into lightweight, type-safe schemas that power anti-hallucination grounding engines, transpilers, IDE stubs, and Model Context Protocol (MCP) servers.

---

## 1. Core Abstraction: The Ghost Protocol & Domain-Specialized Models

At the center of the architecture is the **Ghost Protocol**. Instead of relying on live, heavyweight Python runtime objects (which allocate GPU memory, spawn driver contexts, lock C++ thread state, or depend on gigabyte-sized binary wheels), every framework API, compiler operation, and machine instruction is reduced into a static, serializable representation.

All models inherit from Pydantic V2 schemas defined in `ml_ecosystem_snapshots.models`, `ml_ecosystem_snapshots.serialization`, and `ml_switcheroo_ir.schema.ghost`.

```mermaid
classDiagram
    class GhostOverload {
        +List~GhostParam~ params
        +str returns_type
        +str returns_description
        +str docstring
    }

    class GhostRef {
        +str name
        +str api_path
        +str kind
        +str docstring
        +List~GhostParam~ params
        +str returns_type
        +str returns_description
        +List~str~ raises
        +List~GhostOverload_or_GhostRef~ overloads
        +List~str~ environment_tags
        +List~str~ aliases
        +bool is_public
    }

    class ExtendedGhostRef {
        +List~GhostResult~ returns
        +Dict~str, JsonValue~ domain_metadata
        +str signature_completeness
        +bool is_c_extension
        +List~str~ accepted_kwargs
    }

    class GhostPythonRef {
        +Literal["python"] domain_type
    }

    class GhostIsaRef {
        +Literal["isa"] domain_type
        +List~str~ predicate_guards
        +Dict~str, str~ register_classes
        +Dict~str, JsonValue~ control_codes
        +List~str~ instruction_modifiers
        +Dict~str, JsonValue~ structured_modifiers
        +List~Dict~ structured_operands
        +Dict~str, JsonValue~ vopd_profile
        +List~str~ condition_codes
        +List~str~ supported_architectures
    }

    class GhostMlirRef {
        +Literal["mlir"] domain_type
        +List~str~ traits
        +List~ExtendedGhostParam~ operands
        +Dict~str, JsonValue~ attributes
        +Dict~str, JsonValue~ regions
        +List~str~ successors
        +Dict~str, str~ type_constraints
    }

    class SnapshotEnvelope {
        +str schema_version
        +str target
        +str version
        +str upstream_version
        +str source_type
        +str upstream_commit
        +List~str~ supported_microarchitectures
        +str generated_at
        +Dict~str, JsonValue~ environment
        +Dict~str, List~ categories
    }

    GhostRef o-- GhostOverload
    GhostRef <|-- ExtendedGhostRef
    ExtendedGhostRef <|-- GhostPythonRef
    ExtendedGhostRef <|-- GhostIsaRef
    ExtendedGhostRef <|-- GhostMlirRef
    SnapshotEnvelope o-- ExtendedGhostRef
```

### Core Data Models

- **`GhostParam` & `ExtendedGhostParam`**: Represents an individual parameter or operand. Extends standard Python parameter signatures with:
  - `direction`: `OperandDirection` (`READ`, `WRITE`, `READ_WRITE`, `PREDICATE`) for hardware instructions.
  - `role`: `IRParameterRole` (`OPERAND`, `ATTRIBUTE`, `RESULT`, `SUCCESSOR`, `REGION`) for compiler IRs.
  - `dtypes` & `allowed_dtypes`: Explicit lists of allowed tensor data types (e.g. `["float32", "bfloat16"]`).
  - `rank`: Expected tensor rank or dimensionality constraints.
  - `default`: Sanitized literal default value representation.
- **`GhostOverload`**: Lightweight parameter-signature variant model stored alongside full `GhostRef` entries inside `GhostRef.overloads` (`list[dict[str, JsonValue] | GhostOverload | GhostRef] | None`).
- **`GhostResult`**: Models structured SSA output returns for compiler IR operations (SSA result name, MLIR type like `tensor<?x?xf32>`, and description).
- **`ExtendedGhostRef`**: Adds first-class support for:
  - `returns`: List of `GhostResult` objects, supporting multi-result SSA operations.
  - `domain_metadata`: Structured `dict[str, JsonValue]` holding domain-specific attributes (control code schedules, operand signatures, traits).
  - `signature_completeness`: Classification of signature certainty (`exact`, `heuristic`, or `opaque`).
  - `is_c_extension`: Flag identifying symbols originating from compiled C/C++ extensions.
  - `accepted_kwargs`: Explicit list of recognized keyword arguments extracted from function ASTs that accept generic `**kwargs`.
- **`GhostPythonRef` (`domain_type="python"`)**: High-level Python ML frameworks (PyTorch, JAX, TensorFlow, Keras, Flax NNX, HuggingFace, scikit-learn, etc.).
- **`GhostIsaRef` (`GhostInstructionRef`, `domain_type="isa"`)**: GPU assembly instruction sets (NVIDIA SASS, AMD RDNA/CDNA, NVIDIA PTX).
- **`GhostMlirRef` (`GhostOperationRef`, `domain_type="mlir"`)**: Compiler intermediate representations (Core MLIR dialects and StableHLO).
- **`SnapshotEnvelope`**: Provenance container capturing schema version (`2.0.0`), upstream commit hash, target version, microarchitecture capabilities, host build environment, generation timestamp, and categorized symbol maps.

---

## 2. End-to-End System Architecture

The system operates across three primary stages: **Extraction & Ingestion**, **Grounding & Verification**, and **Downstream Consumption**.

```mermaid
graph TD
    subgraph "Stage 1: Ingestion & Introspection"
        CLI[CLI: capture / pull] --> API[api.py: extract_snapshot]
        SDK[Python SDK] --> API

        API --> ISO{Isolated Subprocess?}
        ISO -->|Yes| PROC[extract_snapshot_isolated]
        ISO -->|No| REG[FRAMEWORK_COLLECTORS]

        PROC --> REG
        REG -->|PyTorch, JAX, TF, Keras...| PY_COL[Python Framework Collectors]
        REG -->|NVIDIA SASS, AMD RDNA, PTX| ISA_COL[ISA Exhaustive Catalogs]
        REG -->|StableHLO, Core MLIR| IR_COL[TableGen / ODS Parsers]

        PY_COL --> GI[introspection/inspector.py: GhostInspector.inspect]
        GI --> UNWRAP[runtime_boundary.py: Decorator Unwrapping]
        UNWRAP --> CDD[cdd_inspector.py: cdd-python AST & Docstrings]
        UNWRAP --> GRIFFE[griffe_inspector.py: Types & Overloads]
        UNWRAP --> RUNTIME[live_inspector.py: Runtime Signature Fallback]
        UNWRAP --> CEXT[c_extension_sig.py: C-Extension Docstring Parser]
        UNWRAP --> KWVIS[ast_kwargs.py: KwargAccessVisitor **kwargs Analysis]

        CDD --> SAN[sanitize.py: Memory Address & Value Sanitizer]
        GRIFFE --> SAN
        RUNTIME --> SAN
        CEXT --> SAN
        KWVIS --> SAN

        ISA_COL --> SAN
        IR_COL --> SAN

        SAN --> DEDUP[consolidate_aliases]
        DEDUP --> ENV[serialization/: SnapshotEnvelope JSON / JSON.GZ]
    end

    subgraph "Stage 2: Storage & Indexing"
        ENV --> DISK[(Disk Cache / Bundled Snapshots)]
        DISK --> FTS[index.py: Ephemeral SQLite FTS5 Index]
        DISK --> ENGINE[grounding/engine.py: In-Memory Target Registry]
    end

    subgraph "Stage 3: Verification, Protocol & Export"
        ENGINE --> G_SDK[grounding/: Grounding SDK]
        G_SDK --> V_PY[validate_python_call]
        G_SDK --> V_ISA[validate_sass / validate_rdna / validate_ptx]
        G_SDK --> V_IR[validate_mlir / validate_stablehlo / validate_wgsl / domain ops]

        DISK --> MCP[mcp_server.py & mcp/: JSON-RPC 2.0 Server]
        MCP --> T_SIG[get_api_signature / search_apis]
        MCP --> T_HAL[check_hallucination / check_code_block]
        MCP --> T_TRANS[translate_concept_arguments / explain_anti_pattern]
        MCP --> T_DOM[23 ISA, Compiler IR & Domain check_* Tools]

        DISK --> DIFF[diff.py: Semantic Snapshot Differ]
        DISK --> STUBS[stubs.py: Synthetic .pyi Generator]
        DISK --> COMP[compliance/: Implementation Verifier]
        DISK --> EXP[export.py & exporters/: Multi-Target Exporter]
        EXP --> EXP_JSON[JSON Schema & OpenAPI]
        EXP --> EXP_PYD[Pydantic V2 Classes]
        EXP --> EXP_PB[Protobuf v3, C++ Headers & TypeScript]
        EXP --> EXP_LLM[Prompt Contexts & Hallucination Guards]
    end
```

---

## 3. Multi-Tiered Introspection & Static Analysis Pipeline (`introspection/`)

Introspecting modern ML libraries is complicated by heavy metaclasses, dynamically injected C++ operators, JIT decorators (`@torch.compile`, `@jax.jit`), and runtime-generated docstrings. `GhostInspector` (`introspection/inspector.py`), coordinated with `introspection/runtime_boundary.py`, `c_extension_sig.py`, and `utils.py`, executes an eight-tiered resolution strategy:

1. **Decorator Unwrapping (`introspection/runtime_boundary.py`):** Recursively unwrap nested wrapper objects up to 10 layers deep in a centralized runtime boundary module (`__wrapped__`, `_python_function`, `_original_fn`, `__original_fn`, `_orig_mod`).
2. **`cdd-python` Static AST Parsing (`introspection/cdd_inspector.py`):** Parses the module AST statically without importing dynamic metaclasses. Extracts structured docstring sections (`params`, `returns`, `raises`), converts parameter defaults into literal AST nodes, and validates Sphinx/reST documentation.
3. **`griffe` AST Engine (`introspection/griffe_inspector.py`):** Complements `cdd-python` by resolving complex PEP-585 / PEP-604 type annotations, resolving module-level `@typing.overload` stacks, and preserving `VAR_POSITIONAL` (`*args`) and `VAR_KEYWORD` (`**kwargs`) parameter roles across Google, NumPy, and Sphinx docstring conventions.
4. **Runtime Signature Fallback (`introspection/live_inspector.py`):** Standard runtime signature extraction via `get_runtime_signature` for pure-Python targets and `@runtime_checkable` `typing.Protocol` contracts when static AST parsing is inapplicable.
5. **C-Extension & PyBind11 Docstring Synthesizer (`c_extension_sig.py:extract_c_extension_signature`):**
   - For compiled native functions (e.g. `torch.relu`, `aten::`, `c10::`) that raise `ValueError: no signature found for builtin`, docstrings are parsed with specialized regular expressions.
   - `_normalize_c_sig_args` translates C++ type syntax (e.g., `const Tensor & self, bool inplace=False`) into valid Python parameter signatures (`self: "Tensor" = False`) and unparses them into AST nodes.
   - Automatically handles numbered overload sets embedded in native docstrings.
6. **AST Keyword Visitor (`introspection/ast_kwargs.py:KwargAccessVisitor`):**
   - When a function signature ends with generic `**kwargs`, the inspector walks the AST of the function body.
   - It identifies all explicit accesses matching `kwargs.get("param")`, `kwargs["param"]`, and `kwargs.pop("param")`, exposing them via `accepted_kwargs` on `ExtendedGhostRef` to prevent false-positive hallucination reports.
7. **Memory Address Scrubbing & Factory Default Sanitization (`introspection/sanitize.py`):**
   - Raw Python defaults often embed memory pointers (e.g., `<function mean at 0x1023a1a60>`, `<Device object at 0x7f...>`).
   - All defaults are processed through deterministic address scrubbing regexes: memory addresses are converted to `<addr>`, and unrepresentable runtime instances are mapped to `<factory_default>` or `<unrepresentable>` to ensure byte-identical snapshot serialization across machines.
8. **TableGen & Pre-Compiled Extraction Tools (`tools/`):**
   - For non-Python targets (SASS, RDNA, PTX, Core MLIR, StableHLO), offline scrapers parse LLVM TableGen (`.td`) definitions, CUDA Toolkit HTML/binary disassembly, and ODS specs into exhaustive local datasets (`*_exhaustive.json`).

---

## 4. Extraction Engine & Subprocess Isolation (`api.py`, `registry.py`, `version_resolver.py`)

The extraction engine manages lifecycle, concurrency, declarative provenance metadata (`FRAMEWORK_PROVENANCE_METADATA`), and environment isolation during snapshot generation:

- **Dynamic Collector Discovery (`get_available_frameworks`):** Uses `pkgutil` and `importlib` to scan the `ml_ecosystem_snapshots.frameworks` namespace, detecting all `collect_*` entry points and mapping canonical aliases.
- **Subprocess Isolation (`extract_snapshot_isolated`):**
  - High-performance ML libraries (PyTorch with CUDA, JAX with Metal, TensorFlow) often initialize non-resettable hardware driver states upon import.
  - `extract_snapshot_isolated` runs the collection logic in an isolated child Python process via `subprocess.run`. This guarantees that memory pre-allocation locks, C++ symbol collisions, and segmentation faults do not pollute or crash the parent process.
- **Concurrent Semantic Tier Processing:** Framework collection functions accept a `SemanticTier` filter (`ARRAY_API`, `NEURAL`, `OPTIMIZER`, `LOSS`, `ACTIVATION`, `EXTRAS`, `DISTRIBUTED`, `DATA`, `HUB`, `METRIC`, `UTIL`, `SCHEDULER`, `MODEL`, `INITIALIZER`, `DATALOADER`). A `ThreadPoolExecutor` distributes tier extraction across threads.
- **Deterministic Alias Deduplication (`consolidate_aliases`):**
  - Resolves duplicate functions exposed across multiple modules (e.g., `torch.add` vs `torch.Tensor.add`).
  - Converts parameter signatures into comparable tuples (including name, kind, default, annotation, direction, role, dtypes, and rank) and collapses duplicates via `GHOST_REF_CONSOLIDATION_SEMIGROUP`, selecting the shortest canonical `api_path` as the primary reference while accumulating `aliases`.
- **Domain Metadata Hashing (`typeclasses._domain_meta_to_key`):** Recursively converts arbitrary domain metadata dictionaries into deterministic sorted tuples for consistent canonical hashing (`GHOST_REF_CANONICAL_HASH`).
- **Offline Mode (`is_offline_mode` / `ML_SNAPSHOTS_OFFLINE`):** Enforces purely local execution, bypassing remote network calls or version checks in air-gapped or sandboxed environments.

---

## 5. Hardware ISAs, Compiler Dialects, and Domain DSL Schemas

`ml-ecosystem-snapshots` provides structured metadata schemas for low-level compilation backends, hardware architectures, and declarative DSLs:

### A. NVIDIA SASS Schema (`frameworks/nvidia_sass.py`)
- **Domain Type:** `GhostIsaRef` (`domain_type="isa"`)
- **Control Code & Scheduling Schema:** SASS instructions capture cycle-accurate scheduling metadata in `control_codes`:
  - `latency_ticks`: Latency cycles before destination registers become ready.
  - `yield_flag`: Warp yield control flag (`Y`).
  - `stall_count`: Cycle stall duration (0–15).
  - `read_barrier_mask` & `write_barrier_mask`: Scoreboarding dependency barrier tokens (0–5).
  - `register_reuse_flags`: Hardware register reuse cache flags (`R*.reuse`).
- **Microarchitectures:** Discrete tagging per NVIDIA compute capability: `["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100"]`. Uniform Registers (`UR`) are restricted to `sm_75+`, and asynchronous warpgroup matrix instructions (`WGMMA`) are restricted to Hopper+ (`sm_90+`).
- **Modifiers & Operands:** Instruction modifiers (e.g. `.SAT`, `.FTZ`, `.F16`) and structured operand records with explicit register classes.

### B. AMD RDNA / CDNA Schema (`frameworks/amd_rdna.py`)
- **Domain Type:** `GhostIsaRef` (`domain_type="isa"`)
- **Encoding Formats:** Models VOP1, VOP2, VOP3, VOP3P, VOPD (dual-issue), SOP1, SOP2, SOPK, SOPP, SMEM, and FLAT instruction profiles.
- **Modifiers:** Supports input negation (`-src`), absolute value (`|src|`), saturation (`clamp`), and output multipliers (`omod:2`, `omod:4`, `omod:div2`).
- **Register Alignment:** Enforces 32-bit `VGPR`, strictly even-aligned 64-bit pairs `V[n:n+1]`, 128-bit quads `V[n:n+3]`, and 256-bit matrix accumulator registers (`a[n:n+3]` on GFX9/CDNA).
- **Dual-Issue Profile (VOPD):** Captures opcode pairing rules and operand constraints for RDNA3/GFX11 and RDNA4/GFX12.

### C. NVIDIA PTX Schema (`frameworks/nvidia_ptx.py`)
- **Domain Type:** `GhostIsaRef` (`domain_type="isa"`)
- **State Spaces:** Models `.reg`, `.sreg`, `.const`, `.global`, `.local`, `.param`, and `.shared` memory spaces.
- **Types & Vector Widths:** Full support for bit types (`.b8`–`.b128`), unsigned/signed integers (`.u8`–`.u64`, `.s8`–`.s64`), standard floats (`.f16`, `.f32`, `.f64`, `.tf32`, `.bf16`), 8-bit floating point formats (`.e4m3`, `.e5m2`), and vector widths (`.v2`, `.v4`).
- **Scopes:** Memory visibility scopes (`.cta`, `.cluster`, `.gpu`, `.sys`).

### D. StableHLO Dialect Schema (`frameworks/stablehlo.py`)
- **Domain Type:** `GhostMlirRef` (`domain_type="mlir"`)
- **Structured Attribute Schemas:**
  - `DotDimensionNumbersAttr`: LHS/RHS batch and contracting dimension arrays.
  - `ConvDimensionNumbersAttr`: Input, kernel, and output spatial/feature dimensions.
  - `ScatterDimensionNumbersAttr` & `GatherDimensionNumbersAttr`: Window dimensions, batching dims, and index mappings.
  - `ComparisonDirectionAttr` & `PrecisionAttr`: Direction enums (`EQ`, `NE`, `GE`, `GT`, `LE`, `LT`) and precision modes (`DEFAULT`, `HIGH`, `HIGHEST`).
- **Regions & Block Arguments:** Enforces block argument signatures for control-flow and higher-order operations (`stablehlo.reduce`, `stablehlo.while`, `stablehlo.sort`).

### E. Core MLIR Dialects Schema (`frameworks/mlir.py`)
- **Domain Type:** `GhostMlirRef` (`domain_type="mlir"`)
- **TableGen ODS Source:** Parsed directly from LLVM TableGen definitions across 9 core dialects: `arith`, `math`, `tensor`, `linalg`, `scf`, `func`, `memref`, `gpu`, and `vector`.
- **Traits & Constraints:** Verification traits (`SameOperandsAndResultType`, `Commutative`) and ODS type constraints (`AnyTensor`, `RankedTensorOf`, `AnyFloat`, `Index`).

### F. Domain-Specific DSLs (`frameworks/html_dsl.py`, `latex_dsl.py`, `tikz.py`)
- Extends the Ghost protocol to declarative non-Python DSLs: standard HTML tags and attributes, LaTeX mathematical macros and environments, and TikZ graphical diagram primitives.

---

## 6. Anti-Hallucination Grounding SDK (`grounding/`)

The Grounding SDK provides programmatic, high-speed validation for LLM-generated code, transpilers, and compiler passes.

### Grounding Engine (`grounding/engine.py`)
- **In-Memory Registry:** Lazily loads and indexes symbol tables from bundled, cached, or custom directory paths.
- **Fuzzy Matching:** Implements Levenshtein distance calculations (`compute_levenshtein`) to suggest closest valid symbols when an invalid or hallucinated API is encountered.
- **Diagnostic System (`grounding/models.py`):** Returns structured `GroundingReport` objects populated with `GroundingDiagnostic` items categorized by `DiagnosticSeverity` (`ERROR`, `WARNING`, `INFO`), complete with actionable suggested fixes.

### Grounding Verifiers
- **Python Framework Call Verification (`validate_python_call`):**
  - Checks target symbol existence against offline snapshots.
  - Verifies positional argument count against parameter specifications.
  - Flags hallucinated keyword arguments.
  - Provides automated cross-framework translations (e.g. flagging `axis` in `torch.sum` and recommending `dim`).
- **Compiler IR & Shader Verification (`validate_stablehlo_op`, `validate_mlir_op`, `verify_wgsl_op`):**
  - Verifies SSA operand counts, mandatory attributes (e.g., `dot_dimension_numbers` on `stablehlo.dot_general`), and WebGPU WGSL storage/barrier operations.
- **Hardware ISA Verification (`validate_sass_instruction`, `validate_rdna_instruction`, `verify_ptx_instruction`):**
  - Confirms instruction availability on the target microarchitecture (e.g., verifying `WGMMA` on `sm_90`, `v_dual_fmac` on `GFX11`, or PTX state spaces/vector widths on target SM).
  - Checks operand count, modifier validity, and register class alignments.
- **Ecosystem Domain Operation Verifiers (`grounding/domain_ops.py`):**
  - Validates operator schemas, operand counts, address spaces, and keyword arguments across ONNX, Apple Metal (MSL), WebAssembly SIMD, WebGL2/GLSL, C++ Runtime, Python Array API, SciPy, TorchVision, TorchAudio, SafeTensors, ATen, NCCL/RCCL, FlashAttention, Chex, Grain, Orbax, and Optax.

---

## 7. Model Context Protocol (MCP) Live Server (`mcp_server.py` & `mcp/`)

The MCP server (`mcp/protocol.py`, `mcp/registry.py`, `mcp/schemas.py`, `mcp/cache.py`, and `mcp/tools/`) exposes the snapshot catalog to LLM coding agents, autonomous CLI tools, and IDE extensions via the JSON-RPC 2.0 Model Context Protocol standard over stdin/stdout.

### Exposed MCP Tool Catalog (29 Tools)

| Tool Name | Purpose | Key Parameters |
|:---|:---|:---|
| `get_api_signature` | Retrieve ground-truth parameter types, defaults, and return specifications | `framework`, `api_path`, `version` |
| `search_apis` | Search operations and mnemonics by keyword or prefix | `framework`, `query`, `limit`, `version` |
| `check_hallucination` | Verify whether an API path or keyword arguments represent LLM hallucinations | `framework`, `api_path`, `kwargs`, `args_count`, `version`, `strict_kwargs`, `kwarg_values` |
| `explain_anti_pattern` | Provide canonical migration advice and rationale for flagged anti-patterns | `framework`, `api_path`, `hallucinated_argument`, `passed_value` |
| `check_code_block` | Batch verify an entire code block for invalid APIs, kwargs, and hardware rules | `code`, `framework`, `version`, `sm_arch`, `gfx_arch` |
| `translate_concept_arguments` | Translate arguments across frameworks using concept parameter mapping schemas | `concept`, `source_framework`, `target_framework`, `source_kwargs` |
| `check_sass_instruction` | Validate SASS instruction legality, modifiers, and operands on target SM | `mnemonic`, `operands`, `modifiers`, `sm_arch`, `control_codes` |
| `check_rdna_instruction` | Validate AMD RDNA instruction legality on target GFX architecture | `mnemonic`, `operands`, `encoding`, `gfx_arch`, `modifiers`, `wave_size` |
| `check_ptx_instruction` | Validate PTX instruction, types, vector widths, and state space on SM | `mnemonic`, `types`, `operands`, `state_space`, `scope`, `vector_width`, `sm_arch` |
| `check_mlir_op` | Validate MLIR dialect operation against ODS traits, operands, and attributes | `op_name`, `operands_count`, `attributes`, `operand_types`, `result_types`, `regions` |
| `check_stablehlo_op` | Validate StableHLO operation attributes and region constraints | `op_name`, `operands_count`, `attributes`, `structured_attributes` |
| `check_wgsl_op` | Validate WebGPU WGSL shader operation against specification | `op_name`, `inputs_count`, `attributes` |
| `check_onnx_op` | Validate ONNX operator inputs, domain, and attributes | `op_name`, `domain`, `inputs_count`, `attributes` |
| `check_metal_op` | Validate Apple Metal MSL compute operation and address space | `op_name`, `address_space`, `inputs_count`, `attributes` |
| `check_wasm_instruction` | Validate WebAssembly 2.0 / SIMD instruction and operand count | `mnemonic`, `operands_count`, `attributes` |
| `check_webgl_op` | Validate WebGL 2.0 / GLSL ES 3.00 operation and arguments | `op_name`, `inputs_count`, `attributes` |
| `check_cpp_op` | Validate C++17 / PyBind11 runtime operation | `op_name`, `inputs_count`, `attributes` |
| `check_array_api_op` | Validate Python Array API Standard operation | `op_name`, `inputs_count`, `attributes` |
| `check_scipy_op` | Validate SciPy scientific computing operation | `op_name`, `inputs_count`, `attributes` |
| `check_torchvision_op` | Validate TorchVision computer vision operation | `op_name`, `inputs_count`, `attributes` |
| `check_torchaudio_op` | Validate TorchAudio audio processing operation | `op_name`, `inputs_count`, `attributes` |
| `check_safetensors_op` | Validate SafeTensors serialization operation | `op_name`, `inputs_count`, `attributes` |
| `check_aten_op` | Validate PyTorch ATen C++ core operator | `op_name`, `inputs_count`, `attributes` |
| `check_nccl_op` | Validate NCCL/RCCL collective communication primitive | `op_name`, `inputs_count`, `attributes` |
| `check_flash_attention_op` | Validate FlashAttention / PagedAttention kernel operation | `op_name`, `inputs_count`, `attributes` |
| `check_chex_op` | Validate Chex assertion, dataclass, fake context, or variant helper | `op_name`, `inputs_count`, `attributes` |
| `check_grain_op` | Validate Grain dataset, sampler, or data loader operation | `op_name`, `inputs_count`, `attributes` |
| `check_orbax_op` | Validate Orbax Checkpoint or Export manager/handler operation | `op_name`, `inputs_count`, `attributes` |
| `check_optax_op` | Validate Optax optimizer, transformation, loss, or schedule | `op_name`, `inputs_count`, `attributes` |

### Concept Argument Translation
Translates semantic parameters between frameworks (e.g. converting `dim` to `axis`, or adjusting `keepdim`/`keepdims` when migrating operations like `reduce_sum`, `matmul`, `softmax`, or `convolution` between PyTorch, JAX, NumPy, TensorFlow, and StableHLO).

---

## 8. Downstream Generation & Integration Pipelines

Captured snapshots power five downstream code and schema generation workflows:

### A. Semantic Snapshot Diffing (`diff.py`)
Computes fine-grained differences between framework snapshots:
- **`added` / `removed`**: Additions and deletions of API paths.
- **`signature_changed`**: Changes in parameter names, order, types, or default values.
- **Breaking vs. Non-Breaking Classification**: Distinguishes breaking changes (e.g., adding mandatory arguments, altering default values, removing instruction modifiers, dropping microarchitecture support) from non-breaking changes (e.g., adding optional parameters with defaults).
- **Changelog Generation (`generate_changelog`)**: Emits structured Markdown changelogs suitable for release notes.

### B. Type Stub Generation (`stubs.py`)
Generates PEP-484 `.pyi` type stub files directly from snapshots:
- Rebuilds module hierarchies, class definitions, method signatures, and function overloads.
- Strips unrepresentable default values while preserving static type hints, enabling IDE autocompletion for frameworks without requiring the native framework wheels to be installed.

### C. Static Compliance Engine (`compliance/`)
Enables cleanroom verification of third-party framework implementations (`compliance/ast_extractor.py`, `compliance/assembly_verifier.py`, `compliance/dtype_rules.py`, `compliance/scorer.py`):
- Compares a candidate library's AST/runtime signatures or assembly files against reference snapshots.
- Emits compliance scores, categorizing missing symbols, signature mismatches, and parameter discrepancies.

### D. Multi-Target Schema & Code Exporting (`export.py` & `exporters/`)
Trans-compiles snapshots into diverse industry-standard interface definitions (`exporters/openapi_schema.py`, `exporters/code_gen.py`, `exporters/prompts.py`):
- **JSON Schema & OpenAPI Specifications**: Via `cdd-python`, generating web service API schemas and REST route definitions.
- **Pydantic V2 Models**: Synthesizes type-safe Pydantic classes with overloaded union variants (`Variant0 | Variant1`) and strict `JsonValue` vararg tuple validation (`tuple[JsonValue, ...]`).
- **Protobuf v3, C++ Headers & TypeScript Interfaces**: Generates `.proto` message/gRPC definitions, C++ struct headers (`to_cpp_header`), and TypeScript interfaces (`to_typescript`).
- **LLM Prompt Context Exporters**:
  - `export_llm_prompt_context`: Compact Markdown specifications with parameter constraints.
  - `export_sass_prompt_context`, `export_ptx_prompt_context` & `export_rdna_prompt_context`: Canonical GPU assembly templates specifying valid register files, state spaces, modifiers, and microarchitectures.
  - `export_mlir_prompt_context` & `export_stablehlo_prompt_context`: Canonical MLIR/StableHLO SSA syntax templates separating runtime operands from compile-time attributes and regions.
  - `export_scoped_prompt_context`: Hierarchical summary index with `COMMON_HALLUCINATION_GUARDS` to prevent context window explosion.

---

## 9. Ephemeral Local SQLite FTS5 Index & Cache (`index.py`)

To achieve sub-millisecond symbol lookups without loading multi-megabyte JSON trees into memory:
- **SQLite FTS5 Full-Text Index (`get_index_db_path`)**: Automatically creates an ephemeral search database in the user's platform-specific cache directory (`XDG_CACHE_HOME` / `~/.cache/ml_ecosystem_snapshots/index.db`).
- **Incremental Indexing**: Uses SHA-256 file hashes (`compute_file_sha256`) to incrementally index bundled and cached JSON snapshots.
- **Integrity Verification**: CLI commands (`verify-local-cache`) validate local cached snapshot files against upstream cryptographic checksums.

---

## 10. CLI Architecture & Path Resolution (`cli/`)

The command-line interface (`cli/parser.py`, `cli/resolver.py`, and `cli/commands/`) provides the user-facing entrypoint for snapshot capture, inspection, and verification.

### Multi-Tier Snapshot Path Resolution (`resolve_snapshot_path`)
Resolves snapshot identifiers (e.g. `torch`, `torch_v2.4.0.json`, or custom paths) across a cascading hierarchy:
1. Exact file path or `.json` suffix.
2. Custom paths defined in `ML_SNAPSHOTS_PATH`.
3. Package-bundled `src/ml_ecosystem_snapshots/snapshots/` and `frameworks/`.
4. Project repository root `snapshots/`.
5. Local cache directory (`~/.cache/ml_ecosystem_snapshots/snapshots`).
6. Current working directory `snapshots/`.

### CLI Subcommands Overview

| Command | Purpose |
|:---|:---|
| `capture` | Introspect installed frameworks and write JSON snapshots |
| `diff` | Compare two snapshots and generate breaking/non-breaking changelogs |
| `generate-stubs` | Synthesize `.pyi` type stubs from a snapshot |
| `export` | Export snapshot to OpenAPI, JSON Schema, Pydantic V2, Protobuf, C++ headers, TypeScript, or LLM prompts |
| `mcp` | Launch the Model Context Protocol JSON-RPC tool server |
| `check` | Verify compliance of a target implementation against a reference snapshot |
| `list-snapshots` | List bundled and cached framework snapshots |
| `pull` | Download an official snapshot asset from GitHub Releases into local cache |
| `download-all` | Download all canonical pre-compiled snapshots into cache |
| `verify-local-cache` | Verify SHA-256 integrity of all cached snapshots |
| `index` | Inspect, clear, or rebuild the local SQLite FTS5 search index |
| `check-sass` | Validate NVIDIA SASS assembly instructions, modifiers, and SM architectures |
| `check-rdna` | Validate AMD RDNA assembly instructions, encodings, and GFX generations |
| `check-ptx` | Validate NVIDIA PTX assembly instructions, state spaces, and SM architectures |
| `check-mlir` | Validate MLIR dialect operations and traits |
| `check-stablehlo` | Validate StableHLO operations, dimension numbers, and block regions |
| `check-wgsl` | Validate WGSL shader or operation against WebGPU specification |
| `check-onnx` | Validate ONNX operator against official specification |
| `check-metal` | Validate Apple Metal MSL compute operation and address space |
| `check-wasm` | Validate WebAssembly 2.0 / SIMD instruction against specification |
| `check-webgl` | Validate WebGL 2.0 / GLSL ES 3.00 operation against specification |
| `check-cpp` | Validate C++17 / PyBind11 runtime operation against specification |
| `check-array-api` | Validate Python Array API Standard operation against specification |
| `check-scipy` | Validate SciPy scientific computing operation against specification |
| `check-torchvision` | Validate TorchVision operation against specification |
| `check-torchaudio` | Validate TorchAudio operation against specification |
| `check-safetensors` | Validate SafeTensors serialization operation against specification |
| `check-aten` | Validate PyTorch ATen C++ operator against specification |
| `check-nccl` | Validate NCCL/RCCL collective communication operation against specification |
| `check-flash-attention` | Validate FlashAttention / PagedAttention kernel operation against specification |
| `check-chex` | Validate Chex assertions, dataclasses, fake contexts, and test variants |
| `check-grain` | Validate Grain (`grain` / `grain.python`) datasets, samplers, and loaders |
| `check-orbax` | Validate Orbax Checkpoint and Orbax Export managers, handlers, and args |
| `check-optax` | Validate Optax optimizers, gradient transformations, losses, and schedules |

---

## 11. Pure Functional Core, Explicit Typeclasses & Phantom Proofs (`identifiers.py`, `typeclasses.py`, `proofs.py`)

Aligned with `ml-switcheroo-ir` v0.0.8, `ml-ecosystem-snapshots` separates its **Pure Functional Core** from its **Imperative Shell** (`cli/`, `mcp/`, `tools/`, `scripts/`):

- **Deep Immutability**: Domain and report schemas across `models.py`, `diff.py`, `compliance/models.py`, `grounding/models.py`, `mcp/protocol.py`, `c_extension_sig.py`, and `introspection/sanitize.py` enforce `pydantic.ConfigDict(frozen=True)` and immutable `tuple[..., ...]` sequence fields.
- **Total Outcomes (`Result[E, A]` & `Validation[E, A]`)**: Core SDK functions (`extract_snapshot`, `extract_snapshot_isolated`, `write_snapshot`, `load_json_data`, `fetch_url`, `hydrate_ghost_ref`, `validate_snapshot_envelope`, `generate_stubs`, `resolve_snapshot_path`, `validate_broadcast_shapes`, `validate_matmul_shapes`) return explicit `Ok` / `Err` outcomes rather than raising exceptions for domain/I/O failures. Batch operations (`extract_all_snapshots`, `hydrate_ghost_refs`, `prove_hydrated_refs`, `prove_determinism_verified`, `prove_compliance_verified`) accumulate *all* diagnostics via applicative `Validation` (`Valid` / `Invalid`).
- **Validated `NewType` Identifiers (`identifiers.py`)**: `FrameworkId`, `SnapshotVersion`, `ApiPath`, `CategoryName`, and `Sha256Hex` wrap validated strings via total `parse_*` smart constructors paired with explicit `Eq`, `Ord` (including `API_PATH_LEN_LEX_ORD`), and `HashableWitness` dictionaries.
- **Explicit Typeclass Witnesses (`typeclasses.py`)**:
  - `DIFF_RESULT_SEMIGROUP` & `DIFF_RESULT_MONOID`: Associatively combine `DiffResult` values via sorted deduplicated set union across symbol-change tuples and rectangular-band `(old_lhs, new_rhs)` composition across `metadata_changed` transitions, with `DiffResult()` as left/right identity.
  - `COMPLIANCE_REPORT_MONOID`, `SNIPPET_REPORT_MONOID`, & `CODE_BLOCK_RESULT_MONOID`: Fold per-module, per-line, and per-snippet verification results via `INT_SUM_MONOID`, `BOOL_ALL_MONOID`, and `tuple_monoid()`.
  - `GHOST_REF_CANONICAL_HASH` & `GHOST_REF_CONSOLIDATION_SEMIGROUP`: Consolidate structurally equivalent `GhostRef` aliases without mutating input models.
  - `SNAPSHOT_ENVELOPE_JSON_CODEC`, `GHOST_REF_JSON_CODEC`, `DIFF_CHANGELOG_PRETTY`, `STUB_MODULE_PRETTY`, & `make_namespace_align_endo`: Explicit `JsonCodec`, `Pretty`, and `Endo[str]` witnesses.
- **Proof-Carrying Phantom Phases (`proofs.py`)**: `ProvenSnapshot[PhaseT]` tracks static verification invariants across `EnvelopeValidPhase` (`prove_envelope_valid`), `HydratedRefsPhase` (`prove_hydrated_refs`), `DeterminismVerifiedPhase` (`prove_determinism_verified`), and `ComplianceVerifiedPhase` (`prove_compliance_verified`), deep-cloning the underlying `SnapshotEnvelope` at minting time.
