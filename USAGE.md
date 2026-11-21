# Usage Guide & Examples

`ml-ecosystem-snapshots` provides a comprehensive command-line interface, a programmatic Grounding SDK, and a Model Context Protocol (MCP) server for capturing, diffing, validating, and exporting framework signatures, compiler IRs, and hardware ISAs.

---

## Table of Contents

1. [Capturing Framework Snapshots](#1-capturing-framework-snapshots)
2. [Offline Snapshots & Cache Management](#2-offline-snapshots--cache-management)
3. [Compliance Checking](#3-compliance-checking)
4. [Structural Diffing & Semantic Changelogs](#4-structural-diffing--semantic-changelogs)
5. [Code & Schema Export](#5-code--schema-export)
6. [Hardware ISA, Compiler IR & Domain CLI Verification](#6-hardware-isa-compiler-ir--domain-cli-verification)
7. [Anti-Hallucination Grounding SDK (Python)](#7-anti-hallucination-grounding-sdk-python)
8. [Model Context Protocol (MCP) Server](#8-model-context-protocol-mcp-server)
9. [Pure Functional SDK: Outcomes, Typeclasses & Phantom Proofs](#9-pure-functional-sdk-outcomes-typeclasses--phantom-proofs)

---

## 1. Capturing Framework Snapshots

The `capture` subcommand inspects installed ML frameworks in your environment, serializing their symbols into deterministic JSON snapshots under the Ghost Protocol.

### Capture Specific Frameworks

```bash
# Capture PyTorch, JAX, and Keras into ./snapshots
ml_ecosystem_snapshots capture torch jax keras --out-dir ./snapshots
```

### Capture All Installed Frameworks

```bash
ml_ecosystem_snapshots capture all --out-dir ./snapshots
```

### Include Non-Public APIs

By default, private methods and functions (prefixed with `_`) are excluded. Use `--include-nonpublic` to extract internal APIs:

```bash
ml_ecosystem_snapshots capture torch --include-nonpublic --out-dir ./snapshots
```

### Capturing Hardware ISAs & Compiler Dialects

The tool captures pre-compiled hardware ISAs and compiler specifications without requiring GPU drivers or LLVM toolchains:

```bash
# Capture NVIDIA SASS assembly catalog
ml_ecosystem_snapshots capture nvidia_sass --out-dir ./snapshots

# Capture AMD RDNA / CDNA assembly catalog
ml_ecosystem_snapshots capture amd_rdna --out-dir ./snapshots

# Capture NVIDIA PTX assembly catalog
ml_ecosystem_snapshots capture nvidia_ptx --out-dir ./snapshots

# Capture StableHLO compiler dialect
ml_ecosystem_snapshots capture stablehlo --out-dir ./snapshots

# Capture Core MLIR dialects
ml_ecosystem_snapshots capture mlir --out-dir ./snapshots

# Capture JAX ecosystem libraries (Chex, Grain, Orbax, Optax)
ml_ecosystem_snapshots capture chex grain orbax optax --out-dir ./snapshots
```

### Regenerating Exhaustive Target Catalogs

To update the exhaustive JSON dumps (e.g., when a new CUDA Toolkit, LLVM release, or StableHLO specification is published), execute the dedicated scraper tools:

```bash
# Rebuild NVIDIA SASS catalog from CUDA Toolkit tables
python -m ml_ecosystem_snapshots.tools.scrape_nvidia_sass

# Rebuild AMD RDNA / CDNA catalog from LLVM TableGen sources
python -m ml_ecosystem_snapshots.tools.scrape_amd_rdna

# Rebuild NVIDIA PTX catalog
python -m ml_ecosystem_snapshots.tools.scrape_nvidia_ptx

# Rebuild Core MLIR dialect catalog
python -m ml_ecosystem_snapshots.tools.scrape_mlir

# Rebuild StableHLO dialect catalog
python -m ml_ecosystem_snapshots.tools.build_stablehlo_snapshot
```

---

## 2. Offline Snapshots & Cache Management

`ml-ecosystem-snapshots` resolves snapshot references across a multi-tier search cascade:
1. Exact file paths (`./snapshots/torch_v2.4.0.json`)
2. Custom paths defined in `$ML_SNAPSHOTS_PATH`
3. Local user cache directory (`~/.cache/ml_ecosystem_snapshots/snapshots/`)
4. Project root `snapshots/`
5. Package-bundled offline snapshots (`src/ml_ecosystem_snapshots/snapshots/` and `frameworks/`)

### Listing Available Snapshots

List all bundled, cached, and local snapshots available offline:

```bash
ml_ecosystem_snapshots list-snapshots
```

### Downloading Pre-Compiled Snapshots

Fetch official pre-compiled snapshot releases from GitHub into your local cache:

```bash
# Pull a specific framework and version
ml_ecosystem_snapshots pull torch@2.4.0

# Pull JAX snapshot
ml_ecosystem_snapshots pull jax@0.4.30

# Download all canonical pre-compiled snapshot assets
ml_ecosystem_snapshots download-all
```

### Cryptographic Cache Integrity Verification

Verify the SHA-256 checksums of all cached snapshot files:

```bash
ml_ecosystem_snapshots verify-local-cache
```

### Managing the Ephemeral SQLite FTS5 Index

The ephemeral SQLite FTS5 search index provides sub-millisecond symbol lookups without loading multi-megabyte JSON trees into memory:

```bash
# Check index status and total indexed symbols
ml_ecosystem_snapshots index --status

# Rebuild the search index across all available snapshots
ml_ecosystem_snapshots index --rebuild

# Clear the index database
ml_ecosystem_snapshots index --clear
```

---

## 3. Compliance Checking

The `check` subcommand performs cleanroom verification of candidate framework implementations (such as transpiled shims, WASM wrappers, or mock modules) against a reference snapshot.

### Checking a Specific Module

Test the compliance of a custom JAX wrapper against a reference JAX snapshot:

```bash
ml_ecosystem_snapshots check \
    jax \
    ~/repos/ml-switcheroo/src/ml_switcheroo/frameworks/jax.py \
    --reference-prefix jax \
    --target-prefix ml_switcheroo.frameworks.jax
```

**Example Output:**
```text
Extracting target APIs from /Users/samuel/repos/ml-switcheroo/src/ml_switcheroo/frameworks/jax.py...
Scoring compliance...

--- Compliance Report ---
Overall Compliance: 82.5%

Breakdown by Module:
  - jax.numpy: 86.5% (45/52)

Missing APIs (2):

|   | Framework | Namespace | Symbol | FQN | Signature | Docstring |
|---|---|---|---|---|---|---|
| [ ] | jax | jax.numpy | fft | jax.numpy.fft | module | Fast Fourier Transform module. |
| [ ] | jax | jax.numpy | linalg | jax.numpy.linalg | module | Linear algebra module. |
```

### Checking an Entire Package Tree

Test an independent project like `onnx9000-jax`:

```bash
ml_ecosystem_snapshots check \
    jax \
    ~/repos/ml-switcheroo/onnx9000/packages/python/onnx9000-jax \
    --reference-prefix jax \
    --target-prefix onnx9000.jax
```

---

## 4. Structural Diffing & Semantic Changelogs

The `diff` subcommand compares two snapshots, identifying additions, removals, parameter shifts, and backwards-incompatible changes.

### Generating a Terminal Summary

```bash
ml_ecosystem_snapshots diff ./snapshots/torch_v2.3.0.json ./snapshots/torch_v2.4.0.json
```

**Output:**
```text
ADDED: 14
  + torch.nn.functional.scaled_dot_product_attention
  + torch.compiler.is_compiling
REMOVED: 2
  - torch.legacy_op
SIGNATURE CHANGED: 5
  ~ torch.sum
```

### Generating a Markdown Changelog

Use `--changelog` to emit release notes with breaking vs. non-breaking classification:

```bash
ml_ecosystem_snapshots diff ./snapshots/jax_v0.4.30.json ./snapshots/jax_v0.4.31.json --changelog
```

**Example Output:**
```markdown
# API Changelog

## Added (2)
- `jax.numpy.trapezoid`
- `jax.experimental.shard_map`

## Removed (0)

## Breaking Signature Changes (1)
- `jax.numpy.pad`: Added required positional parameter `pad_width` without default.

## Non-Breaking Signature Changes (3)
- `jax.numpy.mean`: Added optional keyword argument `where`.
```

---

## 5. Code & Schema Export

Transform framework snapshots into diverse interface formats and schemas.

### Generate Python Type Stubs (`.pyi`)

Create PEP-484 type stub files with reconstructed module trees and overloads:

```bash
ml_ecosystem_snapshots generate-stubs --input torch --out-dir ./stubs/
```

### Export to Pydantic V2 Models

Generate type-safe Pydantic models with `Field(description=...)` and overloaded union variants:

```bash
ml_ecosystem_snapshots export --input torch --format pydantic --out-dir ./models/
```

### Export to OpenAPI 3.0 Specifications

Generate OpenAPI REST interface routes:

```bash
ml_ecosystem_snapshots export --input torch --format openapi --out-dir ./openapi/
```

### Export to JSON Schema

Generate JSON Schema validation specifications:

```bash
ml_ecosystem_snapshots export --input jax --format json_schema --out-dir ./json_schema/
```

### Export to Protobuf v3, C++ Headers & TypeScript Interfaces

Generate `.proto` message definitions, C++ struct headers, and TypeScript interface bindings:

```bash
ml_ecosystem_snapshots export --input torch --format protobuf --out-dir ./proto/
ml_ecosystem_snapshots export --input torch --format cpp_header --out-dir ./include/
ml_ecosystem_snapshots export --input torch --format typescript --out-dir ./ts/
```

### Export LLM & Hardware/Compiler Prompt Contexts

Generate compact Markdown specifications with parameter constraints and built-in hallucination guards (`llm_prompt`, `scoped_prompt`, `sass_prompt`, `ptx_prompt`, `rdna_prompt`, `mlir_prompt`, `stablehlo_prompt`):

```bash
ml_ecosystem_snapshots export --input torch --format llm_prompt --out-dir ./prompts/
ml_ecosystem_snapshots export --input torch --format scoped_prompt --out-dir ./prompts/
ml_ecosystem_snapshots export --input nvidia_sass --format sass_prompt --out-dir ./prompts/
ml_ecosystem_snapshots export --input stablehlo --format stablehlo_prompt --out-dir ./prompts/
```

---

## 6. Hardware ISA, Compiler IR & Domain CLI Verification

Directly validate machine instructions and compiler operations from the terminal.

### NVIDIA SASS Instruction Validation (`check-sass`)

Validate instruction legality, operands, modifiers, and SM architecture support:

```bash
# Verify Hopper asynchronous warpgroup matrix instruction
ml_ecosystem_snapshots check-sass WGMMA --sm-arch sm_90 --modifiers .F16

# Verify single precision fused multiply-add
ml_ecosystem_snapshots check-sass FFMA --operands R0,R1,R2,R3 --modifiers .FTZ --sm-arch sm_80

# Verify an entire SASS assembly file
ml_ecosystem_snapshots check-sass --file kernel.sass --sm-arch sm_90
```

### AMD RDNA / CDNA Instruction Validation (`check-rdna`)

Validate RDNA/CDNA instructions, wave sizes, modifiers, and GFX generations:

```bash
# Verify RDNA3 dual-issue instruction
ml_ecosystem_snapshots check-rdna v_dual_fmac_f32 --gfx-arch GFX11/RDNA3

# Verify vector fused multiply-add (VOP3) with clamp modifier
ml_ecosystem_snapshots check-rdna v_fma_f32 --operands v0,v1,v2,v3 --modifiers clamp --gfx-arch GFX10.3/RDNA2

# Verify an entire RDNA assembly file
ml_ecosystem_snapshots check-rdna --file kernel.s --gfx-arch GFX11/RDNA3
```

### Core MLIR Dialect Verification (`check-mlir`)

Validate MLIR operations against TableGen traits and SSA operand counts:

```bash
# Verify arith.addf operation
ml_ecosystem_snapshots check-mlir arith.addf --operands-count 2

# Verify an entire MLIR text file
ml_ecosystem_snapshots check-mlir --file module.mlir
```

### StableHLO Operation Verification (`check-stablehlo`)

Validate StableHLO operations against dimension number and region constraints:

```bash
# Verify stablehlo.dot_general with mandatory dot_dimension_numbers attribute
ml_ecosystem_snapshots check-stablehlo stablehlo.dot_general \
    --operands-count 2 \
    --attributes dot_dimension_numbers

# Verify an entire StableHLO text file
ml_ecosystem_snapshots check-stablehlo --file graph.mlir
```

### NVIDIA PTX Instruction Verification (`check-ptx`)

Validate PTX instructions, type qualifiers, state spaces, scopes, and SM capabilities:

```bash
# Verify global-memory float load on Ampere
ml_ecosystem_snapshots check-ptx ld --types .f32 --state-space .global --sm-arch sm_80

# Verify Hopper asynchronous warpgroup MMA instruction
ml_ecosystem_snapshots check-ptx wgmma.mma_async --types .f16 --sm-arch sm_90
```

### WebGPU WGSL Shader Verification (`check-wgsl`)

Validate WGSL compute shader intrinsics or `.wgsl` source files:

```bash
# Verify storageStore built-in operation
ml_ecosystem_snapshots check-wgsl storageStore --inputs-count 3

# Verify an entire WGSL shader file
ml_ecosystem_snapshots check-wgsl --file compute.wgsl
```

### Ecosystem & Runtime Domain Verification (`check-*`)

Validate operations across 17 specialized specifications (`check-onnx`, `check-metal`, `check-wasm`, `check-webgl`, `check-cpp`, `check-array-api`, `check-scipy`, `check-torchvision`, `check-torchaudio`, `check-safetensors`, `check-aten`, `check-nccl`, `check-flash-attention`, `check-chex`, `check-grain`, `check-orbax`, `check-optax`):

```bash
# Validate ONNX operator inputs and attributes
ml_ecosystem_snapshots check-onnx Conv --inputs-count 3 --attributes kernel_shape,strides

# Validate Apple Metal MSL operation and address space
ml_ecosystem_snapshots check-metal atomic_fetch_add_explicit --address-space device --inputs-count 3

# Validate WebAssembly SIMD instruction
ml_ecosystem_snapshots check-wasm i32x4.add --operands-count 2

# Validate ATen, NCCL, FlashAttention, Array API, and JAX ecosystem operations
ml_ecosystem_snapshots check-aten add --inputs-count 2 --attributes alpha
ml_ecosystem_snapshots check-nccl all_reduce --inputs-count 3 --attributes op
ml_ecosystem_snapshots check-flash-attention flash_attn_func --inputs-count 3 --attributes causal,softmax_scale
ml_ecosystem_snapshots check-array-api matmul --inputs-count 2
ml_ecosystem_snapshots check-optax adamw --attributes learning_rate,weight_decay
```

---

## 7. Anti-Hallucination Grounding SDK (Python)

Integrate the verification engine directly into your code synthesis pipelines, transpilers, or compiler passes.

```python
from ml_ecosystem_snapshots.grounding import (
    GroundingEngine,
    validate_python_call,
    validate_sass_instruction,
    validate_rdna_instruction,
    validate_stablehlo_op,
)

# Initialize in-memory grounding engine (searches bundled and cached snapshots)
engine = GroundingEngine()

# ---------------------------------------------------------
# 1. Python ML Framework Grounding
# ---------------------------------------------------------
report = validate_python_call(
    framework="torch",
    api_path="torch.sum",
    args=[],
    kwargs={"input": None, "axis": 0},  # Hallucinated: PyTorch uses 'dim'
    engine=engine,
)

print(f"Call Grounded: {report.is_grounded}")  # False
for diag in report.diagnostics:
    print(f"  [{diag.severity}] {diag.message}")
    print(f"  -> Suggested Fix: {diag.suggested_fix}")  # 'dim'

# ---------------------------------------------------------
# 2. NVIDIA SASS Hardware Grounding
# ---------------------------------------------------------
sass_report = validate_sass_instruction(
    mnemonic="WGMMA",
    architecture="sm_80",  # WGMMA requires Hopper+ (sm_90+)
    operands=["R0", "R1", "R2", "R3"],
    engine=engine,
)

if not sass_report.is_grounded:
    for diag in sass_report.diagnostics:
        print(f"SASS Error: {diag.message}")

# ---------------------------------------------------------
# 3. AMD RDNA Hardware Grounding: catching misaligned register pair
# ---------------------------------------------------------
rdna_report = validate_rdna_instruction(
    mnemonic="s_add_u32",
    gfx_arch="GFX11",
    operands=["v[1:2]", "s0", "s1"],  # Register pair v[1:2] is odd-aligned
    engine=engine,
)

if not rdna_report.is_grounded:
    for diag in rdna_report.diagnostics:
        print(f"RDNA Error: {diag.message}")
        print(f"-> Suggested Fix: {diag.suggested_fix}")

# ---------------------------------------------------------
# 4. StableHLO Compiler Dialect Grounding
# ---------------------------------------------------------
hlo_report = validate_stablehlo_op(
    op_name="stablehlo.dot_general",
    operand_types=["tensor<128x64xf32>", "tensor<64x256xf32>"],
    attributes={},  # Missing required 'dot_dimension_numbers'
    engine=engine,
)

if not hlo_report.is_grounded:
    for diag in hlo_report.diagnostics:
        print(f"StableHLO Error: {diag.message}")
```

---

## 8. Model Context Protocol (MCP) Server

The MCP server connects AI coding agents directly to ground-truth signatures and hardware rules via standard JSON-RPC 2.0 (exposing 29 tools across Python APIs, SASS/RDNA/PTX ISAs, MLIR/StableHLO/WGSL, and 17 ecosystem domains — see `README.md` or `tools/list` for the complete catalog).

### Starting the Server

```bash
ml_ecosystem_snapshots mcp
```

### Example JSON-RPC Invocations

#### 1. Look Up API Signature (`get_api_signature`)

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_api_signature",
    "arguments": {
      "framework": "torch",
      "api_path": "torch.nn.functional.linear"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"api_path\": \"torch.nn.functional.linear\",\n  \"kind\": \"function\",\n  \"params\": [\n    {\"name\": \"input\", \"kind\": \"POSITIONAL_OR_KEYWORD\", \"annotation\": \"Tensor\"},\n    {\"name\": \"weight\", \"kind\": \"POSITIONAL_OR_KEYWORD\", \"annotation\": \"Tensor\"},\n    {\"name\": \"bias\", \"kind\": \"POSITIONAL_OR_KEYWORD\", \"annotation\": \"Optional[Tensor]\", \"default\": \"None\"}\n  ],\n  \"returns_type\": \"Tensor\"\n}"
      }
    ]
  }
}
```

#### 2. Check for Hallucinated Arguments (`check_hallucination`)

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "check_hallucination",
    "arguments": {
      "framework": "torch",
      "api_path": "torch.sum",
      "kwargs": ["input", "axis", "keepdim"]
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"api_exists\": true,\n  \"is_hallucinated\": true,\n  \"invalid_kwargs\": [\n    \"axis\"\n  ],\n  \"canonical_params\": [\n    \"dim\",\n    \"dtype\",\n    \"input\",\n    \"keepdim\"\n  ],\n  \"has_unconstrained_kwargs\": false,\n  \"reason\": \"Invalid/hallucinated keyword arguments: axis\",\n  \"snapshot_source\": \"bundled_package\",\n  \"signature_completeness\": \"exact\",\n  \"is_c_extension\": false\n}"
      }
    ]
  }
}
```

#### 3. Cross-Framework Concept Translation (`translate_concept_arguments`)

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "translate_concept_arguments",
    "arguments": {
      "concept": "reduce_sum",
      "source_framework": "torch",
      "target_framework": "jax",
      "source_kwargs": {
        "dim": [1, 2],
        "keepdim": true
      }
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"concept\": \"reduce_sum\",\n  \"source_framework\": \"torch\",\n  \"target_framework\": \"jax\",\n  \"translated_kwargs\": {\n    \"axis\": [\n      1,\n      2\n    ],\n    \"keepdims\": true\n  },\n  \"unmapped_kwargs\": {}\n}"
      }
    ]
  }
}
```

#### 4. Batch Code Block Verification (`check_code_block`)

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "check_code_block",
    "arguments": {
      "code": "import torch\ny = torch.sum(x, axis=1)\nz = torch.relu(y)",
      "framework": "torch"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"total_analyzed\": 2,\n  \"hallucinations_detected\": 1,\n  \"is_valid\": false,\n  \"findings\": [\n    {\n      \"line\": 2,\n      \"type\": \"api_hallucination\",\n      \"framework\": \"torch\",\n      \"target\": \"torch.sum\",\n      \"reason\": \"Invalid/hallucinated keyword arguments: axis\",\n      \"invalid_kwargs\": [\n        \"axis\"\n      ]\n    }\n  ]\n}"
      }
    ]
  }
}
```

---

## 9. Pure Functional SDK: Outcomes, Typeclasses & Phantom Proofs

All core extraction, serialization, hydration, stub generation, and shape/dtype verification APIs return total `Result[E, A]` (`Ok` / `Err`) or error-accumulating `Validation[E, A]` (`Valid` / `Invalid`) values and operate on deeply immutable frozen Pydantic models.

```python
from ml_ecosystem_snapshots import (
    COMPLIANCE_REPORT_MONOID,
    DIFF_RESULT_MONOID,
    SNAPSHOT_ENVELOPE_JSON_CODEC,
    Invalid,
    Ok,
    Valid,
    extract_snapshot,
    prove_determinism_verified,
    prove_hydrated_refs,
)
from ml_ecosystem_snapshots.serialization import hydrate_ghost_refs

# 1. Total extraction + JSON Codec round-trip
res = extract_snapshot("ir")
if isinstance(res, Ok):
    proven_env = res.value
    encoded = SNAPSHOT_ENVELOPE_JSON_CODEC.encode(proven_env.envelope)
    decoded = SNAPSHOT_ENVELOPE_JSON_CODEC.decode(encoded)
    assert isinstance(decoded, Ok)

    # 2. Mint HydratedRefsPhase and DeterminismVerifiedPhase proofs
    hydrated = prove_hydrated_refs(proven_env)
    if isinstance(hydrated, Valid):
        det = prove_determinism_verified(hydrated.value)
        assert isinstance(det, Valid)

# 3. Multi-error applicative Validation across malformed GhostRef payloads
batch_val = hydrate_ghost_refs(["not_a_dict", {"schema_version": "9.9.9"}])
if isinstance(batch_val, Invalid):
    print(f"Accumulated {len(batch_val.errors)} schema errors without short-circuiting.")

# 4. Monoidal aggregation of diffs and compliance reports
empty_diff = DIFF_RESULT_MONOID.empty()
empty_compliance = COMPLIANCE_REPORT_MONOID.empty()
```
