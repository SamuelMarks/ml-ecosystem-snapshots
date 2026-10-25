"""MCP tool module for cross-framework concept argument translation."""

from __future__ import annotations

import json
import os
import typing

import pydantic

from ml_ecosystem_snapshots.mcp.models import ConceptTranslationResult
from ml_ecosystem_snapshots.models import JsonValue


class ConceptParameterSpec(pydantic.BaseModel):
    """Typed parameter translation specification for a mathematical concept."""

    model_config = pydantic.ConfigDict(extra="forbid")

    description: str = ""
    roles: dict[str, dict[str, list[str]]] = pydantic.Field(default_factory=dict)
    defaults: dict[str, dict[str, JsonValue]] | None = None


class ConceptMapSchema(pydantic.BaseModel):
    """Typed concept ontology and parameter translation schema."""

    model_config = pydantic.ConfigDict(extra="allow")

    concepts: dict[str, ConceptParameterSpec] = pydantic.Field(default_factory=dict)


DEFAULT_CONCEPT_MAP: dict[str, JsonValue] = {
    "normalization": {
        "torch": ["torch.nn.LayerNorm", "torch.nn.functional.layer_norm"],
        "jax": ["jax.nn.standardize"],
        "tensorflow": ["tf.keras.layers.LayerNormalization"],
    },
    "layer_norm": {
        "torch": ["torch.nn.LayerNorm", "torch.nn.functional.layer_norm"],
        "jax": ["jax.nn.standardize"],
        "tensorflow": ["tf.keras.layers.LayerNormalization"],
    },
    "rms_norm": {
        "torch": ["torch.nn.RMSNorm"],
        "jax": ["jax.nn.standardize"],
        "tensorflow": ["tf.keras.layers.RMSNormalization"],
    },
    "batch_norm": {
        "torch": [
            "torch.nn.BatchNorm2d",
            "torch.nn.functional.batch_norm",
        ],
        "jax": ["jax.nn.standardize"],
        "tensorflow": ["tf.keras.layers.BatchNormalization"],
    },
    "group_norm": {
        "torch": [
            "torch.nn.GroupNorm",
            "torch.nn.functional.group_norm",
        ],
        "jax": ["jax.nn.standardize"],
        "tensorflow": ["tf.keras.layers.GroupNormalization"],
    },
    "attention": {
        "torch": [
            "torch.nn.functional.scaled_dot_product_attention",
            "torch.nn.MultiheadAttention",
        ],
        "jax": ["jax.nn.dot_product_attention"],
        "tensorflow": ["tf.keras.layers.MultiHeadAttention"],
    },
    "scaled_dot_product_attention": {
        "torch": ["torch.nn.functional.scaled_dot_product_attention"],
        "jax": ["jax.nn.dot_product_attention"],
        "tensorflow": ["tf.keras.layers.MultiHeadAttention"],
    },
    "flash_attention": {
        "torch": ["torch.nn.functional.scaled_dot_product_attention"],
        "jax": ["jax.nn.dot_product_attention"],
    },
    "paged_attention": {
        "torch": ["torch.nn.functional.scaled_dot_product_attention"],
        "jax": ["jax.nn.dot_product_attention"],
    },
    "sliding_window_attention": {
        "torch": ["torch.nn.functional.scaled_dot_product_attention"],
        "jax": ["jax.nn.dot_product_attention"],
    },
    "wgmma_mma_async": {
        "torch": ["torch.cuda.wgmma"],
        "nvidia_sass": ["WGMMA"],
    },
    "activations": {
        "torch": [
            "torch.nn.functional.relu",
            "torch.nn.functional.gelu",
        ],
        "jax": ["jax.nn.relu", "jax.nn.gelu"],
        "tensorflow": ["tf.nn.relu", "tf.nn.gelu"],
    },
    "gelu": {
        "torch": ["torch.nn.functional.gelu"],
        "jax": ["jax.nn.gelu"],
        "tensorflow": ["tf.nn.gelu"],
    },
    "silu": {
        "torch": ["torch.nn.functional.silu"],
        "jax": ["jax.nn.silu"],
        "tensorflow": ["tf.nn.silu"],
    },
    "swiglu": {
        "torch": ["torch.nn.functional.silu"],
        "jax": ["jax.nn.silu"],
    },
    "relu": {
        "torch": ["torch.nn.functional.relu"],
        "jax": ["jax.nn.relu"],
        "tensorflow": ["tf.nn.relu"],
    },
    "quick_gelu": {
        "torch": ["torch.nn.functional.gelu"],
        "jax": ["jax.nn.gelu"],
    },
    "quantization": {
        "torch": [
            "torch.ao.quantization.quantize",
            "torch.quantize_per_tensor",
        ],
        "jax": ["jax.experimental.quantization"],
    },
    "quantize": {
        "torch": ["torch.quantize_per_tensor"],
        "jax": ["jax.experimental.quantization"],
    },
    "dequantize": {
        "torch": ["torch.dequantize"],
        "jax": ["jax.experimental.quantization"],
    },
    "fp8_e4m3": {
        "torch": ["torch.float8_e4m3fn"],
        "nvidia_sass": ["WGMMA_MMA_ASYNC_E4M3"],
    },
    "fp8_e5m2": {
        "torch": ["torch.float8_e5m2"],
        "nvidia_sass": ["WGMMA_MMA_ASYNC_E5M2"],
    },
    "scaled_fp8_quant": {
        "torch": ["torch.quantize_per_tensor"],
    },
    "mx_fp4": {
        "torch": ["torch.float8_e4m3fn"],
        "nvidia_sass": ["MMA_SCALE_FP4"],
    },
    "mxfp8": {
        "torch": ["torch.float8_e4m3fn"],
        "nvidia_sass": ["MMA_SCALE_FP8"],
    },
    "collective": {
        "torch": [
            "torch.distributed.all_reduce",
            "torch.distributed.all_gather",
        ],
        "jax": ["jax.lax.all_gather"],
    },
    "all_reduce": {
        "torch": ["torch.distributed.all_reduce"],
        "jax": ["jax.lax.psum"],
    },
    "all_gather": {
        "torch": ["torch.distributed.all_gather"],
        "jax": ["jax.lax.all_gather"],
    },
    "reduce_scatter": {
        "torch": ["torch.distributed.reduce_scatter"],
        "jax": ["jax.lax.reduce_scatter"],
    },
    "memcpy": {
        "torch": ["torch.Tensor.copy_"],
        "nvidia_sass": ["LDG", "STG"],
    },
    "gather": {
        "torch": ["torch.gather"],
        "jax": ["jax.lax.gather"],
        "tensorflow": ["tf.gather"],
    },
    "scatter": {
        "torch": ["torch.scatter"],
        "jax": ["jax.lax.scatter"],
        "tensorflow": ["tf.scatter_nd"],
    },
    "matmul": {
        "torch": ["torch.matmul", "torch.mm", "torch.bmm"],
        "jax": ["jax.numpy.matmul"],
        "tensorflow": ["tf.linalg.matmul", "tf.matmul"],
        "numpy": ["numpy.matmul"],
    },
    "convolution": {
        "torch": ["torch.nn.functional.conv2d"],
        "jax": ["jax.lax.conv_general_dilated"],
        "tensorflow": ["tf.nn.conv2d"],
        "stablehlo": ["stablehlo.convolution"],
        "amd_rdna": ["v_dot4c_i32_i8"],
    },
    "add": {
        "torch": ["torch.add"],
        "jax": ["jax.numpy.add"],
        "tensorflow": ["tf.add"],
        "numpy": ["numpy.add"],
    },
    "softmax": {
        "torch": ["torch.nn.functional.softmax"],
        "jax": ["jax.nn.softmax"],
        "tensorflow": ["tf.nn.softmax"],
    },
    "adam_optimizer": {
        "torch": ["torch.optim.Adam", "torch.optim.AdamW"],
        "optax": ["optax.adam", "optax.adamw"],
        "keras": ["keras.optimizers.Adam", "keras.optimizers.AdamW"],
    },
    "assert_allclose": {
        "torch": ["torch.testing.assert_close"],
        "chex": ["chex.assert_trees_all_close"],
        "numpy": ["numpy.testing.assert_allclose"],
    },
    "data_loader": {
        "torch": ["torch.utils.data.DataLoader"],
        "grain": ["grain.DataLoader", "grain.python.DataLoader"],
    },
    "_parameter_translations": {
        "matmul": {
            "roles": {
                "lhs": {
                    "torch": ["input", "mat1"],
                    "jax": ["a", "lhs"],
                    "tensorflow": ["a"],
                    "tf": ["a"],
                    "stablehlo": ["lhs"],
                    "numpy": ["a"],
                },
                "rhs": {
                    "torch": ["other", "mat2"],
                    "jax": ["b", "rhs"],
                    "tensorflow": ["b"],
                    "tf": ["b"],
                    "stablehlo": ["rhs"],
                    "numpy": ["b"],
                },
            },
            "defaults": {
                "stablehlo": {
                    "dot_dimension_numbers": {
                        "lhs_contracting_dimensions": [1],
                        "rhs_contracting_dimensions": [0],
                    }
                }
            },
        },
        "reduction": {
            "roles": {
                "dimension": {
                    "torch": ["dim"],
                    "jax": ["axis"],
                    "tensorflow": ["axis"],
                    "tf": ["axis"],
                    "numpy": ["axis"],
                },
                "keepdims": {
                    "torch": ["keepdim"],
                    "jax": ["keepdims"],
                    "tensorflow": ["keepdims"],
                    "tf": ["keepdims"],
                    "numpy": ["keepdims"],
                },
            },
        },
        "convolution": {
            "roles": {
                "input": {
                    "torch": ["input"],
                    "jax": ["lhs"],
                    "tensorflow": ["input"],
                    "tf": ["input"],
                    "stablehlo": ["lhs"],
                },
                "kernel": {
                    "torch": ["weight"],
                    "jax": ["rhs"],
                    "tensorflow": ["filters"],
                    "tf": ["filters"],
                    "stablehlo": ["rhs"],
                },
                "stride": {
                    "torch": ["stride"],
                    "jax": ["window_strides"],
                    "tensorflow": ["strides"],
                    "tf": ["strides"],
                    "stablehlo": ["window_strides"],
                },
                "padding": {
                    "torch": ["padding"],
                    "jax": ["padding"],
                    "tensorflow": ["padding"],
                    "tf": ["padding"],
                    "stablehlo": ["padding"],
                },
            },
        },
        "normalization": {
            "roles": {
                "scale": {
                    "torch": ["weight"],
                    "jax": ["scale"],
                    "tensorflow": ["gamma"],
                    "tf": ["gamma"],
                    "stablehlo": ["scale"],
                },
                "offset": {
                    "torch": ["bias"],
                    "jax": ["bias"],
                    "tensorflow": ["beta"],
                    "tf": ["beta"],
                    "stablehlo": ["offset"],
                },
                "epsilon": {
                    "torch": ["eps"],
                    "jax": ["epsilon"],
                    "tensorflow": ["epsilon"],
                    "tf": ["epsilon"],
                    "stablehlo": ["epsilon"],
                },
            },
        },
        "adam_optimizer": {
            "roles": {
                "learning_rate": {
                    "torch": ["lr"],
                    "optax": ["learning_rate"],
                    "keras": ["learning_rate"],
                },
                "epsilon": {
                    "torch": ["eps"],
                    "optax": ["eps"],
                    "keras": ["epsilon"],
                },
                "weight_decay": {
                    "torch": ["weight_decay"],
                    "optax": ["weight_decay"],
                    "keras": ["weight_decay"],
                },
            },
        },
        "assert_allclose": {
            "roles": {
                "rtol": {
                    "torch": ["rtol"],
                    "chex": ["rtol"],
                    "numpy": ["rtol"],
                },
                "atol": {
                    "torch": ["atol"],
                    "chex": ["atol"],
                    "numpy": ["atol"],
                },
                "message": {
                    "torch": ["msg"],
                    "chex": ["custom_message"],
                    "numpy": ["err_msg"],
                },
            },
        },
        "data_loader": {
            "roles": {
                "data_source": {
                    "torch": ["dataset"],
                    "grain": ["data_source"],
                },
                "sampler": {
                    "torch": ["sampler"],
                    "grain": ["sampler"],
                },
                "workers": {
                    "torch": ["num_workers"],
                    "grain": ["worker_count"],
                },
            },
        },
    },
}


def load_concept_map(
    custom_path: str | None = None,
) -> dict[str, JsonValue]:
    """Load concept ontology mapping from bundled JSON or custom external file with fallback.

    Args:
        custom_path: Optional custom file path to concept map JSON.

    Returns:
        Mapping from concept string to framework-specific API paths.
    """
    if custom_path is not None:
        if os.path.exists(custom_path):
            try:
                with open(custom_path, "r", encoding="utf-8") as f:
                    data: JsonValue = json.load(f)
                    if isinstance(data, dict):
                        loaded_custom: dict[str, JsonValue] = dict(data)
                        return loaded_custom
            except Exception as e:
                print(f"Warning: Failed to load concept map from {custom_path}: {e}")
        return {}

    base_pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.environ.get("ML_SNAPSHOTS_CONCEPT_MAP") or os.path.join(
        base_pkg_dir, "concept_map.json"
    )
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data_bundled: JsonValue = json.load(f)
                if isinstance(data_bundled, dict):
                    loaded_bundled: dict[str, JsonValue] = dict(data_bundled)
                    return loaded_bundled
        except Exception as e:
            print(f"Warning: Failed to load concept map from {path}: {e}")
    return DEFAULT_CONCEPT_MAP


CONCEPT_ALIAS_MAP: dict[str, JsonValue] = load_concept_map()


def translate_concept_arguments(
    concept: str,
    source_framework: str,
    target_framework: str,
    source_kwargs: typing.Mapping[str, JsonValue],
    alias_map: typing.Mapping[str, JsonValue] | None = None,
) -> ConceptTranslationResult:
    """Translate operation arguments across frameworks using concept parameter mapping schemas.

    Translates parameter names and attributes for core concepts such as:
        - 'matmul': translates torch (input, other, mat1, mat2) to jax/numpy (a, b) or stablehlo (lhs, rhs).
        - 'reduce_sum' / 'reduce_mean' / 'reduction': translates dim/keepdim (torch) <-> axis/keepdims (jax, numpy, tf).
        - 'convolution': translates torch/jax/tf conv parameters and layout formats.
        - 'normalization': translates weight/bias/eps (torch) <-> scale/offset/epsilon (jax, stablehlo).
        - 'softmax': translates dim (torch) <-> axis (jax, numpy, tf).

    Args:
        concept: The canonical concept name (e.g. 'matmul', 'reduce_sum', 'softmax', 'convolution').
        source_framework: Source framework identifier (e.g. 'torch', 'jax', 'numpy').
        target_framework: Target framework identifier (e.g. 'jax', 'torch', 'stablehlo').
        source_kwargs: Dictionary of source kwargs or parameter values.
        alias_map: Optional concept alias mapping override (defaults to `CONCEPT_ALIAS_MAP`).

    Returns:
        `ConceptTranslationResult` model with translated target keyword arguments and structured attributes.
    """
    active_alias_map = alias_map if alias_map is not None else CONCEPT_ALIAS_MAP

    c_clean = concept.lower().strip()
    s_fw = source_framework.lower().strip()
    t_fw = target_framework.lower().strip()

    raw_pt = active_alias_map.get("_parameter_translations")
    if not isinstance(raw_pt, dict):
        raw_pt = DEFAULT_CONCEPT_MAP.get("_parameter_translations", {})
    param_translations: dict[str, JsonValue] = (
        dict(raw_pt) if isinstance(raw_pt, dict) else {}
    )
    schema: dict[str, JsonValue] | None = None

    if c_clean in param_translations and isinstance(param_translations[c_clean], dict):
        cand_schema = param_translations[c_clean]
        assert isinstance(cand_schema, dict)
        schema = dict(cand_schema)
    elif "matmul" in c_clean or c_clean in ("mm", "bmm", "dot"):
        raw_matmul = param_translations.get("matmul")
        schema = dict(raw_matmul) if isinstance(raw_matmul, dict) else None
    elif any(
        k in c_clean
        for k in (
            "sum",
            "mean",
            "max",
            "min",
            "prod",
            "argmax",
            "argmin",
            "reduction",
            "softmax",
        )
    ):
        raw_red = param_translations.get("reduction")
        schema = dict(raw_red) if isinstance(raw_red, dict) else None
    elif "conv" in c_clean:
        raw_conv = param_translations.get("convolution")
        schema = dict(raw_conv) if isinstance(raw_conv, dict) else None
    elif "norm" in c_clean:
        raw_norm = param_translations.get("normalization")
        schema = dict(raw_norm) if isinstance(raw_norm, dict) else None

    translated: dict[str, JsonValue] = {}
    unmapped: dict[str, JsonValue] = {}
    mapped_keys: set[str] = set()

    if schema and "roles" in schema and isinstance(schema["roles"], dict):
        roles = schema["roles"]
        for _role, fw_map in roles.items():
            assert isinstance(fw_map, dict)
            raw_src_keys = fw_map.get(s_fw)
            src_keys: list[str] = (
                [str(k) for k in raw_src_keys] if isinstance(raw_src_keys, list) else []
            )
            if not src_keys:
                all_keys: list[str] = []
                for aliases in fw_map.values():
                    assert isinstance(aliases, list)
                    all_keys.extend(str(a) for a in aliases)
                src_keys = all_keys

            for sk in src_keys:
                if sk in source_kwargs:
                    val = source_kwargs[sk]
                    raw_tgt_keys = fw_map.get(t_fw)
                    tgt_keys = (
                        [str(k) for k in raw_tgt_keys]
                        if isinstance(raw_tgt_keys, list)
                        else []
                    )
                    if tgt_keys:
                        translated[tgt_keys[0]] = val
                        mapped_keys.add(sk)
                    elif s_fw == t_fw:
                        translated[sk] = val
                        mapped_keys.add(sk)
                    break

        raw_defaults = schema.get("defaults")
        if (
            isinstance(raw_defaults, dict)
            and t_fw in raw_defaults
            and isinstance(raw_defaults[t_fw], dict)
        ):
            tgt_def = raw_defaults[t_fw]
            assert isinstance(tgt_def, dict)
            translated.update(tgt_def)

    for k, v in source_kwargs.items():
        if k not in mapped_keys:
            if s_fw == t_fw and schema:
                translated[k] = v
            else:
                unmapped[k] = v

    return ConceptTranslationResult(
        concept=concept,
        source_framework=source_framework,
        target_framework=target_framework,
        translated_kwargs=translated,
        unmapped_kwargs=unmapped,
    )


__all__ = [
    "CONCEPT_ALIAS_MAP",
    "ConceptMapSchema",
    "ConceptParameterSpec",
    "DEFAULT_CONCEPT_MAP",
    "load_concept_map",
    "translate_concept_arguments",
]
