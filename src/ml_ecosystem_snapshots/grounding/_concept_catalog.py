"""Canonical cross-framework concept and parameter translation catalog."""

from __future__ import annotations

import pydantic

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


__all__ = ["ConceptMapSchema", "ConceptParameterSpec", "DEFAULT_CONCEPT_MAP"]
