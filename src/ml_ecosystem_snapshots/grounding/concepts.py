"""Cross-framework semantic concept argument translator."""

from __future__ import annotations

import json
import os
import typing

import pydantic

from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.grounding._concept_catalog import (
    DEFAULT_CONCEPT_MAP,
)
from ml_ecosystem_snapshots.grounding.models import ConceptTranslationResult
from ml_ecosystem_snapshots.models import JsonValue, ensure_json_dict, ensure_json_list


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
                with open(custom_path, encoding="utf-8") as f:
                    data: JsonValue = json.load(f)
                    if isinstance(data, dict):
                        loaded_custom: dict[str, JsonValue] = dict(data)
                        return loaded_custom
            except (
                SnapshotError,
                ImportError,
                AttributeError,
                TypeError,
                ValueError,
                LookupError,
                OSError,
                RuntimeError,
                SyntaxError,
                pydantic.ValidationError,
            ) as e:
                print(f"Warning: Failed to load concept map from {custom_path}: {e}")
        return {}

    base_pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.environ.get("ML_SNAPSHOTS_CONCEPT_MAP") or os.path.join(base_pkg_dir, "concept_map.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data_bundled: JsonValue = json.load(f)
                if isinstance(data_bundled, dict):
                    loaded_bundled: dict[str, JsonValue] = dict(data_bundled)
                    return loaded_bundled
        except (
            SnapshotError,
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            LookupError,
            OSError,
            RuntimeError,
            SyntaxError,
            pydantic.ValidationError,
        ) as e:
            print(f"Warning: Failed to load concept map from {path}: {e}")
    return DEFAULT_CONCEPT_MAP


CONCEPT_ALIAS_MAP: dict[str, JsonValue] = load_concept_map()


def _resolve_concept_translation_schema(
    c_clean: str,
    active_alias_map: typing.Mapping[str, JsonValue],
) -> dict[str, JsonValue] | None:
    """Resolve the parameter translation schema dictionary for a normalized concept name.

    Args:
        c_clean: Lowercased and stripped concept name.
        active_alias_map: Active concept ontology dictionary.

    Returns:
        Matching parameter translation schema dictionary, or None if not found.
    """
    raw_pt = active_alias_map.get("_parameter_translations")
    if not isinstance(raw_pt, dict):
        raw_pt = DEFAULT_CONCEPT_MAP.get("_parameter_translations", {})
    param_translations: dict[str, JsonValue] = dict(ensure_json_dict(raw_pt))

    schema_key: str | None = None
    if c_clean in param_translations:
        schema_key = c_clean
    elif "matmul" in c_clean or c_clean in ("mm", "bmm", "dot"):
        schema_key = "matmul"
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
        schema_key = "reduction"
    elif "conv" in c_clean:
        schema_key = "convolution"
    elif "norm" in c_clean:
        schema_key = "normalization"

    if schema_key is not None:
        raw_entry = ensure_json_dict(param_translations.get(schema_key))
        return dict(raw_entry)
    return None


def _resolve_role_source_keys(
    fw_map: dict[str, JsonValue],
    s_fw: str,
) -> list[str]:
    """Return candidate source parameter names for a concept role and source framework.

    Args:
        fw_map: Mapping of framework identifiers to parameter name lists for a single role.
        s_fw: Lowercased source framework identifier.

    Returns:
        List of candidate source parameter names.
    """
    raw_src_keys = fw_map.get(s_fw)
    if isinstance(raw_src_keys, list) and raw_src_keys:
        return [str(k) for k in raw_src_keys]
    all_keys: list[str] = []
    for raw_aliases in fw_map.values():
        aliases = ensure_json_list(raw_aliases)
        all_keys.extend(str(a) for a in aliases)
    return all_keys


def _apply_concept_role_translations(
    schema: dict[str, JsonValue],
    s_fw: str,
    t_fw: str,
    source_kwargs: typing.Mapping[str, JsonValue],
) -> tuple[dict[str, JsonValue], set[str]]:
    """Apply role mappings and target defaults from a concept schema to source kwargs.

    Args:
        schema: Parameter translation schema dictionary containing `roles` and optional `defaults`.
        s_fw: Lowercased source framework identifier.
        t_fw: Lowercased target framework identifier.
        source_kwargs: Source keyword arguments passed by the caller.

    Returns:
        Tuple of `(translated_kwargs_dict, mapped_source_keys_set)`.
    """
    translated: dict[str, JsonValue] = {}
    mapped_keys: set[str] = set()
    roles = ensure_json_dict(schema.get("roles"))
    for _role, raw_fw_map in roles.items():
        fw_map = ensure_json_dict(raw_fw_map)
        src_keys = _resolve_role_source_keys(fw_map, s_fw)
        for sk in src_keys:
            if sk in source_kwargs:
                val = source_kwargs[sk]
                raw_tgt_keys = fw_map.get(t_fw)
                tgt_keys = [str(k) for k in raw_tgt_keys] if isinstance(raw_tgt_keys, list) else []
                if tgt_keys:
                    translated[tgt_keys[0]] = val
                    mapped_keys.add(sk)
                elif s_fw == t_fw:
                    translated[sk] = val
                    mapped_keys.add(sk)
                break

    raw_defaults = schema.get("defaults")
    if isinstance(raw_defaults, dict) and t_fw in raw_defaults:
        tgt_def = ensure_json_dict(raw_defaults[t_fw])
        translated.update(tgt_def)
    return translated, mapped_keys


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

    schema = _resolve_concept_translation_schema(c_clean, active_alias_map)

    translated: dict[str, JsonValue] = {}
    unmapped: dict[str, JsonValue] = {}
    mapped_keys: set[str] = set()

    if schema is not None:
        translated, mapped_keys = _apply_concept_role_translations(schema, s_fw, t_fw, source_kwargs)

    for k, v in source_kwargs.items():
        if k not in mapped_keys:
            if s_fw == t_fw and schema is not None:
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
    "DEFAULT_CONCEPT_MAP",
    "load_concept_map",
    "translate_concept_arguments",
]
