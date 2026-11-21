"""JSON Schema and OpenAPI 3.0 specification emitters for GhostRef models."""

from __future__ import annotations

import typing
from collections import OrderedDict

import pydantic
from ml_switcheroo_ir.schema.ghost import GhostRef

from ml_ecosystem_snapshots.models import JsonValue


class CddIrModel(pydantic.BaseModel):
    """Immutable typed cdd-python Intermediate Representation (IR) model."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str
    type: str = "static"
    doc: str = ""
    params: tuple[dict[str, JsonValue], ...] = ()
    returns: dict[str, dict[str, str]] | None = None

    def param_map(self) -> dict[str, dict[str, str]]:
        """Project `params` tuple entries into a dictionary keyed by parameter name.

        Returns:
            Mapping from parameter name to `{typ, doc, default}` strings.
        """
        out: dict[str, dict[str, str]] = {}
        for entry in self.params:
            name_str = str(entry["name"])
            meta: dict[str, str] = {}
            for k in ("typ", "doc", "default"):
                val = entry.get(k)
                if isinstance(val, str):
                    meta[k] = val
            out[name_str] = meta
        return out

    def to_cdd_dict(self) -> dict[str, JsonValue]:
        """Convert to a mutable dictionary with `OrderedDict` params/returns for `cdd-python`.

        Returns:
            Dictionary suitable for `cdd-python` emitters.
        """
        params_od: dict[str, JsonValue] = OrderedDict((k, dict(v)) for k, v in self.param_map().items())
        returns_od: dict[str, JsonValue] | None = (
            OrderedDict((k, dict(v)) for k, v in self.returns.items()) if self.returns is not None else None
        )
        return {
            "name": self.name,
            "type": self.type,
            "doc": self.doc,
            "params": params_od,
            "returns": returns_od,
        }


def ghost_to_cdd_ir(ref: GhostRef) -> CddIrModel:
    """Convert a `GhostRef` to an immutable `CddIrModel`.

    Args:
        ref: The `GhostRef` instance to convert into a `CddIrModel`.

    Returns:
        Frozen `CddIrModel` representation.
    """
    params_entries: list[dict[str, JsonValue]] = []

    for param in ref.params:
        p_dict: dict[str, JsonValue] = {"name": param.name}
        if param.annotation:
            p_dict["typ"] = param.annotation
        if param.description:
            p_dict["doc"] = param.description
        if param.default is not None:
            p_dict["default"] = str(param.default)
        params_entries.append(p_dict)

    returns_map: dict[str, dict[str, str]] | None = None
    if ref.returns_type or ref.returns_description:
        ret_dict: dict[str, str] = {}
        if ref.returns_type:
            ret_dict["typ"] = ref.returns_type
        if ref.returns_description:
            ret_dict["doc"] = ref.returns_description
        returns_map = {"return_type": ret_dict}

    return CddIrModel(
        name=ref.name,
        type="class" if ref.kind == "class" else "static",
        doc=ref.docstring or "",
        params=tuple(params_entries),
        returns=returns_map,
    )


def to_json_schema(ref: GhostRef) -> dict[str, JsonValue]:
    """Convert a `GhostRef` into a JSON Schema.

    Args:
        ref: The `GhostRef` to convert.

    Returns:
        A dictionary representing the JSON Schema.
    """
    import cdd.json_schema.emit

    ir = _ghost_to_cdd_ir(ref)
    schema: dict[str, JsonValue] = cdd.json_schema.emit.json_schema(ir.to_cdd_dict(), identifier=ref.api_path)
    return schema


def to_openapi(refs: typing.Sequence[GhostRef]) -> dict[str, JsonValue]:
    """Convert a sequence of `GhostRef`s into an OpenAPI specification.

    Args:
        refs: A sequence of `GhostRef`s.

    Returns:
        A dictionary representing the OpenAPI specification.
    """
    import cdd.compound.openapi.emit
    from cdd.compound.openapi.utils.emit_openapi_utils import NameModelRouteIdCrud

    nm_cruds = []
    for ref in refs:
        ir = _ghost_to_cdd_ir(ref)
        route = f"/{ref.api_path.replace('.', '/')}"

        nm_cruds.append(
            NameModelRouteIdCrud(
                name=ref.name,
                model=ir.to_cdd_dict(),
                route=route,
                id=None,
                crud="CRU",
            )
        )

    spec: dict[str, JsonValue] = cdd.compound.openapi.emit.openapi(nm_cruds)
    return spec


_ghost_to_cdd_ir = ghost_to_cdd_ir


__all__ = [
    "CddIrModel",
    "ghost_to_cdd_ir",
    "to_json_schema",
    "to_openapi",
]
