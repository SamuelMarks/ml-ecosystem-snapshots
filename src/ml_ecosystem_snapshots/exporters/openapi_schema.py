"""JSON Schema and OpenAPI 3.0 specification emitters for GhostRef models."""

from __future__ import annotations

import typing
from collections import OrderedDict

import pydantic
from ml_switcheroo_ir.schema.ghost import GhostRef

from ml_ecosystem_snapshots.models import JsonValue


class CddIrModel(pydantic.BaseModel):
    """Typed cdd-python Intermediate Representation (IR) model."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    type: str = "static"
    doc: str = ""
    params: dict[str, dict[str, str]] = pydantic.Field(default_factory=dict)
    returns: dict[str, dict[str, str]] | None = None

    def to_cdd_dict(self) -> dict[str, JsonValue]:
        """Convert to a mutable dictionary with `OrderedDict` params/returns for `cdd-python`.

        Returns:
            Dictionary suitable for `cdd-python` emitters.
        """
        params_od: dict[str, JsonValue] = OrderedDict((k, dict(v)) for k, v in self.params.items())
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
    """Convert a GhostRef to cdd-python Intermediate Representation (IR).

    Args:
        ref: The `GhostRef` instance to convert into a `CddIrModel`.

    Returns:
        The `CddIrModel` representation.
    """
    params_map: dict[str, dict[str, str]] = {}

    for param in ref.params:
        p_dict: dict[str, str] = {}
        if param.annotation:
            p_dict["typ"] = param.annotation
        if param.description:
            p_dict["doc"] = param.description
        if param.default is not None:
            p_dict["default"] = str(param.default)
        params_map[param.name] = p_dict

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
        params=params_map,
        returns=returns_map,
    )


def to_json_schema(ref: GhostRef) -> dict[str, JsonValue]:
    """Convert a GhostRef into a JSON Schema.

    Args:
        ref: The GhostRef to convert.

    Returns:
        A dictionary representing the JSON Schema.
    """
    import cdd.json_schema.emit

    ir = _ghost_to_cdd_ir(ref)
    schema: dict[str, JsonValue] = cdd.json_schema.emit.json_schema(ir.to_cdd_dict(), identifier=ref.api_path)
    return schema


def to_openapi(refs: typing.Sequence[GhostRef]) -> dict[str, JsonValue]:
    """Convert a sequence of GhostRefs into an OpenAPI specification.

    Args:
        refs: A sequence of GhostRefs.

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
