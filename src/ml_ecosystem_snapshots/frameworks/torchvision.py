"""TorchVision API Snapshot Extractor.

Extracts computer vision operations (NMS, RoIAlign, Box IoU), geometric
transforms, and image processing utilities from the TorchVision library.
"""

from __future__ import annotations

import importlib
import types

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    SemanticTier,
)
import pydantic

from ml_ecosystem_snapshots.frameworks._base import CanonicalSubmoduleOpSpec
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector


CANONICAL_TORCHVISION_OPS: list[CanonicalSubmoduleOpSpec] = pydantic.TypeAdapter(
    list[CanonicalSubmoduleOpSpec]
).validate_python(
    [
        # Detection & Bounding Box ops (torchvision.ops)
        {
            "name": "nms",
            "api_path": "torchvision.ops.nms",
            "submodule": "ops",
            "params": [
                {"name": "boxes", "kind": "POSITIONAL_ONLY"},
                {"name": "scores", "kind": "POSITIONAL_ONLY"},
                {"name": "iou_threshold", "kind": "POSITIONAL_ONLY"},
            ],
            "docstring": "Performs non-maximum suppression (NMS) on boxes according to intersection-over-union.",
        },
        {
            "name": "box_iou",
            "api_path": "torchvision.ops.box_iou",
            "submodule": "ops",
            "params": [
                {"name": "boxes1", "kind": "POSITIONAL_ONLY"},
                {"name": "boxes2", "kind": "POSITIONAL_ONLY"},
            ],
            "docstring": "Return intersection-over-union (Jaccard index) between two sets of boxes.",
        },
        {
            "name": "roi_align",
            "api_path": "torchvision.ops.roi_align",
            "submodule": "ops",
            "params": [
                {"name": "input", "kind": "POSITIONAL_ONLY"},
                {"name": "boxes", "kind": "POSITIONAL_ONLY"},
                {"name": "output_size", "kind": "POSITIONAL_ONLY"},
                {"name": "spatial_scale", "kind": "KEYWORD_ONLY", "default": "1.0"},
                {"name": "sampling_ratio", "kind": "KEYWORD_ONLY", "default": "-1"},
                {"name": "aligned", "kind": "KEYWORD_ONLY", "default": "False"},
            ],
            "docstring": "Performs Region of Interest (RoI) Align operator described in Mask R-CNN.",
        },
        {
            "name": "ps_roi_pool",
            "api_path": "torchvision.ops.ps_roi_pool",
            "submodule": "ops",
            "params": [
                {"name": "input", "kind": "POSITIONAL_ONLY"},
                {"name": "boxes", "kind": "POSITIONAL_ONLY"},
                {"name": "output_size", "kind": "POSITIONAL_ONLY"},
                {"name": "spatial_scale", "kind": "KEYWORD_ONLY", "default": "1.0"},
            ],
            "docstring": "Performs Position-Sensitive Region of Interest (RoI) Pool operator.",
        },
        {
            "name": "masks_to_boxes",
            "api_path": "torchvision.ops.masks_to_boxes",
            "submodule": "ops",
            "params": [
                {"name": "masks", "kind": "POSITIONAL_ONLY"},
            ],
            "docstring": "Compute bounding boxes surrounding the masks.",
        },
        {
            "name": "box_convert",
            "api_path": "torchvision.ops.box_convert",
            "submodule": "ops",
            "params": [
                {"name": "boxes", "kind": "POSITIONAL_ONLY"},
                {"name": "in_fmt", "kind": "POSITIONAL_ONLY"},
                {"name": "out_fmt", "kind": "POSITIONAL_ONLY"},
            ],
            "docstring": "Converts boxes from given in_fmt to out_fmt.",
        },
        {
            "name": "clip_boxes_to_image",
            "api_path": "torchvision.ops.clip_boxes_to_image",
            "submodule": "ops",
            "params": [
                {"name": "boxes", "kind": "POSITIONAL_ONLY"},
                {"name": "size", "kind": "POSITIONAL_ONLY"},
            ],
            "docstring": "Clip boxes so that they lie inside an image of given size.",
        },
        # Functional transforms (torchvision.transforms.functional)
        {
            "name": "elastic_transform",
            "api_path": "torchvision.transforms.functional.elastic_transform",
            "submodule": "transforms.functional",
            "params": [
                {"name": "img", "kind": "POSITIONAL_ONLY"},
                {"name": "displacement", "kind": "POSITIONAL_ONLY"},
                {
                    "name": "interpolation",
                    "kind": "KEYWORD_ONLY",
                    "default": "InterpolationMode.BILINEAR",
                },
                {"name": "fill", "kind": "KEYWORD_ONLY", "default": "None"},
            ],
            "docstring": "Transform image using elastic distortion displacement field.",
        },
        {
            "name": "gaussian_blur",
            "api_path": "torchvision.transforms.functional.gaussian_blur",
            "submodule": "transforms.functional",
            "params": [
                {"name": "img", "kind": "POSITIONAL_ONLY"},
                {"name": "kernel_size", "kind": "POSITIONAL_ONLY"},
                {"name": "sigma", "kind": "KEYWORD_ONLY", "default": "None"},
            ],
            "docstring": "Performs Gaussian blurring on the image by given kernel.",
        },
        {
            "name": "perspective",
            "api_path": "torchvision.transforms.functional.perspective",
            "submodule": "transforms.functional",
            "params": [
                {"name": "img", "kind": "POSITIONAL_ONLY"},
                {"name": "startpoints", "kind": "POSITIONAL_ONLY"},
                {"name": "endpoints", "kind": "POSITIONAL_ONLY"},
                {
                    "name": "interpolation",
                    "kind": "KEYWORD_ONLY",
                    "default": "InterpolationMode.BILINEAR",
                },
                {"name": "fill", "kind": "KEYWORD_ONLY", "default": "None"},
            ],
            "docstring": "Perform perspective transform of the given image.",
        },
        {
            "name": "resize",
            "api_path": "torchvision.transforms.functional.resize",
            "submodule": "transforms.functional",
            "params": [
                {"name": "img", "kind": "POSITIONAL_ONLY"},
                {"name": "size", "kind": "POSITIONAL_ONLY"},
                {
                    "name": "interpolation",
                    "kind": "KEYWORD_ONLY",
                    "default": "InterpolationMode.BILINEAR",
                },
                {"name": "max_size", "kind": "KEYWORD_ONLY", "default": "None"},
                {"name": "antialias", "kind": "KEYWORD_ONLY", "default": "True"},
            ],
            "docstring": "Resize the input image to the given size.",
        },
        {
            "name": "crop",
            "api_path": "torchvision.transforms.functional.crop",
            "submodule": "transforms.functional",
            "params": [
                {"name": "img", "kind": "POSITIONAL_ONLY"},
                {"name": "top", "kind": "POSITIONAL_ONLY"},
                {"name": "left", "kind": "POSITIONAL_ONLY"},
                {"name": "height", "kind": "POSITIONAL_ONLY"},
                {"name": "width", "kind": "POSITIONAL_ONLY"},
            ],
            "docstring": "Crop the given image at specified location and output size.",
        },
        {
            "name": "pad",
            "api_path": "torchvision.transforms.functional.pad",
            "submodule": "transforms.functional",
            "params": [
                {"name": "img", "kind": "POSITIONAL_ONLY"},
                {"name": "padding", "kind": "POSITIONAL_ONLY"},
                {"name": "fill", "kind": "KEYWORD_ONLY", "default": "0"},
                {
                    "name": "padding_mode",
                    "kind": "KEYWORD_ONLY",
                    "default": "'constant'",
                },
            ],
            "docstring": "Pad the given image on all sides with specified padding.",
        },
        {
            "name": "rotate",
            "api_path": "torchvision.transforms.functional.rotate",
            "submodule": "transforms.functional",
            "params": [
                {"name": "img", "kind": "POSITIONAL_ONLY"},
                {"name": "angle", "kind": "POSITIONAL_ONLY"},
                {
                    "name": "interpolation",
                    "kind": "KEYWORD_ONLY",
                    "default": "InterpolationMode.NEAREST",
                },
                {"name": "expand", "kind": "KEYWORD_ONLY", "default": "False"},
                {"name": "center", "kind": "KEYWORD_ONLY", "default": "None"},
                {"name": "fill", "kind": "KEYWORD_ONLY", "default": "None"},
            ],
            "docstring": "Rotate the image by angle.",
        },
        {
            "name": "affine",
            "api_path": "torchvision.transforms.functional.affine",
            "submodule": "transforms.functional",
            "params": [
                {"name": "img", "kind": "POSITIONAL_ONLY"},
                {"name": "angle", "kind": "POSITIONAL_ONLY"},
                {"name": "translate", "kind": "POSITIONAL_ONLY"},
                {"name": "scale", "kind": "POSITIONAL_ONLY"},
                {"name": "shear", "kind": "POSITIONAL_ONLY"},
                {
                    "name": "interpolation",
                    "kind": "KEYWORD_ONLY",
                    "default": "InterpolationMode.NEAREST",
                },
                {"name": "fill", "kind": "KEYWORD_ONLY", "default": "None"},
                {"name": "center", "kind": "KEYWORD_ONLY", "default": "None"},
            ],
            "docstring": "Apply affine transformation on the image keeping image center invariant.",
        },
        {
            "name": "normalize",
            "api_path": "torchvision.transforms.functional.normalize",
            "submodule": "transforms.functional",
            "params": [
                {"name": "tensor", "kind": "POSITIONAL_ONLY"},
                {"name": "mean", "kind": "POSITIONAL_ONLY"},
                {"name": "std", "kind": "POSITIONAL_ONLY"},
                {"name": "inplace", "kind": "KEYWORD_ONLY", "default": "False"},
            ],
            "docstring": "Normalize a tensor image with mean and standard deviation.",
        },
        # Utilities (torchvision.utils)
        {
            "name": "draw_bounding_boxes",
            "api_path": "torchvision.utils.draw_bounding_boxes",
            "submodule": "utils",
            "params": [
                {"name": "image", "kind": "POSITIONAL_ONLY"},
                {"name": "boxes", "kind": "POSITIONAL_ONLY"},
                {"name": "labels", "kind": "KEYWORD_ONLY", "default": "None"},
                {"name": "colors", "kind": "KEYWORD_ONLY", "default": "None"},
                {"name": "fill", "kind": "KEYWORD_ONLY", "default": "False"},
                {"name": "width", "kind": "KEYWORD_ONLY", "default": "1"},
                {"name": "font", "kind": "KEYWORD_ONLY", "default": "None"},
                {"name": "font_size", "kind": "KEYWORD_ONLY", "default": "None"},
            ],
            "docstring": "Draws bounding boxes on given image.",
        },
        {
            "name": "make_grid",
            "api_path": "torchvision.utils.make_grid",
            "submodule": "utils",
            "params": [
                {"name": "tensor", "kind": "POSITIONAL_ONLY"},
                {"name": "nrow", "kind": "KEYWORD_ONLY", "default": "8"},
                {"name": "padding", "kind": "KEYWORD_ONLY", "default": "2"},
                {"name": "normalize", "kind": "KEYWORD_ONLY", "default": "False"},
                {"name": "value_range", "kind": "KEYWORD_ONLY", "default": "None"},
                {"name": "scale_each", "kind": "KEYWORD_ONLY", "default": "False"},
                {"name": "pad_value", "kind": "KEYWORD_ONLY", "default": "0"},
            ],
            "docstring": "Make a grid of images.",
        },
        {
            "name": "save_image",
            "api_path": "torchvision.utils.save_image",
            "submodule": "utils",
            "params": [
                {"name": "tensors", "kind": "POSITIONAL_ONLY"},
                {"name": "fp", "kind": "POSITIONAL_ONLY"},
                {"name": "format", "kind": "KEYWORD_ONLY", "default": "None"},
            ],
            "docstring": "Save a given Tensor into an image file.",
        },
    ]
)


def _get_torchvision_submodule(submod: str) -> types.ModuleType | None:
    """Safely import a specific TorchVision submodule.

    Args:
        submod: Name of the TorchVision submodule (e.g. 'ops', 'utils').

    Returns:
        The submodule object or None if not available.
    """
    try:
        return importlib.import_module(f"torchvision.{submod}")
    except Exception:
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Collect TorchVision bounding box, transformation, and image utility APIs.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRef objects representing TorchVision operations.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.NEURAL_OPS,
        SemanticTier.UTIL,
    ):
        return []

    target_submodules = ["ops", "transforms.functional", "utils"]
    loaded_submods: dict[str, types.ModuleType] = {}
    for sub in target_submodules:
        mod = _get_torchvision_submodule(sub)
        if mod is not None:
            loaded_submods[sub] = mod

    from ml_ecosystem_snapshots.frameworks._base import (
        collect_from_submodules_or_canonical,
    )

    return collect_from_submodules_or_canonical(
        loaded_submods=loaded_submods,
        canonical_ops=CANONICAL_TORCHVISION_OPS,
        framework_tag="torchvision",
        default_submodule="ops",
        default_doc_prefix="TorchVision",
        inspector_fn=GhostInspector.inspect,
    )
