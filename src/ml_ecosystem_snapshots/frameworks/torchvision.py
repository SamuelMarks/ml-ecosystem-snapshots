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

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalSubmoduleOpSpec,
    CanonicalSubmoduleOpTuple,
    build_canonical_submodule_ops,
    collect_from_submodules_or_canonical,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

_TORCHVISION_OP_DEFS: list[CanonicalSubmoduleOpTuple] = [
    (
        "nms",
        "torchvision.ops.nms",
        "function",
        "ops",
        [
            ("boxes", "POSITIONAL_ONLY", None),
            ("scores", "POSITIONAL_ONLY", None),
            ("iou_threshold", "POSITIONAL_ONLY", None),
        ],
        "Performs non-maximum suppression (NMS) on boxes according to intersection-over-union.",
    ),
    (
        "box_iou",
        "torchvision.ops.box_iou",
        "function",
        "ops",
        [("boxes1", "POSITIONAL_ONLY", None), ("boxes2", "POSITIONAL_ONLY", None)],
        "Return intersection-over-union (Jaccard index) between two sets of boxes.",
    ),
    (
        "roi_align",
        "torchvision.ops.roi_align",
        "function",
        "ops",
        [
            ("input", "POSITIONAL_ONLY", None),
            ("boxes", "POSITIONAL_ONLY", None),
            ("output_size", "POSITIONAL_ONLY", None),
            ("spatial_scale", "KEYWORD_ONLY", "1.0"),
            ("sampling_ratio", "KEYWORD_ONLY", "-1"),
            ("aligned", "KEYWORD_ONLY", "False"),
        ],
        "Performs Region of Interest (RoI) Align operator described in Mask R-CNN.",
    ),
    (
        "ps_roi_pool",
        "torchvision.ops.ps_roi_pool",
        "function",
        "ops",
        [
            ("input", "POSITIONAL_ONLY", None),
            ("boxes", "POSITIONAL_ONLY", None),
            ("output_size", "POSITIONAL_ONLY", None),
            ("spatial_scale", "KEYWORD_ONLY", "1.0"),
        ],
        "Performs Position-Sensitive Region of Interest (RoI) Pool operator.",
    ),
    (
        "masks_to_boxes",
        "torchvision.ops.masks_to_boxes",
        "function",
        "ops",
        [("masks", "POSITIONAL_ONLY", None)],
        "Compute bounding boxes surrounding the masks.",
    ),
    (
        "box_convert",
        "torchvision.ops.box_convert",
        "function",
        "ops",
        [("boxes", "POSITIONAL_ONLY", None), ("in_fmt", "POSITIONAL_ONLY", None), ("out_fmt", "POSITIONAL_ONLY", None)],
        "Converts boxes from given in_fmt to out_fmt.",
    ),
    (
        "clip_boxes_to_image",
        "torchvision.ops.clip_boxes_to_image",
        "function",
        "ops",
        [("boxes", "POSITIONAL_ONLY", None), ("size", "POSITIONAL_ONLY", None)],
        "Clip boxes so that they lie inside an image of given size.",
    ),
    (
        "elastic_transform",
        "torchvision.transforms.functional.elastic_transform",
        "function",
        "transforms.functional",
        [
            ("img", "POSITIONAL_ONLY", None),
            ("displacement", "POSITIONAL_ONLY", None),
            ("interpolation", "KEYWORD_ONLY", "InterpolationMode.BILINEAR"),
            ("fill", "KEYWORD_ONLY", "None"),
        ],
        "Transform image using elastic distortion displacement field.",
    ),
    (
        "gaussian_blur",
        "torchvision.transforms.functional.gaussian_blur",
        "function",
        "transforms.functional",
        [("img", "POSITIONAL_ONLY", None), ("kernel_size", "POSITIONAL_ONLY", None), ("sigma", "KEYWORD_ONLY", "None")],
        "Performs Gaussian blurring on the image by given kernel.",
    ),
    (
        "perspective",
        "torchvision.transforms.functional.perspective",
        "function",
        "transforms.functional",
        [
            ("img", "POSITIONAL_ONLY", None),
            ("startpoints", "POSITIONAL_ONLY", None),
            ("endpoints", "POSITIONAL_ONLY", None),
            ("interpolation", "KEYWORD_ONLY", "InterpolationMode.BILINEAR"),
            ("fill", "KEYWORD_ONLY", "None"),
        ],
        "Perform perspective transform of the given image.",
    ),
    (
        "resize",
        "torchvision.transforms.functional.resize",
        "function",
        "transforms.functional",
        [
            ("img", "POSITIONAL_ONLY", None),
            ("size", "POSITIONAL_ONLY", None),
            ("interpolation", "KEYWORD_ONLY", "InterpolationMode.BILINEAR"),
            ("max_size", "KEYWORD_ONLY", "None"),
            ("antialias", "KEYWORD_ONLY", "True"),
        ],
        "Resize the input image to the given size.",
    ),
    (
        "crop",
        "torchvision.transforms.functional.crop",
        "function",
        "transforms.functional",
        [
            ("img", "POSITIONAL_ONLY", None),
            ("top", "POSITIONAL_ONLY", None),
            ("left", "POSITIONAL_ONLY", None),
            ("height", "POSITIONAL_ONLY", None),
            ("width", "POSITIONAL_ONLY", None),
        ],
        "Crop the given image at specified location and output size.",
    ),
    (
        "pad",
        "torchvision.transforms.functional.pad",
        "function",
        "transforms.functional",
        [
            ("img", "POSITIONAL_ONLY", None),
            ("padding", "POSITIONAL_ONLY", None),
            ("fill", "KEYWORD_ONLY", "0"),
            ("padding_mode", "KEYWORD_ONLY", "'constant'"),
        ],
        "Pad the given image on all sides with specified padding.",
    ),
    (
        "rotate",
        "torchvision.transforms.functional.rotate",
        "function",
        "transforms.functional",
        [
            ("img", "POSITIONAL_ONLY", None),
            ("angle", "POSITIONAL_ONLY", None),
            ("interpolation", "KEYWORD_ONLY", "InterpolationMode.NEAREST"),
            ("expand", "KEYWORD_ONLY", "False"),
            ("center", "KEYWORD_ONLY", "None"),
            ("fill", "KEYWORD_ONLY", "None"),
        ],
        "Rotate the image by angle.",
    ),
    (
        "affine",
        "torchvision.transforms.functional.affine",
        "function",
        "transforms.functional",
        [
            ("img", "POSITIONAL_ONLY", None),
            ("angle", "POSITIONAL_ONLY", None),
            ("translate", "POSITIONAL_ONLY", None),
            ("scale", "POSITIONAL_ONLY", None),
            ("shear", "POSITIONAL_ONLY", None),
            ("interpolation", "KEYWORD_ONLY", "InterpolationMode.NEAREST"),
            ("fill", "KEYWORD_ONLY", "None"),
            ("center", "KEYWORD_ONLY", "None"),
        ],
        "Apply affine transformation on the image keeping image center invariant.",
    ),
    (
        "normalize",
        "torchvision.transforms.functional.normalize",
        "function",
        "transforms.functional",
        [
            ("tensor", "POSITIONAL_ONLY", None),
            ("mean", "POSITIONAL_ONLY", None),
            ("std", "POSITIONAL_ONLY", None),
            ("inplace", "KEYWORD_ONLY", "False"),
        ],
        "Normalize a tensor image with mean and standard deviation.",
    ),
    (
        "draw_bounding_boxes",
        "torchvision.utils.draw_bounding_boxes",
        "function",
        "utils",
        [
            ("image", "POSITIONAL_ONLY", None),
            ("boxes", "POSITIONAL_ONLY", None),
            ("labels", "KEYWORD_ONLY", "None"),
            ("colors", "KEYWORD_ONLY", "None"),
            ("fill", "KEYWORD_ONLY", "False"),
            ("width", "KEYWORD_ONLY", "1"),
            ("font", "KEYWORD_ONLY", "None"),
            ("font_size", "KEYWORD_ONLY", "None"),
        ],
        "Draws bounding boxes on given image.",
    ),
    (
        "make_grid",
        "torchvision.utils.make_grid",
        "function",
        "utils",
        [
            ("tensor", "POSITIONAL_ONLY", None),
            ("nrow", "KEYWORD_ONLY", "8"),
            ("padding", "KEYWORD_ONLY", "2"),
            ("normalize", "KEYWORD_ONLY", "False"),
            ("value_range", "KEYWORD_ONLY", "None"),
            ("scale_each", "KEYWORD_ONLY", "False"),
            ("pad_value", "KEYWORD_ONLY", "0"),
        ],
        "Make a grid of images.",
    ),
    (
        "save_image",
        "torchvision.utils.save_image",
        "function",
        "utils",
        [("tensors", "POSITIONAL_ONLY", None), ("fp", "POSITIONAL_ONLY", None), ("format", "KEYWORD_ONLY", "None")],
        "Save a given Tensor into an image file.",
    ),
]

CANONICAL_TORCHVISION_OPS: list[CanonicalSubmoduleOpSpec] = build_canonical_submodule_ops(_TORCHVISION_OP_DEFS)


def _get_torchvision_submodule(submod: str) -> types.ModuleType | None:
    """Safely import a specific TorchVision submodule.

    Args:
        submod: Name of the TorchVision submodule (e.g. 'ops', 'utils').

    Returns:
        The submodule object or None if not available.
    """
    try:
        return importlib.import_module(f"torchvision.{submod}")
    except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError, TypeError, ValueError, OSError):
        return None


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
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

    return collect_from_submodules_or_canonical(
        loaded_submods=loaded_submods,
        canonical_ops=CANONICAL_TORCHVISION_OPS,
        framework_tag="torchvision",
        default_submodule="ops",
        default_doc_prefix="TorchVision",
        inspector_fn=GhostInspector.inspect,
    )
