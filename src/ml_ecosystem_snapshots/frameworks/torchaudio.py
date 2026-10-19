"""TorchAudio API Snapshot Extractor.

Extracts audio processing operations (Mel filterbanks, spectrograms, MFCC feature transforms),
resampling filters, and waveform utilities from the TorchAudio library.
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


CANONICAL_TORCHAUDIO_OPS: list[CanonicalSubmoduleOpSpec] = pydantic.TypeAdapter(
    list[CanonicalSubmoduleOpSpec]
).validate_python(
    [
        # Functional audio ops (torchaudio.functional)
        {
            "name": "melscale_fbanks",
            "api_path": "torchaudio.functional.melscale_fbanks",
            "submodule": "functional",
            "params": [
                {"name": "n_freqs", "kind": "POSITIONAL_ONLY"},
                {"name": "f_min", "kind": "POSITIONAL_ONLY"},
                {"name": "f_max", "kind": "POSITIONAL_ONLY"},
                {"name": "n_mels", "kind": "POSITIONAL_ONLY"},
                {"name": "sample_rate", "kind": "POSITIONAL_ONLY"},
                {"name": "norm", "kind": "KEYWORD_ONLY", "default": "None"},
                {"name": "mel_scale", "kind": "KEYWORD_ONLY", "default": "'htk'"},
            ],
            "docstring": "Create a frequency conversion matrix from Hz to Mel.",
        },
        {
            "name": "spectrogram",
            "api_path": "torchaudio.functional.spectrogram",
            "submodule": "functional",
            "params": [
                {"name": "waveform", "kind": "POSITIONAL_ONLY"},
                {"name": "pad", "kind": "POSITIONAL_ONLY"},
                {"name": "window", "kind": "POSITIONAL_ONLY"},
                {"name": "n_fft", "kind": "POSITIONAL_ONLY"},
                {"name": "hop_length", "kind": "POSITIONAL_ONLY"},
                {"name": "win_length", "kind": "POSITIONAL_ONLY"},
                {"name": "power", "kind": "POSITIONAL_ONLY"},
                {"name": "normalized", "kind": "POSITIONAL_ONLY"},
            ],
            "docstring": "Create a spectrogram from a raw audio signal.",
        },
        {
            "name": "amplitude_to_DB",
            "api_path": "torchaudio.functional.amplitude_to_DB",
            "submodule": "functional",
            "params": [
                {"name": "x", "kind": "POSITIONAL_ONLY"},
                {"name": "multiplier", "kind": "POSITIONAL_ONLY"},
                {"name": "amin", "kind": "POSITIONAL_ONLY"},
                {"name": "db_multiplier", "kind": "POSITIONAL_ONLY"},
                {"name": "top_db", "kind": "KEYWORD_ONLY", "default": "None"},
            ],
            "docstring": "Turn a tensor from the power/amplitude scale to the decibel scale.",
        },
        {
            "name": "DB_to_amplitude",
            "api_path": "torchaudio.functional.DB_to_amplitude",
            "submodule": "functional",
            "params": [
                {"name": "x", "kind": "POSITIONAL_ONLY"},
                {"name": "ref", "kind": "POSITIONAL_ONLY"},
                {"name": "power", "kind": "POSITIONAL_ONLY"},
            ],
            "docstring": "Turn a tensor from the decibel scale to power/amplitude scale.",
        },
        {
            "name": "griffinlim",
            "api_path": "torchaudio.functional.griffinlim",
            "submodule": "functional",
            "params": [
                {"name": "specgram", "kind": "POSITIONAL_ONLY"},
                {"name": "window", "kind": "POSITIONAL_ONLY"},
                {"name": "n_fft", "kind": "POSITIONAL_ONLY"},
                {"name": "hop_length", "kind": "POSITIONAL_ONLY"},
                {"name": "win_length", "kind": "POSITIONAL_ONLY"},
                {"name": "power", "kind": "POSITIONAL_ONLY"},
                {"name": "n_iter", "kind": "KEYWORD_ONLY", "default": "32"},
                {"name": "momentum", "kind": "KEYWORD_ONLY", "default": "0.99"},
            ],
            "docstring": "Compute waveform from a linear scale magnitude spectrogram using Griffin-Lim.",
        },
        {
            "name": "resample",
            "api_path": "torchaudio.functional.resample",
            "submodule": "functional",
            "params": [
                {"name": "waveform", "kind": "POSITIONAL_ONLY"},
                {"name": "orig_freq", "kind": "POSITIONAL_ONLY"},
                {"name": "new_freq", "kind": "POSITIONAL_ONLY"},
                {
                    "name": "lowpass_filter_width",
                    "kind": "KEYWORD_ONLY",
                    "default": "6",
                },
                {"name": "rolloff", "kind": "KEYWORD_ONLY", "default": "0.99"},
            ],
            "docstring": "Resamples the waveform at the new frequency using bandlimited sinc interpolation.",
        },
        {
            "name": "compute_deltas",
            "api_path": "torchaudio.functional.compute_deltas",
            "submodule": "functional",
            "params": [
                {"name": "specgram", "kind": "POSITIONAL_ONLY"},
                {"name": "win_length", "kind": "KEYWORD_ONLY", "default": "5"},
                {"name": "mode", "kind": "KEYWORD_ONLY", "default": "'replicate'"},
            ],
            "docstring": "Compute delta coefficients of a tensor, usually a spectrogram.",
        },
        # Audio Transforms (torchaudio.transforms)
        {
            "name": "MFCC",
            "api_path": "torchaudio.transforms.MFCC",
            "submodule": "transforms",
            "params": [
                {
                    "name": "sample_rate",
                    "kind": "POSITIONAL_OR_KEYWORD",
                    "default": "16000",
                },
                {"name": "n_mfcc", "kind": "POSITIONAL_OR_KEYWORD", "default": "40"},
                {"name": "dct_type", "kind": "POSITIONAL_OR_KEYWORD", "default": "2"},
                {"name": "norm", "kind": "POSITIONAL_OR_KEYWORD", "default": "'ortho'"},
                {
                    "name": "log_mels",
                    "kind": "POSITIONAL_OR_KEYWORD",
                    "default": "False",
                },
            ],
            "docstring": "Create Mel-frequency cepstrum coefficients from an audio signal.",
        },
        {
            "name": "MelSpectrogram",
            "api_path": "torchaudio.transforms.MelSpectrogram",
            "submodule": "transforms",
            "params": [
                {
                    "name": "sample_rate",
                    "kind": "POSITIONAL_OR_KEYWORD",
                    "default": "16000",
                },
                {"name": "n_fft", "kind": "POSITIONAL_OR_KEYWORD", "default": "400"},
                {
                    "name": "win_length",
                    "kind": "POSITIONAL_OR_KEYWORD",
                    "default": "None",
                },
                {
                    "name": "hop_length",
                    "kind": "POSITIONAL_OR_KEYWORD",
                    "default": "None",
                },
                {"name": "f_min", "kind": "POSITIONAL_OR_KEYWORD", "default": "0.0"},
                {"name": "f_max", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
                {"name": "n_mels", "kind": "POSITIONAL_OR_KEYWORD", "default": "128"},
            ],
            "docstring": "Create MelSpectrogram for a raw audio signal.",
        },
        {
            "name": "Spectrogram",
            "api_path": "torchaudio.transforms.Spectrogram",
            "submodule": "transforms",
            "params": [
                {"name": "n_fft", "kind": "POSITIONAL_OR_KEYWORD", "default": "400"},
                {
                    "name": "win_length",
                    "kind": "POSITIONAL_OR_KEYWORD",
                    "default": "None",
                },
                {
                    "name": "hop_length",
                    "kind": "POSITIONAL_OR_KEYWORD",
                    "default": "None",
                },
                {"name": "pad", "kind": "POSITIONAL_OR_KEYWORD", "default": "0"},
                {"name": "power", "kind": "POSITIONAL_OR_KEYWORD", "default": "2.0"},
            ],
            "docstring": "Create a spectrogram from a raw audio signal.",
        },
        {
            "name": "AmplitudeToDB",
            "api_path": "torchaudio.transforms.AmplitudeToDB",
            "submodule": "transforms",
            "params": [
                {"name": "top_db", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            ],
            "docstring": "Resample a waveform from one frequency to another.",
        },
    ]
)


def _get_torchaudio_submodule(submod: str) -> types.ModuleType | None:
    """Safely import a specific TorchAudio submodule.

    Args:
        submod: Name of the TorchAudio submodule (e.g. 'functional', 'transforms').

    Returns:
        The submodule object or None if not available.
    """
    try:
        return importlib.import_module(f"torchaudio.{submod}")
    except Exception:
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Collect TorchAudio feature extraction and transformation APIs.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRef objects representing TorchAudio operations.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.NEURAL_OPS,
        SemanticTier.LAYER,
        SemanticTier.UTIL,
    ):
        return []

    target_submodules = ["functional", "transforms"]
    loaded_submods: dict[str, types.ModuleType | object] = {}
    for sub in target_submodules:
        mod = _get_torchaudio_submodule(sub)
        if mod is not None:
            loaded_submods[sub] = mod

    from ml_ecosystem_snapshots.frameworks._base import (
        collect_from_submodules_or_canonical,
    )

    return collect_from_submodules_or_canonical(
        loaded_submods=loaded_submods,
        canonical_ops=CANONICAL_TORCHAUDIO_OPS,
        framework_tag="torchaudio",
        default_submodule="functional",
        default_doc_prefix="TorchAudio",
        inspector_fn=GhostInspector.inspect,
    )
