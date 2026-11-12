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

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalSubmoduleOpSpec,
    CanonicalSubmoduleOpTuple,
    build_canonical_submodule_ops,
    collect_from_submodules_or_canonical,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

_TORCHAUDIO_OP_DEFS: list[CanonicalSubmoduleOpTuple] = [
    (
        "melscale_fbanks",
        "torchaudio.functional.melscale_fbanks",
        "function",
        "functional",
        [
            ("n_freqs", "POSITIONAL_ONLY", None),
            ("f_min", "POSITIONAL_ONLY", None),
            ("f_max", "POSITIONAL_ONLY", None),
            ("n_mels", "POSITIONAL_ONLY", None),
            ("sample_rate", "POSITIONAL_ONLY", None),
            ("norm", "KEYWORD_ONLY", "None"),
            ("mel_scale", "KEYWORD_ONLY", "'htk'"),
        ],
        "Create a frequency conversion matrix from Hz to Mel.",
    ),
    (
        "spectrogram",
        "torchaudio.functional.spectrogram",
        "function",
        "functional",
        [
            ("waveform", "POSITIONAL_ONLY", None),
            ("pad", "POSITIONAL_ONLY", None),
            ("window", "POSITIONAL_ONLY", None),
            ("n_fft", "POSITIONAL_ONLY", None),
            ("hop_length", "POSITIONAL_ONLY", None),
            ("win_length", "POSITIONAL_ONLY", None),
            ("power", "POSITIONAL_ONLY", None),
            ("normalized", "POSITIONAL_ONLY", None),
        ],
        "Create a spectrogram from a raw audio signal.",
    ),
    (
        "amplitude_to_DB",
        "torchaudio.functional.amplitude_to_DB",
        "function",
        "functional",
        [
            ("x", "POSITIONAL_ONLY", None),
            ("multiplier", "POSITIONAL_ONLY", None),
            ("amin", "POSITIONAL_ONLY", None),
            ("db_multiplier", "POSITIONAL_ONLY", None),
            ("top_db", "KEYWORD_ONLY", "None"),
        ],
        "Turn a tensor from the power/amplitude scale to the decibel scale.",
    ),
    (
        "DB_to_amplitude",
        "torchaudio.functional.DB_to_amplitude",
        "function",
        "functional",
        [("x", "POSITIONAL_ONLY", None), ("ref", "POSITIONAL_ONLY", None), ("power", "POSITIONAL_ONLY", None)],
        "Turn a tensor from the decibel scale to power/amplitude scale.",
    ),
    (
        "griffinlim",
        "torchaudio.functional.griffinlim",
        "function",
        "functional",
        [
            ("specgram", "POSITIONAL_ONLY", None),
            ("window", "POSITIONAL_ONLY", None),
            ("n_fft", "POSITIONAL_ONLY", None),
            ("hop_length", "POSITIONAL_ONLY", None),
            ("win_length", "POSITIONAL_ONLY", None),
            ("power", "POSITIONAL_ONLY", None),
            ("n_iter", "KEYWORD_ONLY", "32"),
            ("momentum", "KEYWORD_ONLY", "0.99"),
        ],
        "Compute waveform from a linear scale magnitude spectrogram using Griffin-Lim.",
    ),
    (
        "resample",
        "torchaudio.functional.resample",
        "function",
        "functional",
        [
            ("waveform", "POSITIONAL_ONLY", None),
            ("orig_freq", "POSITIONAL_ONLY", None),
            ("new_freq", "POSITIONAL_ONLY", None),
            ("lowpass_filter_width", "KEYWORD_ONLY", "6"),
            ("rolloff", "KEYWORD_ONLY", "0.99"),
        ],
        "Resamples the waveform at the new frequency using bandlimited sinc interpolation.",
    ),
    (
        "compute_deltas",
        "torchaudio.functional.compute_deltas",
        "function",
        "functional",
        [
            ("specgram", "POSITIONAL_ONLY", None),
            ("win_length", "KEYWORD_ONLY", "5"),
            ("mode", "KEYWORD_ONLY", "'replicate'"),
        ],
        "Compute delta coefficients of a tensor, usually a spectrogram.",
    ),
    (
        "MFCC",
        "torchaudio.transforms.MFCC",
        "function",
        "transforms",
        [
            ("sample_rate", "POSITIONAL_OR_KEYWORD", "16000"),
            ("n_mfcc", "POSITIONAL_OR_KEYWORD", "40"),
            ("dct_type", "POSITIONAL_OR_KEYWORD", "2"),
            ("norm", "POSITIONAL_OR_KEYWORD", "'ortho'"),
            ("log_mels", "POSITIONAL_OR_KEYWORD", "False"),
        ],
        "Create Mel-frequency cepstrum coefficients from an audio signal.",
    ),
    (
        "MelSpectrogram",
        "torchaudio.transforms.MelSpectrogram",
        "function",
        "transforms",
        [
            ("sample_rate", "POSITIONAL_OR_KEYWORD", "16000"),
            ("n_fft", "POSITIONAL_OR_KEYWORD", "400"),
            ("win_length", "POSITIONAL_OR_KEYWORD", "None"),
            ("hop_length", "POSITIONAL_OR_KEYWORD", "None"),
            ("f_min", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("f_max", "POSITIONAL_OR_KEYWORD", "None"),
            ("n_mels", "POSITIONAL_OR_KEYWORD", "128"),
        ],
        "Create MelSpectrogram for a raw audio signal.",
    ),
    (
        "Spectrogram",
        "torchaudio.transforms.Spectrogram",
        "function",
        "transforms",
        [
            ("n_fft", "POSITIONAL_OR_KEYWORD", "400"),
            ("win_length", "POSITIONAL_OR_KEYWORD", "None"),
            ("hop_length", "POSITIONAL_OR_KEYWORD", "None"),
            ("pad", "POSITIONAL_OR_KEYWORD", "0"),
            ("power", "POSITIONAL_OR_KEYWORD", "2.0"),
        ],
        "Create a spectrogram from a raw audio signal.",
    ),
    (
        "AmplitudeToDB",
        "torchaudio.transforms.AmplitudeToDB",
        "function",
        "transforms",
        [("top_db", "POSITIONAL_OR_KEYWORD", "None")],
        "Resample a waveform from one frequency to another.",
    ),
]

CANONICAL_TORCHAUDIO_OPS: list[CanonicalSubmoduleOpSpec] = build_canonical_submodule_ops(_TORCHAUDIO_OP_DEFS)


def _get_torchaudio_submodule(submod: str) -> types.ModuleType | None:
    """Safely import a specific TorchAudio submodule.

    Args:
        submod: Name of the TorchAudio submodule (e.g. 'functional', 'transforms').

    Returns:
        The submodule object or None if not available.
    """
    try:
        return importlib.import_module(f"torchaudio.{submod}")
    except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError, TypeError, ValueError, OSError):
        return None


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
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
    loaded_submods: dict[str, types.ModuleType] = {}
    for sub in target_submodules:
        mod = _get_torchaudio_submodule(sub)
        if mod is not None:
            loaded_submods[sub] = mod

    return collect_from_submodules_or_canonical(
        loaded_submods=loaded_submods,
        canonical_ops=CANONICAL_TORCHAUDIO_OPS,
        framework_tag="torchaudio",
        default_submodule="functional",
        default_doc_prefix="TorchAudio",
        inspector_fn=GhostInspector.inspect,
    )
