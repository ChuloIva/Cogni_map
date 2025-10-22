"""
Cognitive Action Probe System for Gemma-3-4B

This package provides tools for:
- Capturing activations from Gemma-3-4B using nnsight
- Training binary probes for cognitive action detection
- Training sentiment regression probes
- Real-time inference with streaming predictions
- Interactive TUI for token-level visualization

Core Features:
1. Universal Probes - Detect 45 cognitive actions
2. Sentiment Probes - Continuous sentiment scoring
3. Interactive TUI - Token-by-token visualization
"""

from .dataset_utils import load_cognitive_actions_dataset, create_splits
from .probe_models import LinearProbe, MultiHeadProbe, BinaryLinearProbe, BinaryMultiHeadProbe
from .best_probe_loader import load_all_best_probes, get_best_layer_for_action
from .gpu_utils import get_optimal_device, configure_amd_gpu
from .best_multi_probe_inference import BestMultiProbeInferenceEngine
from .streaming_probe_inference import StreamingProbeInferenceEngine

__version__ = "0.1.0"
__all__ = [
    # Dataset utilities
    "load_cognitive_actions_dataset",
    "create_splits",

    # Probe models
    "LinearProbe",
    "MultiHeadProbe",
    "BinaryLinearProbe",
    "BinaryMultiHeadProbe",

    # Probe loading
    "load_all_best_probes",
    "get_best_layer_for_action",

    # GPU utilities
    "get_optimal_device",
    "configure_amd_gpu",

    # Inference engines
    "BestMultiProbeInferenceEngine",
    "StreamingProbeInferenceEngine",
]
