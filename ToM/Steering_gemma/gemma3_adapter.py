"""
Gemma 3 4B Adapter for Improved Steering Vector Training

This adapter addresses the scaling issues found in the base repeng library
for Gemma 3 models by:

1. Normalizing steering vectors to unit norm after training
2. Measuring layer-specific activation magnitudes
3. Providing layer-adaptive coefficient scaling
4. Supporting per-layer coefficient adjustment

Based on best practices from the gemma_specifics.md document.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import sys
import warnings

# Add repeng to path if needed
sys.path.insert(0, '/Users/ivanculo/Desktop/Projects/Cogni_map/brije/ToM/repeng')

from repeng import ControlVector, ControlModel, DatasetEntry


class Gemma3ControlVector(ControlVector):
    """
    Enhanced ControlVector for Gemma 3 with proper normalization.

    Automatically normalizes vectors to unit norm and tracks layer-specific
    activation magnitudes for better coefficient scaling.
    """

    def __init__(
        self,
        model_type: str,
        directions: Dict[int, np.ndarray],
        activation_norms: Optional[Dict[int, float]] = None,
        normalize_vectors: bool = True
    ):
        """
        Args:
            model_type: Model architecture type
            directions: Dictionary mapping layer_id to direction vectors
            activation_norms: Optional dict of measured activation magnitudes per layer
            normalize_vectors: Whether to normalize direction vectors to unit norm
        """
        if normalize_vectors:
            # Normalize each direction vector to unit norm
            normalized_directions = {}
            for layer_id, direction in directions.items():
                norm = np.linalg.norm(direction)
                if norm > 1e-8:  # Avoid division by zero
                    normalized_directions[layer_id] = direction / norm
                else:
                    warnings.warn(f"Layer {layer_id} has near-zero direction vector, skipping normalization")
                    normalized_directions[layer_id] = direction
            directions = normalized_directions

        super().__init__(model_type=model_type, directions=directions)
        self.activation_norms = activation_norms or {}

    @classmethod
    def train(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: List[DatasetEntry],
        measure_activations: bool = True,
        normalize_vectors: bool = True,
        **kwargs
    ) -> "Gemma3ControlVector":
        """
        Train a control vector with Gemma 3 specific enhancements.

        Args:
            model: The model to train against
            tokenizer: Tokenizer for the model
            dataset: Training dataset of positive/negative pairs
            measure_activations: Whether to measure baseline activation norms
            normalize_vectors: Whether to normalize direction vectors
            **kwargs: Additional arguments passed to ControlVector.train

        Returns:
            Gemma3ControlVector with normalized directions and activation norms
        """
        # CRITICAL FIX: Pass the wrapped layer IDs to training
        # Otherwise it trains on ALL layers but only wraps a subset
        if isinstance(model, ControlModel) and 'hidden_layers' not in kwargs:
            kwargs['hidden_layers'] = model.layer_ids
            print(f"Training vector for layers: {model.layer_ids}")

        # Train using base method
        base_vector = ControlVector.train(model, tokenizer, dataset, **kwargs)

        # Measure activation norms if requested
        activation_norms = {}
        if measure_activations and isinstance(model, ControlModel):
            print("Measuring baseline activation magnitudes...")
            activation_norms = cls._measure_activation_norms(
                model, tokenizer, dataset[:min(10, len(dataset))]
            )
            print(f"Activation norms: {activation_norms}")

        return cls(
            model_type=base_vector.model_type,
            directions=base_vector.directions,
            activation_norms=activation_norms,
            normalize_vectors=normalize_vectors
        )

    @staticmethod
    def _measure_activation_norms(
        model: ControlModel,
        tokenizer: PreTrainedTokenizerBase,
        sample_data: List[DatasetEntry],
        num_samples: int = 10
    ) -> Dict[int, float]:
        """
        Measure the typical magnitude of activations at each layer.

        Returns a dictionary mapping layer_id to median activation magnitude.
        """
        from repeng.extract import batched_get_hiddens, model_layer_list

        # Get sample strings
        sample_strs = []
        for entry in sample_data[:num_samples]:
            sample_strs.extend([entry.positive, entry.negative])

        # Get layer IDs
        n_layers = len(model_layer_list(model))
        layer_ids = [i if i >= 0 else n_layers + i for i in model.layer_ids]

        # Get hidden states
        hiddens = batched_get_hiddens(
            model, tokenizer, sample_strs, layer_ids, batch_size=4
        )

        # Calculate median L2 norm for each layer
        norms = {}
        for layer_id, hidden_states in hiddens.items():
            # Calculate L2 norm across the feature dimension for each sample
            sample_norms = np.linalg.norm(hidden_states, axis=1)
            norms[layer_id] = float(np.median(sample_norms))

        return norms

    def get_scaled_coefficient(
        self,
        layer_id: int,
        base_coeff: float = 1.0,
        target_norm: float = 100.0
    ) -> float:
        """
        Get a scaled coefficient for a specific layer based on its activation magnitude.

        Args:
            layer_id: The layer to get coefficient for
            base_coeff: Base coefficient value
            target_norm: Target activation magnitude for scaling

        Returns:
            Scaled coefficient appropriate for the layer's activation magnitude
        """
        if layer_id not in self.activation_norms:
            return base_coeff

        # Scale inversely with activation magnitude
        # Higher activation magnitude = smaller coefficient needed
        layer_norm = self.activation_norms[layer_id]
        if layer_norm > 1e-8:
            scale = target_norm / layer_norm
            return base_coeff * scale
        return base_coeff


class Gemma3ControlModel(ControlModel):
    """
    Enhanced ControlModel for Gemma 3 with per-layer coefficient support.
    """

    def set_control_per_layer(
        self,
        control: Gemma3ControlVector,
        layer_coeffs: Optional[Dict[int, float]] = None,
        base_coeff: float = 1.0,
        use_adaptive_scaling: bool = True,
        **kwargs
    ) -> None:
        """
        Set control with per-layer coefficients.

        Args:
            control: The Gemma3ControlVector to apply
            layer_coeffs: Optional dict of per-layer coefficients
            base_coeff: Base coefficient if layer_coeffs not specified
            use_adaptive_scaling: Whether to use activation-based scaling
            **kwargs: Additional arguments (normalize, operator, etc.)
        """
        raw_control = {}

        for layer_id in self.layer_ids:
            # Determine coefficient for this layer
            if layer_coeffs and layer_id in layer_coeffs:
                coeff = layer_coeffs[layer_id]
            elif use_adaptive_scaling and hasattr(control, 'get_scaled_coefficient'):
                coeff = control.get_scaled_coefficient(layer_id, base_coeff)
            else:
                coeff = base_coeff

            # Apply coefficient to direction
            raw_control[layer_id] = torch.tensor(
                coeff * control.directions[layer_id]
            ).to(self.model.device, dtype=self.model.dtype)

        self.set_raw_control(raw_control, **kwargs)


def create_gemma3_model(
    model_name: str = "google/gemma-3-4b-it",
    layer_range: tuple = (-4, -20),  # (start, end) inclusive
    use_bfloat16: bool = True
) -> tuple[Gemma3ControlModel, PreTrainedTokenizerBase]:
    """
    Create a Gemma3ControlModel with proper configuration.

    Args:
        model_name: HuggingFace model name
        layer_range: Tuple of (start, end) layer indices (negative indexing)
        use_bfloat16: Whether to use bfloat16 (recommended for Gemma 3)

    Returns:
        Tuple of (control_model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from repeng.control import model_layer_list

    print(f"Loading {model_name}...")

    # Determine dtype
    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    print(f"Using dtype: {dtype}")

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0

    # CRITICAL FIX: Gemma3Config is missing num_hidden_layers attribute
    # repeng needs this for training, so we add it manually
    layers = model_layer_list(base_model)
    num_layers = len(layers)

    if not hasattr(base_model.config, 'num_hidden_layers'):
        print(f"Fixing missing num_hidden_layers attribute (setting to {num_layers})")
        base_model.config.num_hidden_layers = num_layers

    # Determine layer IDs
    start, end = layer_range
    layer_ids = list(range(start, end - 1, -1))  # -4 to -20 inclusive

    print(f"Model loaded on device: {base_model.device}")
    print(f"Total layers: {num_layers}")
    print(f"Wrapping layers: {layer_ids}")

    # Create control model
    model = Gemma3ControlModel(base_model, layer_ids)

    return model, tokenizer


def make_dataset_with_truncation(
    template: str,
    pos_personas: List[str],
    neg_personas: List[str],
    suffixes: List[str],
    truncate_suffixes: bool = True,
    min_tokens: int = 1,
    max_truncations: int = 5
) -> List[DatasetEntry]:
    """
    Create dataset with optional suffix truncation for better diversity.

    Args:
        template: String template with {persona} placeholder
        pos_personas: List of positive persona descriptions
        neg_personas: List of negative persona descriptions
        suffixes: List of text suffixes
        truncate_suffixes: Whether to create truncated versions
        min_tokens: Minimum tokens to keep when truncating
        max_truncations: Max truncation points per suffix

    Returns:
        List of DatasetEntry objects
    """
    from transformers import AutoTokenizer

    dataset = []

    # We'll use a simple tokenizer for truncation
    # In practice, you should use the actual model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

    for suffix in suffixes:
        if truncate_suffixes and len(suffix) > 0:
            # Tokenize suffix
            tokens = tokenizer.tokenize(suffix)

            # Create truncated versions
            num_truncations = min(max_truncations, len(tokens))
            for i in range(min_tokens, num_truncations + 1):
                truncated = tokenizer.convert_tokens_to_string(tokens[:i])

                for pos_persona, neg_persona in zip(pos_personas, neg_personas):
                    pos_text = template.format(persona=pos_persona) + " " + truncated
                    neg_text = template.format(persona=neg_persona) + " " + truncated

                    dataset.append(DatasetEntry(positive=pos_text, negative=neg_text))
        else:
            # Use full suffix without truncation
            for pos_persona, neg_persona in zip(pos_personas, neg_personas):
                pos_text = template.format(persona=pos_persona) + " " + suffix
                neg_text = template.format(persona=neg_persona) + " " + suffix

                dataset.append(DatasetEntry(positive=pos_text, negative=neg_text))

    return dataset


# Diagnostic utilities
def analyze_vector_properties(vector: Gemma3ControlVector) -> Dict:
    """Analyze properties of a trained vector."""
    analysis = {
        "num_layers": len(vector.directions),
        "layer_ids": list(vector.directions.keys()),
        "vector_norms": {},
        "activation_norms": vector.activation_norms,
    }

    for layer_id, direction in vector.directions.items():
        analysis["vector_norms"][layer_id] = float(np.linalg.norm(direction))

    return analysis


def print_vector_analysis(vector: Gemma3ControlVector):
    """Print a formatted analysis of vector properties."""
    analysis = analyze_vector_properties(vector)

    print("\n" + "="*80)
    print("VECTOR ANALYSIS")
    print("="*80)
    print(f"Number of layers: {analysis['num_layers']}")
    print(f"Layer IDs: {analysis['layer_ids']}")

    print("\nVector Norms (should be ~1.0 if normalized):")
    for layer_id in sorted(analysis["vector_norms"].keys()):
        norm = analysis["vector_norms"][layer_id]
        print(f"  Layer {layer_id:3d}: {norm:.6f}")

    if analysis["activation_norms"]:
        print("\nActivation Norms (baseline magnitude):")
        for layer_id in sorted(analysis["activation_norms"].keys()):
            norm = analysis["activation_norms"][layer_id]
            print(f"  Layer {layer_id:3d}: {norm:.2f}")

    print("="*80 + "\n")