"""
Multi-probe inference engine that loads the best layer for each cognitive action
"""

from .gpu_utils import configure_amd_gpu
configure_amd_gpu()

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys

# Add nnsight to path
NNSIGHT_PATH = Path(__file__).parent.parent.parent / "third_party" / "nnsight" / "src"
sys.path.insert(0, str(NNSIGHT_PATH))

from nnsight import LanguageModel
from transformers import AutoTokenizer

from .best_probe_loader import load_all_best_probes, get_all_best_layers
from .dataset_utils import get_idx_to_action_mapping


@dataclass
class CognitiveActionPrediction:
    """Single prediction from a probe"""
    action_name: str
    action_idx: int
    confidence: float
    is_active: bool  # Whether confidence >= threshold
    layer: int  # Which layer this probe is from
    auc: float  # AUC performance metric for this probe


class BestMultiProbeInferenceEngine:
    """
    Inference engine that uses the best-performing layer for each cognitive action
    """

    def __init__(
        self,
        probes_base_dir: Path,
        model_name: str = "google/gemma-3-4b-it",
        device: str = None
    ):
        """
        Initialize multi-probe inference engine with best layers

        Args:
            probes_base_dir: Base directory containing probes_binary/ subdirectory
            model_name: Name of the language model
            device: Device to run on (auto-detects if None)
        """
        # Auto-detect device if not provided
        if device is None:
            from gpu_utils import get_optimal_device
            device = get_optimal_device()

        self.probes_base_dir = Path(probes_base_dir)
        self.model_name = model_name
        self.device = device

        print(f"Initializing BestMultiProbeInferenceEngine...")
        print(f"  Probes base dir: {probes_base_dir}")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
        print()

        # Load index to action mapping
        self.idx_to_action = get_idx_to_action_mapping()
        self.action_to_idx = {action: idx for idx, action in self.idx_to_action.items()}

        # Load all probes (best layer for each action)
        self.probes = load_all_best_probes(
            self.probes_base_dir,
            device=device,
            verbose=True
        )

        # Get all unique layers we need
        self.layers_needed = sorted(set(p['layer'] for p in self.probes.values()))
        print(f"\n✓ Will extract activations from {len(self.layers_needed)} layers: {self.layers_needed}")

        # Load model and tokenizer
        print(f"\nLoading language model: {model_name}...")

        # Load model with bf16/fp16 to save memory
        from transformers import AutoModelForCausalLM, AutoConfig
        config = AutoConfig.from_pretrained(model_name)

        # Check if this is a VLM (has vision_config)
        if hasattr(config, 'vision_config'):
            print("Detected vision-language model. Loading text-only (skipping vision tower)...")
            from transformers import Gemma3ForCausalLM
            base_model = Gemma3ForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = LanguageModel(base_model, tokenizer=self.tokenizer)
        else:
            # Regular causal LM
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = LanguageModel(base_model, tokenizer=self.tokenizer)

        print(f"\n✓ Initialized with {len(self.probes)} probes across {len(self.layers_needed)} layers\n")

    def extract_all_layer_activations(self, text: str) -> Dict[int, torch.Tensor]:
        """
        Extract activations from all needed layers

        Args:
            text: Input text

        Returns:
            Dictionary mapping layer_idx -> activations tensor (hidden_dim,)
        """
        # Append special message to create consistent extraction point
        augmented_text = f"{text}\n\nThe cognitive action being demonstrated here is"

        # Extract activations from all needed layers using nnsight
        # Pass text directly to trace() - nnsight handles tokenization
        saved_activations = {}

        with self.model.trace(augmented_text) as tracer:
            for layer_idx in self.layers_needed:
                # Get hidden states from this layer
                hidden_states = self.model.model.layers[layer_idx].output[0]

                # Use last token representation and save
                saved_activations[layer_idx] = hidden_states[:, -1, :].save()

        # After trace exits, saved proxies become tensors - squeeze batch dimension
        return {layer_idx: act.squeeze(0) for layer_idx, act in saved_activations.items()}

    def predict(
        self,
        text: str,
        top_k: int = 5,
        threshold: float = 0.1,
        return_all: bool = False
    ) -> List[CognitiveActionPrediction]:
        """
        Run all probes and return predictions

        Args:
            text: Input text to analyze
            top_k: Number of top predictions to return
            threshold: Minimum confidence threshold
            return_all: If True, return all predictions (not just top-k)

        Returns:
            List of CognitiveActionPrediction objects, sorted by confidence
        """
        # Extract activations from all needed layers (single forward pass)
        layer_activations = self.extract_all_layer_activations(text)

        # Run all probes
        predictions = []

        with torch.no_grad():
            for action_name, probe_info in self.probes.items():
                probe = probe_info['probe']
                layer_idx = probe_info['layer']
                action_idx = self.action_to_idx[action_name]

                # Get activations for this probe's layer
                activations = layer_activations[layer_idx]

                # Get prediction
                logits = probe(activations)
                confidence = torch.sigmoid(logits).item()

                prediction = CognitiveActionPrediction(
                    action_name=action_name,
                    action_idx=action_idx,
                    confidence=confidence,
                    is_active=confidence >= threshold,
                    layer=layer_idx,
                    auc=probe_info['auc']
                )
                predictions.append(prediction)

        # Sort by confidence (descending)
        predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)

        # Return top-k or all
        if return_all:
            return predictions
        else:
            # Return top-k that meet threshold
            top_predictions = [p for p in predictions[:top_k] if p.is_active]
            # If none meet threshold, return top-k anyway
            if len(top_predictions) == 0:
                top_predictions = predictions[:top_k]
            return top_predictions

    def predict_batch(
        self,
        texts: List[str],
        top_k: int = 5,
        threshold: float = 0.1
    ) -> List[List[CognitiveActionPrediction]]:
        """
        Run predictions on multiple texts

        Args:
            texts: List of input texts
            top_k: Number of top predictions per text
            threshold: Minimum confidence threshold

        Returns:
            List of prediction lists (one per text)
        """
        results = []
        for text in texts:
            preds = self.predict(text, top_k=top_k, threshold=threshold)
            results.append(preds)
        return results

    def get_active_actions(
        self,
        text: str,
        threshold: float = 0.5
    ) -> List[str]:
        """
        Get list of active cognitive actions (above threshold)

        Args:
            text: Input text
            threshold: Confidence threshold

        Returns:
            List of action names that are active
        """
        predictions = self.predict(text, top_k=len(self.probes), threshold=threshold, return_all=True)
        return [p.action_name for p in predictions if p.is_active]

    def compare_texts(
        self,
        text1: str,
        text2: str,
        top_k: int = 10
    ) -> Dict:
        """
        Compare cognitive actions in two texts

        Args:
            text1: First text
            text2: Second text
            top_k: Number of top actions to show

        Returns:
            Dictionary with comparison results
        """
        preds1 = self.predict(text1, top_k=len(self.probes), threshold=0.0, return_all=True)
        preds2 = self.predict(text2, top_k=len(self.probes), threshold=0.0, return_all=True)

        # Create confidence maps
        conf1 = {p.action_name: p.confidence for p in preds1}
        conf2 = {p.action_name: p.confidence for p in preds2}

        # Compute differences
        differences = []
        for action_name in self.idx_to_action.values():
            c1 = conf1.get(action_name, 0.0)
            c2 = conf2.get(action_name, 0.0)
            diff = c2 - c1
            differences.append({
                'action': action_name,
                'text1_confidence': c1,
                'text2_confidence': c2,
                'difference': diff
            })

        # Sort by absolute difference
        differences = sorted(differences, key=lambda x: abs(x['difference']), reverse=True)

        return {
            'text1_top_actions': [(p.action_name, p.confidence) for p in preds1[:top_k]],
            'text2_top_actions': [(p.action_name, p.confidence) for p in preds2[:top_k]],
            'biggest_differences': differences[:top_k]
        }

    def get_probe_info(self, action_name: str) -> Dict:
        """Get information about a specific probe"""
        if action_name not in self.probes:
            raise ValueError(f"No probe loaded for action '{action_name}'")

        info = self.probes[action_name]
        return {
            'action': action_name,
            'layer': info['layer'],
            'auc': info['auc'],
            'f1': info['f1'],
            'layer_sensitivity': info['layer_sensitivity']
        }


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Run best multi-probe inference")
    parser.add_argument(
        "--probes-dir",
        type=str,
        required=True,
        help="Base directory containing probes_binary/"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="Language model name"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to analyze"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum confidence threshold"
    )

    args = parser.parse_args()

    # Initialize engine
    engine = BestMultiProbeInferenceEngine(
        probes_base_dir=args.probes_dir,
        model_name=args.model
    )

    # If text provided, analyze it
    if args.text:
        print(f"\nAnalyzing text: \"{args.text}\"\n")
        print("-" * 70)

        predictions = engine.predict(
            args.text,
            top_k=args.top_k,
            threshold=args.threshold
        )

        print(f"\nDetected Cognitive Actions (top {args.top_k}):\n")
        for i, pred in enumerate(predictions, 1):
            marker = "✓" if pred.is_active else " "
            print(f"  {marker} {i}. {pred.action_name:30s} {pred.confidence:.10f}  "
                  f"(Layer {pred.layer}, AUC: {pred.auc:.3f})")

        print()

    else:
        # Interactive mode
        print("\nInteractive mode - enter text to analyze (or 'quit' to exit):\n")

        while True:
            text = input("Text: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                break

            if not text:
                continue

            predictions = engine.predict(
                text,
                top_k=args.top_k,
                threshold=args.threshold
            )

            print(f"\nTop {args.top_k} predictions:")
            for i, pred in enumerate(predictions, 1):
                marker = "✓" if pred.is_active else " "
                print(f"  {marker} {i}. {pred.action_name:30s} {pred.confidence:.10f}  "
                      f"(L{pred.layer})")
            print()


if __name__ == "__main__":
    main()
