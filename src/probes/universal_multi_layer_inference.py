"""
Universal multi-layer inference engine that loads ALL probes across ALL layers
and outputs ranked predictions showing cognitive actions across the full layer hierarchy
"""

from .gpu_utils import configure_amd_gpu
configure_amd_gpu()

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys
from collections import defaultdict

# Add nnsight to path
NNSIGHT_PATH = Path(__file__).parent.parent.parent / "third_party" / "nnsight" / "src"
sys.path.insert(0, str(NNSIGHT_PATH))

from nnsight import LanguageModel
from transformers import AutoTokenizer

from .probe_models import load_probe
from .dataset_utils import get_idx_to_action_mapping


@dataclass
class UniversalPrediction:
    """Single prediction from any probe at any layer"""
    action_name: str
    action_idx: int
    layer: int
    confidence: float
    is_active: bool  # Whether confidence >= threshold
    probe_type: str = "cognitive"  # "cognitive" or "sentiment"


class UniversalMultiLayerInferenceEngine:
    """
    Inference engine that loads ALL probes across ALL layers and outputs
    comprehensive ranked predictions across the full layer hierarchy
    """

    def __init__(
        self,
        probes_base_dir: Path,
        model_name: str = "google/gemma-3-4b-it",
        device: str = None,
        layer_range: Tuple[int, int] = (21, 30),  # (start, end) inclusive
        sentiment_probes_dir: Optional[Path] = None,
        include_sentiment: bool = False
    ):
        """
        Initialize universal multi-layer inference engine

        Args:
            probes_base_dir: Base directory containing layer_XX subdirectories
            model_name: Name of the language model
            device: Device to run on (auto-detects if None)
            layer_range: (start, end) layer range to load (inclusive)
            sentiment_probes_dir: Base directory containing sentiment probes (optional)
            include_sentiment: Whether to include sentiment probes in inference
        """
        # Auto-detect device if not provided
        if device is None:
            from .gpu_utils import get_optimal_device
            device = get_optimal_device()

        self.probes_base_dir = Path(probes_base_dir)
        self.sentiment_probes_dir = Path(sentiment_probes_dir) if sentiment_probes_dir else None
        self.include_sentiment = include_sentiment
        self.model_name = model_name
        self.device = device
        self.layer_start, self.layer_end = layer_range
        self.layers = list(range(self.layer_start, self.layer_end + 1))

        print(f"Initializing UniversalMultiLayerInferenceEngine...")
        print(f"  Probes base dir: {probes_base_dir}")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
        print(f"  Layer range: {self.layer_start}-{self.layer_end} ({len(self.layers)} layers)")
        if include_sentiment:
            print(f"  Sentiment probes: {sentiment_probes_dir}")
        print()

        # Load index to action mapping
        self.idx_to_action = get_idx_to_action_mapping()
        self.action_to_idx = {action: idx for idx, action in self.idx_to_action.items()}

        # Load ALL probes from ALL layers
        print("Loading probes from all layers...")
        self.probes = self._load_all_probes()

        # Load sentiment probes if requested
        self.sentiment_probes = {}
        if include_sentiment and sentiment_probes_dir:
            print("Loading sentiment probes...")
            self.sentiment_probes = self._load_sentiment_probes()

        print(f"\n✓ Loaded {len(self.probes)} cognitive probes across {len(self.layers)} layers")
        print(f"  ({len(self.probes) // len(self.layers)} actions × {len(self.layers)} layers)")
        if include_sentiment:
            print(f"✓ Loaded {len(self.sentiment_probes)} sentiment probes")

        # Load model and tokenizer
        print(f"\nLoading language model: {model_name}...")

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

        print(f"\n✓ Initialization complete!\n")

    def _load_all_probes(self) -> Dict[Tuple[str, int], Dict]:
        """
        Load all probes from all layers

        Returns:
            Dictionary mapping (action_name, layer) -> probe_info
        """
        probes = {}
        total_loaded = 0
        failed = 0

        for layer_idx in self.layers:
            layer_dir = self.probes_base_dir / f"layer_{layer_idx}"

            if not layer_dir.exists():
                print(f"  Warning: Layer directory not found: {layer_dir}")
                continue

            # Load all probe files in this layer directory
            probe_files = sorted(layer_dir.glob("probe_*.pth"))

            for probe_path in probe_files:
                # Extract action name from filename: probe_action_name.pth
                action_name = probe_path.stem.replace("probe_", "")

                if action_name not in self.action_to_idx:
                    continue

                try:
                    probe, metadata = load_probe(probe_path, device=self.device)

                    probes[(action_name, layer_idx)] = {
                        'probe': probe,
                        'layer': layer_idx,
                        'action': action_name,
                        'metadata': metadata
                    }
                    total_loaded += 1

                except Exception as e:
                    print(f"  Warning: Failed to load {probe_path}: {e}")
                    failed += 1

            print(f"  Layer {layer_idx}: loaded {len([p for p in probes if p[1] == layer_idx])} probes")

        if failed > 0:
            print(f"\n  Warning: Failed to load {failed} probes")

        return probes

    def _load_sentiment_probes(self) -> Dict[int, Dict]:
        """
        Load sentiment regression probes from layers 1-11 (default range for sentiment)

        Returns:
            Dictionary mapping layer -> probe_info
        """
        probes = {}
        total_loaded = 0
        failed = 0

        # Sentiment probes: use layers 1-11 by default
        sentiment_layers = range(1, 12)

        for layer_idx in sentiment_layers:
            layer_dir = self.sentiment_probes_dir / f"layer_{layer_idx}"

            if not layer_dir.exists():
                continue

            # Look for sentiment_regression_probe.pth
            probe_path = layer_dir / "sentiment_regression_probe.pth"

            if not probe_path.exists():
                continue

            try:
                # Load the probe using the standard load_probe function
                probe, metadata = load_probe(probe_path, device=self.device)

                probes[layer_idx] = {
                    'probe': probe,
                    'layer': layer_idx,
                    'metadata': metadata
                }
                total_loaded += 1

            except Exception as e:
                print(f"  Warning: Failed to load sentiment probe {probe_path}: {e}")
                failed += 1

        if failed > 0:
            print(f"  Warning: Failed to load {failed} sentiment probes")

        return probes

    def extract_all_layer_activations(self, text: str) -> Dict[int, torch.Tensor]:
        """
        Extract activations from all layers in range

        Args:
            text: Input text

        Returns:
            Dictionary mapping layer_idx -> activations tensor (hidden_dim,)
        """
        # Append special message to create consistent extraction point
        augmented_text = f"{text}\n\nThe cognitive action being demonstrated here is"

        # Extract activations from all layers using nnsight
        saved_activations = {}

        with self.model.trace(augmented_text) as tracer:
            for layer_idx in self.layers:
                # Get hidden states from this layer
                hidden_states = self.model.model.layers[layer_idx].output[0]

                # Use last token representation and save
                saved_activations[layer_idx] = hidden_states[:, -1, :].save()

        # After trace exits, saved proxies become tensors - squeeze batch dimension
        return {layer_idx: act.squeeze(0) for layer_idx, act in saved_activations.items()}

    def predict_all(
        self,
        text: str,
        threshold: float = 0.1,
        top_k: Optional[int] = None
    ) -> List[UniversalPrediction]:
        """
        Run ALL probes across ALL layers and return flat ranked list

        Args:
            text: Input text to analyze
            threshold: Minimum confidence threshold
            top_k: Number of top predictions to return (None = all)

        Returns:
            List of UniversalPrediction objects, sorted by confidence (descending)
        """
        # Extract activations from all layers for cognitive probes (single forward pass)
        layer_activations = self.extract_all_layer_activations(text)

        # Run all cognitive probes
        predictions = []

        with torch.no_grad():
            for (action_name, layer_idx), probe_info in self.probes.items():
                probe = probe_info['probe']
                action_idx = self.action_to_idx[action_name]

                # Get activations for this layer
                activations = layer_activations[layer_idx]

                # Get prediction
                logits = probe(activations)
                confidence = torch.sigmoid(logits).item()

                prediction = UniversalPrediction(
                    action_name=action_name,
                    action_idx=action_idx,
                    layer=layer_idx,
                    confidence=confidence,
                    is_active=confidence >= threshold,
                    probe_type="cognitive"
                )
                predictions.append(prediction)

            # Run sentiment probes if enabled
            if self.include_sentiment:
                # Extract activations for sentiment probes with sentiment-specific prompt
                sentiment_augmented_text = f"{text}\n\nThe sentiment of this section is"

                sentiment_activations = {}
                with self.model.trace(sentiment_augmented_text) as tracer:
                    for layer_idx in self.sentiment_probes.keys():
                        hidden_states = self.model.model.layers[layer_idx].output[0]
                        sentiment_activations[layer_idx] = hidden_states[:, -1, :].save()

                # Squeeze batch dimension
                sentiment_activations = {layer_idx: act.squeeze(0) for layer_idx, act in sentiment_activations.items()}

                # Run each sentiment probe
                for layer_idx, probe_info in self.sentiment_probes.items():
                    if layer_idx not in sentiment_activations:
                        continue

                    probe = probe_info['probe']
                    activations = sentiment_activations[layer_idx]

                    # Get sentiment score (regression output)
                    score = probe(activations).item()

                    prediction = UniversalPrediction(
                        action_name="sentiment",
                        action_idx=-1,  # Special index for sentiment
                        layer=layer_idx,
                        confidence=score,  # Raw sentiment score
                        is_active=abs(score) >= threshold,  # Active if magnitude exceeds threshold
                        probe_type="sentiment"
                    )
                    predictions.append(prediction)

        # Sort by absolute confidence (to handle negative sentiment scores)
        predictions = sorted(predictions, key=lambda x: abs(x.confidence), reverse=True)

        # Apply top_k filter if specified
        if top_k is not None:
            predictions = predictions[:top_k]

        return predictions

    def predict_by_action(
        self,
        text: str,
        threshold: float = 0.1,
        aggregation: str = "max"  # "max", "mean", "all"
    ) -> Dict[str, Dict]:
        """
        Group predictions by action, showing confidence across layers

        Args:
            text: Input text to analyze
            threshold: Minimum confidence threshold
            aggregation: How to aggregate across layers ("max", "mean", "all")

        Returns:
            Dictionary mapping action_name -> {
                'confidences': {layer: confidence},
                'aggregate': aggregated confidence,
                'best_layer': layer with highest confidence
            }
        """
        all_preds = self.predict_all(text, threshold=0.0)  # Get all, filter later

        # Group by action
        action_groups = defaultdict(lambda: {'confidences': {}, 'layers': []})

        for pred in all_preds:
            action_groups[pred.action_name]['confidences'][pred.layer] = pred.confidence
            action_groups[pred.action_name]['layers'].append(pred.layer)

        # Compute aggregates
        result = {}
        for action_name, data in action_groups.items():
            confidences = list(data['confidences'].values())

            if aggregation == "max":
                aggregate = max(confidences)
            elif aggregation == "mean":
                aggregate = sum(confidences) / len(confidences)
            else:  # "all"
                aggregate = confidences

            best_layer = max(data['confidences'].items(), key=lambda x: x[1])[0]

            result[action_name] = {
                'confidences': data['confidences'],
                'aggregate': aggregate,
                'best_layer': best_layer,
                'is_active': aggregate >= threshold if aggregation != "all" else max(confidences) >= threshold
            }

        # Sort by aggregate confidence
        if aggregation != "all":
            result = dict(sorted(result.items(), key=lambda x: x[1]['aggregate'], reverse=True))

        return result

    def predict_by_layer(
        self,
        text: str,
        threshold: float = 0.1
    ) -> Dict[int, List[UniversalPrediction]]:
        """
        Group predictions by layer, showing which actions fire at each layer

        Args:
            text: Input text to analyze
            threshold: Minimum confidence threshold

        Returns:
            Dictionary mapping layer -> list of predictions for that layer
        """
        all_preds = self.predict_all(text, threshold=threshold)

        # Group by layer
        layer_groups = defaultdict(list)
        for pred in all_preds:
            layer_groups[pred.layer].append(pred)

        # Sort predictions within each layer by confidence
        for layer in layer_groups:
            layer_groups[layer] = sorted(layer_groups[layer], key=lambda x: x.confidence, reverse=True)

        return dict(sorted(layer_groups.items()))

    def compare_texts(
        self,
        text1: str,
        text2: str,
        top_k: int = 10
    ) -> Dict:
        """
        Compare cognitive actions across layers for two texts

        Args:
            text1: First text
            text2: Second text
            top_k: Number of top differences to show

        Returns:
            Comparison results
        """
        preds1 = self.predict_by_action(text1, threshold=0.0, aggregation="max")
        preds2 = self.predict_by_action(text2, threshold=0.0, aggregation="max")

        # Compute differences
        differences = []
        for action_name in self.idx_to_action.values():
            conf1 = preds1.get(action_name, {}).get('aggregate', 0.0)
            conf2 = preds2.get(action_name, {}).get('aggregate', 0.0)
            diff = conf2 - conf1

            differences.append({
                'action': action_name,
                'text1_confidence': conf1,
                'text2_confidence': conf2,
                'difference': diff,
                'text1_best_layer': preds1.get(action_name, {}).get('best_layer', None),
                'text2_best_layer': preds2.get(action_name, {}).get('best_layer', None)
            })

        # Sort by absolute difference
        differences = sorted(differences, key=lambda x: abs(x['difference']), reverse=True)

        return {
            'text1_top_actions': [(k, v['aggregate']) for k, v in sorted(preds1.items(), key=lambda x: x[1]['aggregate'], reverse=True)[:top_k]],
            'text2_top_actions': [(k, v['aggregate']) for k, v in sorted(preds2.items(), key=lambda x: x[1]['aggregate'], reverse=True)[:top_k]],
            'biggest_differences': differences[:top_k]
        }


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Run universal multi-layer probe inference")
    parser.add_argument(
        "--probes-dir",
        type=str,
        required=True,
        help="Base directory containing layer_XX subdirectories"
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
        default=20,
        help="Number of top predictions to show"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum confidence threshold"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "by-action", "by-layer"],
        default="all",
        help="Output mode"
    )

    args = parser.parse_args()

    # Initialize engine
    engine = UniversalMultiLayerInferenceEngine(
        probes_base_dir=args.probes_dir,
        model_name=args.model
    )

    # If text provided, analyze it
    if args.text:
        print(f"\nAnalyzing text: \"{args.text}\"\n")
        print("-" * 80)

        if args.mode == "all":
            # Flat ranked list across all layers
            predictions = engine.predict_all(
                args.text,
                threshold=args.threshold,
                top_k=args.top_k
            )

            print(f"\nTop {len(predictions)} predictions across all layers:\n")
            for i, pred in enumerate(predictions, 1):
                marker = "✓" if pred.is_active else " "
                print(f"  {marker} {i:2d}. {pred.action_name:30s} (L{pred.layer:2d})  {pred.confidence:.4f}")

        elif args.mode == "by-action":
            # Grouped by action
            action_preds = engine.predict_by_action(
                args.text,
                threshold=args.threshold,
                aggregation="max"
            )

            print(f"\nTop {args.top_k} actions (max confidence across layers):\n")
            for i, (action_name, data) in enumerate(list(action_preds.items())[:args.top_k], 1):
                marker = "✓" if data['is_active'] else " "
                best_layer = data['best_layer']
                aggregate = data['aggregate']

                # Show top 3 layers for this action
                layer_confs = sorted(data['confidences'].items(), key=lambda x: x[1], reverse=True)[:3]
                layer_str = ", ".join([f"L{l}:{c:.3f}" for l, c in layer_confs])

                print(f"  {marker} {i:2d}. {action_name:30s} max={aggregate:.4f} @ L{best_layer}")
                print(f"       Layers: {layer_str}")

        elif args.mode == "by-layer":
            # Grouped by layer
            layer_preds = engine.predict_by_layer(
                args.text,
                threshold=args.threshold
            )

            print(f"\nPredictions by layer (threshold={args.threshold}):\n")
            for layer, preds in layer_preds.items():
                print(f"\n  Layer {layer} ({len(preds)} active actions):")
                for i, pred in enumerate(preds[:5], 1):  # Show top 5 per layer
                    print(f"    {i}. {pred.action_name:30s} {pred.confidence:.4f}")

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

            predictions = engine.predict_all(
                text,
                threshold=args.threshold,
                top_k=args.top_k
            )

            print(f"\nTop {len(predictions)} predictions across all layers:")
            for i, pred in enumerate(predictions, 1):
                marker = "✓" if pred.is_active else " "
                print(f"  {marker} {i:2d}. {pred.action_name:30s} (L{pred.layer:2d})  {pred.confidence:.4f}")
            print()


if __name__ == "__main__":
    main()
