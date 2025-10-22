"""
Test script for universal multi-layer inference engine
Demonstrates different output modes and compares with single-layer approach
"""

from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_multi_layer_inference import UniversalMultiLayerInferenceEngine


def test_example_texts():
    """Test on various example texts"""

    # Initialize engine
    probes_dir = Path(__file__).parent.parent.parent / "data" / "probes_binary"

    if not probes_dir.exists():
        print(f"Error: Probes directory not found at {probes_dir}")
        print("Please provide the correct path to your probes directory")
        return

    print("="*80)
    print("UNIVERSAL MULTI-LAYER INFERENCE TEST")
    print("="*80)

    engine = UniversalMultiLayerInferenceEngine(
        probes_base_dir=probes_dir,
        model_name="google/gemma-3-4b-it"
    )

    # Test examples
    examples = [
        {
            "text": "I'm thinking about how to solve this complex problem step by step",
            "description": "Analytical reasoning"
        },
        {
            "text": "Looking back on my decision, I realize I should have considered other options",
            "description": "Reflection/metacognition"
        },
        {
            "text": "I feel deeply moved by their story and want to help them",
            "description": "Empathy/emotion"
        },
        {
            "text": "This is just like that other situation we encountered last year",
            "description": "Analogical thinking"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}: {example['description']}")
        print(f"{'='*80}")
        print(f"Text: \"{example['text']}\"\n")

        # Mode 1: Flat ranked list
        print("-" * 80)
        print("MODE 1: Top 15 predictions across ALL layers")
        print("-" * 80)

        preds = engine.predict_all(
            example['text'],
            threshold=0.1,
            top_k=15
        )

        for j, pred in enumerate(preds, 1):
            marker = "✓" if pred.is_active else " "
            print(f"  {marker} {j:2d}. {pred.action_name:30s} (Layer {pred.layer:2d})  {pred.confidence:.4f}")

        # Mode 2: By action (aggregated across layers)
        print("\n" + "-" * 80)
        print("MODE 2: Top 10 actions (max confidence across layers)")
        print("-" * 80)

        action_preds = engine.predict_by_action(
            example['text'],
            threshold=0.1,
            aggregation="max"
        )

        for j, (action_name, data) in enumerate(list(action_preds.items())[:10], 1):
            marker = "✓" if data['is_active'] else " "
            best_layer = data['best_layer']
            aggregate = data['aggregate']

            # Show top 3 layers for this action
            layer_confs = sorted(data['confidences'].items(), key=lambda x: x[1], reverse=True)[:3]
            layer_str = ", ".join([f"L{l}:{c:.3f}" for l, c in layer_confs])

            print(f"  {marker} {j:2d}. {action_name:30s} max={aggregate:.4f} @ Layer {best_layer}")
            print(f"       Active layers: {layer_str}")

        # Mode 3: By layer (which actions fire at each layer)
        print("\n" + "-" * 80)
        print("MODE 3: Predictions grouped by layer (top 3 actions per layer)")
        print("-" * 80)

        layer_preds = engine.predict_by_layer(
            example['text'],
            threshold=0.1
        )

        for layer, preds in layer_preds.items():
            if preds:  # Only show layers with active predictions
                print(f"\n  Layer {layer} ({len(preds)} active):")
                for k, pred in enumerate(preds[:3], 1):  # Top 3 per layer
                    print(f"    {k}. {pred.action_name:30s} {pred.confidence:.4f}")

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")


def compare_with_best_layer():
    """Compare universal multi-layer with best-layer approach"""

    print("\n" + "="*80)
    print("COMPARISON: Universal Multi-Layer vs Best-Layer Approach")
    print("="*80 + "\n")

    probes_dir = Path(__file__).parent.parent.parent / "data" / "probes_binary"

    # Initialize universal engine
    print("Initializing Universal Multi-Layer Engine...")
    universal_engine = UniversalMultiLayerInferenceEngine(
        probes_base_dir=probes_dir,
        model_name="google/gemma-3-4b-it"
    )

    # Try to import and initialize best-layer engine
    try:
        from best_multi_probe_inference import BestMultiProbeInferenceEngine

        print("\nInitializing Best-Layer Engine...")
        best_engine = BestMultiProbeInferenceEngine(
            probes_base_dir=probes_dir.parent,  # Goes to parent, expects probes_binary/ subdir
            model_name="google/gemma-3-4b-it"
        )

        test_text = "I'm analyzing this problem and thinking through different solutions"

        print("\n" + "-"*80)
        print("Test text:", test_text)
        print("-"*80)

        # Universal multi-layer predictions
        print("\nUNIVERSAL MULTI-LAYER (max confidence across layers):")
        universal_preds = universal_engine.predict_by_action(
            test_text,
            threshold=0.1,
            aggregation="max"
        )

        for i, (action, data) in enumerate(list(universal_preds.items())[:10], 1):
            if data['is_active']:
                print(f"  {i:2d}. {action:30s} {data['aggregate']:.4f} @ L{data['best_layer']}")

        # Best-layer predictions
        print("\nBEST-LAYER APPROACH:")
        best_preds = best_engine.predict(
            test_text,
            top_k=10,
            threshold=0.1
        )

        for i, pred in enumerate(best_preds, 1):
            print(f"  {i:2d}. {pred.action_name:30s} {pred.confidence:.4f} @ L{pred.layer}")

        print("\n" + "="*80 + "\n")

    except ImportError:
        print("Note: Could not import BestMultiProbeInferenceEngine for comparison")
        print("Skipping comparison test\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test universal multi-layer inference")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["examples", "compare", "both"],
        default="both",
        help="Test mode"
    )

    args = parser.parse_args()

    if args.mode in ["examples", "both"]:
        test_example_texts()

    if args.mode in ["compare", "both"]:
        compare_with_best_layer()
