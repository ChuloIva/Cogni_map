"""
Utility for loading the best-performing probe for each cognitive action based on layer analysis
"""

import json
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
from .probe_models import load_probe

def load_layer_analysis(probes_base_dir: Path) -> Dict:
    """
    Load the per-action layer analysis JSON

    Args:
        probes_base_dir: Base directory containing probes_binary/

    Returns:
        Dictionary with layer analysis data
    """
    analysis_path = probes_base_dir / "per_action_layer_analysis.json"

    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Layer analysis file not found at {analysis_path}. "
            "Make sure you've trained probes across multiple layers."
        )

    with open(analysis_path, 'r') as f:
        return json.load(f)


def get_best_layer_for_action(action_name: str, probes_base_dir: Path) -> Tuple[int, float, float]:
    """
    Get the best layer for a specific action

    Args:
        action_name: Name of the cognitive action
        probes_base_dir: Base directory containing per_action_layer_analysis.json

    Returns:
        Tuple of (best_layer_idx, best_auc, best_f1)
    """
    analysis = load_layer_analysis(probes_base_dir)

    # Find the action in the analysis
    for action_data in analysis['per_action_best_layers']:
        if action_data['action'] == action_name:
            return (
                action_data['best_layer'],
                action_data['best_auc'],
                action_data['best_f1']
            )

    raise ValueError(f"Action '{action_name}' not found in layer analysis")


def get_all_best_layers(probes_base_dir: Path) -> Dict[str, Dict]:
    """
    Get best layer info for all actions

    Args:
        probes_base_dir: Base directory containing per_action_layer_analysis.json

    Returns:
        Dictionary mapping action_name -> {layer, auc, f1, sensitivity}
    """
    analysis = load_layer_analysis(probes_base_dir)

    best_layers = {}
    for action_data in analysis['per_action_best_layers']:
        best_layers[action_data['action']] = {
            'layer': action_data['best_layer'],
            'auc': action_data['best_auc'],
            'f1': action_data['best_f1'],
            'layer_sensitivity': action_data['layer_sensitivity']
        }

    return best_layers


def load_best_probe_for_action(
    action_name: str,
    probes_base_dir: Path,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple:
    """
    Load the best-performing probe for a specific action

    Args:
        action_name: Name of the cognitive action
        probes_base_dir: Base directory containing probes_binary/
        device: Device to load probe on

    Returns:
        Tuple of (probe_model, metadata, layer_idx, performance_metrics)
    """
    # Get best layer
    layer_idx, best_auc, best_f1 = get_best_layer_for_action(action_name, probes_base_dir)

    # Load probe from that layer
    probe_path = probes_base_dir / f"layer_{layer_idx}" / f"probe_{action_name}.pth"

    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found at {probe_path}")

    probe, metadata = load_probe(probe_path, device=device)

    performance_metrics = {
        'layer': layer_idx,
        'auc': best_auc,
        'f1': best_f1
    }

    return probe, metadata, layer_idx, performance_metrics


def load_all_best_probes(
    probes_base_dir: Path,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Load the best-performing probe for each cognitive action

    Args:
        probes_base_dir: Base directory containing probes_binary/
        device: Device to load probes on
        verbose: Whether to print loading progress

    Returns:
        Dictionary mapping action_name -> {
            'probe': probe_model,
            'layer': best_layer_idx,
            'auc': best_auc,
            'f1': best_f1,
            'metadata': probe_metadata
        }
    """
    best_layers_info = get_all_best_layers(probes_base_dir)

    probes = {}

    if verbose:
        print(f"Loading best probes for {len(best_layers_info)} actions...")
        print(f"Device: {device}")
        print()

    for action_name, layer_info in best_layers_info.items():
        layer_idx = layer_info['layer']
        probe_path = probes_base_dir / f"layer_{layer_idx}" / f"probe_{action_name}.pth"

        if not probe_path.exists():
            if verbose:
                print(f"  ⚠ Warning: Probe not found for '{action_name}' at layer {layer_idx}")
            continue

        try:
            probe, metadata = load_probe(probe_path, device=device)
            probe.eval()

            probes[action_name] = {
                'probe': probe,
                'layer': layer_idx,
                'auc': layer_info['auc'],
                'f1': layer_info['f1'],
                'layer_sensitivity': layer_info['layer_sensitivity'],
                'metadata': metadata
            }

            if verbose and len(probes) % 10 == 0:
                print(f"  Loaded {len(probes)} probes...")

        except Exception as e:
            if verbose:
                print(f"  ⚠ Error loading probe for '{action_name}': {e}")

    if verbose:
        print(f"\n✓ Successfully loaded {len(probes)} probes")

        # Show layer distribution
        layer_counts = {}
        for action_data in probes.values():
            layer = action_data['layer']
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        print(f"\nLayer distribution:")
        for layer in sorted(layer_counts.keys()):
            count = layer_counts[layer]
            bar = "█" * (count // 2)
            print(f"  Layer {layer:2d}: {count:2d} probes {bar}")

    return probes


def print_performance_summary(probes_base_dir: Path, top_n: int = 10):
    """
    Print a summary of probe performance

    Args:
        probes_base_dir: Base directory containing per_action_layer_analysis.json
        top_n: Number of top/bottom performers to show
    """
    best_layers_info = get_all_best_layers(probes_base_dir)

    # Sort by AUC
    sorted_by_auc = sorted(
        best_layers_info.items(),
        key=lambda x: x[1]['auc'],
        reverse=True
    )

    print("="*70)
    print("PROBE PERFORMANCE SUMMARY")
    print("="*70)

    print(f"\nTop {top_n} Best Performing Probes (by AUC):")
    print("-"*70)
    for i, (action, info) in enumerate(sorted_by_auc[:top_n], 1):
        print(f"{i:2d}. {action:35s} Layer {info['layer']:2d}  "
              f"AUC: {info['auc']:.4f}  F1: {info['f1']:.4f}")

    print(f"\nBottom {top_n} Performers (by AUC):")
    print("-"*70)
    for i, (action, info) in enumerate(sorted_by_auc[-top_n:][::-1], 1):
        print(f"{i:2d}. {action:35s} Layer {info['layer']:2d}  "
              f"AUC: {info['auc']:.4f}  F1: {info['f1']:.4f}")

    # Sort by layer sensitivity
    sorted_by_sensitivity = sorted(
        best_layers_info.items(),
        key=lambda x: x[1]['layer_sensitivity'],
        reverse=True
    )

    print(f"\nMost Layer-Sensitive Actions (vary most across layers):")
    print("-"*70)
    for i, (action, info) in enumerate(sorted_by_sensitivity[:top_n], 1):
        print(f"{i:2d}. {action:35s} Sensitivity: {info['layer_sensitivity']:.4f}  "
              f"Best Layer: {info['layer']:2d}")

    print("="*70)


if __name__ == "__main__":
    # Example usage
    import sys

    # Set probes directory
    if len(sys.argv) > 1:
        probes_dir = Path(sys.argv[1])
    else:
        # Default to project structure
        project_root = Path(__file__).parent.parent.parent
        probes_dir = project_root / "data" / "probes_binary"

    print(f"Using probes directory: {probes_dir}\n")

    # Print performance summary
    try:
        print_performance_summary(probes_dir, top_n=10)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Test loading best probes
    print("\n\nTesting best probe loading...")
    print("="*70)

    try:
        probes = load_all_best_probes(probes_dir, verbose=True)
        print(f"\n✓ Successfully loaded {len(probes)} best probes")

        # Show example
        if probes:
            example_action = list(probes.keys())[0]
            example_info = probes[example_action]
            print(f"\nExample - '{example_action}':")
            print(f"  Layer: {example_info['layer']}")
            print(f"  AUC: {example_info['auc']:.4f}")
            print(f"  F1: {example_info['f1']:.4f}")
            print(f"  Probe type: {example_info['probe'].__class__.__name__}")

    except Exception as e:
        print(f"Error loading probes: {e}")
        import traceback
        traceback.print_exc()
