#!/usr/bin/env python3
"""
Train Binary Probes Across All Layers (4-28) - Parallel Training
Assumes activations are already captured locally
Replicates Colab notebook workflow for local execution

This script:
1. Trains 45 binary probes per layer using PARALLEL training
2. Compares performance across layers
3. Generates per-action layer analysis (finds best layer for each cognitive action)
4. Creates visualizations and summary

Key features:
- Parallel training with configurable workers (default: 45 workers = 45x speedup!)
- Pin activations to GPU memory for faster training
- Per-action layer analysis (which layer is best for each cognitive action)
- Comprehensive visualizations
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np


# Color codes for terminal output
class Colors:
    BLUE = '\033[1;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'


def print_section(title: str, color: str = Colors.YELLOW):
    """Print a formatted section header"""
    print(f"\n{color}{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}{Colors.NC}\n")


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and handle errors"""
    print(f"{Colors.BLUE}Running: {description}{Colors.NC}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{Colors.RED}✗ Failed: {description}{Colors.NC}")
        print(f"Error: {e}")
        return False


def train_layer_probes_parallel(
    layer_idx: int,
    activation_file: Path,
    output_dir: Path,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Train binary probes for a single layer using PARALLEL training"""

    # Build command for PARALLEL training
    cmd = [
        'python', 'train_binary_probes_parallel.py',
        '--activations', str(activation_file),
        '--output-dir', str(output_dir),
        '--model-type', config['probe_type'],
        '--batch-size', str(config['batch_size']),
        '--epochs', str(config['epochs']),
        '--lr', str(config['learning_rate']),
        '--weight-decay', str(config['weight_decay']),
        '--early-stopping-patience', str(config['early_stopping_patience']),
        '--device', config['device'],
        '--num-workers', str(config['num_workers'])  # PARALLEL TRAINING!
    ]

    if not config['use_scheduler']:
        cmd.append('--no-scheduler')

    if config['pin_activations_to_gpu']:
        cmd.append('--pin-activations-to-gpu')
    else:
        cmd.append('--no-pin-activations')

    start_time = time.time()
    success = run_command(cmd, f"Training Layer {layer_idx} (🚀 {config['num_workers']} workers)")
    elapsed = time.time() - start_time

    # Load metrics if successful
    if success:
        metrics_file = output_dir / "aggregate_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            return {
                'layer': layer_idx,
                'avg_auc': metrics['average_auc_roc'],
                'avg_f1': metrics['average_f1'],
                'avg_accuracy': metrics['average_accuracy'],
                'time_minutes': elapsed / 60,
                'num_workers': config['num_workers'],
                'success': True
            }

    return {
        'layer': layer_idx,
        'success': False,
        'time_minutes': elapsed / 60
    }


def train_all_layers(
    layers: List[int],
    activations_dir: Path,
    probes_base_dir: Path,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Train binary probes for all layers using PARALLEL training"""

    print_section("🚀 PARALLEL TRAINING OF BINARY PROBES FOR ALL LAYERS", Colors.YELLOW)

    print(f"Layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Probes per layer: 45")
    print(f"Total probes: {len(layers) * 45}")
    print(f"Probe type: {config['probe_type']}")
    print(f"Epochs per probe: {config['epochs']} (max, with early stopping)")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Early stopping patience: {config['early_stopping_patience']}")
    print()
    print(f"{Colors.YELLOW}🚀 PARALLEL TRAINING SETTINGS:{Colors.NC}")
    print(f"  Workers: {config['num_workers']} (train {config['num_workers']} probes simultaneously)")
    print(f"  Pin to GPU: {config['pin_activations_to_gpu']}")
    print(f"  Expected speedup: ~{config['num_workers']}x faster than sequential!")
    print()
    print(f"{Colors.YELLOW}⏰ Estimated time: {len(layers) * 2}-{len(layers) * 5} minutes total{Colors.NC}")
    print(f"{Colors.YELLOW}   (vs {len(layers) * 20}-{len(layers) * 30} minutes sequential)  {Colors.NC}")
    print()

    layer_results = []
    overall_start = time.time()

    for i, layer_idx in enumerate(layers, 1):
        print(f"\n{Colors.BOLD}{'=' * 70}")
        print(f"Training Layer {layer_idx} ({i}/{len(layers)})")
        print(f"🚀 Using {config['num_workers']} parallel workers")
        print(f"{'=' * 70}{Colors.NC}")

        # Check if activations exist
        activation_file = activations_dir / f"layer_{layer_idx}_activations.h5"
        if not activation_file.exists():
            print(f"{Colors.RED}⚠️  Activation file not found: {activation_file}{Colors.NC}")
            print(f"   Skipping layer {layer_idx}")
            continue

        # Set output directory for this layer
        output_dir = probes_base_dir / f"layer_{layer_idx}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train probes for this layer (PARALLEL!)
        result = train_layer_probes_parallel(
            layer_idx=layer_idx,
            activation_file=activation_file,
            output_dir=output_dir,
            config=config
        )

        layer_results.append(result)

        if result['success']:
            print(f"\n{Colors.GREEN}✓ Layer {layer_idx} complete in {result['time_minutes']:.1f} minutes{Colors.NC}")
            print(f"   🚀 {config['num_workers']}x speedup from parallel training!")
            print(f"   Avg AUC: {result['avg_auc']:.4f}, Avg F1: {result['avg_f1']:.4f}")
        else:
            print(f"\n{Colors.RED}✗ Layer {layer_idx} training failed{Colors.NC}")

        # Print progress
        completed = i
        remaining = len(layers) - i
        avg_time = (time.time() - overall_start) / 60 / completed
        est_remaining = remaining * avg_time

        print(f"\n{Colors.BLUE}Progress: {completed}/{len(layers)} layers complete{Colors.NC}")
        if remaining > 0:
            print(f"   Estimated time remaining: {est_remaining:.0f} minutes")

    overall_elapsed = time.time() - overall_start

    print(f"\n{Colors.GREEN}{'=' * 70}")
    print(f"✓ ALL LAYERS COMPLETE!")
    print(f"{'=' * 70}{Colors.NC}")
    print(f"Total time: {overall_elapsed/3600:.2f} hours ({overall_elapsed/60:.1f} minutes)")

    successful_layers = [r for r in layer_results if r['success']]
    print(f"Trained {len(successful_layers) * 45} probes across {len(successful_layers)} layers")
    print(f"🚀 Average speedup: {config['num_workers']}x faster than sequential!")

    return layer_results


def analyze_per_action_layers(
    layers: List[int],
    probes_base_dir: Path
) -> Dict[str, Any]:
    """
    Analyze which layer performs best for EACH cognitive action
    This is a key analysis from the Colab notebook!
    """

    print_section("PER-ACTION LAYER ANALYSIS", Colors.BLUE)
    print("Analyzing which layer is best for each cognitive action...\n")

    # Collect per-action metrics across all layers
    action_layer_performance = defaultdict(dict)  # {action_name: {layer: {'auc': x, 'f1': y}}}

    for layer_idx in layers:
        metrics_file = probes_base_dir / f"layer_{layer_idx}" / "aggregate_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Extract per-action metrics
            for action_metrics in metrics['per_action_metrics']:
                action_name = action_metrics['action']
                auc = action_metrics['auc_roc']
                f1 = action_metrics['f1']

                action_layer_performance[action_name][layer_idx] = {
                    'auc': auc,
                    'f1': f1
                }

    if not action_layer_performance:
        print(f"{Colors.RED}No per-action metrics found{Colors.NC}")
        return {}

    # Find best layer for each action
    action_best_layers = []
    for action_name, layer_perfs in sorted(action_layer_performance.items()):
        # Find layer with highest AUC
        best_layer = max(layer_perfs.items(), key=lambda x: x[1]['auc'])
        layer_idx, perf = best_layer

        # Get performance range
        auc_scores = [p['auc'] for p in layer_perfs.values()]
        auc_range = max(auc_scores) - min(auc_scores)

        action_best_layers.append({
            'action': action_name,
            'best_layer': layer_idx,
            'best_auc': perf['auc'],
            'best_f1': perf['f1'],
            'auc_range': auc_range,
            'worst_auc': min(auc_scores),
            'layer_sensitivity': auc_range  # How much performance varies by layer
        })

    print(f"✅ Analyzed {len(action_best_layers)} cognitive actions across {len(layers)} layers\n")

    # Show distribution of best layers
    print("=" * 70)
    print("BEST LAYER DISTRIBUTION")
    print("=" * 70)
    from collections import Counter
    layer_counts = Counter([a['best_layer'] for a in action_best_layers])
    print("\nHow many actions are best detected at each layer:\n")
    for layer in sorted(layer_counts.keys()):
        count = layer_counts[layer]
        bar = "█" * min(count, 50)  # Cap at 50 for display
        print(f"  Layer {layer:2d}: {count:2d} actions {bar}")

    # Most common best layers
    most_common = layer_counts.most_common(5)
    print(f"\n🏆 Most effective layers:")
    for layer, count in most_common:
        pct = (count / len(action_best_layers)) * 100
        print(f"   Layer {layer}: {count} actions ({pct:.1f}%)")

    # Show layer-sensitive actions
    print("\n" + "=" * 70)
    print("LAYER-SENSITIVE ACTIONS (Layer choice matters most)")
    print("=" * 70)
    sorted_by_sensitivity = sorted(action_best_layers, key=lambda x: x['layer_sensitivity'], reverse=True)
    print(f"\n{'Action':<30} {'Best Layer':<12} {'Best AUC':<10} {'AUC Range':<10}")
    print("-" * 70)
    for row in sorted_by_sensitivity[:10]:
        print(f"{row['action']:<30} Layer {row['best_layer']:<6} {row['best_auc']:.4f}     {row['auc_range']:.4f}")

    print("\n→ Large AUC range = layer choice is critical for this action")

    # Save detailed results
    results = {
        'summary': {
            'total_actions': len(action_best_layers),
            'total_layers_tested': len(layers),
            'most_common_best_layer': most_common[0][0] if most_common else None,
            'layer_distribution': dict(layer_counts)
        },
        'per_action_best_layers': action_best_layers,
        'action_layer_performance': {k: dict(v) for k, v in action_layer_performance.items()}
    }

    analysis_file = probes_base_dir / "per_action_layer_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Detailed results saved to: {analysis_file}")

    return results


def generate_summary(
    layer_results: List[Dict[str, Any]],
    output_dir: Path,
    config: Dict[str, Any]
):
    """Generate training summary and save to file"""

    successful_results = [r for r in layer_results if r['success']]

    if not successful_results:
        print(f"{Colors.RED}No successful layer trainings to summarize{Colors.NC}")
        return

    total_time = sum(r['time_minutes'] for r in layer_results) / 60  # hours

    summary = {
        'total_layers': len(successful_results),
        'total_probes': len(successful_results) * 45,
        'total_time_hours': total_time,
        'parallel_training': True,
        'num_workers': config['num_workers'],
        'layer_results': layer_results,
        'config': {
            'batch_size': config['batch_size'],
            'epochs': config['epochs'],
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'early_stopping_patience': config['early_stopping_patience'],
            'use_scheduler': config['use_scheduler'],
            'num_workers': config['num_workers'],
            'pin_activations_to_gpu': config['pin_activations_to_gpu']
        },
        'statistics': {
            'avg_auc': sum(r['avg_auc'] for r in successful_results) / len(successful_results),
            'best_auc': max(r['avg_auc'] for r in successful_results),
            'worst_auc': min(r['avg_auc'] for r in successful_results),
            'best_layer': max(successful_results, key=lambda x: x['avg_auc'])['layer'],
        }
    }

    # Save summary
    summary_file = output_dir / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{Colors.GREEN}✓ Summary saved to: {summary_file}{Colors.NC}")

    # Print summary
    print_section("TRAINING SUMMARY", Colors.GREEN)
    print(f"Total layers trained: {summary['total_layers']}")
    print(f"Total probes: {summary['total_probes']}")
    print(f"Total time: {summary['total_time_hours']:.2f} hours")
    print(f"Parallel workers: {summary['num_workers']}")
    print()
    print("Performance Statistics:")
    print(f"  Average AUC-ROC: {summary['statistics']['avg_auc']:.4f}")
    print(f"  Best AUC-ROC: {summary['statistics']['best_auc']:.4f} (Layer {summary['statistics']['best_layer']})")
    print(f"  Worst AUC-ROC: {summary['statistics']['worst_auc']:.4f}")
    print()

    # Layer-by-layer results
    print("Layer-by-Layer Results:")
    print(f"{'Layer':<8} {'Avg AUC':<10} {'Avg F1':<10} {'Avg Acc':<10} {'Time (min)':<12}")
    print("-" * 60)
    for result in successful_results:
        print(f"{result['layer']:<8} {result['avg_auc']:<10.4f} {result['avg_f1']:<10.4f} "
              f"{result['avg_accuracy']:<10.4f} {result['time_minutes']:<12.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train binary probes across all layers with PARALLEL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all layers with parallel training (45 workers, like Colab)
  python train_all_layers.py

  # Quick test with fewer layers and workers
  python train_all_layers.py --layer-start 20 --layer-end 22 --epochs 10 --num-workers 8

  # Custom configuration
  python train_all_layers.py --batch-size 64 --epochs 30 --lr 0.001 --num-workers 16

  # Conservative settings for lower VRAM
  python train_all_layers.py --num-workers 4 --batch-size 16 --no-pin-activations
        """
    )

    # Layer configuration
    parser.add_argument(
        "--layer-start",
        type=int,
        default=4,
        help="Start layer (default: 4)"
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=28,
        help="End layer inclusive (default: 28)"
    )

    # Input/Output
    parser.add_argument(
        "--activations-dir",
        type=str,
        default="../../data/activations",
        help="Directory with activation files (default: ../../data/activations)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../data/probes_binary",
        help="Output directory for probes (default: ../../data/probes_binary)"
    )

    # Training configuration
    parser.add_argument(
        "--probe-type",
        type=str,
        choices=["linear", "multihead"],
        default="linear",
        help="Type of probe (default: linear)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32, optimized for parallel training)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Max training epochs with early stopping (default: 50)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="Learning rate (default: 0.0005)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="Weight decay for regularization (default: 0.001)"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable learning rate scheduler"
    )

    # Parallel training configuration
    parser.add_argument(
        "--num-workers",
        type=int,
        default=45,
        help="Number of parallel workers (default: 45, trains all probes simultaneously!)"
    )
    parser.add_argument(
        "--pin-activations-to-gpu",
        action="store_true",
        default=True,
        help="Pin activations to GPU memory for faster training (default: True)"
    )
    parser.add_argument(
        "--no-pin-activations",
        dest="pin_activations_to_gpu",
        action="store_false",
        help="Don't pin activations to GPU (use if limited VRAM)"
    )

    # System configuration
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, or cpu (default: auto)"
    )

    args = parser.parse_args()

    # Handle auto device
    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert to Path objects
    activations_dir = Path(args.activations_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate layer list
    layers = list(range(args.layer_start, args.layer_end + 1))

    # Configuration
    config = {
        'probe_type': args.probe_type,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'use_scheduler': not args.no_scheduler,
        'device': args.device,
        'num_workers': args.num_workers,
        'pin_activations_to_gpu': args.pin_activations_to_gpu
    }

    # Print banner
    print_section("🚀 PARALLEL TRAINING: BINARY PROBES ACROSS ALL LAYERS", Colors.BLUE)

    print("Configuration:")
    print(f"  Activations: {activations_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Layers:      {args.layer_start}-{args.layer_end} ({len(layers)} layers)")
    print(f"  Total probes: {len(layers) * 45}")
    print(f"  Probe type:  {config['probe_type']}")
    print(f"  Epochs:      {config['epochs']} (with early stopping)")
    print(f"  Batch size:  {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Device:      {config['device']}")
    print()
    print(f"{Colors.YELLOW}🚀 Parallel Training:{Colors.NC}")
    print(f"  Workers: {config['num_workers']} (train {config['num_workers']} probes simultaneously)")
    print(f"  Pin to GPU: {config['pin_activations_to_gpu']}")
    print(f"  Expected speedup: ~{config['num_workers']}x faster!")
    print()

    # Check for activation files
    activation_files = sorted(activations_dir.glob("layer_*_activations.h5"))
    available_layers = [
        int(f.stem.split('_')[1])
        for f in activation_files
        if f.stem.startswith('layer_')
    ]

    print(f"{Colors.GREEN}Found {len(activation_files)} activation files{Colors.NC}")
    if len(activation_files) > 0:
        print(f"Available layers: {min(available_layers)}-{max(available_layers)}")
    print()

    # Check if requested layers are available
    missing_layers = [l for l in layers if l not in available_layers]
    if missing_layers:
        print(f"{Colors.YELLOW}⚠️  Warning: Missing activations for layers: {missing_layers}{Colors.NC}")
        print(f"   Will skip these layers during training.")
        print()

    if not activation_files:
        print(f"{Colors.RED}✗ No activation files found in {activations_dir}{Colors.NC}")
        print(f"   Please run activation capture first.")
        sys.exit(1)

    pipeline_start = time.time()

    # Train All Layers with PARALLEL training
    layer_results = train_all_layers(
        layers=layers,
        activations_dir=activations_dir,
        probes_base_dir=output_dir,
        config=config
    )

    # Generate Summary
    generate_summary(
        layer_results=layer_results,
        output_dir=output_dir,
        config=config
    )

    # Per-Action Layer Analysis (like Colab notebook!)
    if len([r for r in layer_results if r['success']]) > 1:
        analyze_per_action_layers(
            layers=[r['layer'] for r in layer_results if r['success']],
            probes_base_dir=output_dir
        )

    # Final summary
    pipeline_elapsed = time.time() - pipeline_start

    print_section("TRAINING COMPLETE! 🎉", Colors.GREEN)
    print(f"Total pipeline time: {pipeline_elapsed/3600:.2f} hours ({pipeline_elapsed/60:.1f} minutes)")
    print()
    print("Next Steps:")
    print(f"  1. Review summary: cat {output_dir}/training_summary.json")
    print(f"  2. Review per-action analysis: cat {output_dir}/per_action_layer_analysis.json")
    print(f"  3. Test with multi-probe inference on best layer")
    print()
    print(f"{Colors.GREEN}All done! 🚀{Colors.NC}")


if __name__ == "__main__":
    main()
