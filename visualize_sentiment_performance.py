"""
Visualize sentiment probe performance across all layers
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")

def load_sentiment_metrics(base_dir='data/sentiment'):
    """Load sentiment metrics for all layers"""
    metrics_by_layer = {}

    for layer_dir in sorted(Path(base_dir).glob('layer_*')):
        layer_num = int(layer_dir.name.split('_')[1])
        metrics_file = layer_dir / 'metrics.json'

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics_by_layer[layer_num] = json.load(f)

    return metrics_by_layer

def create_sentiment_performance_overview(metrics_by_layer, output_path='data/sentiment_probe_performance.png'):
    """Create comprehensive visualization of sentiment probe performance"""

    layers = sorted(metrics_by_layer.keys())

    # Extract metrics
    mse_values = [metrics_by_layer[l]['mse'] for l in layers]
    mae_values = [metrics_by_layer[l]['mae'] for l in layers]
    r2_values = [metrics_by_layer[l]['r2'] for l in layers]
    accuracy_values = [metrics_by_layer[l]['accuracy'] for l in layers]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sentiment Probe Performance Across Model Layers',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: R² Score (primary metric)
    ax1 = axes[0, 0]
    ax1.plot(layers, r2_values, marker='o', linewidth=2.5, markersize=7, color='#2E86AB')
    ax1.fill_between(layers, r2_values, alpha=0.3, color='#2E86AB')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (R²=0)')
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('R² Score by Layer\n(Variance Explained)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Highlight best layer
    best_layer = layers[np.argmax(r2_values)]
    best_r2 = max(r2_values)
    ax1.scatter([best_layer], [best_r2], s=200, c='red', marker='*',
                edgecolors='darkred', linewidths=2, zorder=5, label=f'Best: Layer {best_layer}')

    # Highlight where performance becomes negative
    positive_r2 = [r2 if r2 > 0 else None for r2 in r2_values]
    negative_r2 = [r2 if r2 <= 0 else None for r2 in r2_values]

    if any(r2 <= 0 for r2 in r2_values):
        first_negative = next(i for i, r2 in enumerate(r2_values) if r2 <= 0)
        ax1.axvspan(layers[first_negative], layers[-1], alpha=0.2, color='red',
                   label='Negative R² (worse than baseline)')

    ax1.legend(fontsize=10)

    # Plot 2: MSE (Mean Squared Error)
    ax2 = axes[0, 1]
    ax2.plot(layers, mse_values, marker='s', linewidth=2.5, markersize=7, color='#A23B72')
    ax2.fill_between(layers, mse_values, alpha=0.3, color='#A23B72')
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Squared Error by Layer\n(Lower is Better)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Highlight best (lowest) MSE
    best_mse_layer = layers[np.argmin(mse_values)]
    best_mse = min(mse_values)
    ax2.scatter([best_mse_layer], [best_mse], s=200, c='green', marker='*',
                edgecolors='darkgreen', linewidths=2, zorder=5, label=f'Best: Layer {best_mse_layer}')
    ax2.legend(fontsize=10)

    # Plot 3: MAE (Mean Absolute Error)
    ax3 = axes[1, 0]
    ax3.plot(layers, mae_values, marker='D', linewidth=2.5, markersize=7, color='#F18F01')
    ax3.fill_between(layers, mae_values, alpha=0.3, color='#F18F01')
    ax3.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax3.set_title('Mean Absolute Error by Layer\n(Lower is Better)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Highlight best (lowest) MAE
    best_mae_layer = layers[np.argmin(mae_values)]
    best_mae = min(mae_values)
    ax3.scatter([best_mae_layer], [best_mae], s=200, c='green', marker='*',
                edgecolors='darkgreen', linewidths=2, zorder=5, label=f'Best: Layer {best_mae_layer}')
    ax3.legend(fontsize=10)

    # Plot 4: Accuracy
    ax4 = axes[1, 1]
    ax4.plot(layers, accuracy_values, marker='o', linewidth=2.5, markersize=7, color='#06A77D')
    ax4.fill_between(layers, accuracy_values, alpha=0.3, color='#06A77D')
    ax4.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Classification Accuracy by Layer\n(Directional Prediction)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.5, 1.0])

    # Highlight best accuracy
    best_acc_layer = layers[np.argmax(accuracy_values)]
    best_acc = max(accuracy_values)
    ax4.scatter([best_acc_layer], [best_acc], s=200, c='red', marker='*',
                edgecolors='darkred', linewidths=2, zorder=5, label=f'Best: Layer {best_acc_layer}')
    ax4.legend(fontsize=10)

    # Add summary text
    summary_text = f"""
    Best R² Score: Layer {best_layer} ({best_r2:.4f})
    Best MSE: Layer {best_mse_layer} ({best_mse:.4f})
    Best MAE: Layer {best_mae_layer} ({best_mae:.4f})
    Best Accuracy: Layer {best_acc_layer} ({best_acc:.4f})
    Layers analyzed: {min(layers)}-{max(layers)}
    """
    fig.text(0.02, 0.02, summary_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='bottom', family='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved sentiment performance visualization to {output_path}")
    plt.close()

def create_combined_metrics_plot(metrics_by_layer, output_path='data/sentiment_all_metrics_combined.png'):
    """Create a single plot showing all metrics together"""

    layers = sorted(metrics_by_layer.keys())

    # Extract metrics
    mse_values = np.array([metrics_by_layer[l]['mse'] for l in layers])
    mae_values = np.array([metrics_by_layer[l]['mae'] for l in layers])
    r2_values = np.array([metrics_by_layer[l]['r2'] for l in layers])
    accuracy_values = np.array([metrics_by_layer[l]['accuracy'] for l in layers])

    # Normalize metrics for comparison (0-1 scale, inverted for errors)
    # For R² and accuracy: higher is better (keep as is, scaled to 0-1)
    # For MSE and MAE: lower is better (invert and scale)

    r2_normalized = (r2_values - min(r2_values)) / (max(r2_values) - min(r2_values)) if max(r2_values) != min(r2_values) else r2_values
    accuracy_normalized = (accuracy_values - min(accuracy_values)) / (max(accuracy_values) - min(accuracy_values)) if max(accuracy_values) != min(accuracy_values) else accuracy_values

    # For errors: invert (1 - normalized) so lower error = higher score
    mse_normalized = 1 - ((mse_values - min(mse_values)) / (max(mse_values) - min(mse_values))) if max(mse_values) != min(mse_values) else mse_values
    mae_normalized = 1 - ((mae_values - min(mae_values)) / (max(mae_values) - min(mae_values))) if max(mae_values) != min(mae_values) else mae_values

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(layers, r2_normalized, marker='o', linewidth=2.5, markersize=7,
           label='R² Score (Normalized)', color='#2E86AB')
    ax.plot(layers, accuracy_normalized, marker='s', linewidth=2.5, markersize=7,
           label='Accuracy (Normalized)', color='#06A77D')
    ax.plot(layers, mse_normalized, marker='D', linewidth=2.5, markersize=7,
           label='MSE (Inverted & Normalized)', color='#A23B72')
    ax.plot(layers, mae_normalized, marker='^', linewidth=2.5, markersize=7,
           label='MAE (Inverted & Normalized)', color='#F18F01')

    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized Performance (0-1, Higher is Better)', fontsize=13, fontweight='bold')
    ax.set_title('All Sentiment Probe Metrics Across Layers (Normalized)',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Add vertical line at best R² layer
    best_layer = layers[np.argmax(r2_values)]
    ax.axvline(x=best_layer, color='red', linestyle='--', linewidth=2, alpha=0.5,
              label=f'Best R² Layer ({best_layer})')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined metrics plot to {output_path}")
    plt.close()

def create_good_vs_bad_layers(metrics_by_layer, output_path='data/sentiment_good_vs_bad_layers.png'):
    """Compare good performing layers vs poor performing layers"""

    layers = sorted(metrics_by_layer.keys())
    r2_values = [metrics_by_layer[l]['r2'] for l in layers]

    # Split into good (R² > 0.5) and bad (R² < 0) layers
    good_layers = [(l, metrics_by_layer[l]) for l in layers if metrics_by_layer[l]['r2'] > 0.5]
    bad_layers = [(l, metrics_by_layer[l]) for l in layers if metrics_by_layer[l]['r2'] < 0]
    moderate_layers = [(l, metrics_by_layer[l]) for l in layers
                      if 0 <= metrics_by_layer[l]['r2'] <= 0.5]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Sentiment Probe Performance: Layer Quality Comparison',
                 fontsize=16, fontweight='bold')

    # Plot 1: Good layers
    if good_layers:
        ax1 = axes[0]
        good_layer_nums = [l for l, _ in good_layers]
        good_r2 = [m['r2'] for _, m in good_layers]
        good_mae = [m['mae'] for _, m in good_layers]

        ax1_twin = ax1.twinx()

        p1 = ax1.bar([str(l) for l in good_layer_nums], good_r2, alpha=0.7, color='#2E86AB', label='R² Score')
        p2 = ax1_twin.plot([str(l) for l in good_layer_nums], good_mae, marker='o',
                          color='#F18F01', linewidth=2.5, markersize=8, label='MAE')

        ax1.set_xlabel('Layer', fontsize=11, fontweight='bold')
        ax1.set_ylabel('R² Score', fontsize=11, fontweight='bold', color='#2E86AB')
        ax1_twin.set_ylabel('MAE', fontsize=11, fontweight='bold', color='#F18F01')
        ax1.set_title(f'Good Layers (R² > 0.5)\nCount: {len(good_layers)}',
                     fontsize=12, fontweight='bold', color='green')
        ax1.tick_params(axis='y', labelcolor='#2E86AB')
        ax1_twin.tick_params(axis='y', labelcolor='#F18F01')
        ax1.grid(True, alpha=0.3)

        # Rotate x labels if many layers
        if len(good_layer_nums) > 10:
            ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Moderate layers
    if moderate_layers:
        ax2 = axes[1]
        mod_layer_nums = [l for l, _ in moderate_layers]
        mod_r2 = [m['r2'] for _, m in moderate_layers]
        mod_mae = [m['mae'] for _, m in moderate_layers]

        ax2_twin = ax2.twinx()

        p1 = ax2.bar([str(l) for l in mod_layer_nums], mod_r2, alpha=0.7, color='#F18F01', label='R² Score')
        p2 = ax2_twin.plot([str(l) for l in mod_layer_nums], mod_mae, marker='s',
                          color='#A23B72', linewidth=2.5, markersize=8, label='MAE')

        ax2.set_xlabel('Layer', fontsize=11, fontweight='bold')
        ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold', color='#F18F01')
        ax2_twin.set_ylabel('MAE', fontsize=11, fontweight='bold', color='#A23B72')
        ax2.set_title(f'Moderate Layers (0 ≤ R² ≤ 0.5)\nCount: {len(moderate_layers)}',
                     fontsize=12, fontweight='bold', color='orange')
        ax2.tick_params(axis='y', labelcolor='#F18F01')
        ax2_twin.tick_params(axis='y', labelcolor='#A23B72')
        ax2.grid(True, alpha=0.3)

    # Plot 3: Bad layers
    if bad_layers:
        ax3 = axes[2]
        bad_layer_nums = [l for l, _ in bad_layers]
        bad_r2 = [m['r2'] for _, m in bad_layers]
        bad_mae = [m['mae'] for _, m in bad_layers]

        ax3_twin = ax3.twinx()

        p1 = ax3.bar([str(l) for l in bad_layer_nums], bad_r2, alpha=0.7, color='#C73E1D', label='R² Score')
        p2 = ax3_twin.plot([str(l) for l in bad_layer_nums], bad_mae, marker='D',
                          color='#06A77D', linewidth=2.5, markersize=8, label='MAE')

        ax3.set_xlabel('Layer', fontsize=11, fontweight='bold')
        ax3.set_ylabel('R² Score', fontsize=11, fontweight='bold', color='#C73E1D')
        ax3_twin.set_ylabel('MAE', fontsize=11, fontweight='bold', color='#06A77D')
        ax3.set_title(f'Poor Layers (R² < 0)\nCount: {len(bad_layers)}',
                     fontsize=12, fontweight='bold', color='red')
        ax3.tick_params(axis='y', labelcolor='#C73E1D')
        ax3_twin.tick_params(axis='y', labelcolor='#06A77D')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved layer quality comparison to {output_path}")
    plt.close()

def create_training_curves(metrics_by_layer, sample_layers=[1, 7, 15, 20, 28],
                          output_path='data/sentiment_training_curves.png'):
    """Show training curves for selected layers"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Curves for Selected Sentiment Probe Layers',
                 fontsize=16, fontweight='bold')

    axes_flat = axes.flatten()

    for idx, layer in enumerate(sample_layers):
        if layer not in metrics_by_layer:
            continue

        metrics = metrics_by_layer[layer]
        if 'history' not in metrics:
            continue

        ax = axes_flat[idx]
        history = metrics['history']

        # Plot validation MSE
        val_mse = history.get('val_mse', [])
        epochs = list(range(1, len(val_mse) + 1))

        ax.plot(epochs, val_mse, linewidth=2, color='#2E86AB', label='Validation MSE')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=11, fontweight='bold')

        # Highlight best epoch
        if 'best_epoch' in history:
            best_epoch = history['best_epoch']
            best_val_mse = history.get('best_val_mse', val_mse[best_epoch] if best_epoch < len(val_mse) else None)
            if best_val_mse is not None:
                ax.scatter([best_epoch + 1], [best_val_mse], s=150, c='red', marker='*',
                          edgecolors='darkred', linewidths=2, zorder=5)
                ax.axvline(x=best_epoch + 1, color='red', linestyle='--', alpha=0.5)

        # Add final metrics
        final_r2 = metrics['r2']
        final_mae = metrics['mae']

        title = f"Layer {layer}\nR²={final_r2:.3f}, MAE={final_mae:.3f}"
        color = 'green' if final_r2 > 0.5 else ('orange' if final_r2 > 0 else 'red')
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # Hide extra subplot
    if len(sample_layers) < 6:
        axes_flat[-1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {output_path}")
    plt.close()

def main():
    """Main function to generate all visualizations"""
    print("Loading sentiment metrics from all layers...")
    metrics_by_layer = load_sentiment_metrics()

    if not metrics_by_layer:
        print("Error: No sentiment metrics found!")
        return

    print(f"Found metrics for {len(metrics_by_layer)} layers")

    print("\nGenerating performance overview...")
    create_sentiment_performance_overview(metrics_by_layer)

    print("\nGenerating combined metrics plot...")
    create_combined_metrics_plot(metrics_by_layer)

    print("\nGenerating layer quality comparison...")
    create_good_vs_bad_layers(metrics_by_layer)

    print("\nGenerating training curves for selected layers...")
    create_training_curves(metrics_by_layer)

    print("\n✓ All visualizations completed!")
    print("\nGenerated files:")
    print("  - data/sentiment_probe_performance.png")
    print("  - data/sentiment_all_metrics_combined.png")
    print("  - data/sentiment_good_vs_bad_layers.png")
    print("  - data/sentiment_training_curves.png")

if __name__ == '__main__':
    main()