"""
Visualize cognitive action probe performance across all layers
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def load_all_layer_metrics(base_dir='data/probes_binary'):
    """Load aggregate metrics for all layers"""
    metrics_by_layer = {}

    for layer_dir in sorted(Path(base_dir).glob('layer_*')):
        layer_num = int(layer_dir.name.split('_')[1])
        metrics_file = layer_dir / 'aggregate_metrics.json'

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics_by_layer[layer_num] = json.load(f)

    return metrics_by_layer

def create_performance_visualization(metrics_by_layer, output_path='data/cognitive_probe_performance_by_layer.png'):
    """Create comprehensive visualization of probe performance across layers"""

    # Extract data
    layers = sorted(metrics_by_layer.keys())
    avg_auc_roc = [metrics_by_layer[l]['average_auc_roc'] for l in layers]
    avg_f1 = [metrics_by_layer[l]['average_f1'] for l in layers]
    avg_accuracy = [metrics_by_layer[l]['average_accuracy'] for l in layers]
    avg_precision = [metrics_by_layer[l]['average_precision'] for l in layers]
    avg_recall = [metrics_by_layer[l]['average_recall'] for l in layers]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cognitive Action Probe Performance Across Model Layers',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: AUC-ROC across layers
    ax1 = axes[0, 0]
    ax1.plot(layers, avg_auc_roc, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax1.fill_between(layers, avg_auc_roc, alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average AUC-ROC', fontsize=12, fontweight='bold')
    ax1.set_title('Average AUC-ROC by Layer\n(Discrimination Ability)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(avg_auc_roc) - 0.02, max(avg_auc_roc) + 0.02])

    # Highlight best layer
    best_layer = layers[np.argmax(avg_auc_roc)]
    best_auc = max(avg_auc_roc)
    ax1.scatter([best_layer], [best_auc], s=200, c='red', marker='*',
                edgecolors='darkred', linewidths=2, zorder=5, label=f'Best: Layer {best_layer}')
    ax1.legend(fontsize=10)

    # Plot 2: F1 Score across layers
    ax2 = axes[0, 1]
    ax2.plot(layers, avg_f1, marker='s', linewidth=2, markersize=6, color='#A23B72')
    ax2.fill_between(layers, avg_f1, alpha=0.3, color='#A23B72')
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Average F1 Score by Layer\n(Precision-Recall Balance)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Highlight best layer for F1
    best_layer_f1 = layers[np.argmax(avg_f1)]
    best_f1 = max(avg_f1)
    ax2.scatter([best_layer_f1], [best_f1], s=200, c='red', marker='*',
                edgecolors='darkred', linewidths=2, zorder=5, label=f'Best: Layer {best_layer_f1}')
    ax2.legend(fontsize=10)

    # Plot 3: Precision and Recall
    ax3 = axes[1, 0]
    ax3.plot(layers, avg_precision, marker='o', linewidth=2, markersize=6,
             color='#F18F01', label='Precision')
    ax3.plot(layers, avg_recall, marker='s', linewidth=2, markersize=6,
             color='#C73E1D', label='Recall')
    ax3.fill_between(layers, avg_precision, alpha=0.2, color='#F18F01')
    ax3.fill_between(layers, avg_recall, alpha=0.2, color='#C73E1D')
    ax3.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Precision vs Recall by Layer', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11, loc='best')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Accuracy across layers
    ax4 = axes[1, 1]
    ax4.plot(layers, avg_accuracy, marker='D', linewidth=2, markersize=6, color='#06A77D')
    ax4.fill_between(layers, avg_accuracy, alpha=0.3, color='#06A77D')
    ax4.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Average Accuracy by Layer', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([min(avg_accuracy) - 0.005, 1.0])

    # Add summary text
    summary_text = f"""
    Best AUC-ROC: Layer {best_layer} ({best_auc:.4f})
    Best F1 Score: Layer {best_layer_f1} ({best_f1:.4f})
    Layers analyzed: {min(layers)}-{max(layers)}
    """
    fig.text(0.02, 0.02, summary_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='bottom', family='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()

    return best_layer, best_auc

def create_top_actions_heatmap(metrics_by_layer, top_n=10, output_path='data/top_actions_heatmap.png'):
    """Create heatmap showing top N actions' AUC-ROC across layers"""

    # Find top actions based on best layer performance
    action_best_scores = {}

    for layer_num, metrics in metrics_by_layer.items():
        for action_data in metrics['per_action_metrics']:
            action = action_data['action']
            auc = action_data['auc_roc']

            if action not in action_best_scores:
                action_best_scores[action] = auc
            else:
                action_best_scores[action] = max(action_best_scores[action], auc)

    # Get top N actions
    top_actions = sorted(action_best_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_action_names = [action for action, _ in top_actions]

    # Build matrix: rows = actions, columns = layers
    layers = sorted(metrics_by_layer.keys())
    matrix = []

    for action_name in top_action_names:
        row = []
        for layer_num in layers:
            # Find this action's AUC in this layer
            action_data = next((a for a in metrics_by_layer[layer_num]['per_action_metrics']
                               if a['action'] == action_name), None)
            if action_data:
                row.append(action_data['auc_roc'])
            else:
                row.append(0)
        matrix.append(row)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(18, 10))

    # Format action names for display
    formatted_names = [name.replace('_', ' ').title() for name in top_action_names]

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0.7, vmax=1.0)

    # Set ticks
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=10)
    ax.set_yticks(range(len(formatted_names)))
    ax.set_yticklabels(formatted_names, fontsize=11)

    # Labels
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cognitive Action', fontsize=13, fontweight='bold')
    ax.set_title(f'Top {top_n} Cognitive Actions: AUC-ROC Performance Across Layers',
                 fontsize=15, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('AUC-ROC Score', fontsize=12, fontweight='bold')

    # Add values in cells
    for i in range(len(formatted_names)):
        for j in range(len(layers)):
            value = matrix[i][j]
            if value > 0.85:
                text_color = 'white'
            else:
                text_color = 'black'
            text = ax.text(j, i, f'{value:.3f}', ha="center", va="center",
                          color=text_color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()

def create_category_performance(metrics_by_layer, output_path='data/category_performance.png'):
    """Create visualization showing performance by category across layers"""

    # Define categories
    categories = {
        'Metacognitive': ['reconsidering', 'updating_beliefs', 'suspending_judgment', 'meta_awareness',
                         'metacognitive_monitoring', 'metacognitive_regulation', 'self_questioning',
                         'cognition_awareness', 'perspective_taking', 'noticing', 'accepting',
                         'questioning', 'pattern_recognition'],
        'Analytical': ['understanding', 'applying', 'analyzing', 'evaluating', 'abstracting',
                      'concretizing', 'connecting', 'distinguishing', 'zooming_out', 'zooming_in',
                      'convergent_thinking', 'remembering'],
        'Creative': ['creating', 'divergent_thinking', 'hypothesis_generation',
                    'counterfactual_reasoning', 'analogical_thinking', 'reframing'],
        'Emotional': ['emotional_reappraisal', 'emotion_receiving', 'emotion_responding',
                     'emotion_valuing', 'emotion_organizing', 'emotion_characterizing',
                     'situation_selection', 'situation_modification', 'attentional_deployment',
                     'response_modulation', 'emotion_perception', 'emotion_facilitation',
                     'emotion_understanding', 'emotion_management']
    }

    layers = sorted(metrics_by_layer.keys())
    category_scores = {cat: [] for cat in categories.keys()}

    for layer_num in layers:
        metrics = metrics_by_layer[layer_num]

        for category, actions in categories.items():
            # Calculate average AUC-ROC for this category in this layer
            aucs = []
            for action_data in metrics['per_action_metrics']:
                if action_data['action'] in actions:
                    aucs.append(action_data['auc_roc'])

            category_scores[category].append(np.mean(aucs) if aucs else 0)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

    for (category, scores), color in zip(category_scores.items(), colors):
        ax.plot(layers, scores, marker='o', linewidth=2.5, markersize=7,
               label=category, color=color)
        ax.fill_between(layers, scores, alpha=0.15, color=color)

    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average AUC-ROC', fontsize=13, fontweight='bold')
    ax.set_title('Cognitive Action Performance by Category Across Layers',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved category performance to {output_path}")
    plt.close()

def main():
    """Main function to generate all visualizations"""
    print("Loading metrics from all layers...")
    metrics_by_layer = load_all_layer_metrics()

    if not metrics_by_layer:
        print("Error: No metrics found!")
        return

    print(f"Found metrics for {len(metrics_by_layer)} layers")

    print("\nGenerating performance visualization...")
    best_layer, best_auc = create_performance_visualization(metrics_by_layer)

    print(f"\nBest performing layer: {best_layer} (AUC-ROC: {best_auc:.4f})")

    print("\nGenerating top actions heatmap...")
    create_top_actions_heatmap(metrics_by_layer, top_n=15)

    print("\nGenerating category performance visualization...")
    create_category_performance(metrics_by_layer)

    print("\n✓ All visualizations completed!")
    print("\nGenerated files:")
    print("  - data/cognitive_probe_performance_by_layer.png")
    print("  - data/top_actions_heatmap.png")
    print("  - data/category_performance.png")

if __name__ == '__main__':
    main()