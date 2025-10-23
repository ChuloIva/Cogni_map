"""
Visualize individual cognitive action performance across all layers
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")

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

def create_all_actions_grid(metrics_by_layer, output_path='data/all_cognitive_actions_performance.png'):
    """Create a grid of plots showing AUC-ROC for all 45 cognitive actions"""

    # Get all action names from first layer
    first_layer = min(metrics_by_layer.keys())
    all_actions = [action['action'] for action in metrics_by_layer[first_layer]['per_action_metrics']]
    all_actions = sorted(all_actions)

    layers = sorted(metrics_by_layer.keys())

    # Create a grid: 9 rows x 5 columns for 45 actions
    fig, axes = plt.subplots(9, 5, figsize=(24, 32))
    fig.suptitle('Individual Cognitive Action Performance Across All Layers (AUC-ROC)',
                 fontsize=20, fontweight='bold', y=0.995)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Define category colors
    category_colors = {
        'Metacognitive': '#2E86AB',
        'Analytical': '#A23B72',
        'Creative': '#F18F01',
        'Emotional': '#06A77D'
    }

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

    # Map actions to categories
    action_to_category = {}
    for category, actions in categories.items():
        for action in actions:
            action_to_category[action] = category

    # Plot each action
    for idx, action_name in enumerate(all_actions):
        ax = axes_flat[idx]

        # Collect AUC-ROC values across layers for this action
        auc_values = []
        for layer_num in layers:
            action_data = next((a for a in metrics_by_layer[layer_num]['per_action_metrics']
                               if a['action'] == action_name), None)
            if action_data:
                auc_values.append(action_data['auc_roc'])
            else:
                auc_values.append(0)

        # Get category and color
        category = action_to_category.get(action_name, 'Unknown')
        color = category_colors.get(category, '#666666')

        # Plot
        ax.plot(layers, auc_values, linewidth=2, color=color, alpha=0.8)
        ax.fill_between(layers, auc_values, alpha=0.3, color=color)

        # Find best layer for this action
        best_idx = np.argmax(auc_values)
        best_layer = layers[best_idx]
        best_auc = auc_values[best_idx]

        # Highlight best point
        ax.scatter([best_layer], [best_auc], s=80, c='red', marker='*',
                  edgecolors='darkred', linewidths=1.5, zorder=5)

        # Format action name for title
        title = action_name.replace('_', ' ').title()
        ax.set_title(f"{title}\nBest: L{best_layer} ({best_auc:.3f})",
                    fontsize=9, fontweight='bold', color=color)

        # Set limits and labels
        ax.set_ylim([0.5, 1.05])
        ax.set_xlim([min(layers), max(layers)])
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=7)

        # Only show x-label on bottom row
        if idx >= 40:
            ax.set_xlabel('Layer', fontsize=8)

        # Only show y-label on leftmost column
        if idx % 5 == 0:
            ax.set_ylabel('AUC-ROC', fontsize=8)

    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=category)
                      for category, color in category_colors.items()]
    fig.legend(handles=legend_elements, loc='upper right',
              fontsize=12, framealpha=0.9, bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved all actions visualization to {output_path}")
    plt.close()

def create_actions_heatmap_full(metrics_by_layer, output_path='data/all_actions_heatmap_full.png'):
    """Create a comprehensive heatmap for ALL 45 actions"""

    # Get all actions
    first_layer = min(metrics_by_layer.keys())
    all_actions = sorted([action['action'] for action in metrics_by_layer[first_layer]['per_action_metrics']])

    layers = sorted(metrics_by_layer.keys())

    # Build matrix: rows = actions, columns = layers
    matrix = []
    for action_name in all_actions:
        row = []
        for layer_num in layers:
            action_data = next((a for a in metrics_by_layer[layer_num]['per_action_metrics']
                               if a['action'] == action_name), None)
            if action_data:
                row.append(action_data['auc_roc'])
            else:
                row.append(0)
        matrix.append(row)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(20, 18))

    # Format action names
    formatted_names = [name.replace('_', ' ').title() for name in all_actions]

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)

    # Set ticks
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=9)
    ax.set_yticks(range(len(formatted_names)))
    ax.set_yticklabels(formatted_names, fontsize=9)

    # Labels
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cognitive Action', fontsize=14, fontweight='bold')
    ax.set_title('All 45 Cognitive Actions: AUC-ROC Performance Across All Layers',
                 fontsize=16, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('AUC-ROC Score', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved full heatmap to {output_path}")
    plt.close()

def create_best_worst_comparison(metrics_by_layer, output_path='data/best_worst_actions.png'):
    """Compare the 10 best and 10 worst performing actions"""

    # Calculate average AUC-ROC for each action across all layers
    action_avg_auc = {}
    first_layer = min(metrics_by_layer.keys())
    all_actions = [action['action'] for action in metrics_by_layer[first_layer]['per_action_metrics']]

    layers = sorted(metrics_by_layer.keys())

    for action_name in all_actions:
        aucs = []
        for layer_num in layers:
            action_data = next((a for a in metrics_by_layer[layer_num]['per_action_metrics']
                               if a['action'] == action_name), None)
            if action_data:
                aucs.append(action_data['auc_roc'])
        action_avg_auc[action_name] = np.mean(aucs)

    # Sort and get top/bottom 10
    sorted_actions = sorted(action_avg_auc.items(), key=lambda x: x[1], reverse=True)
    best_10 = sorted_actions[:10]
    worst_10 = sorted_actions[-10:]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Best vs Worst Performing Cognitive Actions (Average AUC-ROC Across All Layers)',
                 fontsize=16, fontweight='bold')

    # Plot best 10
    for action_name, avg_auc in best_10:
        aucs = []
        for layer_num in layers:
            action_data = next((a for a in metrics_by_layer[layer_num]['per_action_metrics']
                               if a['action'] == action_name), None)
            if action_data:
                aucs.append(action_data['auc_roc'])

        formatted_name = action_name.replace('_', ' ').title()
        ax1.plot(layers, aucs, linewidth=2, marker='o', markersize=4,
                label=f"{formatted_name} ({avg_auc:.3f})", alpha=0.8)

    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax1.set_title('Top 10 Best Performing Actions', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.7, 1.05])

    # Plot worst 10
    for action_name, avg_auc in worst_10:
        aucs = []
        for layer_num in layers:
            action_data = next((a for a in metrics_by_layer[layer_num]['per_action_metrics']
                               if a['action'] == action_name), None)
            if action_data:
                aucs.append(action_data['auc_roc'])

        formatted_name = action_name.replace('_', ' ').title()
        ax2.plot(layers, aucs, linewidth=2, marker='o', markersize=4,
                label=f"{formatted_name} ({avg_auc:.3f})", alpha=0.8)

    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax2.set_title('Top 10 Worst Performing Actions', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.5, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved best/worst comparison to {output_path}")
    plt.close()

def main():
    """Main function to generate all visualizations"""
    print("Loading metrics from all layers...")
    metrics_by_layer = load_all_layer_metrics()

    if not metrics_by_layer:
        print("Error: No metrics found!")
        return

    print(f"Found metrics for {len(metrics_by_layer)} layers")

    print("\nGenerating individual action grid visualization...")
    create_all_actions_grid(metrics_by_layer)

    print("\nGenerating full heatmap for all 45 actions...")
    create_actions_heatmap_full(metrics_by_layer)

    print("\nGenerating best vs worst comparison...")
    create_best_worst_comparison(metrics_by_layer)

    print("\n✓ All visualizations completed!")
    print("\nGenerated files:")
    print("  - data/all_cognitive_actions_performance.png (9x5 grid of all actions)")
    print("  - data/all_actions_heatmap_full.png (comprehensive heatmap)")
    print("  - data/best_worst_actions.png (top 10 vs bottom 10)")

if __name__ == '__main__':
    main()