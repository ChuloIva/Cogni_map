import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import argparse
import glob

DATA_DIR = '../../data'
RESULTS_DIR = os.path.join(DATA_DIR, 'results')


def load_summary_results(summary_file: str) -> Dict:
    """Load summary results from JSON file"""
    with open(summary_file, 'r') as f:
        return json.load(f)


def create_condition_breakdown_plot(summary: Dict, output_file: str = None):
    """
    Create a bar plot showing accuracy breakdown by condition

    Args:
        summary: Summary dictionary from evaluation
        output_file: Path to save plot (if None, displays plot)
    """
    # Extract data
    conditions = []
    accuracies = []

    for result in summary['condition_results']:
        condition_name = result['condition']
        # Parse condition name for better formatting
        parts = condition_name.split('_')
        # Format: "0/1 backward/forward belief/action true/false"
        formatted = f"{parts[0]} {parts[1]}\n{parts[2]} {parts[3]}"
        conditions.append(formatted)
        accuracies.append(result['accuracy'] * 100)

    # Create DataFrame
    df = pd.DataFrame({
        'Condition': conditions,
        'Accuracy (%)': accuracies
    })

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df, x='Condition', y='Accuracy (%)', ax=ax, palette='viridis')

    ax.set_title(f"Accuracy by Condition - {summary['model_name']}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 100)

    # Add accuracy values on top of bars
    for i, (condition, accuracy) in enumerate(zip(conditions, accuracies)):
        ax.text(i, accuracy + 1, f'{accuracy:.1f}%', ha='center', va='bottom', fontsize=9)

    # Add horizontal line for overall accuracy
    overall_acc = summary['overall_accuracy'] * 100
    ax.axhline(y=overall_acc, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall_acc:.1f}%')
    ax.legend()

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def create_heatmap(summary: Dict, output_file: str = None):
    """
    Create a heatmap showing accuracy across different condition dimensions

    Args:
        summary: Summary dictionary from evaluation
        output_file: Path to save plot (if None, displays plot)
    """
    # Parse conditions into structured format
    data_rows = []
    for result in summary['condition_results']:
        parts = result['condition'].split('_')
        data_rows.append({
            'order_init': parts[0],
            'direction': parts[1],
            'variable': parts[2],
            'belief_type': parts[3],
            'accuracy': result['accuracy'] * 100
        })

    df = pd.DataFrame(data_rows)

    # Create pivot tables for different views
    # View 1: Direction x Belief Type
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Heatmap 1: Direction vs Belief Type (averaged across other dimensions)
    pivot1 = df.groupby(['direction', 'belief_type'])['accuracy'].mean().unstack()
    sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                ax=axes[0, 0], cbar_kws={'label': 'Accuracy (%)'})
    axes[0, 0].set_title('Direction vs Belief Type', fontweight='bold')
    axes[0, 0].set_xlabel('Belief Type')
    axes[0, 0].set_ylabel('Direction')

    # Heatmap 2: Variable vs Belief Type
    pivot2 = df.groupby(['variable', 'belief_type'])['accuracy'].mean().unstack()
    sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                ax=axes[0, 1], cbar_kws={'label': 'Accuracy (%)'})
    axes[0, 1].set_title('Variable vs Belief Type', fontweight='bold')
    axes[0, 1].set_xlabel('Belief Type')
    axes[0, 1].set_ylabel('Variable')

    # Heatmap 3: Order Init vs Direction
    pivot3 = df.groupby(['order_init', 'direction'])['accuracy'].mean().unstack()
    sns.heatmap(pivot3, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                ax=axes[1, 0], cbar_kws={'label': 'Accuracy (%)'})
    axes[1, 0].set_title('Order Init vs Direction', fontweight='bold')
    axes[1, 0].set_xlabel('Direction')
    axes[1, 0].set_ylabel('Order Init')

    # Heatmap 4: Variable vs Direction
    pivot4 = df.groupby(['variable', 'direction'])['accuracy'].mean().unstack()
    sns.heatmap(pivot4, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                ax=axes[1, 1], cbar_kws={'label': 'Accuracy (%)'})
    axes[1, 1].set_title('Variable vs Direction', fontweight='bold')
    axes[1, 1].set_xlabel('Direction')
    axes[1, 1].set_ylabel('Variable')

    fig.suptitle(f'Accuracy Heatmaps - {summary["model_name"]}', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def create_comparison_plot(summary_files: List[str], output_file: str = None):
    """
    Create a comparison plot for multiple models

    Args:
        summary_files: List of summary JSON file paths
        output_file: Path to save plot (if None, displays plot)
    """
    summaries = [load_summary_results(f) for f in summary_files]

    # Prepare data
    data_rows = []
    for summary in summaries:
        model_name = summary['model_name'].split('/')[-1]  # Get short name
        for result in summary['condition_results']:
            data_rows.append({
                'model': model_name,
                'condition': result['condition'],
                'accuracy': result['accuracy'] * 100
            })

    df = pd.DataFrame(data_rows)

    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(16, 6))

    # Format condition names
    condition_map = {}
    for condition in df['condition'].unique():
        parts = condition.split('_')
        formatted = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
        condition_map[condition] = formatted

    df['condition_formatted'] = df['condition'].map(condition_map)

    # Plot
    sns.barplot(data=df, x='condition_formatted', y='accuracy', hue='model', ax=ax)

    ax.set_title('Model Comparison Across Conditions', fontsize=14, fontweight='bold')
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 100)

    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def create_summary_table(summary: Dict, output_file: str = None):
    """
    Create a formatted table of results

    Args:
        summary: Summary dictionary from evaluation
        output_file: Path to save table (CSV format)
    """
    # Create DataFrame
    rows = []
    for result in summary['condition_results']:
        rows.append({
            'Condition': result['condition'],
            'Accuracy (%)': f"{result['accuracy'] * 100:.2f}",
            'Correct': result['num_correct'],
            'Total': result['num_samples']
        })

    df = pd.DataFrame(rows)

    # Add summary row
    summary_row = pd.DataFrame([{
        'Condition': 'OVERALL',
        'Accuracy (%)': f"{summary['overall_accuracy'] * 100:.2f}",
        'Correct': summary['total_correct'],
        'Total': summary['total_samples']
    }])

    df = pd.concat([df, summary_row], ignore_index=True)

    # Print to console
    print("\n" + "="*70)
    print(f"Results Summary - {summary['model_name']}")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    # Save to CSV
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nTable saved to: {output_file}")

    return df


def visualize_all(summary_file: str, output_dir: str = None):
    """
    Create all visualizations for a summary file

    Args:
        summary_file: Path to summary JSON file
        output_dir: Directory to save plots (if None, uses same dir as summary)
    """
    summary = load_summary_results(summary_file)

    if output_dir is None:
        output_dir = os.path.dirname(summary_file)

    # Create base filename
    model_name = summary['model_name'].replace('/', '_')
    base_name = os.path.join(output_dir, f"{model_name}_temp{summary['temperature']}")

    print(f"\nGenerating visualizations for {summary['model_name']}...")

    # Create all plots
    create_condition_breakdown_plot(summary, f"{base_name}_breakdown.png")
    create_heatmap(summary, f"{base_name}_heatmap.png")
    create_summary_table(summary, f"{base_name}_table.csv")

    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Theory of Mind evaluation results'
    )
    parser.add_argument(
        '--summary',
        type=str,
        required=True,
        help='Path to summary JSON file or pattern (e.g., "results/summary_*.json")'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save visualizations (default: same as summary file)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple models (use wildcard in --summary)'
    )

    args = parser.parse_args()

    # Handle wildcards
    summary_files = glob.glob(args.summary)

    if not summary_files:
        print(f"No summary files found matching: {args.summary}")
        return

    if args.compare and len(summary_files) > 1:
        print(f"\nComparing {len(summary_files)} models...")
        output_file = os.path.join(
            args.output_dir or RESULTS_DIR,
            'model_comparison.png'
        )
        create_comparison_plot(summary_files, output_file)
    else:
        # Visualize each summary file
        for summary_file in summary_files:
            visualize_all(summary_file, args.output_dir)


if __name__ == '__main__':
    main()
