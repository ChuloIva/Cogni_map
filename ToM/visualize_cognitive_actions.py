"""
Advanced Visualization Suite for Cognitive Action Analysis

This script creates comprehensive visualizations from cognitive action evaluation data,
including:
1. Complete cognitive action comparison (all 45 actions)
2. Category-based analysis (metacognitive, analytical, creative, emotional, memory)
3. Temporal dynamics (at question vs after answers)
4. Network/relationship visualizations
5. Statistical distributions
6. Interactive comparison plots

Usage:
    python visualize_cognitive_actions.py --input results/single_vector_test_summary.json
    python visualize_cognitive_actions.py --input results/my_summary.json --output-dir custom_viz
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from probes.action_categories import ACTION_TO_CATEGORY, CATEGORY_TAGS, get_action_category

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class CognitiveActionVisualizer:
    """Create comprehensive visualizations for cognitive action analysis"""

    def __init__(self, data_path: str, output_dir: str = "visualizations", raw_csv_path: str = None):
        """
        Initialize visualizer

        Args:
            data_path: Path to summary JSON file
            output_dir: Directory to save visualizations
            raw_csv_path: Optional path to raw CSV with baseline/steered values
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load summary data
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)

        # Load raw CSV data if provided
        self.raw_data = None
        if raw_csv_path:
            self.raw_csv_path = Path(raw_csv_path)
            self.raw_data = self._load_raw_csv()
            print(f"Loaded raw CSV with {len(self.raw_data)} samples")

        # Extract all cognitive actions
        self.all_actions = sorted(self.data['mean_diff_at_question'].keys())

        # Categorize actions
        self.action_categories = {
            action: get_action_category(action)
            for action in self.all_actions
        }

        # Group by category
        self.categories = defaultdict(list)
        for action, category in self.action_categories.items():
            self.categories[category].append(action)

        print(f"Loaded data from {self.data_path}")
        print(f"Total cognitive actions: {len(self.all_actions)}")
        print(f"Categories: {dict([(cat, len(actions)) for cat, actions in self.categories.items()])}")

    def _load_raw_csv(self) -> List[Dict]:
        """Load raw CSV data with baseline and steered values"""
        import csv

        rows = []
        with open(self.raw_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse JSON fields
                row['baseline_activations_at_question'] = json.loads(row['baseline_activations_at_question'])
                row['baseline_activations_after_true'] = json.loads(row['baseline_activations_after_true'])
                row['baseline_activations_after_wrong'] = json.loads(row['baseline_activations_after_wrong'])
                row['steered_activations_at_question'] = json.loads(row['steered_activations_at_question'])
                row['steered_activations_after_true'] = json.loads(row['steered_activations_after_true'])
                row['steered_activations_after_wrong'] = json.loads(row['steered_activations_after_wrong'])
                rows.append(row)

        return rows

    def create_all_visualizations(self):
        """Generate all visualization types"""
        print("\nGenerating visualizations...")
        print("=" * 80)

        # Baseline vs Steered focused visualizations (if raw data available)
        if self.raw_data:
            print("\n🎯 BASELINE vs STEERED COMPARISON PLOTS")
            print("-" * 80)

            # B1. Side-by-side comparison
            print("B1. Creating baseline vs steered side-by-side comparison...")
            self.viz_baseline_vs_steered_sidebyside()

            # B2. Direct comparison bars
            print("B2. Creating baseline vs steered direct comparison...")
            self.viz_baseline_vs_steered_bars()

            # B3. Grouped category comparison
            print("B3. Creating category-wise baseline vs steered...")
            self.viz_category_baseline_vs_steered()

            # B4. Accuracy impact analysis
            print("B4. Creating accuracy impact analysis...")
            self.viz_accuracy_impact_analysis()

            # B5. Sample-by-sample comparison
            print("B5. Creating sample-level analysis...")
            self.viz_sample_level_comparison()

            print()

        # Original visualizations
        print("📊 GENERAL ANALYSIS PLOTS")
        print("-" * 80)

        # 1. Complete action comparison
        print("1. Creating complete action comparison...")
        self.viz_complete_action_comparison()

        # 2. Category-based analysis
        print("2. Creating category-based analysis...")
        self.viz_category_analysis()

        # 3. Temporal dynamics
        print("3. Creating temporal dynamics visualization...")
        self.viz_temporal_dynamics()

        # 4. Radar/spider chart
        print("4. Creating category radar chart...")
        self.viz_category_radar()

        # 5. Heatmap grid
        print("5. Creating comprehensive heatmap...")
        self.viz_comprehensive_heatmap()

        # 6. Diverging bar chart
        print("6. Creating diverging bar charts...")
        self.viz_diverging_bars()

        # 7. Scatter plot (baseline vs steered effect)
        print("7. Creating scatter analysis...")
        self.viz_scatter_analysis()

        # 8. Distribution plots
        print("8. Creating distribution plots...")
        self.viz_distributions()

        # 9. Top changes summary
        print("9. Creating top changes summary...")
        self.viz_top_changes_summary()

        # 10. Category comparison grid
        print("10. Creating category comparison grid...")
        self.viz_category_grid()

        print("\n" + "=" * 80)
        print(f"✓ All visualizations saved to: {self.output_dir}")

    # =========================================================================
    # BASELINE vs STEERED COMPARISON VISUALIZATIONS
    # =========================================================================

    def viz_baseline_vs_steered_sidebyside(self):
        """
        B1: Side-by-side comparison of baseline and steered activation levels
        Shows actual activation counts (not differences) for all actions
        """
        fig, axes = plt.subplots(1, 3, figsize=(24, 18))

        timepoints = [
            ('at_question', 'At Question'),
            ('after_true', 'After True Answer'),
            ('after_wrong', 'After Wrong Answer')
        ]

        for ax, (time_key, title) in zip(axes, timepoints):
            # Calculate mean activations for baseline and steered
            baseline_means = defaultdict(list)
            steered_means = defaultdict(list)

            for row in self.raw_data:
                for action in self.all_actions:
                    baseline_means[action].append(
                        row[f'baseline_activations_{time_key}'].get(action, 0)
                    )
                    steered_means[action].append(
                        row[f'steered_activations_{time_key}'].get(action, 0)
                    )

            # Calculate means
            baseline_avg = {action: np.mean(vals) for action, vals in baseline_means.items()}
            steered_avg = {action: np.mean(vals) for action, vals in steered_means.items()}

            # Sort by the maximum of either baseline or steered
            sorted_actions = sorted(
                self.all_actions,
                key=lambda a: max(baseline_avg[a], steered_avg[a]),
                reverse=True
            )

            y_pos = np.arange(len(sorted_actions))
            width = 0.35

            # Plot baseline and steered side by side
            baseline_vals = [baseline_avg[a] for a in sorted_actions]
            steered_vals = [steered_avg[a] for a in sorted_actions]

            bars1 = ax.barh(y_pos - width/2, baseline_vals, width,
                           label='Baseline', color='#3498db', alpha=0.8,
                           edgecolor='black', linewidth=0.5)
            bars2 = ax.barh(y_pos + width/2, steered_vals, width,
                           label='Steered', color='#e74c3c', alpha=0.8,
                           edgecolor='black', linewidth=0.5)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_actions, fontsize=7)
            ax.set_xlabel('Mean Layer Count', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(axis='x', alpha=0.3)

            # Highlight actions with big differences
            for i, action in enumerate(sorted_actions):
                diff = steered_avg[action] - baseline_avg[action]
                if abs(diff) > 0.5:
                    # Add arrow showing direction of change
                    x_start = baseline_avg[action]
                    x_end = steered_avg[action]
                    if diff > 0:
                        ax.annotate('', xy=(x_end, i + width/2), xytext=(x_start, i - width/2),
                                  arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.6))
                    else:
                        ax.annotate('', xy=(x_end, i + width/2), xytext=(x_start, i - width/2),
                                  arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.6))

        plt.suptitle('Baseline vs Steered: Side-by-Side Comparison\n' +
                    f'Direct Comparison of Activation Levels (n={len(self.raw_data)} samples)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'B1_baseline_vs_steered_sidebyside.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_baseline_vs_steered_bars(self):
        """
        B2: Grouped bar chart showing baseline vs steered for top actions
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))

        # Calculate overall means
        baseline_means = defaultdict(list)
        steered_means = defaultdict(list)

        for row in self.raw_data:
            for action in self.all_actions:
                baseline_means[action].append(
                    row['baseline_activations_at_question'].get(action, 0)
                )
                steered_means[action].append(
                    row['steered_activations_at_question'].get(action, 0)
                )

        baseline_avg = {action: np.mean(vals) for action, vals in baseline_means.items()}
        steered_avg = {action: np.mean(vals) for action, vals in steered_means.items()}

        # Top increases (steered > baseline)
        ax = axes[0, 0]
        increases = sorted(
            [(action, steered_avg[action] - baseline_avg[action])
             for action in self.all_actions],
            key=lambda x: x[1],
            reverse=True
        )[:15]

        actions = [item[0] for item in increases]
        baseline_vals = [baseline_avg[a] for a in actions]
        steered_vals = [steered_avg[a] for a in actions]

        x = np.arange(len(actions))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, steered_vals, width, label='Steered',
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Mean Layer Count', fontweight='bold')
        ax.set_title('Top 15 Increased Actions (Steered > Baseline)', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(actions, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (b, s) in enumerate(zip(baseline_vals, steered_vals)):
            ax.text(i, max(b, s), f'+{s-b:.2f}', ha='center', va='bottom',
                   fontsize=8, fontweight='bold', color='green')

        # Top decreases (baseline > steered)
        ax = axes[0, 1]
        decreases = sorted(
            [(action, steered_avg[action] - baseline_avg[action])
             for action in self.all_actions],
            key=lambda x: x[1]
        )[:15]

        actions = [item[0] for item in decreases]
        baseline_vals = [baseline_avg[a] for a in actions]
        steered_vals = [steered_avg[a] for a in actions]

        x = np.arange(len(actions))

        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, steered_vals, width, label='Steered',
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Mean Layer Count', fontweight='bold')
        ax.set_title('Top 15 Decreased Actions (Baseline > Steered)', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(actions, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (b, s) in enumerate(zip(baseline_vals, steered_vals)):
            ax.text(i, max(b, s), f'{s-b:.2f}', ha='center', va='bottom',
                   fontsize=8, fontweight='bold', color='red')

        # Scatter: baseline vs steered
        ax = axes[1, 0]

        all_baseline = [baseline_avg[a] for a in self.all_actions]
        all_steered = [steered_avg[a] for a in self.all_actions]
        colors = [self._get_category_color(self.action_categories[a]) for a in self.all_actions]

        ax.scatter(all_baseline, all_steered, c=colors, s=120, alpha=0.7,
                  edgecolors='black', linewidth=0.5)

        # Add diagonal line (no change)
        lims = [0, max(max(all_baseline), max(all_steered))]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='No change')

        # Add correlation
        corr = np.corrcoef(all_baseline, all_steered)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               fontsize=13, fontweight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_xlabel('Baseline Mean Layer Count', fontweight='bold')
        ax.set_ylabel('Steered Mean Layer Count', fontweight='bold')
        ax.set_title('Baseline vs Steered Scatter', fontweight='bold', pad=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Percent change analysis
        ax = axes[1, 1]

        percent_changes = []
        for action in self.all_actions:
            b = baseline_avg[action]
            s = steered_avg[action]
            if b > 0:
                pct = ((s - b) / b) * 100
            elif s > 0:
                pct = 100  # Infinite increase
            else:
                pct = 0
            percent_changes.append((action, pct))

        # Sort and take top positive and negative
        sorted_pct = sorted(percent_changes, key=lambda x: abs(x[1]), reverse=True)[:20]

        actions = [item[0] for item in sorted_pct]
        pcts = [item[1] for item in sorted_pct]
        colors_bar = ['green' if p > 0 else 'red' for p in pcts]

        ax.barh(range(len(actions)), pcts, color=colors_bar, alpha=0.7,
               edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(actions)))
        ax.set_yticklabels(actions, fontsize=9)
        ax.set_xlabel('Percent Change (%)', fontweight='bold')
        ax.set_title('Top 20 Actions by Percent Change', fontweight='bold', pad=10)
        ax.axvline(0, color='black', linestyle='-', linewidth=1.5)
        ax.grid(axis='x', alpha=0.3)

        plt.suptitle('Baseline vs Steered: Grouped Comparisons',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'B2_baseline_vs_steered_bars.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_category_baseline_vs_steered(self):
        """
        B3: Category-wise baseline vs steered comparison
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        # Calculate category averages
        categories = sorted(self.categories.keys())

        for idx, category in enumerate(categories):
            if idx >= len(axes):
                break

            ax = axes[idx]
            actions = self.categories[category]

            # Calculate means for this category
            baseline_means = []
            steered_means = []
            action_labels = []

            for action in actions:
                baseline_vals = [row['baseline_activations_at_question'].get(action, 0)
                               for row in self.raw_data]
                steered_vals = [row['steered_activations_at_question'].get(action, 0)
                              for row in self.raw_data]

                baseline_means.append(np.mean(baseline_vals))
                steered_means.append(np.mean(steered_vals))
                action_labels.append(action)

            # Sort by difference
            sorted_idx = sorted(range(len(actions)),
                              key=lambda i: abs(steered_means[i] - baseline_means[i]),
                              reverse=True)

            sorted_actions = [action_labels[i] for i in sorted_idx]
            sorted_baseline = [baseline_means[i] for i in sorted_idx]
            sorted_steered = [steered_means[i] for i in sorted_idx]

            x = np.arange(len(sorted_actions))
            width = 0.35

            bars1 = ax.bar(x - width/2, sorted_baseline, width, label='Baseline',
                          color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, sorted_steered, width, label='Steered',
                          color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

            ax.set_ylabel('Mean Layer Count', fontweight='bold', fontsize=10)
            ax.set_title(f'{category.upper()}\n({len(actions)} actions)',
                        fontweight='bold', pad=10,
                        bbox=dict(boxstyle='round', facecolor=self._get_category_color(category),
                                alpha=0.3))
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_actions, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        # Hide unused subplots
        for idx in range(len(categories), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Category-Wise Baseline vs Steered Comparison',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'B3_category_baseline_vs_steered.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_accuracy_impact_analysis(self):
        """
        B4: Analyze cognitive actions in relation to accuracy improvement
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # Separate samples into improved vs not improved
        improved_samples = [row for row in self.raw_data if row['accuracy_improvement'] == 'True']
        not_improved_samples = [row for row in self.raw_data
                               if row['accuracy_improvement'] == 'False']

        print(f"      Improved: {len(improved_samples)}, Not improved: {len(not_improved_samples)}")

        # Plot 1: Average difference in improved vs not improved samples
        ax = axes[0, 0]

        improved_diffs = defaultdict(list)
        not_improved_diffs = defaultdict(list)

        for row in improved_samples:
            for action in self.all_actions:
                diff = (row['steered_activations_at_question'].get(action, 0) -
                       row['baseline_activations_at_question'].get(action, 0))
                improved_diffs[action].append(diff)

        for row in not_improved_samples:
            for action in self.all_actions:
                diff = (row['steered_activations_at_question'].get(action, 0) -
                       row['baseline_activations_at_question'].get(action, 0))
                not_improved_diffs[action].append(diff)

        # Calculate means
        improved_means = {a: np.mean(diffs) if diffs else 0
                         for a, diffs in improved_diffs.items()}
        not_improved_means = {a: np.mean(diffs) if diffs else 0
                            for a, diffs in not_improved_diffs.items()}

        # Find actions with biggest difference between groups
        group_diffs = {a: improved_means[a] - not_improved_means[a]
                      for a in self.all_actions}

        top_group_diffs = sorted(group_diffs.items(), key=lambda x: abs(x[1]), reverse=True)[:15]

        actions = [item[0] for item in top_group_diffs]
        improved_vals = [improved_means[a] for a in actions]
        not_improved_vals = [not_improved_means[a] for a in actions]

        x = np.arange(len(actions))
        width = 0.35

        bars1 = ax.bar(x - width/2, improved_vals, width, label='Accuracy Improved',
                      color='green', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, not_improved_vals, width, label='No Improvement',
                      color='gray', alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Mean Activation Difference', fontweight='bold')
        ax.set_title('Top Actions Distinguishing Improved vs Not Improved Samples',
                    fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(actions, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        # Plot 2: Accuracy by activation level (for top actions)
        ax = axes[0, 1]

        # Choose top 3 most changed actions
        top_actions = sorted(self.data['mean_diff_at_question'].items(),
                           key=lambda x: abs(x[1]), reverse=True)[:3]

        for action, _ in top_actions:
            # Group samples by steered activation level
            low_activation = []
            high_activation = []

            for row in self.raw_data:
                steered_act = row['steered_activations_at_question'].get(action, 0)
                is_correct = row['steered_correct'] == 'True'

                if steered_act < np.median([r['steered_activations_at_question'].get(action, 0)
                                           for r in self.raw_data]):
                    low_activation.append(1 if is_correct else 0)
                else:
                    high_activation.append(1 if is_correct else 0)

            low_acc = np.mean(low_activation) if low_activation else 0
            high_acc = np.mean(high_activation) if high_activation else 0

            color = self._get_category_color(self.action_categories[action])

            x_pos = list(top_actions).index((action, _))
            ax.bar([x_pos - 0.2, x_pos + 0.2], [low_acc, high_acc],
                  width=0.35, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('Accuracy: Low vs High Activation (Top 3 Actions)',
                    fontweight='bold', pad=10)
        ax.set_xticks(range(3))
        ax.set_xticklabels([action for action, _ in top_actions], fontsize=10)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='gray', alpha=0.7, label='Low Activation'),
                          Patch(facecolor='gray', alpha=0.9, label='High Activation')]
        ax.legend(handles=legend_elements)

        # Plot 3: Baseline accuracy vs steered accuracy
        ax = axes[1, 0]

        baseline_correct = [row['baseline_correct'] == 'True' for row in self.raw_data]
        steered_correct = [row['steered_correct'] == 'True' for row in self.raw_data]

        baseline_acc = np.mean(baseline_correct)
        steered_acc = np.mean(steered_correct)

        bars = ax.bar(['Baseline', 'Steered'], [baseline_acc, steered_acc],
                     color=['#3498db', '#e74c3c'], alpha=0.8,
                     edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title(f'Overall Accuracy Comparison\n(n={len(self.raw_data)} samples)',
                    fontweight='bold', pad=10)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, [baseline_acc, steered_acc])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1%}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')

        # Add improvement text
        improvement = steered_acc - baseline_acc
        ax.text(0.5, 0.5, f'Improvement: {improvement:+.1%}',
               ha='center', fontsize=14, fontweight='bold',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightgreen' if improvement > 0 else 'lightcoral',
                        alpha=0.7, edgecolor='black', linewidth=2))

        # Plot 4: Probability distributions
        ax = axes[1, 1]

        baseline_prob_true = [float(row['baseline_prob_true']) for row in self.raw_data]
        baseline_prob_wrong = [float(row['baseline_prob_wrong']) for row in self.raw_data]
        steered_prob_true = [float(row['steered_prob_true']) for row in self.raw_data]
        steered_prob_wrong = [float(row['steered_prob_wrong']) for row in self.raw_data]

        data_to_plot = [baseline_prob_true, steered_prob_true,
                       baseline_prob_wrong, steered_prob_wrong]
        positions = [1, 2, 4, 5]
        colors_box = ['#3498db', '#e74c3c', '#3498db', '#e74c3c']

        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                       patch_artist=True, showmeans=True)

        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks([1.5, 4.5])
        ax.set_xticklabels(['P(True Answer)', 'P(Wrong Answer)'], fontweight='bold')
        ax.set_ylabel('Probability', fontweight='bold')
        ax.set_title('Probability Distributions: Baseline vs Steered', fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#3498db', alpha=0.7, label='Baseline'),
                          Patch(facecolor='#e74c3c', alpha=0.7, label='Steered')]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.suptitle('Accuracy Impact Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'B4_accuracy_impact_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_sample_level_comparison(self):
        """
        B5: Sample-level analysis showing variation across samples
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # Get top 5 most affected actions
        top_actions = sorted(self.data['mean_diff_at_question'].items(),
                           key=lambda x: abs(x[1]), reverse=True)[:5]

        # Plot 1: Sample-by-sample for top action
        ax = axes[0, 0]

        action = top_actions[0][0]
        sample_ids = range(min(30, len(self.raw_data)))  # First 30 samples

        baseline_vals = [self.raw_data[i]['baseline_activations_at_question'].get(action, 0)
                        for i in sample_ids]
        steered_vals = [self.raw_data[i]['steered_activations_at_question'].get(action, 0)
                       for i in sample_ids]

        ax.plot(sample_ids, baseline_vals, 'o-', label='Baseline', color='#3498db',
               linewidth=2, markersize=6, alpha=0.7)
        ax.plot(sample_ids, steered_vals, 's-', label='Steered', color='#e74c3c',
               linewidth=2, markersize=6, alpha=0.7)

        ax.set_xlabel('Sample ID', fontweight='bold')
        ax.set_ylabel('Layer Count', fontweight='bold')
        ax.set_title(f'Sample-by-Sample Comparison\nTop Action: {action}',
                    fontweight='bold', pad=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Variance comparison
        ax = axes[0, 1]

        baseline_vars = []
        steered_vars = []
        action_labels = []

        for action, _ in top_actions:
            baseline_vals = [row['baseline_activations_at_question'].get(action, 0)
                           for row in self.raw_data]
            steered_vals = [row['steered_activations_at_question'].get(action, 0)
                          for row in self.raw_data]

            baseline_vars.append(np.std(baseline_vals))
            steered_vars.append(np.std(steered_vals))
            action_labels.append(action)

        x = np.arange(len(action_labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_vars, width, label='Baseline',
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, steered_vars, width, label='Steered',
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Standard Deviation', fontweight='bold')
        ax.set_title('Variability: Baseline vs Steered (Top 5 Actions)',
                    fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(action_labels, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Plot 3: Heatmap of top actions across all samples
        ax = axes[1, 0]

        # Create difference matrix (steered - baseline)
        diff_matrix = []
        for action, _ in top_actions[:10]:  # Top 10 for readability
            diffs = [
                (row['steered_activations_at_question'].get(action, 0) -
                 row['baseline_activations_at_question'].get(action, 0))
                for row in self.raw_data[:30]  # First 30 samples
            ]
            diff_matrix.append(diffs)

        diff_matrix = np.array(diff_matrix)

        im = ax.imshow(diff_matrix, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=3)

        ax.set_yticks(range(len(top_actions[:10])))
        ax.set_yticklabels([action for action, _ in top_actions[:10]], fontsize=9)
        ax.set_xlabel('Sample ID', fontweight='bold')
        ax.set_title('Difference Heatmap (Steered - Baseline)\nTop 10 Actions × First 30 Samples',
                    fontweight='bold', pad=10)

        plt.colorbar(im, ax=ax, label='Layer Count Difference')

        # Plot 4: Consistency score
        ax = axes[1, 1]

        # Calculate consistency: how often does steering increase/decrease each action
        consistency_scores = []

        for action in self.all_actions:
            diffs = [
                (row['steered_activations_at_question'].get(action, 0) -
                 row['baseline_activations_at_question'].get(action, 0))
                for row in self.raw_data
            ]

            # Consistency = % of samples with same sign as mean
            mean_diff = np.mean(diffs)
            if mean_diff != 0:
                same_sign = sum(1 for d in diffs if np.sign(d) == np.sign(mean_diff))
                consistency = (same_sign / len(diffs)) * 100
            else:
                consistency = 0

            consistency_scores.append((action, consistency, abs(mean_diff)))

        # Sort by consistency
        sorted_consistency = sorted(consistency_scores, key=lambda x: x[1], reverse=True)[:20]

        actions = [item[0] for item in sorted_consistency]
        consistencies = [item[1] for item in sorted_consistency]
        colors = ['green' if item[2] > 0 else 'red' for item in sorted_consistency]

        ax.barh(range(len(actions)), consistencies, color=colors, alpha=0.7,
               edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(actions)))
        ax.set_yticklabels(actions, fontsize=8)
        ax.set_xlabel('Consistency Score (%)', fontweight='bold')
        ax.set_title('Top 20 Most Consistent Changes\n(% of samples with same direction)',
                    fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3)

        plt.suptitle('Sample-Level Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'B5_sample_level_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    # =========================================================================
    # ORIGINAL VISUALIZATIONS
    # =========================================================================

    def viz_complete_action_comparison(self):
        """
        Visualization 1: Complete comparison of all 45 cognitive actions
        Shows baseline vs steered for all three timepoints
        """
        fig, axes = plt.subplots(1, 3, figsize=(24, 16))

        timepoints = [
            ('mean_diff_at_question', 'At Question'),
            ('mean_diff_after_true', 'After True Answer'),
            ('mean_diff_after_wrong', 'After Wrong Answer')
        ]

        for ax, (key, title) in zip(axes, timepoints):
            # Get all actions sorted by absolute difference
            diffs = self.data[key]
            sorted_actions = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)

            actions = [a[0] for a in sorted_actions]
            values = [a[1] for a in sorted_actions]

            # Color by category
            colors = [self._get_category_color(self.action_categories[action])
                     for action in actions]

            # Create horizontal bar chart
            y_pos = np.arange(len(actions))
            bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(actions, fontsize=8)
            ax.set_xlabel('Layer Count Difference (Steered - Baseline)', fontsize=10, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.grid(axis='x', alpha=0.3)

            # Add value labels for top changes
            for i, (action, val) in enumerate(sorted_actions[:10]):
                if abs(val) > 0.01:
                    ax.text(val, i, f' {val:.2f}',
                           va='center', ha='left' if val > 0 else 'right',
                           fontsize=7, fontweight='bold')

        # Create legend for categories
        category_colors = {cat: self._get_category_color(cat)
                          for cat in set(self.action_categories.values())}
        legend_elements = [plt.Rectangle((0,0),1,1, fc=color, alpha=0.7, edgecolor='black', linewidth=0.5,
                                        label=f'{cat.title()} ({len(self.categories[cat])})')
                          for cat, color in sorted(category_colors.items())]
        fig.legend(handles=legend_elements, loc='upper center', ncol=6,
                  fontsize=10, frameon=True, fancybox=True, shadow=True)

        plt.suptitle('Complete Cognitive Action Comparison (All 45 Actions)\n' +
                    f'Steering Effect Across All Timepoints (n={self.data["num_samples"]})',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        output_path = self.output_dir / '01_complete_action_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_category_analysis(self):
        """
        Visualization 2: Category-based aggregated analysis
        Shows mean effect per category
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        timepoints = [
            ('mean_diff_at_question', 'At Question'),
            ('mean_diff_after_true', 'After True Answer'),
            ('mean_diff_after_wrong', 'After Wrong Answer')
        ]

        # Calculate category averages
        category_stats = defaultdict(lambda: defaultdict(list))

        for category, actions in self.categories.items():
            for key, _ in timepoints:
                for action in actions:
                    if action in self.data[key]:
                        category_stats[category][key].append(self.data[key][action])

        # Plot 1: Average effect per category across timepoints
        ax = axes[0, 0]
        categories = sorted(category_stats.keys())
        x = np.arange(len(timepoints))
        width = 0.15

        for i, category in enumerate(categories):
            means = [np.mean(category_stats[category][key]) if category_stats[category][key] else 0
                    for key, _ in timepoints]
            offset = (i - len(categories)/2) * width
            bars = ax.bar(x + offset, means, width,
                         label=category.title(),
                         color=self._get_category_color(category),
                         alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Timepoint', fontweight='bold')
        ax.set_ylabel('Mean Layer Count Difference', fontweight='bold')
        ax.set_title('Mean Steering Effect by Category', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([t[1] for t in timepoints], rotation=15, ha='right')
        ax.legend(loc='best', fontsize=9)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        # Plot 2: Distribution of effects per category (at question)
        ax = axes[0, 1]
        data_for_violin = []
        labels_for_violin = []

        for category in categories:
            values = category_stats[category]['mean_diff_at_question']
            if values:
                data_for_violin.append(values)
                labels_for_violin.append(f'{category.title()}\n(n={len(values)})')

        parts = ax.violinplot(data_for_violin, positions=range(len(data_for_violin)),
                             showmeans=True, showmedians=True)

        # Color violin plots
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self._get_category_color(categories[i]))
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(labels_for_violin)))
        ax.set_xticklabels(labels_for_violin, fontsize=9)
        ax.set_ylabel('Layer Count Difference', fontweight='bold')
        ax.set_title('Distribution of Effects by Category (At Question)', fontweight='bold', pad=10)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        # Plot 3: Category composition (pie chart)
        ax = axes[1, 0]
        sizes = [len(actions) for actions in self.categories.values()]
        colors = [self._get_category_color(cat) for cat in self.categories.keys()]
        labels = [f'{cat.title()}\n({len(self.categories[cat])} actions)'
                 for cat in self.categories.keys()]

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                           startangle=90, textprops={'fontsize': 9, 'fontweight': 'bold'})
        ax.set_title('Cognitive Action Category Distribution', fontweight='bold', pad=10)

        # Plot 4: Heatmap of category × timepoint
        ax = axes[1, 1]
        heatmap_data = []

        for category in categories:
            row = [np.mean(category_stats[category][key]) if category_stats[category][key] else 0
                  for key, _ in timepoints]
            heatmap_data.append(row)

        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-1.5, vmax=1.5)

        ax.set_xticks(np.arange(len(timepoints)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels([t[1] for t in timepoints], rotation=15, ha='right')
        ax.set_yticklabels([cat.title() for cat in categories])

        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(timepoints)):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')

        ax.set_title('Category × Timepoint Heatmap', fontweight='bold', pad=10)
        plt.colorbar(im, ax=ax, label='Mean Layer Count Difference')

        plt.suptitle('Category-Based Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / '02_category_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_temporal_dynamics(self):
        """
        Visualization 3: Temporal dynamics showing how effects change across timepoints
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # Get top actions by maximum absolute effect across any timepoint
        max_effects = {}
        for action in self.all_actions:
            max_effect = max(
                abs(self.data['mean_diff_at_question'].get(action, 0)),
                abs(self.data['mean_diff_after_true'].get(action, 0)),
                abs(self.data['mean_diff_after_wrong'].get(action, 0))
            )
            max_effects[action] = max_effect

        top_actions = sorted(max_effects.items(), key=lambda x: x[1], reverse=True)[:20]

        # Plot 1: Line plot showing trajectory for top actions
        ax = axes[0, 0]

        for action, _ in top_actions[:15]:
            values = [
                self.data['mean_diff_at_question'].get(action, 0),
                self.data['mean_diff_after_true'].get(action, 0),
                self.data['mean_diff_after_wrong'].get(action, 0)
            ]
            color = self._get_category_color(self.action_categories[action])
            ax.plot([0, 1, 2], values, marker='o', label=action,
                   color=color, alpha=0.7, linewidth=2, markersize=6)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['At Question', 'After True', 'After Wrong'])
        ax.set_ylabel('Layer Count Difference', fontweight='bold')
        ax.set_title('Temporal Trajectory (Top 15 Actions)', fontweight='bold', pad=10)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Change from "at question" to "after answers"
        ax = axes[0, 1]

        changes_to_true = []
        changes_to_wrong = []
        labels = []

        for action, _ in top_actions[:15]:
            at_q = self.data['mean_diff_at_question'].get(action, 0)
            after_true = self.data['mean_diff_after_true'].get(action, 0)
            after_wrong = self.data['mean_diff_after_wrong'].get(action, 0)

            changes_to_true.append(after_true - at_q)
            changes_to_wrong.append(after_wrong - at_q)
            labels.append(action)

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.barh(x - width/2, changes_to_true, width, label='Change to True Answer',
                       color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
        bars2 = ax.barh(x + width/2, changes_to_wrong, width, label='Change to Wrong Answer',
                       color='red', alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Change in Layer Count Difference', fontweight='bold')
        ax.set_title('Dynamic Changes: From Question to Answer', fontweight='bold', pad=10)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        # Plot 3: Stability metric (variance across timepoints)
        ax = axes[1, 0]

        stability = {}
        for action in self.all_actions:
            values = [
                self.data['mean_diff_at_question'].get(action, 0),
                self.data['mean_diff_after_true'].get(action, 0),
                self.data['mean_diff_after_wrong'].get(action, 0)
            ]
            stability[action] = np.std(values)

        sorted_stability = sorted(stability.items(), key=lambda x: x[1], reverse=True)[:20]

        actions = [s[0] for s in sorted_stability]
        stds = [s[1] for s in sorted_stability]
        colors = [self._get_category_color(self.action_categories[action]) for action in actions]

        bars = ax.barh(range(len(actions)), stds, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(actions)))
        ax.set_yticklabels(actions, fontsize=9)
        ax.set_xlabel('Temporal Variability (Std Dev)', fontweight='bold')
        ax.set_title('Most Variable Actions Across Timepoints', fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3)

        # Plot 4: Correlation between timepoints
        ax = axes[1, 1]

        # Scatter: at_question vs after_true
        x_vals = [self.data['mean_diff_at_question'].get(a, 0) for a in self.all_actions]
        y_vals = [self.data['mean_diff_after_true'].get(a, 0) for a in self.all_actions]
        colors = [self._get_category_color(self.action_categories[a]) for a in self.all_actions]

        ax.scatter(x_vals, y_vals, c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

        # Add diagonal line
        min_val = min(min(x_vals), min(y_vals))
        max_val = max(max(x_vals), max(y_vals))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

        # Add correlation coefficient
        corr = np.corrcoef(x_vals, y_vals)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               fontsize=12, fontweight='bold', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Effect at Question', fontweight='bold')
        ax.set_ylabel('Effect after True Answer', fontweight='bold')
        ax.set_title('Correlation: At Question vs After True', fontweight='bold', pad=10)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax.grid(True, alpha=0.3)

        plt.suptitle('Temporal Dynamics Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / '03_temporal_dynamics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_category_radar(self):
        """
        Visualization 4: Radar/spider chart for category comparison
        """
        categories = sorted(self.categories.keys())

        # Calculate mean effects per category for each timepoint
        timepoints = [
            ('mean_diff_at_question', 'At Question'),
            ('mean_diff_after_true', 'After True'),
            ('mean_diff_after_wrong', 'After Wrong')
        ]

        fig = plt.figure(figsize=(18, 6))

        for idx, (key, title) in enumerate(timepoints):
            ax = fig.add_subplot(1, 3, idx + 1, projection='polar')

            # Calculate mean for each category
            values = []
            for category in categories:
                actions = self.categories[category]
                category_values = [self.data[key].get(action, 0) for action in actions]
                values.append(np.mean(category_values) if category_values else 0)

            # Number of variables
            num_vars = len(categories)

            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

            # Complete the circle
            values += values[:1]
            angles += angles[:1]

            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
            ax.fill(angles, values, alpha=0.25, color='steelblue')

            # Fix axis to go in the right order
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # Draw axis lines for each angle and label
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([cat.title() for cat in categories], fontsize=10)

            # Set title
            ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

            # Add grid
            ax.grid(True)

            # Set y-axis limits
            max_val = max(abs(min(values)), abs(max(values)))
            ax.set_ylim(-max_val * 1.2, max_val * 1.2)

        plt.suptitle('Category Radar Charts Across Timepoints', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / '04_category_radar.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_comprehensive_heatmap(self):
        """
        Visualization 5: Comprehensive heatmap of all actions × all timepoints
        """
        fig, ax = plt.subplots(figsize=(12, 20))

        # Prepare data matrix
        timepoints = [
            'mean_diff_at_question',
            'mean_diff_after_true',
            'mean_diff_after_wrong'
        ]

        # Sort actions by category then by mean absolute effect
        sorted_actions = []
        for category in sorted(self.categories.keys()):
            category_actions = self.categories[category]
            # Sort within category by absolute mean effect
            category_sorted = sorted(category_actions,
                                   key=lambda a: abs(self.data['mean_diff_at_question'].get(a, 0)),
                                   reverse=True)
            sorted_actions.extend(category_sorted)

        # Build matrix
        data_matrix = []
        for action in sorted_actions:
            row = [self.data[tp].get(action, 0) for tp in timepoints]
            data_matrix.append(row)

        data_matrix = np.array(data_matrix)

        # Create heatmap
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=-2.5, vmax=2.5)

        # Set ticks
        ax.set_xticks(np.arange(len(timepoints)))
        ax.set_yticks(np.arange(len(sorted_actions)))

        ax.set_xticklabels(['At Question', 'After True', 'After Wrong'], fontsize=11, fontweight='bold')
        ax.set_yticklabels(sorted_actions, fontsize=8)

        # Add category separators
        y_pos = 0
        for category in sorted(self.categories.keys()):
            num_actions = len(self.categories[category])
            if y_pos > 0:
                ax.axhline(y=y_pos - 0.5, color='black', linewidth=2)

            # Add category label
            ax.text(-0.7, y_pos + num_actions/2 - 0.5, category.upper(),
                   rotation=90, va='center', ha='center',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=self._get_category_color(category),
                           alpha=0.7, edgecolor='black'))

            y_pos += num_actions

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Layer Count Difference (Steered - Baseline)',
                      fontsize=11, fontweight='bold')

        # Title
        ax.set_title('Comprehensive Heatmap: All 45 Cognitive Actions × Timepoints\n' +
                    f'Grouped by Category (n={self.data["num_samples"]} samples)',
                    fontsize=14, fontweight='bold', pad=15)

        plt.tight_layout()

        output_path = self.output_dir / '05_comprehensive_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_diverging_bars(self):
        """
        Visualization 6: Diverging bar charts for positive/negative effects
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 18))

        timepoints = [
            ('mean_diff_at_question', 'At Question'),
            ('mean_diff_after_true', 'After True Answer'),
            ('mean_diff_after_wrong', 'After Wrong Answer')
        ]

        for ax, (key, title) in zip(axes, timepoints):
            # Get all actions sorted by value
            diffs = self.data[key]
            sorted_items = sorted(diffs.items(), key=lambda x: x[1])

            actions = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]

            # Create colors based on positive/negative and category
            colors = []
            for action in actions:
                base_color = self._get_category_color(self.action_categories[action])
                colors.append(base_color)

            # Create diverging bar chart
            y_pos = np.arange(len(actions))
            bars = ax.barh(y_pos, values, color=colors, alpha=0.8,
                          edgecolor='black', linewidth=0.5)

            # Styling
            ax.set_yticks(y_pos)
            ax.set_yticklabels(actions, fontsize=7)
            ax.set_xlabel('Layer Count Difference (Steered - Baseline)',
                         fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
            ax.grid(axis='x', alpha=0.3, linestyle='--')

            # Add annotations for extreme values
            max_idx = values.index(max(values))
            min_idx = values.index(min(values))

            if values[max_idx] > 0.1:
                ax.annotate(f'{values[max_idx]:.2f}',
                          xy=(values[max_idx], max_idx),
                          xytext=(5, 0), textcoords='offset points',
                          fontsize=8, fontweight='bold', color='darkgreen')

            if values[min_idx] < -0.1:
                ax.annotate(f'{values[min_idx]:.2f}',
                          xy=(values[min_idx], min_idx),
                          xytext=(-5, 0), textcoords='offset points',
                          ha='right', fontsize=8, fontweight='bold', color='darkred')

        plt.suptitle('Diverging Bar Charts: All Cognitive Actions\n' +
                    'Sorted by Steering Effect (Most Negative to Most Positive)',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / '06_diverging_bars.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_scatter_analysis(self):
        """
        Visualization 7: Scatter plot analysis comparing different timepoints
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        # Plot 1: At Question vs After True
        ax = axes[0, 0]
        x = [self.data['mean_diff_at_question'].get(a, 0) for a in self.all_actions]
        y = [self.data['mean_diff_after_true'].get(a, 0) for a in self.all_actions]
        colors = [self._get_category_color(self.action_categories[a]) for a in self.all_actions]

        ax.scatter(x, y, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Add diagonal
        lims = [min(min(x), min(y)), max(max(x), max(y))]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5)

        # Add correlation
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_xlabel('At Question', fontsize=11, fontweight='bold')
        ax.set_ylabel('After True Answer', fontsize=11, fontweight='bold')
        ax.set_title('At Question vs After True Answer', fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Plot 2: At Question vs After Wrong
        ax = axes[0, 1]
        y = [self.data['mean_diff_after_wrong'].get(a, 0) for a in self.all_actions]

        ax.scatter(x, y, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5)

        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_xlabel('At Question', fontsize=11, fontweight='bold')
        ax.set_ylabel('After Wrong Answer', fontsize=11, fontweight='bold')
        ax.set_title('At Question vs After Wrong Answer', fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Plot 3: After True vs After Wrong
        ax = axes[1, 0]
        x = [self.data['mean_diff_after_true'].get(a, 0) for a in self.all_actions]
        y = [self.data['mean_diff_after_wrong'].get(a, 0) for a in self.all_actions]

        ax.scatter(x, y, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        lims = [min(min(x), min(y)), max(max(x), max(y))]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5)

        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_xlabel('After True Answer', fontsize=11, fontweight='bold')
        ax.set_ylabel('After Wrong Answer', fontsize=11, fontweight='bold')
        ax.set_title('After True vs After Wrong Answer', fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Plot 4: Quadrant analysis (At Question vs mean of after answers)
        ax = axes[1, 1]
        x = [self.data['mean_diff_at_question'].get(a, 0) for a in self.all_actions]
        y = [(self.data['mean_diff_after_true'].get(a, 0) +
              self.data['mean_diff_after_wrong'].get(a, 0)) / 2
             for a in self.all_actions]

        scatter = ax.scatter(x, y, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Label quadrants
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.text(xlim[1]*0.7, ylim[1]*0.7, 'Strong &\nPersistent',
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(xlim[0]*0.7, ylim[1]*0.7, 'Grows After\nQuestion',
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        ax.text(xlim[0]*0.7, ylim[0]*0.7, 'Weak &\nPersistent',
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        ax.text(xlim[1]*0.7, ylim[0]*0.7, 'Weakens After\nQuestion',
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        ax.set_xlabel('Effect at Question', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Effect After Answers', fontsize=11, fontweight='bold')
        ax.set_title('Quadrant Analysis: Initial vs Sustained Effects', fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=1.5)
        ax.grid(True, alpha=0.3)

        plt.suptitle('Scatter Analysis: Relationship Between Timepoints',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / '07_scatter_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_distributions(self):
        """
        Visualization 8: Distribution analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Histogram of all effects at question
        ax = axes[0, 0]
        values = list(self.data['mean_diff_at_question'].values())

        ax.hist(values, bins=30, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero effect')
        ax.axvline(np.mean(values), color='green', linestyle='--', linewidth=2,
                  label=f'Mean = {np.mean(values):.3f}')
        ax.axvline(np.median(values), color='orange', linestyle='--', linewidth=2,
                  label=f'Median = {np.median(values):.3f}')

        ax.set_xlabel('Layer Count Difference', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of Effects at Question', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Plot 2: Box plots by timepoint
        ax = axes[0, 1]

        data_by_timepoint = [
            list(self.data['mean_diff_at_question'].values()),
            list(self.data['mean_diff_after_true'].values()),
            list(self.data['mean_diff_after_wrong'].values())
        ]

        bp = ax.boxplot(data_by_timepoint, labels=['At Question', 'After True', 'After Wrong'],
                       patch_artist=True, showmeans=True, meanline=True)

        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Layer Count Difference', fontsize=11, fontweight='bold')
        ax.set_title('Distribution Comparison Across Timepoints', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Plot 3: Cumulative distribution
        ax = axes[1, 0]

        for key, label, color in [
            ('mean_diff_at_question', 'At Question', 'blue'),
            ('mean_diff_after_true', 'After True', 'green'),
            ('mean_diff_after_wrong', 'After Wrong', 'red')
        ]:
            values = sorted(self.data[key].values())
            y = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, y, label=label, linewidth=2, color=color, alpha=0.7)

        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Layer Count Difference', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
        ax.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Category-wise distributions
        ax = axes[1, 1]

        category_data = []
        category_labels = []

        for category in sorted(self.categories.keys()):
            actions = self.categories[category]
            values = [self.data['mean_diff_at_question'].get(a, 0) for a in actions]
            category_data.append(values)
            category_labels.append(f'{category.title()}\n(n={len(values)})')

        bp = ax.boxplot(category_data, labels=category_labels,
                       patch_artist=True, showmeans=True)

        # Color by category
        for patch, category in zip(bp['boxes'], sorted(self.categories.keys())):
            patch.set_facecolor(self._get_category_color(category))
            patch.set_alpha(0.7)

        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Layer Count Difference (At Question)', fontsize=11, fontweight='bold')
        ax.set_title('Distribution by Category', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

        plt.suptitle('Statistical Distributions', fontsize=15, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / '08_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_top_changes_summary(self):
        """
        Visualization 9: Summary of top changes with annotations
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Top increases
        ax1 = fig.add_subplot(gs[0, :2])
        top_increases = sorted(self.data['mean_diff_at_question'].items(),
                              key=lambda x: x[1], reverse=True)[:15]

        actions = [t[0] for t in top_increases]
        values = [t[1] for t in top_increases]
        colors = [self._get_category_color(self.action_categories[a]) for a in actions]

        bars = ax1.barh(range(len(actions)), values, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=0.5)
        ax1.set_yticks(range(len(actions)))
        ax1.set_yticklabels(actions, fontsize=10)
        ax1.set_xlabel('Layer Count Increase', fontsize=11, fontweight='bold')
        ax1.set_title('Top 15 Increased Cognitive Actions (At Question)',
                     fontsize=13, fontweight='bold', pad=10)
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax1.text(val, i, f'  {val:.2f}', va='center', fontsize=9, fontweight='bold')

        # Top decreases
        ax2 = fig.add_subplot(gs[1, :2])
        top_decreases = sorted(self.data['mean_diff_at_question'].items(),
                              key=lambda x: x[1])[:15]

        actions = [t[0] for t in top_decreases]
        values = [t[1] for t in top_decreases]
        colors = [self._get_category_color(self.action_categories[a]) for a in actions]

        bars = ax2.barh(range(len(actions)), values, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=0.5)
        ax2.set_yticks(range(len(actions)))
        ax2.set_yticklabels(actions, fontsize=10)
        ax2.set_xlabel('Layer Count Decrease', fontsize=11, fontweight='bold')
        ax2.set_title('Top 15 Decreased Cognitive Actions (At Question)',
                     fontsize=13, fontweight='bold', pad=10)
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax2.text(val, i, f'{val:.2f}  ', va='center', ha='right',
                    fontsize=9, fontweight='bold')

        # Stats summary
        ax3 = fig.add_subplot(gs[:2, 2])
        ax3.axis('off')

        stats_text = f"""
EVALUATION SUMMARY

Sample Size: {self.data['num_samples']}

ACCURACY
Baseline:  {self.data['baseline_accuracy']:.2%}
Steered:   {self.data['steered_accuracy']:.2%}
Improvement: {self.data['accuracy_improvement']:+.2%}
Samples improved: {self.data['num_improved']}

PROBABILITIES
Baseline:
  p(true):  {self.data['baseline_prob_true_avg']:.4f}
  p(wrong): {self.data['baseline_prob_wrong_avg']:.4f}

Steered:
  p(true):  {self.data['steered_prob_true_avg']:.4f}
  p(wrong): {self.data['steered_prob_wrong_avg']:.4f}

COGNITIVE ACTIONS
Total: {len(self.all_actions)}

By Category:
"""
        for category in sorted(self.categories.keys()):
            stats_text += f"  {category.title()}: {len(self.categories[category])}\n"

        ax3.text(0.1, 0.95, stats_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5,
                         edgecolor='black', linewidth=2))

        # Comparison table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')

        # Create table data
        table_data = [['Cognitive Action', 'Category', 'At Question', 'After True', 'After Wrong', 'Mean Abs']]

        # Get top actions by mean absolute effect
        mean_abs = {}
        for action in self.all_actions:
            mean_abs[action] = np.mean([
                abs(self.data['mean_diff_at_question'].get(action, 0)),
                abs(self.data['mean_diff_after_true'].get(action, 0)),
                abs(self.data['mean_diff_after_wrong'].get(action, 0))
            ])

        top_actions = sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)[:12]

        for action, _ in top_actions:
            row = [
                action,
                self.action_categories[action][:4].upper(),
                f"{self.data['mean_diff_at_question'].get(action, 0):+.2f}",
                f"{self.data['mean_diff_after_true'].get(action, 0):+.2f}",
                f"{self.data['mean_diff_after_wrong'].get(action, 0):+.2f}",
                f"{mean_abs[action]:.2f}"
            ]
            table_data.append(row)

        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.25, 0.1, 0.13, 0.13, 0.13, 0.1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color rows by category
        for i, (action, _) in enumerate(top_actions, 1):
            category = self.action_categories[action]
            color = self._get_category_color(category)
            table[(i, 1)].set_facecolor(color)
            table[(i, 1)].set_alpha(0.5)

        plt.suptitle('Top Changes Summary', fontsize=16, fontweight='bold')

        output_path = self.output_dir / '09_top_changes_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_category_grid(self):
        """
        Visualization 10: Category comparison grid
        """
        categories = sorted(self.categories.keys())
        n_cats = len(categories)

        fig, axes = plt.subplots(n_cats, 3, figsize=(18, 4 * n_cats))

        if n_cats == 1:
            axes = axes.reshape(1, -1)

        timepoints = [
            ('mean_diff_at_question', 'At Question'),
            ('mean_diff_after_true', 'After True'),
            ('mean_diff_after_wrong', 'After Wrong')
        ]

        for row, category in enumerate(categories):
            actions = sorted(self.categories[category])
            color = self._get_category_color(category)

            for col, (key, title) in enumerate(timepoints):
                ax = axes[row, col]

                # Get values for this category
                values = [self.data[key].get(action, 0) for action in actions]

                # Sort by value
                sorted_pairs = sorted(zip(actions, values), key=lambda x: abs(x[1]), reverse=True)
                sorted_actions = [p[0] for p in sorted_pairs]
                sorted_values = [p[1] for p in sorted_pairs]

                # Plot
                bars = ax.barh(range(len(sorted_actions)), sorted_values,
                             color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

                ax.set_yticks(range(len(sorted_actions)))
                ax.set_yticklabels(sorted_actions, fontsize=8)
                ax.set_xlabel('Layer Count Diff', fontsize=9, fontweight='bold')
                ax.axvline(0, color='black', linestyle='--', linewidth=1)
                ax.grid(axis='x', alpha=0.3)

                # Title
                if row == 0:
                    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

                # Category label on left
                if col == 0:
                    ax.set_ylabel(f'{category.upper()}\n({len(actions)} actions)',
                                fontsize=10, fontweight='bold', rotation=0,
                                ha='right', va='center', labelpad=40)

        plt.suptitle('Category Comparison Grid: All Actions by Category and Timepoint',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / '10_category_grid.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    # =========================================================================
    # PAPER-SPECIFIC VISUALIZATIONS
    # =========================================================================

    def viz_radar_baseline_vs_steered_paper(self):
        """
        Paper Fig: Radar charts showing baseline vs steered for each category
        """
        if not self.raw_data:
            print("   ⚠ Skipping: Requires raw CSV data")
            return

        categories = sorted(self.categories.keys())

        # Calculate mean effects per category for each timepoint
        timepoints = [
            ('at_question', 'At Question'),
            ('after_true', 'After True'),
            ('after_wrong', 'After Wrong')
        ]

        fig = plt.figure(figsize=(18, 6))

        for idx, (time_key, title) in enumerate(timepoints):
            ax = fig.add_subplot(1, 3, idx + 1, projection='polar')

            # Calculate baseline and steered means for each category
            baseline_values = []
            steered_values = []

            for category in categories:
                actions = self.categories[category]

                baseline_cat_vals = []
                steered_cat_vals = []

                for action in actions:
                    for row in self.raw_data:
                        baseline_cat_vals.append(
                            row[f'baseline_activations_{time_key}'].get(action, 0)
                        )
                        steered_cat_vals.append(
                            row[f'steered_activations_{time_key}'].get(action, 0)
                        )

                baseline_values.append(np.mean(baseline_cat_vals) if baseline_cat_vals else 0)
                steered_values.append(np.mean(steered_cat_vals) if steered_cat_vals else 0)

            # Number of variables
            num_vars = len(categories)

            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

            # Complete the circle
            baseline_values += baseline_values[:1]
            steered_values += steered_values[:1]
            angles += angles[:1]

            # Plot baseline and steered
            ax.plot(angles, baseline_values, 'o-', linewidth=2, color='#3498db',
                   label='Baseline', alpha=0.8)
            ax.fill(angles, baseline_values, alpha=0.15, color='#3498db')

            ax.plot(angles, steered_values, 's-', linewidth=2, color='#e74c3c',
                   label='Steered', alpha=0.8)
            ax.fill(angles, steered_values, alpha=0.15, color='#e74c3c')

            # Fix axis to go in the right order
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # Draw axis lines for each angle and label
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([cat.title() for cat in categories], fontsize=10)

            # Set title
            ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

            # Add grid and legend
            ax.grid(True)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

        plt.suptitle('Category Comparison: Baseline vs Steered', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'paper_radar_baseline_vs_steered.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_category_analysis_paper(self):
        """
        Paper Fig: Simplified category analysis with just top 2 charts
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        timepoints = [
            ('mean_diff_at_question', 'At Question'),
            ('mean_diff_after_true', 'After True Answer'),
            ('mean_diff_after_wrong', 'After Wrong Answer')
        ]

        # Calculate category averages
        category_stats = defaultdict(lambda: defaultdict(list))

        for category, actions in self.categories.items():
            for key, _ in timepoints:
                for action in actions:
                    if action in self.data[key]:
                        category_stats[category][key].append(self.data[key][action])

        # Plot 1: Average effect per category across timepoints
        ax = axes[0]
        categories = sorted(category_stats.keys())
        x = np.arange(len(timepoints))
        width = 0.15

        for i, category in enumerate(categories):
            means = [np.mean(category_stats[category][key]) if category_stats[category][key] else 0
                    for key, _ in timepoints]
            offset = (i - len(categories)/2) * width
            bars = ax.bar(x + offset, means, width,
                         label=category.title(),
                         color=self._get_category_color(category),
                         alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Timepoint', fontweight='bold', fontsize=12)
        ax.set_ylabel('Mean Layer Count Difference', fontweight='bold', fontsize=12)
        ax.set_title('Mean Steering Effect by Category', fontweight='bold', pad=10, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([t[1] for t in timepoints], rotation=15, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        # Plot 2: Distribution of effects per category (at answer level)
        ax = axes[1]
        data_for_violin = []
        labels_for_violin = []

        for category in categories:
            # Combine both correct and incorrect answer effects
            values = (category_stats[category]['mean_diff_after_true'] +
                     category_stats[category]['mean_diff_after_wrong'])
            if values:
                data_for_violin.append(values)
                labels_for_violin.append(f'{category.title()}\\n(n={len(values)})')

        parts = ax.violinplot(data_for_violin, positions=range(len(data_for_violin)),
                             showmeans=True, showmedians=True)

        # Color violin plots
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self._get_category_color(categories[i]))
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(labels_for_violin)))
        ax.set_xticklabels(labels_for_violin, fontsize=10)
        ax.set_ylabel('Layer Count Difference', fontweight='bold', fontsize=12)
        ax.set_title('Distribution of Effects by Category (At Answer)', fontweight='bold',
                    pad=10, fontsize=13)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Category-Based Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'paper_category_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_complete_action_comparison_nonzero_paper(self):
        """
        Paper Fig: Complete comparison of cognitive actions (excluding zeros)
        """
        fig, axes = plt.subplots(1, 3, figsize=(24, 14))

        timepoints = [
            ('mean_diff_at_question', 'At Question'),
            ('mean_diff_after_true', 'After True Answer'),
            ('mean_diff_after_wrong', 'After Wrong Answer')
        ]

        for ax, (key, title) in zip(axes, timepoints):
            # Get all actions with non-zero differences
            diffs = {action: diff for action, diff in self.data[key].items()
                    if abs(diff) > 0.001}  # Filter out near-zero values

            if not diffs:
                ax.text(0.5, 0.5, 'No non-zero values', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.axis('off')
                continue

            sorted_actions = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)

            actions = [a[0] for a in sorted_actions]
            values = [a[1] for a in sorted_actions]

            # Color by category
            colors = [self._get_category_color(self.action_categories[action])
                     for action in actions]

            # Create horizontal bar chart
            y_pos = np.arange(len(actions))
            bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(actions, fontsize=8)
            ax.set_xlabel('Layer Count Difference (Steered - Baseline)', fontsize=10, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.grid(axis='x', alpha=0.3)

            # Add value labels for significant changes
            for i, (action, val) in enumerate(sorted_actions[:10]):
                if abs(val) > 0.01:
                    ax.text(val, i, f' {val:.2f}',
                           va='center', ha='left' if val > 0 else 'right',
                           fontsize=7, fontweight='bold')

        # Create legend for categories
        category_colors = {cat: self._get_category_color(cat)
                          for cat in set(self.action_categories.values())}
        legend_elements = [plt.Rectangle((0,0),1,1, fc=color, alpha=0.7, edgecolor='black', linewidth=0.5,
                                        label=f'{cat.title()} ({len(self.categories[cat])})')
                          for cat, color in sorted(category_colors.items())]
        fig.legend(handles=legend_elements, loc='upper center', ncol=6,
                  fontsize=10, frameon=True, fancybox=True, shadow=True)

        plt.suptitle('Cognitive Action Comparison (Non-Zero Actions Only)\\n' +
                    f'Steering Effect Across All Timepoints (n={self.data["num_samples"]})',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        output_path = self.output_dir / 'paper_complete_comparison_nonzero.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def viz_comprehensive_heatmap_nonzero_paper(self):
        """
        Paper Fig: Comprehensive heatmap (excluding zero-value actions)
        """
        fig, ax = plt.subplots(figsize=(12, 16))

        # Prepare data matrix
        timepoints = [
            'mean_diff_at_question',
            'mean_diff_after_true',
            'mean_diff_after_wrong'
        ]

        # Filter actions that have at least one non-zero value
        nonzero_actions = []
        for action in self.all_actions:
            has_nonzero = any(abs(self.data[tp].get(action, 0)) > 0.001
                            for tp in timepoints)
            if has_nonzero:
                nonzero_actions.append(action)

        # Sort actions by category then by mean absolute effect
        sorted_actions = []
        for category in sorted(self.categories.keys()):
            category_actions = [a for a in self.categories[category] if a in nonzero_actions]
            # Sort within category by absolute mean effect
            category_sorted = sorted(category_actions,
                                   key=lambda a: abs(self.data['mean_diff_at_question'].get(a, 0)),
                                   reverse=True)
            sorted_actions.extend(category_sorted)

        if not sorted_actions:
            print("   ⚠ No non-zero actions found")
            return

        # Build matrix
        data_matrix = []
        for action in sorted_actions:
            row = [self.data[tp].get(action, 0) for tp in timepoints]
            data_matrix.append(row)

        data_matrix = np.array(data_matrix)

        # Create heatmap
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=-2.5, vmax=2.5)

        # Set ticks
        ax.set_xticks(np.arange(len(timepoints)))
        ax.set_yticks(np.arange(len(sorted_actions)))

        ax.set_xticklabels(['At Question', 'After True', 'After Wrong'], fontsize=11, fontweight='bold')
        ax.set_yticklabels(sorted_actions, fontsize=8)

        # Add category separators
        y_pos = 0
        for category in sorted(self.categories.keys()):
            category_actions = [a for a in self.categories[category] if a in nonzero_actions]
            num_actions = len(category_actions)
            if num_actions == 0:
                continue

            if y_pos > 0:
                ax.axhline(y=y_pos - 0.5, color='black', linewidth=2)

            # Add category label
            ax.text(-0.7, y_pos + num_actions/2 - 0.5, category.upper(),
                   rotation=90, va='center', ha='center',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=self._get_category_color(category),
                           alpha=0.7, edgecolor='black'))

            y_pos += num_actions

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Layer Count Difference (Steered - Baseline)',
                      fontsize=11, fontweight='bold')

        # Title
        ax.set_title(f'Comprehensive Heatmap: Active Cognitive Actions × Timepoints\\n' +
                    f'Grouped by Category (n={self.data["num_samples"]} samples, {len(sorted_actions)} active actions)',
                    fontsize=14, fontweight='bold', pad=15)

        plt.tight_layout()

        output_path = self.output_dir / 'paper_heatmap_nonzero.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_path}")

    def create_paper_visualizations(self):
        """Generate paper-specific visualizations"""
        print("\\n📄 PAPER-SPECIFIC VISUALIZATIONS")
        print("-" * 80)

        print("P1. Creating baseline vs steered radar charts...")
        self.viz_radar_baseline_vs_steered_paper()

        print("P2. Creating simplified category analysis...")
        self.viz_category_analysis_paper()

        print("P3. Creating complete comparison (non-zero)...")
        self.viz_complete_action_comparison_nonzero_paper()

        print("P4. Creating comprehensive heatmap (non-zero)...")
        self.viz_comprehensive_heatmap_nonzero_paper()

        print()

    def _get_category_color(self, category: str) -> str:
        """Get consistent color for each category"""
        color_map = {
            'metacognitive': '#9b59b6',  # Purple
            'analytical': '#3498db',      # Blue
            'creative': '#f39c12',        # Orange
            'emotional': '#e74c3c',       # Red
            'memory': '#2ecc71',          # Green
            'other': '#95a5a6'            # Gray
        }
        return color_map.get(category, '#95a5a6')


def main():
    parser = argparse.ArgumentParser(
        description='Create comprehensive visualizations for cognitive action analysis'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='results/single_vector_test_summary.json',
        help='Path to summary JSON file'
    )
    parser.add_argument(
        '--raw-csv',
        type=str,
        default=None,
        help='Path to raw CSV file with baseline/steered data (enables baseline vs steered plots)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='visualizations',
        help='Output directory for visualizations'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("COGNITIVE ACTION VISUALIZATION SUITE")
    print("=" * 80)
    print(f"\nInput: {args.input}")
    if args.raw_csv:
        print(f"Raw CSV: {args.raw_csv}")
    print(f"Output: {args.output_dir}")

    # Create visualizer
    visualizer = CognitiveActionVisualizer(args.input, args.output_dir, args.raw_csv)

    # Generate all visualizations
    visualizer.create_all_visualizations()

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print(f"All plots saved to: {visualizer.output_dir}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()