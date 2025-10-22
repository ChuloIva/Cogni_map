"""
Theory of Mind Analysis & Visualization
=======================================

Comprehensive analysis and visualization tools for ToM experiments.

Visualizations:
1. Heatmaps of cognitive actions by task type
2. Token-level activation timelines
3. Differential activation (ToM vs control) plots
4. Network graphs of cognitive action co-occurrence
5. Layer preference analysis
6. Accuracy vs. cognitive signature correlation
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import sys

# Add paths
PROBES_PATH = Path(__file__).parent.parent / "probes"
sys.path.insert(0, str(PROBES_PATH))

from tom_tasks import ToMTask, ToMTaskType
from tom_inference import ToMCognitiveSignature, ToMExperimentResult


class ToMAnalyzer:
    """Analyze and visualize ToM experiment results"""

    def __init__(self, output_dir: Path):
        """
        Initialize analyzer

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        plt.rcParams['font.size'] = 10

        # ToM-specific actions
        self.tom_actions = [
            "perspective_taking",
            "hypothesis_generation",
            "metacognitive_monitoring",
            "distinguishing",
            "updating_beliefs",
            "counterfactual_reasoning",
            "suspending_judgment"
        ]

    def create_comprehensive_report(
        self,
        result: ToMExperimentResult,
        tasks: List[ToMTask]
    ):
        """
        Create comprehensive analysis report with all visualizations

        Args:
            result: ToMExperimentResult object
            tasks: Original list of ToMTask objects
        """
        print(f"\n{'='*80}")
        print(f"CREATING COMPREHENSIVE ToM ANALYSIS REPORT")
        print(f"{'='*80}\n")

        # 1. Heatmap: Cognitive Actions x Task Types
        print("1. Generating cognitive action heatmap...")
        self.plot_action_by_tasktype_heatmap(result)

        # 2. Differential activations
        print("2. Generating differential activation plots...")
        self.plot_differential_activations(result)

        # 3. Layer preference analysis
        print("3. Generating layer preference plots...")
        self.plot_layer_preferences(result)

        # 4. ToM specificity distribution
        print("4. Generating ToM specificity distribution...")
        self.plot_specificity_distribution(result)

        # 5. Expected vs. detected actions
        print("5. Generating expected action coverage...")
        self.plot_expected_coverage(result)

        # 6. Token-level activation timeline (sample tasks)
        print("6. Generating token activation timelines...")
        self.plot_token_timelines(result, n_samples=3)

        # 7. Cognitive action network
        print("7. Generating cognitive action network...")
        self.plot_action_cooccurrence_network(result)

        # 8. Summary statistics table
        print("8. Creating summary statistics...")
        self.create_summary_table(result, tasks)

        print(f"\n✓ All visualizations saved to {self.output_dir}")
        print(f"  Generated 8 visualizations + 1 summary report")

    def plot_action_by_tasktype_heatmap(self, result: ToMExperimentResult):
        """Heatmap showing which cognitive actions activate for each task type"""
        # Collect data: task_type x action -> average confidence
        task_types = list(set(sig.task_type for sig in result.task_results))
        all_actions = set()
        for sig in result.task_results:
            all_actions.update(sig.differential_actions.keys())

        # Focus on ToM-relevant actions
        relevant_actions = sorted([
            a for a in all_actions
            if a in self.tom_actions or "emotion" in a or "perspective" in a
        ])

        # Build matrix
        matrix = []
        for task_type in sorted(task_types, key=lambda x: x.value):
            row = []
            type_sigs = [s for s in result.task_results if s.task_type == task_type]

            for action in relevant_actions:
                # Average differential for this action in this task type
                diffs = [s.differential_actions.get(action, 0.0) for s in type_sigs]
                avg_diff = np.mean(diffs) if diffs else 0.0
                row.append(avg_diff)

            matrix.append(row)

        matrix = np.array(matrix)

        # Plot
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(
            matrix,
            xticklabels=relevant_actions,
            yticklabels=[t.value.replace('_', ' ').title() for t in sorted(task_types, key=lambda x: x.value)],
            cmap='RdYlGn',
            center=0,
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Differential Activation (Test - Control)'},
            ax=ax
        )
        ax.set_title('Cognitive Action Signatures by ToM Task Type', fontsize=16, fontweight='bold')
        ax.set_xlabel('Cognitive Action', fontsize=12)
        ax.set_ylabel('Task Type', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / '01_action_by_tasktype_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_differential_activations(self, result: ToMExperimentResult):
        """Plot top differential activations (ToM-specific)"""
        # Get top 20 ToM-specific actions
        top_actions = result.tom_action_rankings[:20]

        actions = [action for action, _ in top_actions]
        differentials = [diff for _, diff in top_actions]

        # Color by whether it's expected ToM action
        colors = ['#2ecc71' if action in self.tom_actions else '#3498db'
                  for action in actions]

        fig, ax = plt.subplots(figsize=(14, 10))
        y_pos = np.arange(len(actions))
        ax.barh(y_pos, differentials, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(actions)
        ax.set_xlabel('Average Differential Activation (Test - Control)', fontsize=12)
        ax.set_title('Top 20 ToM-Specific Cognitive Actions', fontsize=16, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Expected ToM Action'),
            Patch(facecolor='#3498db', label='Other Action')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(self.output_dir / '02_differential_activations.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_layer_preferences(self, result: ToMExperimentResult):
        """Plot which layers prefer ToM reasoning"""
        layers = sorted(result.tom_layer_preferences.keys())
        activations = [result.tom_layer_preferences[layer] for layer in layers]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(layers, activations, marker='o', linewidth=2, markersize=8, color='#e74c3c')
        ax.fill_between(layers, activations, alpha=0.3, color='#e74c3c')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Average Confidence', fontsize=12)
        ax.set_title('ToM Cognitive Action Activation by Layer', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(layers)

        # Highlight peak layers
        peak_layer = max(result.tom_layer_preferences.items(), key=lambda x: x[1])[0]
        ax.axvline(x=peak_layer, color='red', linestyle='--', alpha=0.5,
                   label=f'Peak Layer: {peak_layer}')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / '03_layer_preferences.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_specificity_distribution(self, result: ToMExperimentResult):
        """Distribution of ToM specificity scores"""
        specificities = [sig.tom_specificity_score for sig in result.task_results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram
        ax = axes[0]
        ax.hist(specificities, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(specificities), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(specificities):.3f}')
        ax.set_xlabel('ToM Specificity Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of ToM Specificity Scores', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Box plot by task type
        ax = axes[1]
        task_types = sorted(set(sig.task_type.value for sig in result.task_results))
        data_by_type = [
            [sig.tom_specificity_score for sig in result.task_results if sig.task_type.value == tt]
            for tt in task_types
        ]

        bp = ax.boxplot(data_by_type, labels=[tt.replace('_', '\n') for tt in task_types],
                        patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#9b59b6')
            patch.set_alpha(0.7)

        ax.set_ylabel('ToM Specificity Score', fontsize=12)
        ax.set_title('ToM Specificity by Task Type', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / '04_specificity_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_expected_coverage(self, result: ToMExperimentResult):
        """Plot expected action coverage"""
        coverages = [sig.expected_action_coverage for sig in result.task_results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Overall distribution
        ax = axes[0]
        ax.hist(coverages, bins=20, color='#1abc9c', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(coverages), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(coverages):.1%}')
        ax.set_xlabel('Expected Action Coverage', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Expected Action Coverage', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # By task type
        ax = axes[1]
        task_types = sorted(set(sig.task_type.value for sig in result.task_results))
        type_coverages = [
            np.mean([sig.expected_action_coverage for sig in result.task_results
                     if sig.task_type.value == tt])
            for tt in task_types
        ]

        bars = ax.bar(range(len(task_types)), type_coverages, color='#1abc9c', alpha=0.7)
        ax.set_xticks(range(len(task_types)))
        ax.set_xticklabels([tt.replace('_', '\n') for tt in task_types])
        ax.set_ylabel('Average Coverage', fontsize=12)
        ax.set_title('Expected Action Coverage by Task Type', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / '05_expected_coverage.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_token_timelines(self, result: ToMExperimentResult, n_samples: int = 3):
        """Plot token-level activation timelines for sample tasks"""
        # Select diverse samples
        samples = []
        for task_type in ToMTaskType:
            type_sigs = [s for s in result.task_results if s.task_type == task_type]
            if type_sigs:
                samples.append(type_sigs[0])
            if len(samples) >= n_samples:
                break

        fig, axes = plt.subplots(n_samples, 1, figsize=(16, 4*n_samples))
        if n_samples == 1:
            axes = [axes]

        for idx, sig in enumerate(samples):
            ax = axes[idx]

            # Get critical tokens
            if sig.critical_tokens:
                positions = [pos for pos, _, _ in sig.critical_tokens]
                tokens = [tok for _, tok, _ in sig.critical_tokens]
                action_counts = [len(acts) for _, _, acts in sig.critical_tokens]

                ax.scatter(positions, action_counts, s=100, alpha=0.7, color='#e74c3c')

                # Annotate some tokens
                for i, (pos, tok, count) in enumerate(zip(positions, tokens, action_counts)):
                    if i % 3 == 0:  # Annotate every 3rd token to avoid clutter
                        ax.annotate(tok, (pos, count), fontsize=8,
                                   xytext=(5, 5), textcoords='offset points')

            ax.set_xlabel('Token Position', fontsize=11)
            ax.set_ylabel('# ToM Actions Active', fontsize=11)
            ax.set_title(f'{sig.task_type.value.replace("_", " ").title()} - Task {sig.task_id}',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / '06_token_timelines.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_action_cooccurrence_network(self, result: ToMExperimentResult):
        """Network graph of cognitive action co-occurrence"""
        try:
            import networkx as nx
        except ImportError:
            print("  Warning: networkx not available, skipping network plot")
            return

        # Build co-occurrence matrix
        cooccurrence = defaultdict(Counter)

        for sig in result.task_results:
            active_actions = [pred.action_name for pred in sig.test_predictions if pred.is_active]
            # Count co-occurrences
            for i, action1 in enumerate(active_actions):
                for action2 in active_actions[i+1:]:
                    cooccurrence[action1][action2] += 1
                    cooccurrence[action2][action1] += 1

        # Build graph (focus on ToM actions)
        G = nx.Graph()

        # Add nodes
        for action in self.tom_actions:
            if action in cooccurrence:
                G.add_node(action)

        # Add edges (only strong co-occurrences)
        threshold = 3  # Minimum co-occurrence count
        for action1 in self.tom_actions:
            if action1 in cooccurrence:
                for action2, count in cooccurrence[action1].items():
                    if action2 in self.tom_actions and count >= threshold:
                        G.add_edge(action1, action2, weight=count)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 14))

        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#3498db',
                              alpha=0.7, ax=ax)

        # Draw edges (width based on weight)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*0.5 for w in weights],
                              alpha=0.5, ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

        ax.set_title('ToM Cognitive Action Co-occurrence Network', fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / '07_action_network.png', dpi=150, bbox_inches='tight')
        plt.close()

    def create_summary_table(self, result: ToMExperimentResult, tasks: List[ToMTask]):
        """Create text summary table"""
        summary_lines = []
        summary_lines.append("="*80)
        summary_lines.append("THEORY OF MIND EXPERIMENT - SUMMARY REPORT")
        summary_lines.append("="*80)
        summary_lines.append("")

        # Overall metrics
        summary_lines.append("OVERALL METRICS")
        summary_lines.append("-"*80)
        summary_lines.append(f"Total Tasks Analyzed: {len(result.task_results)}")
        summary_lines.append(f"Average ToM Specificity: {result.avg_tom_specificity:.3f}")
        summary_lines.append(f"Average Expected Coverage: {result.avg_expected_coverage:.1%}")
        summary_lines.append("")

        # By task type
        summary_lines.append("RESULTS BY TASK TYPE")
        summary_lines.append("-"*80)
        for task_type, stats in sorted(result.by_task_type.items()):
            summary_lines.append(f"\n{task_type.upper().replace('_', ' ')}:")
            summary_lines.append(f"  Tasks: {stats['n']}")
            summary_lines.append(f"  Avg Specificity: {stats['avg_specificity']:.3f}")
            summary_lines.append(f"  Avg Coverage: {stats['avg_coverage']:.1%}")

        # Top ToM actions
        summary_lines.append("\n\nTOP 15 ToM-SPECIFIC COGNITIVE ACTIONS")
        summary_lines.append("-"*80)
        for i, (action, diff) in enumerate(result.tom_action_rankings[:15], 1):
            marker = "***" if action in self.tom_actions else "   "
            summary_lines.append(f"{marker} {i:2d}. {action:35s} {diff:+.4f}")

        # Layer analysis
        summary_lines.append("\n\nLAYER ACTIVATION PROFILE")
        summary_lines.append("-"*80)
        for layer, conf in sorted(result.tom_layer_preferences.items()):
            bar_length = int(conf * 50)
            bar = "█" * bar_length
            summary_lines.append(f"Layer {layer:2d}: {bar:50s} {conf:.4f}")

        # Task examples
        summary_lines.append("\n\nEXAMPLE TASK ANALYSES")
        summary_lines.append("-"*80)

        for task_type in ToMTaskType:
            type_sigs = [s for s in result.task_results if s.task_type == task_type]
            if type_sigs:
                sig = type_sigs[0]
                task = next((t for t in tasks if t.task_id == sig.task_id), None)
                if task:
                    summary_lines.append(f"\n{task_type.value.upper().replace('_', ' ')}:")
                    summary_lines.append(f"Task ID: {sig.task_id}")
                    summary_lines.append(f"ToM Specificity: {sig.tom_specificity_score:.3f}")
                    summary_lines.append(f"Expected Coverage: {sig.expected_action_coverage:.1%}")
                    summary_lines.append(f"Expected Actions: {', '.join(sig.expected_actions)}")
                    summary_lines.append(f"Detected: {', '.join(sig.detected_expected_actions)}")

        summary_lines.append("\n" + "="*80)
        summary_lines.append("END OF REPORT")
        summary_lines.append("="*80)

        # Save
        summary_text = "\n".join(summary_lines)
        with open(self.output_dir / '08_summary_report.txt', 'w') as f:
            f.write(summary_text)

        print(summary_text)


def main():
    """Example usage"""
    # Load results
    results_path = Path("output/tom_experiments/example_results.json")

    if results_path.exists():
        print("Loading experiment results...")
        # Note: In real usage, you'd reconstruct ToMExperimentResult from JSON
        # For now, this is a placeholder
        print(f"Results file found at {results_path}")
        print("Run tom_inference.py first to generate results")
    else:
        print(f"No results found at {results_path}")
        print("Run tom_inference.py first to generate experiment results")


if __name__ == "__main__":
    main()
