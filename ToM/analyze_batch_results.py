#!/usr/bin/env python3
"""
Analyze Batch Evaluation Results

This script analyzes results from batch_evaluate_steering_intensities.py and generates:
- Intensity vs accuracy curves for each vector/condition pair
- Comparative visualizations across steering vectors
- Optimal intensity recommendations
- Summary statistics and reports
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, List
import argparse


class BatchResultAnalyzer:
    """Analyzes batch evaluation results and generates reports"""

    def __init__(self, batch_results_dir: str):
        """
        Initialize analyzer

        Args:
            batch_results_dir: Path to batch results directory
        """
        self.results_dir = Path(batch_results_dir)

        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

        print(f"\n{'='*80}")
        print(f"BATCH RESULTS ANALYZER")
        print(f"{'='*80}")
        print(f"Results directory: {self.results_dir}")
        print(f"{'='*80}\n")

        # Load metadata
        metadata_path = self.results_dir / "batch_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ Loaded metadata: {self.metadata['summary']['completed']} experiments")
        else:
            self.metadata = None
            print("⚠ No metadata file found")

        # Load all result files
        self.results_data = self._load_all_results()
        print(f"✓ Loaded {len(self.results_data)} result files\n")

    def _load_all_results(self) -> pd.DataFrame:
        """Load all summary JSON files into a DataFrame"""
        records = []

        summary_files = sorted(self.results_dir.glob("*_summary.json"))

        for summary_file in summary_files:
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)

                # Parse filename to extract parameters
                # Format: vectortype__condition__coeffXXX_summary.json
                filename = summary_file.stem.replace("_summary", "")
                parts = filename.split("__")

                if len(parts) >= 3:
                    vector_type = parts[0]
                    condition = parts[1]
                    intensity_str = parts[2].replace("coeff", "")

                    record = {
                        'vector_type': vector_type,
                        'condition': condition,
                        'intensity': int(intensity_str),
                        'baseline_accuracy': summary['baseline_accuracy'],
                        'steered_accuracy': summary['steered_accuracy'],
                        'accuracy_improvement': summary['accuracy_improvement'],
                        'num_samples': summary['num_samples'],
                        'num_improved': summary['num_improved'],
                        'filename': summary_file.name
                    }

                    records.append(record)

            except Exception as e:
                print(f"Warning: Failed to load {summary_file.name}: {e}")

        return pd.DataFrame(records)

    def generate_intensity_curves(self):
        """Generate intensity vs accuracy curves for each vector/condition pair"""
        if self.results_data.empty:
            print("No results data to plot")
            return

        # Group by vector and condition
        grouped = self.results_data.groupby(['vector_type', 'condition'])

        num_pairs = len(grouped)
        cols = 3
        rows = (num_pairs + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
        axes = axes.flatten() if num_pairs > 1 else [axes]

        for idx, ((vector_type, condition), group) in enumerate(grouped):
            ax = axes[idx]

            # Sort by intensity
            group_sorted = group.sort_values('intensity')

            # Plot baseline and steered accuracies
            ax.plot(
                group_sorted['intensity'],
                group_sorted['baseline_accuracy'],
                'o--',
                label='Baseline',
                color='#3498db',
                linewidth=2,
                markersize=6
            )
            ax.plot(
                group_sorted['intensity'],
                group_sorted['steered_accuracy'],
                'o-',
                label='Steered',
                color='#2ecc71',
                linewidth=2,
                markersize=6
            )

            # Find optimal intensity
            best_idx = group_sorted['steered_accuracy'].idxmax()
            best_intensity = group_sorted.loc[best_idx, 'intensity']
            best_accuracy = group_sorted.loc[best_idx, 'steered_accuracy']

            ax.axvline(
                x=best_intensity,
                color='red',
                linestyle=':',
                alpha=0.5,
                label=f'Optimal: {best_intensity}'
            )

            ax.set_xlabel('Steering Intensity', fontsize=10)
            ax.set_ylabel('Accuracy', fontsize=10)
            ax.set_title(
                f'{vector_type}\n{condition}',
                fontsize=10,
                fontweight='bold'
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1])

        # Hide unused subplots
        for idx in range(num_pairs, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        output_path = self.results_dir / "intensity_curves.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved intensity curves to: {output_path}")
        plt.close()

    def generate_improvement_heatmap(self):
        """Generate heatmap of accuracy improvements"""
        if self.results_data.empty:
            print("No results data to plot")
            return

        # Pivot data for heatmap
        pivot_data = self.results_data.pivot_table(
            values='accuracy_improvement',
            index='vector_type',
            columns='intensity',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(14, 8))

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Accuracy Improvement'},
            ax=ax
        )

        ax.set_title('Accuracy Improvement by Vector Type and Steering Intensity', fontsize=14, fontweight='bold')
        ax.set_xlabel('Steering Intensity', fontsize=12)
        ax.set_ylabel('Vector Type', fontsize=12)

        plt.tight_layout()

        output_path = self.results_dir / "improvement_heatmap.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved improvement heatmap to: {output_path}")
        plt.close()

    def generate_optimal_intensity_report(self) -> pd.DataFrame:
        """Generate report of optimal intensities for each vector/condition pair"""
        if self.results_data.empty:
            print("No results data available")
            return pd.DataFrame()

        # Find optimal intensity for each vector/condition pair
        optimal_records = []

        grouped = self.results_data.groupby(['vector_type', 'condition'])

        for (vector_type, condition), group in grouped:
            # Find intensity with highest steered accuracy
            best_idx = group['steered_accuracy'].idxmax()
            best_row = group.loc[best_idx]

            optimal_records.append({
                'vector_type': vector_type,
                'condition': condition,
                'optimal_intensity': best_row['intensity'],
                'baseline_accuracy': best_row['baseline_accuracy'],
                'best_steered_accuracy': best_row['steered_accuracy'],
                'max_improvement': best_row['accuracy_improvement'],
                'num_samples': best_row['num_samples']
            })

        optimal_df = pd.DataFrame(optimal_records)

        # Save to CSV
        output_path = self.results_dir / "optimal_intensities.csv"
        optimal_df.to_csv(output_path, index=False)
        print(f"✓ Saved optimal intensities report to: {output_path}")

        return optimal_df

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if self.results_data.empty:
            print("No results data available")
            return

        report_lines = [
            "="*80,
            "BATCH EVALUATION SUMMARY REPORT",
            "="*80,
            ""
        ]

        # Overall statistics
        report_lines.extend([
            "OVERALL STATISTICS:",
            f"  Total experiments: {len(self.results_data)}",
            f"  Unique vectors: {self.results_data['vector_type'].nunique()}",
            f"  Unique conditions: {self.results_data['condition'].nunique()}",
            f"  Intensity range: {self.results_data['intensity'].min()} - {self.results_data['intensity'].max()}",
            ""
        ])

        # Average improvements by vector type
        report_lines.append("AVERAGE IMPROVEMENT BY VECTOR TYPE:")
        vector_stats = self.results_data.groupby('vector_type').agg({
            'accuracy_improvement': ['mean', 'std', 'max'],
            'steered_accuracy': 'mean'
        }).round(4)

        for vector_type in vector_stats.index:
            mean_imp = vector_stats.loc[vector_type, ('accuracy_improvement', 'mean')]
            std_imp = vector_stats.loc[vector_type, ('accuracy_improvement', 'std')]
            max_imp = vector_stats.loc[vector_type, ('accuracy_improvement', 'max')]
            report_lines.append(
                f"  {vector_type:30s} | Mean: {mean_imp:+.4f} ± {std_imp:.4f} | Max: {max_imp:+.4f}"
            )

        report_lines.append("")

        # Best overall results
        report_lines.append("TOP 10 BEST STEERING RESULTS:")
        top_results = self.results_data.nlargest(10, 'steered_accuracy')

        for idx, row in top_results.iterrows():
            report_lines.append(
                f"  {row['vector_type']:20s} | {row['condition']:35s} | "
                f"Intensity: {row['intensity']:4d} | Accuracy: {row['steered_accuracy']:.4f}"
            )

        report_lines.append("")

        # Intensity trends
        report_lines.append("ACCURACY BY INTENSITY (AVERAGED ACROSS ALL VECTORS/CONDITIONS):")
        intensity_stats = self.results_data.groupby('intensity').agg({
            'steered_accuracy': 'mean',
            'accuracy_improvement': 'mean'
        }).round(4)

        for intensity in sorted(intensity_stats.index):
            acc = intensity_stats.loc[intensity, 'steered_accuracy']
            imp = intensity_stats.loc[intensity, 'accuracy_improvement']
            report_lines.append(f"  Intensity {intensity:4d}: Accuracy {acc:.4f} | Improvement {imp:+.4f}")

        report_lines.append("")
        report_lines.append("="*80)

        # Save report
        report_text = "\n".join(report_lines)
        output_path = self.results_dir / "summary_report.txt"

        with open(output_path, 'w') as f:
            f.write(report_text)

        print(f"✓ Saved summary report to: {output_path}")

        # Also print to console
        print("\n" + report_text)

    def generate_all_analyses(self):
        """Run all analysis and visualization functions"""
        print("\nGenerating analyses...\n")

        self.generate_intensity_curves()
        self.generate_improvement_heatmap()
        optimal_df = self.generate_optimal_intensity_report()
        self.generate_summary_report()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print(f"All results saved to: {self.results_dir}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze batch evaluation results'
    )
    parser.add_argument(
        'results_dir',
        type=str,
        help='Path to batch results directory (e.g., batch_results/batch_20240101_120000)'
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = BatchResultAnalyzer(args.results_dir)

    # Run all analyses
    analyzer.generate_all_analyses()


if __name__ == '__main__':
    main()