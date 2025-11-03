#!/usr/bin/env python3
"""
Generate paper-specific visualizations for cognitive action analysis
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualize_cognitive_actions import CognitiveActionVisualizer

def main():
    # Paths
    summary_json = "ToM/results/Forward_false_1000_summary.json"
    raw_csv = "ToM/results/Forward_false_1000_raw.csv"
    output_dir = "ToM/visualizations/paper"

    print("\n" + "=" * 80)
    print("PAPER-SPECIFIC VISUALIZATIONS")
    print("=" * 80)
    print(f"\nInput JSON: {summary_json}")
    print(f"Raw CSV: {raw_csv}")
    print(f"Output: {output_dir}\n")

    # Create visualizer
    visualizer = CognitiveActionVisualizer(summary_json, output_dir, raw_csv)

    # Generate paper visualizations
    visualizer.create_paper_visualizations()

    print("\n" + "=" * 80)
    print("✓ PAPER VISUALIZATIONS COMPLETE!")
    print(f"All plots saved to: {output_dir}")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()