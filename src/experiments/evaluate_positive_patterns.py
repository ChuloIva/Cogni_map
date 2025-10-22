"""
Comprehensive evaluation of positive_patterns.jsonl using dual probe inference

This script performs token-by-token and whole-string analysis on:
1. Positive thought patterns
2. Negative thought patterns
3. Transformation statements

Generates rich visualizations and comprehensive HTML reports.
"""

import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "probes"))

from gpu_utils import configure_amd_gpu
configure_amd_gpu()

import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import probe engines
from streaming_probe_inference import StreamingProbeInferenceEngine, StreamingPrediction, AggregatedPrediction
from universal_multi_layer_inference import UniversalMultiLayerInferenceEngine, UniversalPrediction
from visualization_suite import VisualizationSuite
from report_generator import HTMLReportGenerator


@dataclass
class PatternEntry:
    """Single entry from positive_patterns.jsonl"""
    positive_thought_pattern: str
    reference_negative_example: str
    reference_transformed_example: str
    cognitive_pattern_name: str
    cognitive_pattern_type: str
    pattern_description: str
    source_question: str
    model: str
    timestamp: str
    metadata: Dict

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class StatementAnalysis:
    """Analysis results for a single statement"""
    text: str
    statement_type: str  # 'positive', 'negative', 'transformed'

    # Streaming analysis (token-by-token)
    streaming_predictions: List[AggregatedPrediction]
    streaming_top_actions: List[Tuple[str, float]]  # (action, confidence)

    # Universal analysis (whole-string)
    universal_predictions: List[UniversalPrediction]
    universal_top_actions: List[Tuple[str, float]]  # (action, confidence)

    # Metadata
    num_tokens: int
    cognitive_pattern_name: str
    cognitive_pattern_type: str


@dataclass
class EntryComparison:
    """Comparison results for a single entry (positive vs negative vs transformed)"""
    entry_id: int
    cognitive_pattern_name: str
    cognitive_pattern_type: str

    positive_analysis: StatementAnalysis
    negative_analysis: StatementAnalysis
    transformed_analysis: StatementAnalysis

    # Comparative metrics
    action_differences: Dict[str, Dict[str, float]]  # action -> {pos, neg, trans, pos-neg, trans-neg}
    unique_to_positive: List[str]
    unique_to_negative: List[str]
    transformation_bridge_actions: List[str]  # Actions that appear in trans but not neg


class PositivePatternsEvaluator:
    """
    Comprehensive evaluator for positive_patterns.jsonl

    Performs dual analysis (streaming + universal) on all statement types
    and generates rich visualizations and reports.
    """

    def __init__(
        self,
        data_path: Path,
        probes_dir: Path,
        output_dir: Path,
        sample_size: Optional[int] = None,
        layer_range: Tuple[int, int] = (21, 30),
        threshold: float = 0.1,
        device: str = None
    ):
        """
        Initialize evaluator

        Args:
            data_path: Path to positive_patterns.jsonl
            probes_dir: Path to probes_binary directory
            output_dir: Where to save results
            sample_size: Number of entries to analyze (None = all)
            layer_range: (start, end) layer range for probes
            threshold: Confidence threshold for active predictions
            device: Device to run on (auto-detects if None)
        """
        self.data_path = Path(data_path)
        self.probes_dir = Path(probes_dir)
        self.output_dir = Path(output_dir)
        self.sample_size = sample_size
        self.layer_range = layer_range
        self.threshold = threshold

        if device is None:
            from gpu_utils import get_optimal_device
            device = get_optimal_device()
        self.device = device

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)

        print(f"Initializing PositivePatternsEvaluator")
        print(f"  Data: {self.data_path}")
        print(f"  Probes: {self.probes_dir}")
        print(f"  Output: {self.output_dir}")
        print(f"  Sample size: {sample_size if sample_size else 'all'}")
        print(f"  Layer range: {layer_range}")
        print(f"  Device: {device}\n")

        # Load data
        self.entries = self._load_data()
        print(f"✓ Loaded {len(self.entries)} entries\n")

        # Load shared model ONCE
        self._load_shared_model()

        # Initialize inference engines with shared model
        print("Initializing inference engines...")

        # Create streaming engine WITHOUT loading model
        from streaming_probe_inference import StreamingProbeInferenceEngine
        self.streaming_engine = StreamingProbeInferenceEngine.__new__(StreamingProbeInferenceEngine)
        self.streaming_engine.probes_base_dir = self.probes_dir
        self.streaming_engine.model_name = "google/gemma-3-4b-it"
        self.streaming_engine.device = self.device
        self.streaming_engine.verbose = False
        self.streaming_engine.layer_start, self.streaming_engine.layer_end = self.layer_range
        self.streaming_engine.layers_needed = list(range(self.layer_range[0], self.layer_range[1] + 1))

        # Share model and tokenizer
        self.streaming_engine.model = self.shared_model
        self.streaming_engine.tokenizer = self.shared_tokenizer

        # Load probes and mappings
        from dataset_utils import get_idx_to_action_mapping
        self.streaming_engine.idx_to_action = get_idx_to_action_mapping()
        self.streaming_engine.action_to_idx = {action: idx for idx, action in self.streaming_engine.idx_to_action.items()}
        self.streaming_engine.probes = self.streaming_engine._load_all_probes()

        print(f"✓ Streaming engine ready ({len(self.streaming_engine.probes)} probes)")

        # Create universal engine WITHOUT loading model
        from universal_multi_layer_inference import UniversalMultiLayerInferenceEngine
        self.universal_engine = UniversalMultiLayerInferenceEngine.__new__(UniversalMultiLayerInferenceEngine)
        self.universal_engine.probes_base_dir = self.probes_dir
        self.universal_engine.model_name = "google/gemma-3-4b-it"
        self.universal_engine.device = self.device
        self.universal_engine.layer_start, self.universal_engine.layer_end = self.layer_range
        self.universal_engine.layers = list(range(self.layer_range[0], self.layer_range[1] + 1))

        # Share model and tokenizer
        self.universal_engine.model = self.shared_model
        self.universal_engine.tokenizer = self.shared_tokenizer

        # Load probes and mappings
        self.universal_engine.idx_to_action = get_idx_to_action_mapping()
        self.universal_engine.action_to_idx = {action: idx for idx, action in self.universal_engine.idx_to_action.items()}
        self.universal_engine.probes = self.universal_engine._load_all_probes()

        print(f"✓ Universal engine ready ({len(self.universal_engine.probes)} probes)")
        print("\n✓ Both engines sharing same model instance\n")

        # Storage for results
        self.comparisons: List[EntryComparison] = []

    def _load_shared_model(self):
        """Load model once and share between engines"""
        print("Loading shared language model...")

        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
        from nnsight import LanguageModel

        config = AutoConfig.from_pretrained("google/gemma-3-4b-it")

        # Check if this is a VLM (has vision_config)
        if hasattr(config, 'vision_config'):
            print("Detected vision-language model. Loading text-only...")
            from transformers import Gemma3ForCausalLM
            base_model = Gemma3ForCausalLM.from_pretrained(
                "google/gemma-3-4b-it",
                device_map=self.device,
                torch_dtype="auto"
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-3-4b-it",
                device_map=self.device,
                torch_dtype="auto"
            )

        self.shared_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
        self.shared_model = LanguageModel(base_model, tokenizer=self.shared_tokenizer)

        print(f"✓ Loaded shared model\n")

    def _load_data(self) -> List[PatternEntry]:
        """Load and parse JSONL data"""
        entries = []
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f):
                if self.sample_size and i >= self.sample_size:
                    break
                data = json.loads(line)
                entries.append(PatternEntry.from_dict(data))
        return entries

    def analyze_statement(
        self,
        text: str,
        statement_type: str,
        cognitive_pattern_name: str,
        cognitive_pattern_type: str
    ) -> StatementAnalysis:
        """
        Analyze a single statement with both streaming and universal inference

        Args:
            text: Statement text
            statement_type: 'positive', 'negative', or 'transformed'
            cognitive_pattern_name: Pattern name for metadata
            cognitive_pattern_type: Pattern type for metadata

        Returns:
            StatementAnalysis object
        """
        # Streaming analysis (token-by-token)
        streaming_preds = self.streaming_engine.predict_streaming(
            text,
            top_k=len(self.streaming_engine.probes),  # Get all
            threshold=0.0,  # Get all, filter later
            show_realtime=False
        )

        # Aggregate by action
        aggregated = self.streaming_engine.aggregate_predictions(
            streaming_preds,
            threshold=self.threshold
        )

        # Extract top actions (streaming)
        streaming_top = [
            (pred.action_name, pred.max_confidence)
            for pred in aggregated[:10]
            if pred.is_active
        ]

        # Universal analysis (whole-string)
        universal_preds = self.universal_engine.predict_all(
            text,
            threshold=self.threshold,
            top_k=None  # Get all
        )

        # Extract top actions (universal)
        universal_top = [
            (pred.action_name, pred.confidence)
            for pred in universal_preds[:10]
            if pred.is_active
        ]

        # Count tokens
        inputs = self.streaming_engine.tokenizer(text, return_tensors="pt")
        num_tokens = inputs['input_ids'].shape[1]

        return StatementAnalysis(
            text=text,
            statement_type=statement_type,
            streaming_predictions=aggregated,
            streaming_top_actions=streaming_top,
            universal_predictions=universal_preds,
            universal_top_actions=universal_top,
            num_tokens=num_tokens,
            cognitive_pattern_name=cognitive_pattern_name,
            cognitive_pattern_type=cognitive_pattern_type
        )

    def compare_entry(self, entry: PatternEntry, entry_id: int) -> EntryComparison:
        """
        Analyze all three statement types for a single entry and compare

        Args:
            entry: PatternEntry object
            entry_id: Entry index

        Returns:
            EntryComparison object
        """
        # Analyze each statement type
        positive_analysis = self.analyze_statement(
            entry.positive_thought_pattern,
            "positive",
            entry.cognitive_pattern_name,
            entry.cognitive_pattern_type
        )

        negative_analysis = self.analyze_statement(
            entry.reference_negative_example,
            "negative",
            entry.cognitive_pattern_name,
            entry.cognitive_pattern_type
        )

        transformed_analysis = self.analyze_statement(
            entry.reference_transformed_example,
            "transformed",
            entry.cognitive_pattern_name,
            entry.cognitive_pattern_type
        )

        # Compute comparative metrics
        action_differences = self._compute_action_differences(
            positive_analysis,
            negative_analysis,
            transformed_analysis
        )

        # Find unique actions
        pos_actions = set(a for a, _ in positive_analysis.streaming_top_actions)
        neg_actions = set(a for a, _ in negative_analysis.streaming_top_actions)
        trans_actions = set(a for a, _ in transformed_analysis.streaming_top_actions)

        unique_to_positive = list(pos_actions - neg_actions)
        unique_to_negative = list(neg_actions - pos_actions)
        transformation_bridge = list(trans_actions - neg_actions)

        return EntryComparison(
            entry_id=entry_id,
            cognitive_pattern_name=entry.cognitive_pattern_name,
            cognitive_pattern_type=entry.cognitive_pattern_type,
            positive_analysis=positive_analysis,
            negative_analysis=negative_analysis,
            transformed_analysis=transformed_analysis,
            action_differences=action_differences,
            unique_to_positive=unique_to_positive,
            unique_to_negative=unique_to_negative,
            transformation_bridge_actions=transformation_bridge
        )

    def _compute_action_differences(
        self,
        pos: StatementAnalysis,
        neg: StatementAnalysis,
        trans: StatementAnalysis
    ) -> Dict[str, Dict[str, float]]:
        """Compute confidence differences for all actions"""
        # Collect all actions
        all_actions = set()

        for pred in pos.streaming_predictions:
            all_actions.add(pred.action_name)
        for pred in neg.streaming_predictions:
            all_actions.add(pred.action_name)
        for pred in trans.streaming_predictions:
            all_actions.add(pred.action_name)

        # Create confidence lookup
        pos_conf = {p.action_name: p.max_confidence for p in pos.streaming_predictions}
        neg_conf = {p.action_name: p.max_confidence for p in neg.streaming_predictions}
        trans_conf = {p.action_name: p.max_confidence for p in trans.streaming_predictions}

        # Compute differences
        differences = {}
        for action in all_actions:
            p = pos_conf.get(action, 0.0)
            n = neg_conf.get(action, 0.0)
            t = trans_conf.get(action, 0.0)

            differences[action] = {
                'pos': p,
                'neg': n,
                'trans': t,
                'pos_minus_neg': p - n,
                'trans_minus_neg': t - n,
                'pos_minus_trans': p - t
            }

        return differences

    def run_analysis(self):
        """Run complete analysis on all entries"""
        print("=" * 80)
        print("RUNNING COMPREHENSIVE ANALYSIS")
        print("=" * 80)
        print()

        # Analyze each entry
        for i, entry in enumerate(tqdm(self.entries, desc="Analyzing entries")):
            comparison = self.compare_entry(entry, i)
            self.comparisons.append(comparison)

        print(f"\n✓ Analyzed {len(self.comparisons)} entries\n")

    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics across all entries"""
        print("Generating summary statistics...")

        # Aggregate action differences across all entries
        action_stats = defaultdict(lambda: {
            'pos_conf': [],
            'neg_conf': [],
            'trans_conf': [],
            'pos_minus_neg': [],
            'trans_minus_neg': []
        })

        for comp in self.comparisons:
            for action, diffs in comp.action_differences.items():
                action_stats[action]['pos_conf'].append(diffs['pos'])
                action_stats[action]['neg_conf'].append(diffs['neg'])
                action_stats[action]['trans_conf'].append(diffs['trans'])
                action_stats[action]['pos_minus_neg'].append(diffs['pos_minus_neg'])
                action_stats[action]['trans_minus_neg'].append(diffs['trans_minus_neg'])

        # Compute statistics
        summary_data = []
        for action, stats in action_stats.items():
            summary_data.append({
                'action': action,
                'pos_mean': np.mean(stats['pos_conf']),
                'pos_std': np.std(stats['pos_conf']),
                'neg_mean': np.mean(stats['neg_conf']),
                'neg_std': np.std(stats['neg_conf']),
                'trans_mean': np.mean(stats['trans_conf']),
                'trans_std': np.std(stats['trans_conf']),
                'pos_minus_neg_mean': np.mean(stats['pos_minus_neg']),
                'pos_minus_neg_std': np.std(stats['pos_minus_neg']),
                'trans_minus_neg_mean': np.mean(stats['trans_minus_neg']),
                'trans_minus_neg_std': np.std(stats['trans_minus_neg']),
                'count': len(stats['pos_conf'])
            })

        df = pd.DataFrame(summary_data)
        df = df.sort_values('pos_minus_neg_mean', ascending=False)

        # Save to CSV
        output_path = self.output_dir / "action_summary_statistics.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Saved summary statistics to {output_path}\n")

        return df

    def export_detailed_results(self):
        """Export all results to CSV files"""
        print("Exporting detailed results...")

        # Statement-level data
        statement_data = []
        for comp in self.comparisons:
            for analysis in [comp.positive_analysis, comp.negative_analysis, comp.transformed_analysis]:
                for pred in analysis.streaming_predictions[:10]:  # Top 10
                    statement_data.append({
                        'entry_id': comp.entry_id,
                        'statement_type': analysis.statement_type,
                        'cognitive_pattern_name': comp.cognitive_pattern_name,
                        'cognitive_pattern_type': comp.cognitive_pattern_type,
                        'action': pred.action_name,
                        'max_confidence': pred.max_confidence,
                        'mean_confidence': pred.mean_confidence,
                        'layer_count': pred.layer_count,
                        'best_layer': pred.best_layer,
                        'num_tokens': analysis.num_tokens
                    })

        df_statements = pd.DataFrame(statement_data)
        output_path = self.output_dir / "statement_level_analysis.csv"
        df_statements.to_csv(output_path, index=False)
        print(f"✓ Saved statement-level data to {output_path}")

        # Comparison-level data
        comparison_data = []
        for comp in self.comparisons:
            comparison_data.append({
                'entry_id': comp.entry_id,
                'cognitive_pattern_name': comp.cognitive_pattern_name,
                'cognitive_pattern_type': comp.cognitive_pattern_type,
                'num_unique_to_positive': len(comp.unique_to_positive),
                'num_unique_to_negative': len(comp.unique_to_negative),
                'num_transformation_bridge': len(comp.transformation_bridge_actions),
                'unique_to_positive': ', '.join(comp.unique_to_positive[:5]),
                'unique_to_negative': ', '.join(comp.unique_to_negative[:5]),
                'transformation_bridge': ', '.join(comp.transformation_bridge_actions[:5])
            })

        df_comparisons = pd.DataFrame(comparison_data)
        output_path = self.output_dir / "entry_comparisons.csv"
        df_comparisons.to_csv(output_path, index=False)
        print(f"✓ Saved comparison data to {output_path}\n")

        return df_statements, df_comparisons


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate positive_patterns.jsonl with dual probe inference")
    parser.add_argument(
        "--data",
        type=str,
        default="data/positive_patterns.jsonl",
        help="Path to positive_patterns.jsonl"
    )
    parser.add_argument(
        "--probes",
        type=str,
        default="data/probes_binary",
        help="Path to probes_binary directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/positive_patterns_analysis",
        help="Output directory"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size (None = all)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Confidence threshold"
    )

    args = parser.parse_args()

    # Resolve paths
    data_path = PROJECT_ROOT / args.data
    probes_dir = PROJECT_ROOT / args.probes
    output_dir = PROJECT_ROOT / args.output

    # Create evaluator
    evaluator = PositivePatternsEvaluator(
        data_path=data_path,
        probes_dir=probes_dir,
        output_dir=output_dir,
        sample_size=args.sample,
        threshold=args.threshold
    )

    # Run analysis
    evaluator.run_analysis()

    # Generate results
    print("\n" + "=" * 80)
    print("GENERATING RESULTS")
    print("=" * 80)
    print()

    summary_df = evaluator.generate_summary_statistics()
    statement_df, comparison_df = evaluator.export_detailed_results()

    # Generate visualizations
    viz_suite = VisualizationSuite(output_dir)
    viz_suite.generate_all_visualizations(summary_df, statement_df)

    # Generate HTML report
    report_gen = HTMLReportGenerator(output_dir)
    report_path = report_gen.generate_report(
        summary_df=summary_df,
        statement_df=statement_df,
        comparison_df=comparison_df,
        num_entries=len(evaluator.entries),
        sample_size=args.sample
    )

    # Print top findings
    print("\n" + "=" * 80)
    print("TOP FINDINGS")
    print("=" * 80)
    print()

    print("Actions most associated with POSITIVE patterns:")
    print(summary_df.nlargest(10, 'pos_minus_neg_mean')[['action', 'pos_minus_neg_mean', 'pos_mean', 'neg_mean']].to_string(index=False))

    print("\n\nActions most associated with NEGATIVE patterns:")
    print(summary_df.nsmallest(10, 'pos_minus_neg_mean')[['action', 'pos_minus_neg_mean', 'pos_mean', 'neg_mean']].to_string(index=False))

    print("\n\nActions most associated with TRANSFORMATION:")
    print(summary_df.nlargest(10, 'trans_minus_neg_mean')[['action', 'trans_minus_neg_mean', 'trans_mean', 'neg_mean']].to_string(index=False))

    print("\n\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"HTML Report: {report_path}")
    print(f"Figures: {output_dir / 'figures'}")
    print(f"CSV Files: {output_dir}")
    print("\nOpen the HTML report in your browser for interactive visualizations!")


if __name__ == "__main__":
    main()
