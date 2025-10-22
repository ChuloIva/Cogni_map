"""
Theory of Mind Inference Engine
================================

Extends StreamingProbeInferenceEngine to analyze cognitive processes
during ToM reasoning.

Key capabilities:
- Process ToM tasks and track cognitive actions
- Compare control vs. test conditions
- Identify ToM-specific cognitive signatures
- Analyze token-level activations at critical reasoning moments
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import json

# Add probes to path
PROBES_PATH = Path(__file__).parent.parent / "probes"
sys.path.insert(0, str(PROBES_PATH))

from streaming_probe_inference import (
    StreamingProbeInferenceEngine,
    AggregatedPrediction,
    StreamingPrediction
)
from tom_tasks import ToMTask, ToMTaskType, TaskDifficulty


@dataclass
class ToMCognitiveSignature:
    """Cognitive signature for a ToM task"""
    task_id: str
    task_type: ToMTaskType

    # Predictions for test scenario (requiring ToM)
    test_predictions: List[AggregatedPrediction]

    # Predictions for control scenario (no ToM required)
    control_predictions: List[AggregatedPrediction]

    # Differential activations (test - control)
    differential_actions: Dict[str, float]

    # Token-level analysis for critical moments
    critical_tokens: List[Tuple[int, str, List[str]]]  # (position, token, active_actions)

    # Expected vs. actual cognitive actions
    expected_actions: List[str]
    detected_expected_actions: List[str]
    unexpected_actions: List[str]

    # Metrics
    tom_specificity_score: float  # How much more ToM actions in test vs control
    expected_action_coverage: float  # % of expected actions detected


@dataclass
class ToMExperimentResult:
    """Results for a full ToM experiment"""
    task_results: List[ToMCognitiveSignature]

    # Aggregate statistics
    avg_tom_specificity: float
    avg_expected_coverage: float

    # By task type
    by_task_type: Dict[str, Dict[str, float]]

    # Most ToM-specific cognitive actions
    tom_action_rankings: List[Tuple[str, float]]  # (action, avg_differential)

    # Layer analysis
    tom_layer_preferences: Dict[int, float]  # Which layers activate for ToM


class ToMInferenceEngine:
    """
    Theory of Mind inference engine

    Uses cognitive action probes to analyze ToM reasoning patterns
    """

    def __init__(
        self,
        probes_base_dir: Path,
        model_name: str = "google/gemma-3-4b-it",
        device: str = None,
        verbose: bool = True
    ):
        """
        Initialize ToM inference engine

        Args:
            probes_base_dir: Base directory containing probes
            model_name: Language model to use
            device: Device (auto-detect if None)
            verbose: Print progress
        """
        self.verbose = verbose

        if verbose:
            print("Initializing Theory of Mind Inference Engine...")

        # Initialize base streaming engine
        self.engine = StreamingProbeInferenceEngine(
            probes_base_dir=probes_base_dir,
            model_name=model_name,
            device=device,
            verbose=verbose,
            layer_range=(21, 30)  # Focus on higher layers for complex reasoning
        )

        # ToM-specific cognitive actions (expected to be highly active)
        self.tom_actions = [
            "perspective_taking",
            "hypothesis_generation",
            "metacognitive_monitoring",
            "distinguishing",
            "updating_beliefs",
            "counterfactual_reasoning",
            "suspending_judgment",
            "emotion_perception",
            "emotion_understanding"
        ]

        if verbose:
            print(f"✓ ToM Inference Engine ready")
            print(f"  Tracking {len(self.tom_actions)} ToM-specific cognitive actions")

    def analyze_task(
        self,
        task: ToMTask,
        threshold: float = 0.1,
        show_realtime: bool = False,
        generate_answer: bool = True
    ) -> ToMCognitiveSignature:
        """
        Analyze a single ToM task

        Args:
            task: ToMTask to analyze
            threshold: Confidence threshold for active predictions
            show_realtime: Show real-time streaming output
            generate_answer: If True, generate model response; if False, just scan the question

        Returns:
            ToMCognitiveSignature with analysis results
        """
        if self.verbose and not show_realtime:
            print(f"\nAnalyzing {task.task_type.value} task: {task.task_id}")

        # Construct prompts
        if generate_answer:
            # Generate actual model response
            test_prompt = (
                f"You are reasoning about a theory of mind scenario. "
                f"Think step-by-step about what different people know and believe.\n\n"
                f"Scenario: {task.scenario}\n\n"
                f"Question: {task.question}\n\n"
                f"Provide your answer and explain your reasoning:"
            )
            control_prompt = (
                f"You are reasoning about a scenario.\n\n"
                f"Scenario: {task.control_scenario}\n\n"
                f"Question: {task.question}\n\n"
                f"Provide your answer and explain your reasoning:"
            )
        else:
            # Just scan the question text (old behavior)
            test_prompt = f"{task.scenario}\n\nQuestion: {task.question}"
            control_prompt = f"{task.control_scenario}\n\nQuestion: {task.question}"

        # Run inference on test scenario
        if self.verbose and not show_realtime:
            mode_str = " with generation" if generate_answer else ""
            print(f"  Running test scenario (ToM required){mode_str}...")

        if generate_answer:
            # Generate response while tracking cognitive actions
            test_response, test_preds_raw = self.engine.generate_with_cognitive_tracking(
                test_prompt,
                max_new_tokens=256,
                threshold=0.0,  # Get all predictions for proper aggregation
                show_realtime=show_realtime,
                temperature=0.7
            )
            if self.verbose:
                print(f"\n  Model's Answer: {test_response[:200]}{'...' if len(test_response) > 200 else ''}\n")
        else:
            # Get ALL predictions (not just top_k) for proper aggregation
            test_preds_raw = self.engine.predict_streaming(
                test_prompt,
                top_k=len(self.engine.probes),  # Get all predictions
                threshold=0.0,  # Get all, filter during aggregation
                show_realtime=show_realtime
            )

        # Aggregate predictions by action across layers (like Interactive_TUI.py)
        test_agg = self.engine.aggregate_predictions(test_preds_raw, threshold=threshold)

        # Run inference on control scenario
        if self.verbose and not show_realtime:
            mode_str = " with generation" if generate_answer else ""
            print(f"  Running control scenario (no ToM){mode_str}...")

        if generate_answer:
            control_response, control_preds_raw = self.engine.generate_with_cognitive_tracking(
                control_prompt,
                max_new_tokens=256,
                threshold=0.0,  # Get all predictions for proper aggregation
                show_realtime=False,  # Don't show control in realtime
                temperature=0.7
            )
        else:
            # Get ALL predictions (not just top_k) for proper aggregation
            control_preds_raw = self.engine.predict_streaming(
                control_prompt,
                top_k=len(self.engine.probes),  # Get all predictions
                threshold=0.0,  # Get all, filter during aggregation
                show_realtime=False  # Don't show control in realtime
            )

        # Aggregate predictions by action across layers (like Interactive_TUI.py)
        control_agg = self.engine.aggregate_predictions(control_preds_raw, threshold=threshold)

        # Calculate differential activations (test - control)
        # This now compares aggregated predictions across all layers
        differential = self._calculate_differential(test_agg, control_agg)

        # Identify critical tokens (tokens with high aggregated ToM action activation)
        # Pass aggregated predictions to analyze token-level patterns
        critical_tokens = self._find_critical_tokens(test_agg, threshold)

        # Compare expected vs. actual actions (using aggregated data)
        expected = set(task.expected_cognitive_actions)
        detected_actions = {pred.action_name for pred in test_agg if pred.is_active}
        detected_expected = list(expected & detected_actions)
        unexpected = list(detected_actions - expected)

        # Calculate metrics
        tom_specificity = self._calculate_tom_specificity(differential)
        expected_coverage = len(detected_expected) / len(expected) if expected else 0.0

        if self.verbose and not show_realtime:
            print(f"  ✓ ToM Specificity Score: {tom_specificity:.3f}")
            print(f"  ✓ Expected Action Coverage: {expected_coverage:.1%}")
            print(f"  ✓ Detected {len(detected_expected)}/{len(expected)} expected actions")

        return ToMCognitiveSignature(
            task_id=task.task_id,
            task_type=task.task_type,
            test_predictions=test_agg,
            control_predictions=control_agg,
            differential_actions=differential,
            critical_tokens=critical_tokens,
            expected_actions=list(expected),
            detected_expected_actions=detected_expected,
            unexpected_actions=unexpected,
            tom_specificity_score=tom_specificity,
            expected_action_coverage=expected_coverage
        )

    def analyze_task_suite(
        self,
        tasks: List[ToMTask],
        threshold: float = 0.1,
        save_path: Optional[Path] = None
    ) -> ToMExperimentResult:
        """
        Analyze a full suite of ToM tasks

        Args:
            tasks: List of ToMTask objects
            threshold: Confidence threshold
            save_path: Optional path to save results

        Returns:
            ToMExperimentResult with aggregate analysis
        """
        print(f"\n{'='*80}")
        print(f"THEORY OF MIND EXPERIMENT")
        print(f"{'='*80}")
        print(f"Analyzing {len(tasks)} ToM tasks...")

        results = []
        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] {task.task_type.value} - {task.task_id}")
            signature = self.analyze_task(task, threshold=threshold)
            results.append(signature)

        # Aggregate statistics
        experiment_result = self._aggregate_results(results)

        # Print summary
        self._print_summary(experiment_result)

        # Save if requested
        if save_path:
            self._save_results(experiment_result, save_path)

        return experiment_result

    def _calculate_differential(
        self,
        test_preds: List[AggregatedPrediction],
        control_preds: List[AggregatedPrediction]
    ) -> Dict[str, float]:
        """Calculate differential activation (test - control)"""
        test_dict = {pred.action_name: pred.max_confidence for pred in test_preds}
        control_dict = {pred.action_name: pred.max_confidence for pred in control_preds}

        differential = {}
        all_actions = set(test_dict.keys()) | set(control_dict.keys())

        for action in all_actions:
            test_conf = test_dict.get(action, 0.0)
            control_conf = control_dict.get(action, 0.0)
            differential[action] = test_conf - control_conf

        return differential

    def _find_critical_tokens(
        self,
        aggregated_predictions: List[AggregatedPrediction],
        threshold: float
    ) -> List[Tuple[int, str, List[str]]]:
        """
        Find tokens with high ToM action activation using aggregated predictions

        This analyzes token-level activations from aggregated (multi-layer) predictions,
        identifying critical moments where multiple ToM actions activate.

        Args:
            aggregated_predictions: List of AggregatedPrediction objects
            threshold: Confidence threshold for considering an action active

        Returns:
            List of (token_position, token_text, active_actions) for critical tokens
        """
        critical = []

        # Group predictions by token position, using aggregated max confidence
        token_activations = {}

        for agg_pred in aggregated_predictions:
            # Only consider ToM-specific actions
            if agg_pred.action_name not in self.tom_actions:
                continue

            # Only consider actions that are active (above threshold)
            if not agg_pred.is_active:
                continue

            # Look at token activations from the best layer for this action
            for layer_pred in agg_pred.layer_predictions:
                if layer_pred.layer == agg_pred.best_layer:
                    # Analyze token-level activations from this layer
                    for token_act in layer_pred.token_activations:
                        pos = token_act.token_position
                        if pos not in token_activations:
                            token_activations[pos] = {
                                'token': token_act.token_text,
                                'actions': set(),
                                'confidences': {}
                            }

                        # Use the aggregated max confidence for this action
                        if agg_pred.max_confidence >= threshold:
                            token_activations[pos]['actions'].add(agg_pred.action_name)
                            token_activations[pos]['confidences'][agg_pred.action_name] = agg_pred.max_confidence

        # Find tokens with multiple ToM actions (critical reasoning moments)
        for pos, data in sorted(token_activations.items()):
            if len(data['actions']) >= 2:  # At least 2 ToM actions
                # Sort actions by confidence for this token
                sorted_actions = sorted(
                    data['actions'],
                    key=lambda a: data['confidences'].get(a, 0.0),
                    reverse=True
                )
                critical.append((pos, data['token'], sorted_actions))

        return critical

    def _calculate_tom_specificity(self, differential: Dict[str, float]) -> float:
        """Calculate how ToM-specific the activations are"""
        tom_differential = [differential[action] for action in self.tom_actions if action in differential]

        if not tom_differential:
            return 0.0

        # Average positive differential for ToM actions
        positive_diffs = [d for d in tom_differential if d > 0]
        return np.mean(positive_diffs) if positive_diffs else 0.0

    def _aggregate_results(self, results: List[ToMCognitiveSignature]) -> ToMExperimentResult:
        """Aggregate results across all tasks"""
        # Overall metrics
        avg_specificity = np.mean([r.tom_specificity_score for r in results])
        avg_coverage = np.mean([r.expected_action_coverage for r in results])

        # By task type
        by_task_type = {}
        for task_type in ToMTaskType:
            type_results = [r for r in results if r.task_type == task_type]
            if type_results:
                by_task_type[task_type.value] = {
                    'n': len(type_results),
                    'avg_specificity': np.mean([r.tom_specificity_score for r in type_results]),
                    'avg_coverage': np.mean([r.expected_action_coverage for r in type_results])
                }

        # Action rankings (average differential across all tasks)
        action_diffs = {}
        for result in results:
            for action, diff in result.differential_actions.items():
                if action not in action_diffs:
                    action_diffs[action] = []
                action_diffs[action].append(diff)

        tom_rankings = []
        for action, diffs in action_diffs.items():
            avg_diff = np.mean(diffs)
            tom_rankings.append((action, avg_diff))

        tom_rankings.sort(key=lambda x: x[1], reverse=True)

        # Layer preferences
        layer_activations = {i: [] for i in range(21, 31)}
        for result in results:
            for pred in result.test_predictions:
                for layer in pred.layers:
                    layer_activations[layer].append(pred.max_confidence)

        layer_preferences = {
            layer: np.mean(acts) if acts else 0.0
            for layer, acts in layer_activations.items()
        }

        return ToMExperimentResult(
            task_results=results,
            avg_tom_specificity=avg_specificity,
            avg_expected_coverage=avg_coverage,
            by_task_type=by_task_type,
            tom_action_rankings=tom_rankings[:20],
            tom_layer_preferences=layer_preferences
        )

    def _print_summary(self, result: ToMExperimentResult):
        """Print experiment summary"""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*80}")

        print(f"\nOverall Metrics:")
        print(f"  Average ToM Specificity: {result.avg_tom_specificity:.3f}")
        print(f"  Average Expected Coverage: {result.avg_expected_coverage:.1%}")

        print(f"\nBy Task Type:")
        for task_type, stats in result.by_task_type.items():
            print(f"  {task_type}:")
            print(f"    Tasks: {stats['n']}")
            print(f"    Specificity: {stats['avg_specificity']:.3f}")
            print(f"    Coverage: {stats['avg_coverage']:.1%}")

        print(f"\nTop 10 ToM-Specific Cognitive Actions:")
        for i, (action, diff) in enumerate(result.tom_action_rankings[:10], 1):
            marker = "✓" if action in [
                "perspective_taking", "hypothesis_generation", "metacognitive_monitoring",
                "distinguishing", "counterfactual_reasoning"
            ] else " "
            print(f"  {marker} {i:2d}. {action:30s} {diff:+.3f}")

        print(f"\nLayer Activation Profile:")
        for layer, avg_conf in sorted(result.tom_layer_preferences.items()):
            bar_length = int(avg_conf * 50)
            bar = "█" * bar_length
            print(f"  Layer {layer}: {bar} {avg_conf:.3f}")

    def _save_results(self, result: ToMExperimentResult, save_path: Path):
        """Save results to JSON"""
        # Convert to serializable format
        results_dict = {
            'avg_tom_specificity': float(result.avg_tom_specificity),
            'avg_expected_coverage': float(result.avg_expected_coverage),
            'by_task_type': result.by_task_type,
            'tom_action_rankings': [
                {'action': action, 'differential': float(diff)}
                for action, diff in result.tom_action_rankings
            ],
            'tom_layer_preferences': {
                str(layer): float(conf)
                for layer, conf in result.tom_layer_preferences.items()
            },
            'task_results': [
                {
                    'task_id': sig.task_id,
                    'task_type': sig.task_type.value,
                    'tom_specificity_score': float(sig.tom_specificity_score),
                    'expected_action_coverage': float(sig.expected_action_coverage),
                    'expected_actions': sig.expected_actions,
                    'detected_expected_actions': sig.detected_expected_actions,
                    'unexpected_actions': sig.unexpected_actions,
                    'differential_actions': {k: float(v) for k, v in sig.differential_actions.items()},
                    'critical_tokens': [
                        {'position': pos, 'token': tok, 'actions': acts}
                        for pos, tok, acts in sig.critical_tokens
                    ]
                }
                for sig in result.task_results
            ]
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n✓ Results saved to {save_path}")


def main():
    """Example usage"""
    from tom_tasks import ToMTaskGenerator

    # Initialize
    engine = ToMInferenceEngine(
        probes_base_dir=Path("data/probes_binary"),
        model_name="google/gemma-3-4b-it",
        verbose=True
    )

    # Load tasks
    task_path = Path("data/tom_tasks/tom_task_suite.json")
    if task_path.exists():
        generator = ToMTaskGenerator()
        tasks = generator.load_tasks(task_path)
        print(f"\nLoaded {len(tasks)} ToM tasks")

        # Analyze a few example tasks (not all to save time)
        example_tasks = tasks[:5]  # First 5 tasks

        result = engine.analyze_task_suite(
            example_tasks,
            threshold=0.1,
            save_path=Path("output/tom_experiments/example_results.json")
        )
    else:
        print(f"Task suite not found at {task_path}")
        print("Run tom_tasks.py first to generate tasks")


if __name__ == "__main__":
    main()
