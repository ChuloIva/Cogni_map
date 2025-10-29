"""
Evaluate Cognitive Action Activation Patterns in ToM-Steered vs Baseline Models

This script compares how Theory of Mind steering vectors affect cognitive action
activation patterns when processing ToM benchmark stories.

Key features:
- Loads ToM benchmark stories from BigToM dataset
- Evaluates both baseline (no steering) and steered (with ToM vector) conditions
- Captures cognitive action activations at TWO points:
  1. After story + question (before answer generation)
  2. After story + question + generated answer
- Generates comprehensive analysis, visualizations, and CSV reports
"""

import os
import sys
import csv
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

# IMPORTANT: Configure AMD GPU BEFORE importing torch
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
try:
    from probes.gpu_utils import configure_amd_gpu, get_optimal_device
    configure_amd_gpu()
except ImportError:
    print("⚠️  gpu_utils not found - skipping AMD GPU configuration")

# Now safe to import torch
import torch
from tqdm import tqdm

# Define get_optimal_device fallback if import failed
if 'get_optimal_device' not in dir():
    def get_optimal_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

# Import required components
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns

# Add repeng to path
sys.path.insert(0, str(Path(__file__).parent / "repeng"))
from repeng import ControlVector, ControlModel

# Add nnsight to path
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "nnsight" / "src"))
from nnsight import LanguageModel

# Import probe components
from probes.probe_models import load_probe
from probes.dataset_utils import get_idx_to_action_mapping
from probes.action_categories import get_action_category, CATEGORY_TAGS

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


@dataclass
class ActivationResult:
    """Container for activation analysis results"""
    story: str
    question: str
    true_answer: str
    wrong_answer: str

    # Baseline condition
    baseline_answer: str
    baseline_correct: bool
    baseline_activations_at_question: Dict[str, float]
    baseline_activations_after_answer: Dict[str, float]

    # Steered condition
    steered_answer: str
    steered_correct: bool
    steered_activations_at_question: Dict[str, float]
    steered_activations_after_answer: Dict[str, float]

    # Differences
    diff_at_question: Dict[str, float]
    diff_after_answer: Dict[str, float]

    accuracy_improvement: bool  # Whether steering improved correctness


class CognitiveActionEvaluator:
    """
    Evaluates cognitive action activation patterns with ToM steering
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        steering_vector_path: str = "steering_vectors/tom_general.gguf",
        probes_dir: str = "../trained_probes",
        benchmark_dir: str = "procedural-evals-tom/data/conditions",
        results_dir: str = "results",
        steering_coeff: float = 1.5,
        probe_layer_range: Tuple[int, int] = (21, 30),
        steering_layer_range: List[int] = None,
        device: str = None
    ):
        """
        Initialize the evaluator

        Args:
            model_name: HuggingFace model name
            steering_vector_path: Path to .gguf steering vector
            probes_dir: Directory containing trained cognitive action probes
            benchmark_dir: Directory containing ToM benchmark conditions
            results_dir: Where to save results
            steering_coeff: Steering vector coefficient (strength)
            probe_layer_range: (start, end) layers for probes
            steering_layer_range: Layers for steering (default: -5 to -18)
            device: Device to use (auto-detect if None)
        """
        self.model_name = model_name
        self.steering_coeff = steering_coeff
        self.probe_layer_range = probe_layer_range
        self.steering_layer_range = steering_layer_range or list(range(-5, -18, -1))

        # Resolve paths
        script_dir = Path(__file__).parent
        self.steering_vector_path = script_dir / steering_vector_path
        self.probes_dir = Path(probes_dir)
        self.benchmark_dir = script_dir / benchmark_dir
        self.results_dir = script_dir / results_dir
        self.results_dir.mkdir(exist_ok=True)

        # Auto-detect device
        self.device = device or get_optimal_device()

        print(f"\n{'='*80}")
        print(f"Initializing Cognitive Action Evaluator")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Steering vector: {self.steering_vector_path}")
        print(f"Steering coefficient: {steering_coeff}")
        print(f"Steering layers: {self.steering_layer_range}")
        print(f"Probe layers: {probe_layer_range}")
        print(f"{'='*80}\n")

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load steering vector
        print(f"Loading steering vector from {self.steering_vector_path}...")
        self.tom_vector = ControlVector.import_gguf(str(self.steering_vector_path))
        print(f"✓ Loaded steering vector with {len(self.tom_vector.directions)} layer directions")

        # Initialize model (single instance, reused for both conditions)
        self.base_model = None  # Full model with LM head for generation
        self.model = None  # nnsight wrapper for activation tracing
        self.control_model = None  # Control wrapper for steering

        # Load action mapping and probes
        print("Loading action mappings...")
        self.idx_to_action = get_idx_to_action_mapping()
        self.action_to_idx = {action: idx for idx, action in self.idx_to_action.items()}
        print(f"✓ Loaded {len(self.idx_to_action)} cognitive actions")

        print(f"\nLoading probes from {self.probes_dir}...")
        self.probes = self._load_all_probes()
        print(f"✓ Loaded {len(self.probes)} probes across layers")

        print("✓ Initialization complete\n")

    def _load_all_probes(self) -> Dict[Tuple[str, int], Dict]:
        """
        Load all cognitive action probes from all layers

        Returns:
            Dictionary mapping (action_name, layer) -> probe_info
        """
        probes = {}
        layers = range(self.probe_layer_range[0], self.probe_layer_range[1] + 1)

        for layer_idx in layers:
            layer_dir = self.probes_dir / f"layer_{layer_idx}"

            if not layer_dir.exists():
                print(f"  Warning: Layer directory not found: {layer_dir}")
                continue

            # Load all probe files in this layer
            probe_files = sorted(layer_dir.glob("probe_*.pth"))

            for probe_path in probe_files:
                # Extract action name from filename: probe_action_name.pth
                action_name = probe_path.stem.replace("probe_", "")

                if action_name not in self.action_to_idx:
                    continue

                try:
                    probe, metadata = load_probe(probe_path, device=self.device)

                    probes[(action_name, layer_idx)] = {
                        'probe': probe,
                        'layer': layer_idx,
                        'action': action_name,
                        'metadata': metadata
                    }

                except Exception as e:
                    print(f"  Warning: Failed to load {probe_path}: {e}")

            if layer_idx == layers[0]:  # Report first layer
                num_probes = len([p for p in probes if p[1] == layer_idx])
                print(f"  Layer {layer_idx}: loaded {num_probes} probes")

        return probes

    def _run_probes_on_activations(
        self,
        layer_activations: Dict[int, torch.Tensor]
    ) -> Dict[str, Dict]:
        """
        Run all probes on extracted activations and aggregate by action

        Args:
            layer_activations: Dict mapping layer_idx -> activation tensor

        Returns:
            Dict mapping action_name -> {
                'confidences': {layer: confidence},
                'aggregate': max confidence across layers,
                'best_layer': layer with highest confidence
            }
        """
        action_results = defaultdict(lambda: {'confidences': {}, 'layers': []})

        with torch.no_grad():
            for (action_name, layer_idx), probe_info in self.probes.items():
                if layer_idx not in layer_activations:
                    continue

                probe = probe_info['probe']
                activations = layer_activations[layer_idx]

                # Ensure activations are on correct device
                if activations.device != self.device:
                    activations = activations.to(self.device)

                # Get prediction
                logits = probe(activations)
                confidence = torch.sigmoid(logits).item()

                action_results[action_name]['confidences'][layer_idx] = confidence
                action_results[action_name]['layers'].append(layer_idx)

        # Compute aggregates (max across layers)
        result = {}
        for action_name, data in action_results.items():
            confidences = list(data['confidences'].values())

            if confidences:
                aggregate = max(confidences)
                best_layer = max(data['confidences'].items(), key=lambda x: x[1])[0]

                result[action_name] = {
                    'confidences': data['confidences'],
                    'aggregate': aggregate,
                    'best_layer': best_layer
                }

        return result

    def _load_model(self):
        """Load model (text-only, skip vision tower) - reused for both conditions"""
        if self.model is not None:
            return

        print("Loading model (text-only)...")

        # Import necessary classes
        from transformers import AutoConfig, Gemma3ForCausalLM

        # Check if this is a VLM
        config = AutoConfig.from_pretrained(self.model_name)

        if hasattr(config, 'vision_config'):
            print("  Detected VLM - loading text-only (skipping vision tower)...")
            base_model = Gemma3ForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device  # Use single device, no splitting
            )
        else:
            print("  Loading standard causal LM...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device  # Use single device, no splitting
            )

        base_model.eval()

        # Store base model for generation (has .generate() method)
        self.base_model = base_model

        # Wrap with nnsight LanguageModel for activation tracing
        self.model = LanguageModel(base_model, tokenizer=self.tokenizer)
        print("✓ Model loaded\n")

    def _apply_steering(self):
        """Apply steering vector to the model"""
        if self.control_model is not None:
            print("  Steering already applied")
            return

        print(f"  Applying ToM steering (coeff={self.steering_coeff})...")
        # Apply ControlModel to base_model (modifies in-place, affects both base_model and nnsight wrapper)
        self.control_model = ControlModel(self.base_model, self.steering_layer_range)
        self.control_model.set_control(self.tom_vector, coeff=self.steering_coeff)
        print("  ✓ Steering applied")

    def _remove_steering(self):
        """Remove steering vector from the model"""
        if self.control_model is None:
            return

        print("  Removing ToM steering...")
        self.control_model.reset()
        self.control_model = None
        print("  ✓ Steering removed")

    def _clear_gpu_memory(self):
        """Aggressively clear GPU memory"""
        import gc

        # Clear Python garbage
        gc.collect()

        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_tom_benchmark_samples(
        self,
        condition: str = "0_backward_belief_false_belief",
        num_samples: int = 10,
        offset: int = 0
    ) -> List[Dict]:
        """
        Load samples from ToM benchmark

        Args:
            condition: Condition directory name
            num_samples: Number of samples to load
            offset: Start offset in dataset

        Returns:
            List of sample dictionaries
        """
        csv_path = self.benchmark_dir / condition / "stories.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Benchmark condition not found: {csv_path}")

        print(f"Loading ToM benchmark samples from: {condition}")
        print(f"  Path: {csv_path}")

        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            rows = list(reader)

        total_available = len(rows)
        num_samples = min(num_samples, total_available - offset)

        samples = []
        for row in rows[offset:offset + num_samples]:
            if len(row) != 4:
                continue

            samples.append({
                'story': row[0],
                'question': row[1],
                'true_answer': row[2],
                'wrong_answer': row[3]
            })

        print(f"✓ Loaded {len(samples)} samples (offset={offset}, total={total_available})\n")
        return samples

    def generate_answer(
        self,
        model: LanguageModel,
        story: str,
        question: str,
        max_tokens: int = 100
    ) -> str:
        """
        Generate answer using the model

        Args:
            model: nnsight LanguageModel (baseline or steered)
            story: Story text
            question: Question text
            max_tokens: Maximum tokens to generate

        Returns:
            Generated answer string
        """
        # Format prompt
        prompt = f"Story: {story}\n\nQuestion: {question}\n\nAnswer:"

        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)

        # Generate
        with torch.no_grad():
            # Use stored base_model (has .generate() method)
            outputs = self.base_model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,  # Deterministic
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only new tokens
        answer = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return answer.strip()

    def capture_activations_at_question(
        self,
        model: LanguageModel,
        story: str,
        question: str
    ) -> Dict[str, Dict]:
        """
        Capture cognitive action activations after story+question, before generation

        Args:
            model: nnsight LanguageModel
            story: Story text
            question: Question text

        Returns:
            Dict mapping action_name -> activation info
        """
        # Format the prompt (same as generation)
        prompt = f"Story: {story}\n\nQuestion: {question}\n\nAnswer:"

        # Augment for probe extraction
        augmented = f"{prompt}\n\nThe cognitive action being demonstrated here is"

        # Extract activations using nnsight
        saved_activations = {}

        with model.trace(augmented) as tracer:
            for layer_idx in range(self.probe_layer_range[0], self.probe_layer_range[1] + 1):
                hidden_states = model.model.layers[layer_idx].output[0]
                saved_activations[layer_idx] = hidden_states[:, -1, :].save()

        # Convert to dict format
        layer_activations = {
            layer_idx: act.squeeze(0).cpu()
            for layer_idx, act in saved_activations.items()
        }

        # Run probes on activations
        action_predictions = self._run_probes_on_activations(layer_activations)

        return action_predictions

    def capture_activations_after_answer(
        self,
        model: LanguageModel,
        story: str,
        question: str,
        answer: str
    ) -> Dict[str, Dict]:
        """
        Capture cognitive action activations after full story+question+answer

        Args:
            model: nnsight LanguageModel
            story: Story text
            question: Question text
            answer: Generated answer

        Returns:
            Dict mapping action_name -> activation info
        """
        # Full text including answer
        full_text = f"Story: {story}\n\nQuestion: {question}\n\nAnswer: {answer}"

        # Augment for probe extraction
        augmented = f"{full_text}\n\nThe cognitive action being demonstrated here is"

        # Extract activations
        saved_activations = {}

        with model.trace(augmented) as tracer:
            for layer_idx in range(self.probe_layer_range[0], self.probe_layer_range[1] + 1):
                hidden_states = model.model.layers[layer_idx].output[0]
                saved_activations[layer_idx] = hidden_states[:, -1, :].save()

        # Convert to dict format
        layer_activations = {
            layer_idx: act.squeeze(0).cpu()
            for layer_idx, act in saved_activations.items()
        }

        # Run probes on activations
        action_predictions = self._run_probes_on_activations(layer_activations)

        return action_predictions

    def grade_answer(
        self,
        predicted: str,
        true_answer: str,
        wrong_answer: str
    ) -> bool:
        """
        Grade whether predicted answer is correct

        Args:
            predicted: Predicted answer
            true_answer: Correct answer
            wrong_answer: Incorrect answer

        Returns:
            True if correct, False otherwise
        """
        pred_lower = predicted.lower()
        true_lower = true_answer.lower()
        wrong_lower = wrong_answer.lower()

        # Simple keyword matching
        true_keywords = set(true_lower.split())
        wrong_keywords = set(wrong_lower.split())
        pred_keywords = set(pred_lower.split())

        true_overlap = len(true_keywords & pred_keywords)
        wrong_overlap = len(wrong_keywords & pred_keywords)

        return true_overlap > wrong_overlap

    def evaluate_sample(
        self,
        sample: Dict
    ) -> ActivationResult:
        """
        Evaluate a single sample with both baseline and steered conditions

        Args:
            sample: Sample dict with story, question, answers

        Returns:
            ActivationResult with complete analysis
        """
        story = sample['story']
        question = sample['question']
        true_answer = sample['true_answer']
        wrong_answer = sample['wrong_answer']

        # Load model (once, reused for both conditions)
        self._load_model()

        print(f"\n{'='*60}")
        print(f"Evaluating sample: {story[:60]}...")
        print(f"{'='*60}")

        # ============================================================
        # BASELINE CONDITION (No Steering)
        # ============================================================
        print("\n[BASELINE CONDITION - No Steering]")

        # 1. Capture activations at question
        print("  1/4 Capturing activations at question...")
        baseline_act_q = self.capture_activations_at_question(
            self.model, story, question
        )
        self._clear_gpu_memory()  # Clean up after activation capture

        # 2. Generate answer
        print("  2/4 Generating answer...")
        baseline_answer = self.generate_answer(
            self.model, story, question
        )
        self._clear_gpu_memory()  # Clean up after generation

        # 3. Capture activations after answer
        print("  3/4 Capturing activations after answer...")
        baseline_act_a = self.capture_activations_after_answer(
            self.model, story, question, baseline_answer
        )
        self._clear_gpu_memory()  # Clean up after activation capture

        # 4. Grade answer
        print("  4/4 Grading answer...")
        baseline_correct = self.grade_answer(
            baseline_answer, true_answer, wrong_answer
        )
        print(f"  ✓ Baseline complete (correct={baseline_correct})")

        # ============================================================
        # STEERED CONDITION (With ToM Steering)
        # ============================================================
        print("\n[STEERED CONDITION - With ToM Steering]")

        # Apply steering
        self._apply_steering()

        # 1. Capture activations at question
        print("  1/4 Capturing activations at question...")
        steered_act_q = self.capture_activations_at_question(
            self.model, story, question
        )
        self._clear_gpu_memory()  # Clean up after activation capture

        # 2. Generate answer
        print("  2/4 Generating answer...")
        steered_answer = self.generate_answer(
            self.model, story, question
        )
        self._clear_gpu_memory()  # Clean up after generation

        # 3. Capture activations after answer
        print("  3/4 Capturing activations after answer...")
        steered_act_a = self.capture_activations_after_answer(
            self.model, story, question, steered_answer
        )
        self._clear_gpu_memory()  # Clean up after activation capture

        # 4. Grade answer
        print("  4/4 Grading answer...")
        steered_correct = self.grade_answer(
            steered_answer, true_answer, wrong_answer
        )
        print(f"  ✓ Steered complete (correct={steered_correct})")

        # Remove steering for next iteration
        self._remove_steering()

        # COMPUTE DIFFERENCES
        diff_at_q = self._compute_activation_diff(baseline_act_q, steered_act_q)
        diff_at_a = self._compute_activation_diff(baseline_act_a, steered_act_a)

        # Create result
        result = ActivationResult(
            story=story,
            question=question,
            true_answer=true_answer,
            wrong_answer=wrong_answer,
            baseline_answer=baseline_answer,
            baseline_correct=baseline_correct,
            baseline_activations_at_question=self._extract_aggregates(baseline_act_q),
            baseline_activations_after_answer=self._extract_aggregates(baseline_act_a),
            steered_answer=steered_answer,
            steered_correct=steered_correct,
            steered_activations_at_question=self._extract_aggregates(steered_act_q),
            steered_activations_after_answer=self._extract_aggregates(steered_act_a),
            diff_at_question=diff_at_q,
            diff_after_answer=diff_at_a,
            accuracy_improvement=(steered_correct and not baseline_correct)
        )

        return result

    def _compute_activation_diff(
        self,
        baseline: Dict[str, Dict],
        steered: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Compute activation differences (steered - baseline)"""
        diff = {}
        all_actions = set(baseline.keys()) | set(steered.keys())

        for action in all_actions:
            baseline_val = baseline.get(action, {}).get('aggregate', 0.0)
            steered_val = steered.get(action, {}).get('aggregate', 0.0)
            diff[action] = steered_val - baseline_val

        return diff

    def _extract_aggregates(self, activations: Dict[str, Dict]) -> Dict[str, float]:
        """Extract aggregate values from activation dict"""
        return {
            action: data.get('aggregate', 0.0)
            for action, data in activations.items()
        }

    def evaluate_multiple_samples(
        self,
        samples: List[Dict],
        verbose: bool = True
    ) -> List[ActivationResult]:
        """
        Evaluate multiple samples

        Args:
            samples: List of sample dicts
            verbose: Show progress bar

        Returns:
            List of ActivationResult objects
        """
        results = []

        iterator = tqdm(samples, desc="Evaluating samples") if verbose else samples

        for sample in iterator:
            result = self.evaluate_sample(sample)
            results.append(result)

        return results

    def generate_summary_statistics(
        self,
        results: List[ActivationResult]
    ) -> Dict:
        """
        Generate summary statistics from results

        Args:
            results: List of ActivationResult objects

        Returns:
            Summary statistics dict
        """
        # Accuracy metrics
        baseline_accuracy = sum(r.baseline_correct for r in results) / len(results)
        steered_accuracy = sum(r.steered_correct for r in results) / len(results)
        improvements = sum(r.accuracy_improvement for r in results)

        # Aggregate activation differences
        all_diff_q = defaultdict(list)
        all_diff_a = defaultdict(list)

        for result in results:
            for action, diff in result.diff_at_question.items():
                all_diff_q[action].append(diff)
            for action, diff in result.diff_after_answer.items():
                all_diff_a[action].append(diff)

        # Compute mean differences
        mean_diff_q = {
            action: np.mean(diffs)
            for action, diffs in all_diff_q.items()
        }
        mean_diff_a = {
            action: np.mean(diffs)
            for action, diffs in all_diff_a.items()
        }

        # Sort by absolute difference
        top_diff_q = sorted(
            mean_diff_q.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:20]

        top_diff_a = sorted(
            mean_diff_a.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:20]

        summary = {
            'num_samples': len(results),
            'baseline_accuracy': baseline_accuracy,
            'steered_accuracy': steered_accuracy,
            'accuracy_improvement': steered_accuracy - baseline_accuracy,
            'num_improved': improvements,
            'top_differences_at_question': top_diff_q,
            'top_differences_after_answer': top_diff_a,
            'mean_diff_at_question': mean_diff_q,
            'mean_diff_after_answer': mean_diff_a
        }

        return summary

    def save_results(
        self,
        results: List[ActivationResult],
        summary: Dict,
        prefix: str = "cognitive_eval"
    ):
        """
        Save results to files

        Args:
            results: List of ActivationResult objects
            summary: Summary statistics dict
            prefix: Filename prefix
        """
        # Save raw results as CSV
        csv_path = self.results_dir / f"{prefix}_raw.csv"
        with open(csv_path, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
                writer.writeheader()
                for result in results:
                    # Convert to dict but keep nested dicts as JSON strings
                    row = {}
                    for k, v in asdict(result).items():
                        if isinstance(v, dict):
                            row[k] = json.dumps(v)
                        else:
                            row[k] = v
                    writer.writerow(row)

        print(f"✓ Saved raw results to: {csv_path}")

        # Save summary as JSON
        json_path = self.results_dir / f"{prefix}_summary.json"

        # Convert numpy types to Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_for_json(item) for item in obj]
            return obj

        summary_json = convert_for_json(summary)

        with open(json_path, 'w') as f:
            json.dump(summary_json, f, indent=2)

        print(f"✓ Saved summary to: {json_path}")

    def generate_visualizations(
        self,
        summary: Dict,
        prefix: str = "cognitive_eval"
    ):
        """
        Generate visualization plots

        Args:
            summary: Summary statistics dict
            prefix: Filename prefix
        """
        # Top differences bar plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # At question
        actions_q = [item[0] for item in summary['top_differences_at_question'][:15]]
        diffs_q = [item[1] for item in summary['top_differences_at_question'][:15]]

        colors_q = ['green' if d > 0 else 'red' for d in diffs_q]
        ax1.barh(actions_q, diffs_q, color=colors_q, alpha=0.7)
        ax1.set_xlabel('Activation Difference (Steered - Baseline)')
        ax1.set_title('Top 15 Cognitive Action Differences at Question Point')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax1.grid(axis='x', alpha=0.3)

        # After answer
        actions_a = [item[0] for item in summary['top_differences_after_answer'][:15]]
        diffs_a = [item[1] for item in summary['top_differences_after_answer'][:15]]

        colors_a = ['green' if d > 0 else 'red' for d in diffs_a]
        ax2.barh(actions_a, diffs_a, color=colors_a, alpha=0.7)
        ax2.set_xlabel('Activation Difference (Steered - Baseline)')
        ax2.set_title('Top 15 Cognitive Action Differences After Answer')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        plot_path = self.results_dir / f"{prefix}_differences.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved differences plot to: {plot_path}")
        plt.close()

        # Accuracy comparison
        fig, ax = plt.subplots(figsize=(8, 6))

        accuracies = [summary['baseline_accuracy'], summary['steered_accuracy']]
        labels = ['Baseline', 'Steered']
        colors = ['#3498db', '#2ecc71']

        bars = ax.bar(labels, accuracies, color=colors, alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title('ToM Task Accuracy: Baseline vs Steered')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom'
            )

        # Add improvement annotation
        improvement = summary['accuracy_improvement']
        ax.text(
            0.5, max(accuracies) + 0.05,
            f'Improvement: {improvement:+.2%}',
            ha='center',
            fontsize=12,
            fontweight='bold'
        )

        plot_path = self.results_dir / f"{prefix}_accuracy.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved accuracy plot to: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate cognitive action activation patterns with ToM steering'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='google/gemma-3-4b-it',
        help='Model name'
    )
    parser.add_argument(
        '--steering-vector',
        type=str,
        default='steering_vectors/tom_general_chat.gguf',
        help='Path to steering vector .gguf file'
    )
    parser.add_argument(
        '--probes-dir',
        type=str,
        default='data/probes_binary',
        help='Directory containing trained probes'
    )
    parser.add_argument(
        '--condition',
        type=str,
        default='0_backward_belief_false_belief',
        help='ToM benchmark condition to evaluate'
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=50,
        help='Number of samples to evaluate'
    )
    parser.add_argument(
        '--offset', '-o',
        type=int,
        default=0,
        help='Offset in dataset'
    )
    parser.add_argument(
        '--steering-coeff',
        type=float,
        default=1.3,
        help='Steering coefficient (strength)'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='cognitive_eval',
        help='Output filename prefix'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("COGNITIVE ACTION EVALUATION WITH TOM STEERING")
    print("="*80 + "\n")

    # Initialize evaluator
    evaluator = CognitiveActionEvaluator(
        model_name=args.model,
        steering_vector_path=args.steering_vector,
        probes_dir=args.probes_dir,
        steering_coeff=args.steering_coeff
    )

    # Load benchmark samples
    samples = evaluator.load_tom_benchmark_samples(
        condition=args.condition,
        num_samples=args.num_samples,
        offset=args.offset
    )

    # Evaluate all samples
    print(f"Evaluating {len(samples)} samples...\n")
    results = evaluator.evaluate_multiple_samples(samples)

    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summary = evaluator.generate_summary_statistics(results)

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Samples evaluated: {summary['num_samples']}")
    print(f"Baseline accuracy: {summary['baseline_accuracy']:.2%}")
    print(f"Steered accuracy: {summary['steered_accuracy']:.2%}")
    print(f"Accuracy improvement: {summary['accuracy_improvement']:+.2%}")
    print(f"Samples improved: {summary['num_improved']}")
    print("\nTop 10 cognitive action differences at question:")
    for action, diff in summary['top_differences_at_question'][:10]:
        print(f"  {action:40s} {diff:+.4f}")
    print("\nTop 10 cognitive action differences after answer:")
    for action, diff in summary['top_differences_after_answer'][:10]:
        print(f"  {action:40s} {diff:+.4f}")
    print("="*80 + "\n")

    # Save results
    print("Saving results...")
    evaluator.save_results(results, summary, prefix=args.output_prefix)

    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.generate_visualizations(summary, prefix=args.output_prefix)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {evaluator.results_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
