"""
Evaluate Cognitive Action Activation Patterns in Baseline Model (No Steering)

This script analyzes how cognitive action activation patterns differ between
correct and incorrect answers on ToM benchmark stories.

Key features:
- Loads ToM benchmark stories from BigToM dataset
- Evaluates baseline model (no steering) on multiple samples
- Captures cognitive action activations at TWO points:
  1. After story + question (before answer generation)
  2. After story + question + generated answer
- Compares activation patterns between correct vs incorrect answers
- Generates comprehensive analysis, visualizations, and CSV reports

Usage examples:
  # Evaluate 100 samples from a specific condition
  python evaluate_cognitive_actions_baseline.py \
    --condition 0_forward_belief_true_belief \
    --num-samples 100

  # Evaluate with custom offset
  python evaluate_cognitive_actions_baseline.py \
    --condition 0_backward_belief_false_belief \
    --num-samples 50 \
    --offset 20

  # Custom output prefix
  python evaluate_cognitive_actions_baseline.py \
    --condition 0_forward_belief_true_belief \
    --num-samples 100 \
    --output-prefix forward_belief
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

    # Model response
    generated_answer: str
    is_correct: bool

    # Activations at two timepoints
    activations_at_question: Dict[str, int]  # Layer counts
    activations_after_answer: Dict[str, int]  # Layer counts


class BaselineCognitiveEvaluator:
    """
    Evaluates cognitive action activation patterns in baseline model,
    comparing correct vs incorrect answers
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        probes_dir: str = "../trained_probes",
        benchmark_dir: str = "procedural-evals-tom/data/conditions",
        results_dir: str = "results",
        probe_layer_range: Tuple[int, int] = (10, 20),
        device: str = None
    ):
        """
        Initialize the evaluator

        Args:
            model_name: HuggingFace model name
            probes_dir: Directory containing trained cognitive action probes
            benchmark_dir: Directory containing ToM benchmark conditions
            results_dir: Where to save results
            probe_layer_range: (start, end) layers for probes
            device: Device to use (auto-detect if None)
        """
        self.model_name = model_name
        self.probe_layer_range = probe_layer_range

        # Resolve paths
        script_dir = Path(__file__).parent
        self.probes_dir = Path(probes_dir)
        self.benchmark_dir = script_dir / benchmark_dir
        self.results_dir = script_dir / results_dir
        self.results_dir.mkdir(exist_ok=True)

        # Auto-detect device
        self.device = device or get_optimal_device()

        print(f"\n{'='*80}")
        print(f"Initializing Baseline Cognitive Action Evaluator")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Probe layers: {probe_layer_range}")
        print(f"{'='*80}\n")

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize model
        self.base_model = None  # Full model with LM head for generation
        self.model = None  # nnsight wrapper for activation tracing

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
        layer_activations: Dict[int, torch.Tensor],
        activation_threshold: float = 0.001
    ) -> Dict[str, Dict]:
        """
        Run all probes on extracted activations and count activated layers per action

        Args:
            layer_activations: Dict mapping layer_idx -> activation tensor
            activation_threshold: Confidence threshold to consider a layer "activated"

        Returns:
            Dict mapping action_name -> {
                'confidences': {layer: confidence},
                'activated_layers': list of layers where confidence > threshold,
                'layer_count': number of activated layers,
                'all_layers_tested': list of all layers tested
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

        # Compute layer counts based on activation threshold
        result = {}
        for action_name, data in action_results.items():
            confidences = data['confidences']

            if confidences:
                # Count layers where confidence exceeds threshold
                activated_layers = [
                    layer for layer, conf in confidences.items()
                    if conf > activation_threshold
                ]

                result[action_name] = {
                    'confidences': confidences,
                    'activated_layers': activated_layers,
                    'layer_count': len(activated_layers),
                    'all_layers_tested': data['layers']
                }

        return result

    def _load_model(self):
        """Load model (text-only, skip vision tower)"""
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
        num_samples: int = 100,
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
        max_tokens: int = 300
    ) -> str:
        """
        Generate answer using the model

        Args:
            model: nnsight LanguageModel
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
        attention_mask = inputs['attention_mask'].to(self.device)

        # Generate
        with torch.no_grad():
            # Use stored base_model (has .generate() method)
            outputs = self.base_model.generate(
                input_ids,
                attention_mask=attention_mask,
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
        Evaluate a single sample

        Args:
            sample: Sample dict with story, question, answers

        Returns:
            ActivationResult with complete analysis
        """
        story = sample['story']
        question = sample['question']
        true_answer = sample['true_answer']
        wrong_answer = sample['wrong_answer']

        # Load model
        self._load_model()

        # 1. Capture activations at question
        activations_q = self.capture_activations_at_question(
            self.model, story, question
        )
        self._clear_gpu_memory()

        # 2. Generate answer
        generated_answer = self.generate_answer(
            self.model, story, question
        )
        self._clear_gpu_memory()

        # 3. Capture activations after answer
        activations_a = self.capture_activations_after_answer(
            self.model, story, question, generated_answer
        )
        self._clear_gpu_memory()

        # 4. Grade answer
        is_correct = self.grade_answer(
            generated_answer, true_answer, wrong_answer
        )

        # Create result
        result = ActivationResult(
            story=story,
            question=question,
            true_answer=true_answer,
            wrong_answer=wrong_answer,
            generated_answer=generated_answer,
            is_correct=is_correct,
            activations_at_question=self._extract_aggregates(activations_q),
            activations_after_answer=self._extract_aggregates(activations_a)
        )

        return result

    def _extract_aggregates(self, activations: Dict[str, Dict]) -> Dict[str, int]:
        """Extract layer counts from activation dict"""
        return {
            action: data.get('layer_count', 0)
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
        Generate summary statistics comparing correct vs incorrect answers

        Args:
            results: List of ActivationResult objects

        Returns:
            Summary statistics dict
        """
        # Split results by correctness
        correct_results = [r for r in results if r.is_correct]
        incorrect_results = [r for r in results if not r.is_correct]

        # Overall accuracy
        accuracy = len(correct_results) / len(results) if results else 0.0

        # Aggregate activations for correct answers
        correct_act_q = defaultdict(list)
        correct_act_a = defaultdict(list)

        for result in correct_results:
            for action, count in result.activations_at_question.items():
                correct_act_q[action].append(count)
            for action, count in result.activations_after_answer.items():
                correct_act_a[action].append(count)

        # Aggregate activations for incorrect answers
        incorrect_act_q = defaultdict(list)
        incorrect_act_a = defaultdict(list)

        for result in incorrect_results:
            for action, count in result.activations_at_question.items():
                incorrect_act_q[action].append(count)
            for action, count in result.activations_after_answer.items():
                incorrect_act_a[action].append(count)

        # Compute mean counts
        correct_mean_q = {
            action: np.mean(counts) for action, counts in correct_act_q.items()
        }
        correct_mean_a = {
            action: np.mean(counts) for action, counts in correct_act_a.items()
        }

        incorrect_mean_q = {
            action: np.mean(counts) for action, counts in incorrect_act_q.items()
        }
        incorrect_mean_a = {
            action: np.mean(counts) for action, counts in incorrect_act_a.items()
        }

        # Compute differences (correct - incorrect)
        all_actions = set(correct_mean_q.keys()) | set(incorrect_mean_q.keys()) | \
                      set(correct_mean_a.keys()) | set(incorrect_mean_a.keys())

        diff_q = {}
        diff_a = {}

        for action in all_actions:
            c_q = correct_mean_q.get(action, 0.0)
            i_q = incorrect_mean_q.get(action, 0.0)
            diff_q[action] = c_q - i_q

            c_a = correct_mean_a.get(action, 0.0)
            i_a = incorrect_mean_a.get(action, 0.0)
            diff_a[action] = c_a - i_a

        # Sort by absolute difference
        top_diff_q = sorted(
            diff_q.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:20]

        top_diff_a = sorted(
            diff_a.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:20]

        summary = {
            'num_samples': len(results),
            'num_correct': len(correct_results),
            'num_incorrect': len(incorrect_results),
            'accuracy': accuracy,
            'correct_mean_at_question': correct_mean_q,
            'correct_mean_after_answer': correct_mean_a,
            'incorrect_mean_at_question': incorrect_mean_q,
            'incorrect_mean_after_answer': incorrect_mean_a,
            'diff_at_question': diff_q,
            'diff_after_answer': diff_a,
            'top_differences_at_question': top_diff_q,
            'top_differences_after_answer': top_diff_a
        }

        return summary

    def save_results(
        self,
        results: List[ActivationResult],
        summary: Dict,
        prefix: str = "baseline_eval"
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
        prefix: str = "baseline_eval"
    ):
        """
        Generate visualization plots

        Args:
            summary: Summary statistics dict
            prefix: Filename prefix
        """
        # Top differences bar plot (Correct - Incorrect)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # At question
        actions_q = [item[0] for item in summary['top_differences_at_question'][:15]]
        diffs_q = [item[1] for item in summary['top_differences_at_question'][:15]]

        colors_q = ['green' if d > 0 else 'red' for d in diffs_q]
        ax1.barh(actions_q, diffs_q, color=colors_q, alpha=0.7)
        ax1.set_xlabel('Layer Count Difference (Correct - Incorrect)')
        ax1.set_title('Top 15 Cognitive Action Layer Count Differences at Question Point')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax1.grid(axis='x', alpha=0.3)

        # After answer
        actions_a = [item[0] for item in summary['top_differences_after_answer'][:15]]
        diffs_a = [item[1] for item in summary['top_differences_after_answer'][:15]]

        colors_a = ['green' if d > 0 else 'red' for d in diffs_a]
        ax2.barh(actions_a, diffs_a, color=colors_a, alpha=0.7)
        ax2.set_xlabel('Layer Count Difference (Correct - Incorrect)')
        ax2.set_title('Top 15 Cognitive Action Layer Count Differences After Answer')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        plot_path = self.results_dir / f"{prefix}_differences.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved differences plot to: {plot_path}")
        plt.close()

        # Accuracy visualization
        fig, ax = plt.subplots(figsize=(8, 6))

        categories = ['Correct', 'Incorrect']
        counts = [summary['num_correct'], summary['num_incorrect']]
        colors = ['#2ecc71', '#e74c3c']

        bars = ax.bar(categories, counts, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'Answer Correctness Distribution (Accuracy: {summary["accuracy"]:.2%})')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom'
            )

        plot_path = self.results_dir / f"{prefix}_accuracy.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved accuracy plot to: {plot_path}")
        plt.close()

    def generate_heatmap_visualization(
        self,
        results: List[ActivationResult],
        summary: Dict,
        prefix: str = "baseline_eval",
        top_n: int = 30
    ):
        """
        Generate heatmap comparing correct vs incorrect layer counts

        Args:
            results: List of ActivationResult objects
            summary: Summary statistics dict
            prefix: Filename prefix
            top_n: Number of top actions to display
        """
        # Get mean activations for each action
        correct_mean_q = summary['correct_mean_at_question']
        correct_mean_a = summary['correct_mean_after_answer']
        incorrect_mean_q = summary['incorrect_mean_at_question']
        incorrect_mean_a = summary['incorrect_mean_after_answer']

        # Compute total absolute difference for sorting
        all_actions = set(correct_mean_q.keys()) | set(incorrect_mean_q.keys()) | \
                      set(correct_mean_a.keys()) | set(incorrect_mean_a.keys())

        action_stats = {}
        for action in all_actions:
            c_q = correct_mean_q.get(action, 0.0)
            i_q = incorrect_mean_q.get(action, 0.0)
            c_a = correct_mean_a.get(action, 0.0)
            i_a = incorrect_mean_a.get(action, 0.0)

            total_diff = abs(c_q - i_q) + abs(c_a - i_a)

            action_stats[action] = {
                'correct_q': c_q,
                'incorrect_q': i_q,
                'correct_a': c_a,
                'incorrect_a': i_a,
                'total_diff': total_diff
            }

        # Sort by total difference and take top N
        sorted_actions = sorted(
            action_stats.items(),
            key=lambda x: x[1]['total_diff'],
            reverse=True
        )[:top_n]

        # Prepare data for heatmap
        actions = [action for action, _ in sorted_actions]
        data_matrix = []

        for action, stats in sorted_actions:
            row = [
                stats['correct_q'],
                stats['incorrect_q'],
                stats['correct_a'],
                stats['incorrect_a']
            ]
            data_matrix.append(row)

        data_matrix = np.array(data_matrix)

        # Create heatmap
        _, ax = plt.subplots(figsize=(10, max(12, top_n * 0.4)))

        # Use a diverging colormap
        sns.heatmap(
            data_matrix,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            xticklabels=['Correct\n(at Question)', 'Incorrect\n(at Question)',
                         'Correct\n(after Answer)', 'Incorrect\n(after Answer)'],
            yticklabels=actions,
            cbar_kws={'label': 'Mean Layer Count'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )

        ax.set_title(f'Cognitive Action Layer Counts: Correct vs Incorrect (Top {top_n})',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Condition', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cognitive Action', fontsize=11, fontweight='bold')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=0, ha='center')
        plt.yticks(rotation=0, fontsize=9)

        plt.tight_layout()

        plot_path = self.results_dir / f"{prefix}_heatmap.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved heatmap to: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate baseline cognitive action patterns (correct vs incorrect answers)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='google/gemma-3-4b-it',
        help='Model name'
    )
    parser.add_argument(
        '--probes-dir',
        type=str,
        default='../data/probes_binary',
        help='Directory containing trained probes'
    )
    parser.add_argument(
        '--condition',
        type=str,
        default='0_forward_belief_true_belief',
        help='ToM benchmark condition to evaluate'
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=100,
        help='Number of samples to evaluate'
    )
    parser.add_argument(
        '--offset', '-o',
        type=int,
        default=0,
        help='Offset in dataset'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='baseline_eval',
        help='Output filename prefix'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("BASELINE COGNITIVE ACTION EVALUATION (CORRECT VS INCORRECT)")
    print("="*80 + "\n")

    # Initialize evaluator
    evaluator = BaselineCognitiveEvaluator(
        model_name=args.model,
        probes_dir=args.probes_dir
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
    print(f"Correct answers: {summary['num_correct']}")
    print(f"Incorrect answers: {summary['num_incorrect']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print("\nTop 10 cognitive action differences at question (Correct - Incorrect):")
    for action, diff in summary['top_differences_at_question'][:10]:
        print(f"  {action:40s} {diff:+.4f}")
    print("\nTop 10 cognitive action differences after answer (Correct - Incorrect):")
    for action, diff in summary['top_differences_after_answer'][:10]:
        print(f"  {action:40s} {diff:+.4f}")
    print("="*80 + "\n")

    # Save results
    print("Saving results...")
    evaluator.save_results(results, summary, prefix=args.output_prefix)

    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.generate_visualizations(summary, prefix=args.output_prefix)
    evaluator.generate_heatmap_visualization(results, summary, prefix=args.output_prefix)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {evaluator.results_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()