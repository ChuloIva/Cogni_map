"""
Evaluate Cognitive Action Activation Patterns in ToM-Steered vs Baseline Models

This script compares how Theory of Mind steering vectors affect cognitive action
activation patterns when processing ToM benchmark stories using ANSWER RANKING BY PROBABILITY
(as described in the BigToM paper).

Key features:
- Loads ToM benchmark stories from BigToM dataset
- Evaluates both baseline (no steering) and steered (with ToM vector) conditions
- Uses ANSWER RANKING BY PROBABILITY instead of text generation:
  * Formats questions as multiple choice (a/b options)
  * Calculates p(letter='a') and p(letter='b') from model logits
  * Ranks answers by probability (no text parsing needed)
- Captures cognitive action activations at THREE points:
  1. After story + question (before answer selection)
  2. After story + question + true answer
  3. After story + question + wrong answer
- Generates comprehensive analysis, visualizations, and CSV reports
- Supports combining multiple steering vectors with custom coefficients

Vector combination examples:
  # Single vector with coefficient
  python evaluate_cognitive_actions_with_steering.py --steering-coeff 1.5

  # Combine two vectors with equal weight
  python evaluate_cognitive_actions_with_steering.py \
    --steering-vectors vec1.gguf vec2.gguf

  # Combine vectors with custom coefficients (weighted sum)
  python evaluate_cognitive_actions_with_steering.py \
    --steering-vectors vec1.gguf vec2.gguf \
    --steering-coeffs 1.5 -0.5

  # Add vectors together (positive coefficients strengthen)
  python evaluate_cognitive_actions_with_steering.py \
    --steering-vectors tom_core.gguf tom_direction.gguf \
    --steering-coeffs 2.0 1.0

  # Subtract vectors (negative coefficients reverse effect)
  python evaluate_cognitive_actions_with_steering.py \
    --steering-vectors good_vec.gguf bad_vec.gguf \
    --steering-coeffs 1.0 -1.0
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
    baseline_selected: str  # Which answer was selected
    baseline_correct: bool
    baseline_prob_true: float  # Probability assigned to true answer
    baseline_prob_wrong: float  # Probability assigned to wrong answer
    baseline_activations_at_question: Dict[str, int]  # Layer counts
    baseline_activations_after_true: Dict[str, int]  # Layer counts after true answer
    baseline_activations_after_wrong: Dict[str, int]  # Layer counts after wrong answer

    # Steered condition
    steered_selected: str  # Which answer was selected
    steered_correct: bool
    steered_prob_true: float  # Probability assigned to true answer
    steered_prob_wrong: float  # Probability assigned to wrong answer
    steered_activations_at_question: Dict[str, int]  # Layer counts
    steered_activations_after_true: Dict[str, int]  # Layer counts after true answer
    steered_activations_after_wrong: Dict[str, int]  # Layer counts after wrong answer

    # Differences (layer count differences) - now for both answer paths
    diff_at_question: Dict[str, int]
    diff_after_true: Dict[str, int]
    diff_after_wrong: Dict[str, int]

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
        steering_coeff: float = 0,
        probe_layer_range: Tuple[int, int] = (10, 20),
        steering_layer_range: List[int] = None,
        device: str = None,
        steering_vector_paths: List[str] = None,
        steering_coeffs: List[float] = None
    ):
        """
        Initialize the evaluator

        Args:
            model_name: HuggingFace model name
            steering_vector_path: Path to .gguf steering vector (used if steering_vector_paths is None)
            probes_dir: Directory containing trained cognitive action probes
            benchmark_dir: Directory containing ToM benchmark conditions
            results_dir: Where to save results
            steering_coeff: Steering vector coefficient (strength) (used if steering_coeffs is None)
            probe_layer_range: (start, end) layers for probes
            steering_layer_range: Layers for steering (default: -5 to -18)
            device: Device to use (auto-detect if None)
            steering_vector_paths: List of paths to .gguf steering vectors to combine (optional)
            steering_coeffs: List of coefficients for each vector (optional, must match steering_vector_paths length)
        """
        self.model_name = model_name
        self.steering_coeff = steering_coeff
        self.probe_layer_range = probe_layer_range
        self.steering_layer_range = steering_layer_range or list(range(-4, -20, -1))

        # Resolve paths
        script_dir = Path(__file__).parent
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

        # Handle multiple vectors vs single vector
        if steering_vector_paths is not None:
            self.steering_vector_paths = [script_dir / path for path in steering_vector_paths]
            self.steering_coeffs = steering_coeffs or [1.0] * len(steering_vector_paths)

            if len(self.steering_coeffs) != len(self.steering_vector_paths):
                raise ValueError(f"Number of coefficients ({len(self.steering_coeffs)}) must match number of vectors ({len(self.steering_vector_paths)})")

            print(f"Combining {len(self.steering_vector_paths)} steering vectors:")
            for path, coeff in zip(self.steering_vector_paths, self.steering_coeffs):
                print(f"  - {path.name} (coeff={coeff})")
        else:
            self.steering_vector_paths = [script_dir / steering_vector_path]
            self.steering_coeffs = [steering_coeff]
            print(f"Steering vector: {self.steering_vector_paths[0]}")
            print(f"Steering coefficient: {steering_coeff}")

        print(f"Steering layers: {self.steering_layer_range}")
        print(f"Probe layers: {probe_layer_range}")
        print(f"{'='*80}\n")

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load steering vector(s)
        self.tom_vector = self._load_and_combine_vectors()

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

    def _load_and_combine_vectors(self) -> ControlVector:
        """
        Load and combine multiple steering vectors with their coefficients

        Returns:
            Combined ControlVector
        """
        if len(self.steering_vector_paths) == 1:
            # Single vector case
            print(f"Loading steering vector from {self.steering_vector_paths[0]}...")
            vector = ControlVector.import_gguf(str(self.steering_vector_paths[0]))
            print(f"✓ Loaded steering vector with {len(vector.directions)} layer directions")
            return vector

        # Multiple vectors case - combine them
        print(f"Loading and combining {len(self.steering_vector_paths)} steering vectors...")

        # Load first vector
        combined_vector = ControlVector.import_gguf(str(self.steering_vector_paths[0]))
        print(f"  ✓ Loaded {self.steering_vector_paths[0].name} ({len(combined_vector.directions)} layers)")

        # Apply first coefficient
        if self.steering_coeffs[0] != 1.0:
            combined_vector = self.steering_coeffs[0] * combined_vector
            print(f"    Applied coefficient: {self.steering_coeffs[0]}")

        # Load and combine remaining vectors
        for i, (path, coeff) in enumerate(zip(self.steering_vector_paths[1:], self.steering_coeffs[1:]), 1):
            vector = ControlVector.import_gguf(str(path))
            print(f"  ✓ Loaded {path.name} ({len(vector.directions)} layers)")

            # Apply coefficient and add to combined vector
            if coeff != 1.0:
                combined_vector = combined_vector + (coeff * vector)
                print(f"    Applied coefficient: {coeff}")
            else:
                combined_vector = combined_vector + vector

        print(f"✓ Combined vector created with {len(combined_vector.directions)} layer directions")
        return combined_vector

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

        # When using multiple vectors, coefficients are already baked into the combined vector
        # so we apply with coefficient 1.0. For single vector, use the specified coefficient.
        if len(self.steering_vector_paths) > 1:
            apply_coeff = 1.0
            print(f"  Applying combined ToM steering (coefficients already applied)...")
        else:
            apply_coeff = self.steering_coeff
            print(f"  Applying ToM steering (coeff={apply_coeff})...")

        # Apply ControlModel to base_model (modifies in-place, affects both base_model and nnsight wrapper)
        self.control_model = ControlModel(self.base_model, self.steering_layer_range)
        self.control_model.set_control(self.tom_vector, coeff=apply_coeff)
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

    def format_mcq_prompt(
        self,
        story: str,
        question: str,
        true_answer: str,
        wrong_answer: str,
        randomize: bool = True
    ) -> Tuple[str, str]:
        """
        Format question as multiple choice with a/b options

        Args:
            story: Story text
            question: Question text
            true_answer: Correct answer
            wrong_answer: Incorrect answer
            randomize: Whether to randomize answer positions

        Returns:
            Tuple of (formatted_prompt, true_position) where true_position is 'a' or 'b'
        """
        # Randomize answer positions
        if randomize:
            if random.random() < 0.5:
                option_a = true_answer
                option_b = wrong_answer
                true_position = 'a'
            else:
                option_a = wrong_answer
                option_b = true_answer
                true_position = 'b'
        else:
            option_a = true_answer
            option_b = wrong_answer
            true_position = 'a'

        # Format as MCQ
        prompt = (
            f"Story: {story}\n\n"
            f"Question: {question}\n"
            f"Choose one of the following:\n"
            f"a) {option_a}\n"
            f"b) {option_b}\n\n"
            f"Please answer with the letter of your choice (a or b).\n"
            f"Answer:"
        )

        return prompt, true_position

    def calculate_letter_probability(
        self,
        prompt: str,
        letter: str
    ) -> float:
        """
        Calculate probability that model assigns to a specific letter answer ('a' or 'b')

        Args:
            prompt: Formatted MCQ prompt ending with "Answer:"
            letter: Either 'a' or 'b'

        Returns:
            Probability of the letter token
        """
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

        # Tokenize the prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Get the token ID for the letter (try both with and without space)
        letter_tokens = [
            self.tokenizer.encode(letter, add_special_tokens=False),
            self.tokenizer.encode(f" {letter}", add_special_tokens=False),
            self.tokenizer.encode(f"{letter})", add_special_tokens=False),
            self.tokenizer.encode(f" {letter})", add_special_tokens=False),
        ]

        # Get model logits at the last position (where answer would be)
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits[0, -1, :]  # Last position logits

        # Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get max probability across all letter token variants
        max_prob = 0.0
        for token_ids in letter_tokens:
            if token_ids:  # If tokenization succeeded
                token_id = token_ids[0]  # Take first token
                token_prob = probs[token_id].item()
                max_prob = max(max_prob, token_prob)

        return max_prob

    def select_answer_by_probability(
        self,
        model: LanguageModel,
        story: str,
        question: str,
        true_answer: str,
        wrong_answer: str
    ) -> Tuple[str, Dict[str, float]]:
        """
        Select answer by ranking probabilities (instead of generating text)

        Args:
            model: nnsight LanguageModel (baseline or steered)
            story: Story text
            question: Question text
            true_answer: Correct answer text
            wrong_answer: Incorrect answer text

        Returns:
            Tuple of (selected_answer, probability_dict) where probability_dict contains
            {'prob_a': float, 'prob_b': float, 'true_position': 'a'/'b', 'selected': 'a'/'b'}
        """
        # Format as MCQ
        mcq_prompt, true_position = self.format_mcq_prompt(
            story, question, true_answer, wrong_answer, randomize=True
        )

        # Calculate probabilities for both letters
        prob_a = self.calculate_letter_probability(mcq_prompt, 'a')
        prob_b = self.calculate_letter_probability(mcq_prompt, 'b')

        # Determine selected answer
        selected_letter = 'a' if prob_a > prob_b else 'b'
        selected_answer = true_answer if selected_letter == true_position else wrong_answer

        probability_dict = {
            'prob_a': prob_a,
            'prob_b': prob_b,
            'true_position': true_position,
            'selected': selected_letter,
            'prob_true': prob_a if true_position == 'a' else prob_b,
            'prob_wrong': prob_b if true_position == 'a' else prob_a
        }

        return selected_answer, probability_dict

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
        true_answer: str,
        wrong_answer: str
    ) -> Dict[str, Dict[str, Dict]]:
        """
        Capture cognitive action activations after both possible answer completions

        Args:
            model: nnsight LanguageModel
            story: Story text
            question: Question text
            true_answer: Correct answer text
            wrong_answer: Incorrect answer text

        Returns:
            Dict with structure: {
                'true_answer': {action_name -> activation info},
                'wrong_answer': {action_name -> activation info}
            }
        """
        results = {}

        # Capture activations for both answer completions
        for answer_type, answer_text in [('true_answer', true_answer), ('wrong_answer', wrong_answer)]:
            # Full text including answer
            full_text = f"Story: {story}\n\nQuestion: {question}\n\nAnswer: {answer_text}"

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
            results[answer_type] = action_predictions

        return results

    def grade_by_probability(
        self,
        probability_dict: Dict[str, float]
    ) -> bool:
        """
        Grade answer correctness based on probability ranking

        Args:
            probability_dict: Dictionary containing probability information from select_answer_by_probability()

        Returns:
            True if correct answer had higher probability, False otherwise
        """
        return probability_dict['prob_true'] > probability_dict['prob_wrong']

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

        # 2. Select answer by probability
        print("  2/4 Ranking answers by probability...")
        baseline_selected, baseline_probs = self.select_answer_by_probability(
            self.model, story, question, true_answer, wrong_answer
        )
        self._clear_gpu_memory()  # Clean up after probability calculation

        # 3. Capture activations after both possible answers
        print("  3/4 Capturing activations after both answers...")
        baseline_act_a = self.capture_activations_after_answer(
            self.model, story, question, true_answer, wrong_answer
        )
        self._clear_gpu_memory()  # Clean up after activation capture

        # 4. Grade answer
        print("  4/4 Grading answer...")
        baseline_correct = self.grade_by_probability(baseline_probs)
        print(f"  ✓ Baseline complete (correct={baseline_correct}, p_true={baseline_probs['prob_true']:.3f}, p_wrong={baseline_probs['prob_wrong']:.3f})")

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

        # 2. Select answer by probability
        print("  2/4 Ranking answers by probability...")
        steered_selected, steered_probs = self.select_answer_by_probability(
            self.model, story, question, true_answer, wrong_answer
        )
        self._clear_gpu_memory()  # Clean up after probability calculation

        # 3. Capture activations after both possible answers
        print("  3/4 Capturing activations after both answers...")
        steered_act_a = self.capture_activations_after_answer(
            self.model, story, question, true_answer, wrong_answer
        )
        self._clear_gpu_memory()  # Clean up after activation capture

        # 4. Grade answer
        print("  4/4 Grading answer...")
        steered_correct = self.grade_by_probability(steered_probs)
        print(f"  ✓ Steered complete (correct={steered_correct}, p_true={steered_probs['prob_true']:.3f}, p_wrong={steered_probs['prob_wrong']:.3f})")

        # Remove steering for next iteration
        self._remove_steering()

        # COMPUTE DIFFERENCES
        diff_at_q = self._compute_activation_diff(baseline_act_q, steered_act_q)
        diff_after_true = self._compute_activation_diff(
            baseline_act_a['true_answer'],
            steered_act_a['true_answer']
        )
        diff_after_wrong = self._compute_activation_diff(
            baseline_act_a['wrong_answer'],
            steered_act_a['wrong_answer']
        )

        # Create result
        result = ActivationResult(
            story=story,
            question=question,
            true_answer=true_answer,
            wrong_answer=wrong_answer,
            baseline_selected=baseline_selected,
            baseline_correct=baseline_correct,
            baseline_prob_true=baseline_probs['prob_true'],
            baseline_prob_wrong=baseline_probs['prob_wrong'],
            baseline_activations_at_question=self._extract_aggregates(baseline_act_q),
            baseline_activations_after_true=self._extract_aggregates(baseline_act_a['true_answer']),
            baseline_activations_after_wrong=self._extract_aggregates(baseline_act_a['wrong_answer']),
            steered_selected=steered_selected,
            steered_correct=steered_correct,
            steered_prob_true=steered_probs['prob_true'],
            steered_prob_wrong=steered_probs['prob_wrong'],
            steered_activations_at_question=self._extract_aggregates(steered_act_q),
            steered_activations_after_true=self._extract_aggregates(steered_act_a['true_answer']),
            steered_activations_after_wrong=self._extract_aggregates(steered_act_a['wrong_answer']),
            diff_at_question=diff_at_q,
            diff_after_true=diff_after_true,
            diff_after_wrong=diff_after_wrong,
            accuracy_improvement=(steered_correct and not baseline_correct)
        )

        return result

    def _compute_activation_diff(
        self,
        baseline: Dict[str, Dict],
        steered: Dict[str, Dict]
    ) -> Dict[str, int]:
        """Compute activation differences (steered - baseline) based on layer counts"""
        diff = {}
        all_actions = set(baseline.keys()) | set(steered.keys())

        for action in all_actions:
            baseline_count = baseline.get(action, {}).get('layer_count', 0)
            steered_count = steered.get(action, {}).get('layer_count', 0)
            diff[action] = steered_count - baseline_count

        return diff

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

        # Probability metrics
        baseline_prob_true_avg = np.mean([r.baseline_prob_true for r in results])
        baseline_prob_wrong_avg = np.mean([r.baseline_prob_wrong for r in results])
        steered_prob_true_avg = np.mean([r.steered_prob_true for r in results])
        steered_prob_wrong_avg = np.mean([r.steered_prob_wrong for r in results])

        # Aggregate activation differences
        all_diff_q = defaultdict(list)
        all_diff_true = defaultdict(list)
        all_diff_wrong = defaultdict(list)

        for result in results:
            for action, diff in result.diff_at_question.items():
                all_diff_q[action].append(diff)
            for action, diff in result.diff_after_true.items():
                all_diff_true[action].append(diff)
            for action, diff in result.diff_after_wrong.items():
                all_diff_wrong[action].append(diff)

        # Compute mean differences
        mean_diff_q = {
            action: np.mean(diffs)
            for action, diffs in all_diff_q.items()
        }
        mean_diff_true = {
            action: np.mean(diffs)
            for action, diffs in all_diff_true.items()
        }
        mean_diff_wrong = {
            action: np.mean(diffs)
            for action, diffs in all_diff_wrong.items()
        }

        # Sort by absolute difference
        top_diff_q = sorted(
            mean_diff_q.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:20]

        top_diff_true = sorted(
            mean_diff_true.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:20]

        top_diff_wrong = sorted(
            mean_diff_wrong.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:20]

        summary = {
            'num_samples': len(results),
            'baseline_accuracy': baseline_accuracy,
            'steered_accuracy': steered_accuracy,
            'accuracy_improvement': steered_accuracy - baseline_accuracy,
            'num_improved': improvements,
            'baseline_prob_true_avg': baseline_prob_true_avg,
            'baseline_prob_wrong_avg': baseline_prob_wrong_avg,
            'steered_prob_true_avg': steered_prob_true_avg,
            'steered_prob_wrong_avg': steered_prob_wrong_avg,
            'top_differences_at_question': top_diff_q,
            'top_differences_after_true': top_diff_true,
            'top_differences_after_wrong': top_diff_wrong,
            'mean_diff_at_question': mean_diff_q,
            'mean_diff_after_true': mean_diff_true,
            'mean_diff_after_wrong': mean_diff_wrong
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
        # Top differences bar plot (now with 3 subplots)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # At question
        actions_q = [item[0] for item in summary['top_differences_at_question'][:15]]
        diffs_q = [item[1] for item in summary['top_differences_at_question'][:15]]

        colors_q = ['green' if d > 0 else 'red' for d in diffs_q]
        ax1.barh(actions_q, diffs_q, color=colors_q, alpha=0.7)
        ax1.set_xlabel('Layer Count Difference (Steered - Baseline)')
        ax1.set_title('Top 15 Cognitive Action Layer Count Differences at Question Point')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax1.grid(axis='x', alpha=0.3)

        # After true answer
        actions_true = [item[0] for item in summary['top_differences_after_true'][:15]]
        diffs_true = [item[1] for item in summary['top_differences_after_true'][:15]]

        colors_true = ['green' if d > 0 else 'red' for d in diffs_true]
        ax2.barh(actions_true, diffs_true, color=colors_true, alpha=0.7)
        ax2.set_xlabel('Layer Count Difference (Steered - Baseline)')
        ax2.set_title('Top 15 Cognitive Action Layer Count Differences After True Answer')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax2.grid(axis='x', alpha=0.3)

        # After wrong answer
        actions_wrong = [item[0] for item in summary['top_differences_after_wrong'][:15]]
        diffs_wrong = [item[1] for item in summary['top_differences_after_wrong'][:15]]

        colors_wrong = ['green' if d > 0 else 'red' for d in diffs_wrong]
        ax3.barh(actions_wrong, diffs_wrong, color=colors_wrong, alpha=0.7)
        ax3.set_xlabel('Layer Count Difference (Steered - Baseline)')
        ax3.set_title('Top 15 Cognitive Action Layer Count Differences After Wrong Answer')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax3.grid(axis='x', alpha=0.3)

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

    def generate_heatmap_visualization(
        self,
        results: List[ActivationResult],
        prefix: str = "cognitive_eval",
        top_n: int = 30
    ):
        """
        Generate heatmap comparing baseline vs steered layer counts for all cognitive actions

        Args:
            results: List of ActivationResult objects
            prefix: Filename prefix
            top_n: Number of top actions to display (based on absolute difference)
        """
        # Aggregate layer counts across all samples for each action
        baseline_counts_q = defaultdict(list)
        steered_counts_q = defaultdict(list)
        baseline_counts_true = defaultdict(list)
        steered_counts_true = defaultdict(list)
        baseline_counts_wrong = defaultdict(list)
        steered_counts_wrong = defaultdict(list)

        for result in results:
            # At question
            for action, count in result.baseline_activations_at_question.items():
                baseline_counts_q[action].append(count)
            for action, count in result.steered_activations_at_question.items():
                steered_counts_q[action].append(count)

            # After true answer
            for action, count in result.baseline_activations_after_true.items():
                baseline_counts_true[action].append(count)
            for action, count in result.steered_activations_after_true.items():
                steered_counts_true[action].append(count)

            # After wrong answer
            for action, count in result.baseline_activations_after_wrong.items():
                baseline_counts_wrong[action].append(count)
            for action, count in result.steered_activations_after_wrong.items():
                steered_counts_wrong[action].append(count)

        # Compute mean counts for each action
        all_actions = set(baseline_counts_q.keys()) | set(steered_counts_q.keys()) | \
                      set(baseline_counts_true.keys()) | set(steered_counts_true.keys()) | \
                      set(baseline_counts_wrong.keys()) | set(steered_counts_wrong.keys())

        action_stats = {}
        for action in all_actions:
            baseline_q_mean = np.mean(baseline_counts_q[action]) if baseline_counts_q[action] else 0.0
            steered_q_mean = np.mean(steered_counts_q[action]) if steered_counts_q[action] else 0.0
            baseline_true_mean = np.mean(baseline_counts_true[action]) if baseline_counts_true[action] else 0.0
            steered_true_mean = np.mean(steered_counts_true[action]) if steered_counts_true[action] else 0.0
            baseline_wrong_mean = np.mean(baseline_counts_wrong[action]) if baseline_counts_wrong[action] else 0.0
            steered_wrong_mean = np.mean(steered_counts_wrong[action]) if steered_counts_wrong[action] else 0.0

            # Total absolute difference across all timepoints
            total_diff = (abs(steered_q_mean - baseline_q_mean) +
                         abs(steered_true_mean - baseline_true_mean) +
                         abs(steered_wrong_mean - baseline_wrong_mean))

            action_stats[action] = {
                'baseline_q': baseline_q_mean,
                'steered_q': steered_q_mean,
                'baseline_true': baseline_true_mean,
                'steered_true': steered_true_mean,
                'baseline_wrong': baseline_wrong_mean,
                'steered_wrong': steered_wrong_mean,
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
                stats['baseline_q'],
                stats['steered_q'],
                stats['baseline_true'],
                stats['steered_true'],
                stats['baseline_wrong'],
                stats['steered_wrong']
            ]
            data_matrix.append(row)

        data_matrix = np.array(data_matrix)

        # Create heatmap
        _, ax = plt.subplots(figsize=(14, max(12, top_n * 0.4)))

        # Use a diverging colormap
        sns.heatmap(
            data_matrix,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            xticklabels=['Baseline\n(at Question)', 'Steered\n(at Question)',
                         'Baseline\n(after True)', 'Steered\n(after True)',
                         'Baseline\n(after Wrong)', 'Steered\n(after Wrong)'],
            yticklabels=actions,
            cbar_kws={'label': 'Mean Layer Count'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )

        ax.set_title(f'Cognitive Action Layer Counts: Baseline vs Steered (Top {top_n})',
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
        default='steering_vectors/tom_caa_forward_belief_new.gguf',
        help='Path to steering vector .gguf file (ignored if --steering-vectors is provided)'
    )
    parser.add_argument(
        '--steering-vectors',
        type=str,
        nargs='+',
        default=None,
        help='Multiple steering vector paths to combine (e.g., --steering-vectors vec1.gguf vec2.gguf)'
    )
    parser.add_argument(
        '--steering-coeffs',
        type=float,
        nargs='+',
        default=None,
        help='Coefficients for each steering vector (e.g., --steering-coeffs 1.0 -0.5). Must match number of vectors.'
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
        default=10,
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
        default=None,
        help='Steering coefficient (strength)'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='procedural_vector',
        help='Output filename prefix'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("COGNITIVE ACTION EVALUATION WITH TOM STEERING")
    print("="*80 + "\n")

    # Initialize evaluator
    evaluator_kwargs = {
        'model_name': args.model,
        'steering_vector_path': args.steering_vector,
        'probes_dir': args.probes_dir,
        'steering_vector_paths': args.steering_vectors,
        'steering_coeffs': args.steering_coeffs
    }

    # Only pass steering_coeff if explicitly provided
    if args.steering_coeff is not None:
        evaluator_kwargs['steering_coeff'] = args.steering_coeff

    evaluator = CognitiveActionEvaluator(**evaluator_kwargs)

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
    print(f"\nAccuracy Metrics:")
    print(f"  Baseline accuracy: {summary['baseline_accuracy']:.2%}")
    print(f"  Steered accuracy: {summary['steered_accuracy']:.2%}")
    print(f"  Accuracy improvement: {summary['accuracy_improvement']:+.2%}")
    print(f"  Samples improved: {summary['num_improved']}")
    print(f"\nProbability Metrics:")
    print(f"  Baseline - Avg p(true): {summary['baseline_prob_true_avg']:.4f}, Avg p(wrong): {summary['baseline_prob_wrong_avg']:.4f}")
    print(f"  Steered  - Avg p(true): {summary['steered_prob_true_avg']:.4f}, Avg p(wrong): {summary['steered_prob_wrong_avg']:.4f}")
    print("\nTop 10 cognitive action differences at question:")
    for action, diff in summary['top_differences_at_question'][:10]:
        print(f"  {action:40s} {diff:+.4f}")
    print("\nTop 10 cognitive action differences after true answer:")
    for action, diff in summary['top_differences_after_true'][:10]:
        print(f"  {action:40s} {diff:+.4f}")
    print("\nTop 10 cognitive action differences after wrong answer:")
    for action, diff in summary['top_differences_after_wrong'][:10]:
        print(f"  {action:40s} {diff:+.4f}")
    print("="*80 + "\n")

    # Save results
    print("Saving results...")
    evaluator.save_results(results, summary, prefix=args.output_prefix)

    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.generate_visualizations(summary, prefix=args.output_prefix)
    evaluator.generate_heatmap_visualization(results, prefix=args.output_prefix)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {evaluator.results_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
