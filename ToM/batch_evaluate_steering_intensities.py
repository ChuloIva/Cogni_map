#!/usr/bin/env python3
"""
Batch Evaluation Script for Steering Vector Intensity Analysis

This script runs systematic evaluations of ToM steering vectors across:
- Multiple steering vector intensities (500 to 2000)
- Multiple matched conditions from the ToM benchmark
- 100 samples per condition

The results are saved with descriptive filenames for later analysis.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import json
from tqdm import tqdm

# Define steering vector to condition mappings
VECTOR_CONDITION_MAPPINGS = {
    "tom_backward_belief.gguf": [
        "0_backward_belief_false_belief",
        "0_backward_belief_true_belief",
        "1_backward_belief_false_belief",
        "1_backward_belief_true_belief",
    ],
    "tom_forward_belief.gguf": [
        "0_forward_belief_false_belief",
        "0_forward_belief_true_belief",
        "1_forward_belief_false_belief",
        "1_forward_belief_true_belief",
    ],
    "tom_forward_action.gguf": [
        "0_forward_action_false_belief",
        "0_forward_action_true_belief",
        "1_forward_action_false_belief",
        "1_forward_action_true_belief",
    ],
    "tom_general.gguf": [
        # Test general vector on all condition types (sample a few)
        "0_backward_belief_false_belief",
        "0_forward_action_false_belief",
        "0_forward_belief_false_belief",
    ],
}

# Additional vectors to test on select conditions
ADDITIONAL_VECTORS = {
    "tom_core_capabilities.gguf": ["0_backward_belief_false_belief"],
    "tom_direction.gguf": [
        "0_forward_action_false_belief",
        "0_backward_belief_false_belief",
    ],
    "tom_belief_type.gguf": [
        "0_backward_belief_false_belief",
        "0_forward_belief_false_belief",
    ],
}

# Combine all mappings
VECTOR_CONDITION_MAPPINGS.update(ADDITIONAL_VECTORS)


class BatchEvaluator:
    """
    Manages batch evaluation of steering vectors with varying intensities
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        probes_dir: str = "data/probes_binary",
        num_samples: int = 100,
        intensity_range: Tuple[int, int, int] = (500, 2000, 250),
        dry_run: bool = False
    ):
        """
        Initialize batch evaluator

        Args:
            model_name: HuggingFace model name
            probes_dir: Directory containing trained probes
            num_samples: Number of samples per evaluation
            intensity_range: (min, max, step) for steering coefficients
            dry_run: If True, print commands without executing
        """
        self.model_name = model_name
        self.probes_dir = probes_dir
        self.num_samples = num_samples
        self.intensity_range = intensity_range
        self.dry_run = dry_run

        # Setup paths
        self.script_dir = Path(__file__).parent
        self.eval_script = self.script_dir / "evaluate_cognitive_actions_with_steering.py"
        self.results_base_dir = self.script_dir / "batch_results"

        # Create timestamped results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.results_base_dir / f"batch_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Generate intensity values
        min_val, max_val, step = intensity_range
        self.intensities = list(range(min_val, max_val + 1, step))

        print(f"\n{'='*80}")
        print(f"BATCH EVALUATION CONFIGURATION")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Probes directory: {probes_dir}")
        print(f"Samples per evaluation: {num_samples}")
        print(f"Steering intensities: {self.intensities}")
        print(f"Results directory: {self.results_dir}")
        print(f"Dry run: {dry_run}")
        print(f"{'='*80}\n")

        # Track experiment metadata
        self.experiments = []
        self.completed = []
        self.failed = []

    def generate_experiments(self) -> List[Dict]:
        """
        Generate all experiment configurations

        Returns:
            List of experiment dicts with parameters
        """
        experiments = []

        for vector_name, conditions in VECTOR_CONDITION_MAPPINGS.items():
            for condition in conditions:
                for intensity in self.intensities:
                    experiment = {
                        'vector': vector_name,
                        'condition': condition,
                        'intensity': intensity,
                        'num_samples': self.num_samples,
                        'output_prefix': self._generate_output_prefix(
                            vector_name, condition, intensity
                        )
                    }
                    experiments.append(experiment)

        return experiments

    def _generate_output_prefix(
        self,
        vector_name: str,
        condition: str,
        intensity: int
    ) -> str:
        """Generate descriptive output filename prefix"""
        # Extract vector type (e.g., "backward_belief" from "tom_backward_belief.gguf")
        vector_type = vector_name.replace("tom_", "").replace(".gguf", "")

        # Create compact but descriptive prefix
        prefix = f"{vector_type}__{condition}__coeff{intensity}"

        return prefix

    def run_experiment(self, experiment: Dict) -> bool:
        """
        Run a single experiment

        Args:
            experiment: Experiment configuration dict

        Returns:
            True if successful, False otherwise
        """
        vector_path = f"steering_vectors/{experiment['vector']}"

        # Build command
        cmd = [
            sys.executable,
            str(self.eval_script),
            "--model", self.model_name,
            "--steering-vector", vector_path,
            "--probes-dir", self.probes_dir,
            "--condition", experiment['condition'],
            "--num-samples", str(experiment['num_samples']),
            "--steering-coeff", str(experiment['intensity']),
            "--output-prefix", experiment['output_prefix']
        ]

        print(f"\n{'='*80}")
        print(f"Running: {experiment['vector'].replace('.gguf', '')} | "
              f"{experiment['condition']} | intensity={experiment['intensity']}")
        print(f"{'='*80}")

        if self.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}\n")
            return True

        try:
            # Run with streaming output and progress bar
            process = subprocess.Popen(
                cmd,
                cwd=self.script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Progress bar for examples
            pbar = tqdm(
                total=experiment['num_samples'],
                desc="Processing examples",
                unit="example",
                leave=False
            )

            passed = 0
            failed = 0

            # Read output line by line
            while True:
                line = process.stdout.readline()
                if not line:
                    break

                # Detect when a sample completes (look for the completion marker)
                if "✓ Baseline complete" in line or "✓ Steered complete" in line:
                    # Each sample has both baseline and steered, so update on steered
                    if "Steered complete" in line:
                        pbar.update(1)
                        # Check if correct
                        if "correct=True" in line:
                            passed += 1
                        else:
                            failed += 1
                        pbar.set_postfix({'passed': passed, 'failed': failed})

            pbar.close()

            # Wait for process to complete
            returncode = process.wait(timeout=3600)

            if returncode == 0:
                print(f"✓ Completed: {passed} passed, {failed} failed")

                # Move results to batch results directory
                self._move_results(experiment['output_prefix'])

                return True
            else:
                stderr = process.stderr.read()
                print(f"✗ Experiment failed with return code {returncode}")
                if stderr:
                    print(f"STDERR: {stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"✗ Experiment timed out after 1 hour")
            process.kill()
            return False
        except Exception as e:
            print(f"✗ Experiment failed with error: {e}")
            return False

    def _move_results(self, output_prefix: str):
        """Move result files to batch results directory"""
        results_dir = self.script_dir / "results"

        # Files to move
        file_patterns = [
            f"{output_prefix}_raw.csv",
            f"{output_prefix}_summary.json",
            f"{output_prefix}_differences.png",
            f"{output_prefix}_accuracy.png"
        ]

        for pattern in file_patterns:
            src = results_dir / pattern
            if src.exists():
                dst = self.results_dir / pattern
                src.rename(dst)
                print(f"  Moved: {pattern}")

    def run_all_experiments(self):
        """Run all experiments in sequence"""
        experiments = self.generate_experiments()

        print(f"\n{'='*80}")
        print(f"STARTING BATCH EVALUATION")
        print(f"{'='*80}")
        print(f"Total experiments: {len(experiments)}")
        print(f"Estimated time: ~{len(experiments) * 5} minutes (5 min/experiment)")
        print(f"{'='*80}\n")

        # Use tqdm for progress tracking
        pbar = tqdm(experiments, desc="Overall Progress", unit="experiment")

        for i, experiment in enumerate(pbar, 1):
            # Update progress bar description with current experiment details
            pbar.set_description(
                f"Exp {i}/{len(experiments)}: {experiment['vector'].replace('.gguf', '')} | "
                f"{experiment['condition']} | intensity={experiment['intensity']}"
            )

            success = self.run_experiment(experiment)

            experiment['success'] = success
            experiment['timestamp'] = datetime.now().isoformat()
            self.experiments.append(experiment)

            if success:
                self.completed.append(experiment)
            else:
                self.failed.append(experiment)

            # Update progress bar postfix with success/failure counts
            pbar.set_postfix({
                'completed': len(self.completed),
                'failed': len(self.failed)
            })

            # Save progress after each experiment
            self._save_metadata()

        self._print_final_summary()

    def _save_metadata(self):
        """Save experiment metadata and progress"""
        metadata = {
            'configuration': {
                'model_name': self.model_name,
                'probes_dir': self.probes_dir,
                'num_samples': self.num_samples,
                'intensities': self.intensities,
                'timestamp': datetime.now().isoformat()
            },
            'experiments': self.experiments,
            'summary': {
                'total': len(self.experiments),
                'completed': len(self.completed),
                'failed': len(self.failed)
            }
        }

        metadata_path = self.results_dir / "batch_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _print_final_summary(self):
        """Print final summary of all experiments"""
        print(f"\n\n{'='*80}")
        print(f"BATCH EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total experiments: {len(self.experiments)}")
        print(f"Completed successfully: {len(self.completed)}")
        print(f"Failed: {len(self.failed)}")
        print(f"\nResults saved to: {self.results_dir}")

        if self.failed:
            print(f"\n{'='*80}")
            print(f"FAILED EXPERIMENTS:")
            print(f"{'='*80}")
            for exp in self.failed:
                print(f"  - {exp['vector']} / {exp['condition']} / intensity={exp['intensity']}")

        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluation of steering vectors with varying intensities'
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
        default='data/probes_binary',
        help='Directory containing trained probes'
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=100,
        help='Number of samples per evaluation'
    )
    parser.add_argument(
        '--intensity-min',
        type=int,
        default=500,
        help='Minimum steering intensity'
    )
    parser.add_argument(
        '--intensity-max',
        type=int,
        default=2000,
        help='Maximum steering intensity'
    )
    parser.add_argument(
        '--intensity-step',
        type=int,
        default=250,
        help='Step size for steering intensity'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )
    parser.add_argument(
        '--vector',
        type=str,
        help='Run only specific steering vector (e.g., tom_backward_belief.gguf)'
    )
    parser.add_argument(
        '--condition',
        type=str,
        help='Run only specific condition (e.g., 0_backward_belief_false_belief)'
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = BatchEvaluator(
        model_name=args.model,
        probes_dir=args.probes_dir,
        num_samples=args.num_samples,
        intensity_range=(args.intensity_min, args.intensity_max, args.intensity_step),
        dry_run=args.dry_run
    )

    # Filter experiments if specific vector/condition requested
    if args.vector or args.condition:
        experiments = evaluator.generate_experiments()

        filtered_experiments = []
        for exp in experiments:
            if args.vector and exp['vector'] != args.vector:
                continue
            if args.condition and exp['condition'] != args.condition:
                continue
            filtered_experiments.append(exp)

        print(f"\nFiltered to {len(filtered_experiments)} experiments")

        # Replace mapping with filtered experiments
        VECTOR_CONDITION_MAPPINGS.clear()

        for exp in filtered_experiments:
            if exp['vector'] not in VECTOR_CONDITION_MAPPINGS:
                VECTOR_CONDITION_MAPPINGS[exp['vector']] = []
            if exp['condition'] not in VECTOR_CONDITION_MAPPINGS[exp['vector']]:
                VECTOR_CONDITION_MAPPINGS[exp['vector']].append(exp['condition'])

    # Run all experiments
    evaluator.run_all_experiments()


if __name__ == '__main__':
    main()
