import os
import random
import csv
import json
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple

# Configure AMD GPU BEFORE importing torch
try:
    from gpu_utils import configure_amd_gpu, get_optimal_device
    configure_amd_gpu()
except ImportError:
    print("⚠️  gpu_utils.py not found - skipping AMD GPU configuration")
    def get_optimal_device():
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)  # Go up to 'code' directory

DATA_DIR = os.path.join(CODE_DIR, '..', 'data')
CONDITION_DIR = os.path.join(DATA_DIR, 'conditions')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
PROMPT_DIR = os.path.join(CODE_DIR, 'prompt_instructions')
random.seed(0)


class GemmaEvaluator:
    """Evaluator for Gemma models on Theory of Mind tasks"""

    def __init__(self, model_name: str = "google/gemma-3-4b-it", temperature: float = 0.0,
                 max_tokens: int = 100, device: str = None):
        """
        Initialize the Gemma evaluator

        Args:
            model_name: HuggingFace model name or path
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            device: Device to run model on (auto-detect if None)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Auto-detect device
        if device is None:
            self.device = get_optimal_device()
        else:
            self.device = device

        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        print(f"Target device: {self.device}")
        print(f"{'='*60}\n")

        # Load tokenizer and model with progress bar
        with tqdm(total=2, desc="Loading model components") as pbar:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=os.getenv('HUGGINGFACE_TOKEN')
            )
            pbar.update(1)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=os.getenv('HUGGINGFACE_TOKEN'),
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            pbar.update(1)

        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"Model loaded successfully!")

        # Load instruction
        with open(f'{PROMPT_DIR}/evaluate.txt', 'r') as f:
            self.instruction = f.read()

    def format_prompt(self, story: str, question: str, mcq: bool = True) -> str:
        """Format the prompt for Gemma chat template"""
        if mcq:
            prompt = f"{self.instruction}\n\nStory: {story}\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"{self.instruction}\n\nStory: {story}\nQuestion: {question}\nAnswer:"
        return prompt

    def predict_answer(self, story: str, question: str) -> str:
        """Generate answer for a given story and question"""
        prompt = self.format_prompt(story, question)

        # Format with chat template if available
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
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            if self.temperature > 0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

        # Decode
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def grade_answer(self, predicted_answer: str, true_answer: str,
                     wrong_answer: str, answer_key: str, negative_answer_key: str) -> bool:
        """
        Grade the predicted answer

        Args:
            predicted_answer: The model's predicted answer
            true_answer: The correct answer (with prefix like 'a)')
            wrong_answer: The incorrect answer (with prefix like 'b)')
            answer_key: The key for correct answer ('a)' or 'b)')
            negative_answer_key: The key for incorrect answer

        Returns:
            True if answer is correct, False otherwise
        """
        predicted_lower = predicted_answer.lower()

        # Check for explicit answer key
        if answer_key in predicted_lower:
            return True
        elif negative_answer_key in predicted_lower:
            return False

        # Check for answer content match (fuzzy matching)
        # Remove the prefix from true/wrong answers for content comparison
        true_content = re.sub(r'^[ab]\)\s*', '', true_answer, flags=re.IGNORECASE).lower()
        wrong_content = re.sub(r'^[ab]\)\s*', '', wrong_answer, flags=re.IGNORECASE).lower()

        # Simple keyword matching - check if key phrases from true answer appear
        true_keywords = set(true_content.split())
        wrong_keywords = set(wrong_content.split())
        pred_keywords = set(predicted_lower.split())

        true_overlap = len(true_keywords & pred_keywords)
        wrong_overlap = len(wrong_keywords & pred_keywords)

        if true_overlap > wrong_overlap:
            return True
        elif wrong_overlap > true_overlap:
            return False

        # If still unclear, return None to indicate uncertain grading
        return None


def evaluate_condition(evaluator: GemmaEvaluator, init_belief: str,
                      variable: str, condition: str, num_samples: int = None,
                      verbose: bool = False, offset: int = 0) -> Dict:
    """
    Evaluate model on a specific condition

    Args:
        evaluator: GemmaEvaluator instance
        init_belief: Initial belief condition (e.g., '0_backward', '1_forward')
        variable: Variable type ('belief' or 'action')
        condition: Condition type ('true_belief' or 'false_belief')
        num_samples: Number of samples to evaluate (None = all)
        verbose: Print detailed output
        offset: Start from this offset in the dataset

    Returns:
        Dictionary with results
    """

    # Load condition data
    csv_name = os.path.join(CONDITION_DIR, f'{init_belief}_{variable}_{condition}/stories.csv')
    print(f"\nLoading condition: {init_belief}_{variable}_{condition}")

    with open(csv_name, "r") as f:
        reader = csv.reader(f, delimiter=";")
        condition_rows = list(reader)

    # Determine number of samples
    total_rows = len(condition_rows)
    if num_samples is None:
        num_samples = total_rows - offset
    else:
        num_samples = min(num_samples, total_rows - offset)

    print(f"Evaluating {num_samples} samples (offset: {offset}, total available: {total_rows})")

    predicted_answers = []
    graded_answers = []
    uncertain_grades = []

    # Evaluate each sample with progress bar
    for idx, row in enumerate(tqdm(condition_rows[offset:offset+num_samples],
                                    desc=f"Evaluating {init_belief}_{variable}_{condition}")):
        story = row[0]
        question_orig = row[1]
        true_answer_base, wrong_answer_base = row[2], row[3]

        # Shuffle answers
        answers = [true_answer_base, wrong_answer_base]
        random.shuffle(answers)

        # Create MCQ format
        question = f"{question_orig}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"

        # Get prediction
        predicted_answer = evaluator.predict_answer(story, question)

        # Determine answer keys
        if answers[0] == true_answer_base:
            answer_key = 'a)'
            negative_answer_key = 'b)'
            true_answer = 'a) ' + true_answer_base
            wrong_answer = 'b) ' + wrong_answer_base
        else:
            answer_key = 'b)'
            negative_answer_key = 'a)'
            true_answer = 'b) ' + true_answer_base
            wrong_answer = 'a) ' + wrong_answer_base

        # Grade answer
        is_correct = evaluator.grade_answer(
            predicted_answer, true_answer, wrong_answer,
            answer_key, negative_answer_key
        )

        if is_correct is None:
            uncertain_grades.append(idx + offset)
            is_correct = False  # Count as incorrect if uncertain

        predicted_answers.append(predicted_answer)
        graded_answers.append('True' if is_correct else 'False')

        if verbose:
            print(f"\n{'='*60}")
            print(f"Sample {idx + offset + 1}")
            print(f"Story: {story[:100]}...")
            print(f"Question: {question}")
            print(f"True answer: {true_answer}")
            print(f"Predicted: {predicted_answer}")
            print(f"Correct: {is_correct}")
            print(f"{'='*60}")

    # Save results
    model_name_safe = evaluator.model_name.replace('/', '_')
    result_dir = os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}')
    os.makedirs(result_dir, exist_ok=True)

    prediction_file = os.path.join(
        result_dir,
        f'prediction_{model_name_safe}_{evaluator.temperature}_{variable}_{condition}_{offset}_{num_samples}.csv'
    )
    accuracy_file = os.path.join(
        result_dir,
        f'accuracy_{model_name_safe}_{evaluator.temperature}_{variable}_{condition}_{offset}_{num_samples}.csv'
    )

    # Save predictions
    with open(prediction_file, "w") as f:
        writer = csv.writer(f, delimiter=";")
        for predicted_answer in predicted_answers:
            writer.writerow([predicted_answer])

    # Save graded answers
    with open(accuracy_file, "w") as f:
        writer = csv.writer(f, delimiter=";")
        for graded_answer in graded_answers:
            writer.writerow([graded_answer])

    # Calculate accuracy
    accuracy = graded_answers.count('True') / len(graded_answers)

    # Print results
    print("\n" + "="*60)
    print(" "*20 + "RESULTS")
    print("="*60)
    print(f"Model: {evaluator.model_name}")
    print(f"Temperature: {evaluator.temperature}")
    print(f"Condition: {init_belief}_{variable}_{condition}")
    print(f"Samples evaluated: {num_samples}")
    print(f"Accuracy: {accuracy:.2%} ({graded_answers.count('True')}/{len(graded_answers)})")
    if uncertain_grades:
        print(f"Uncertain grades: {len(uncertain_grades)} samples")
    print("="*60 + "\n")

    return {
        'condition': f'{init_belief}_{variable}_{condition}',
        'accuracy': accuracy,
        'num_samples': num_samples,
        'num_correct': graded_answers.count('True'),
        'uncertain_grades': len(uncertain_grades)
    }


def evaluate_all_conditions(model_name: str, temperature: float = 0.0,
                           num_samples: int = 100, max_tokens: int = 100,
                           verbose: bool = False, offset: int = 0) -> Dict:
    """
    Evaluate model on all conditions

    Returns:
        Dictionary with aggregated results
    """

    # Load model once for all conditions
    print("\n" + "="*60)
    print("Initializing model...")
    print("="*60)

    evaluator = GemmaEvaluator(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Define all conditions
    init_beliefs = ['0_backward', '0_forward', '1_backward', '1_forward']
    variables = ['belief', 'action']
    conditions = ['true_belief', 'false_belief']

    all_results = []

    print("\n" + "="*60)
    print("Starting evaluation on all conditions")
    print("="*60)

    # Create overall progress bar
    total_conditions = len(init_beliefs) * len(variables) * len(conditions)

    with tqdm(total=total_conditions, desc="Overall progress") as pbar:
        for init_belief in init_beliefs:
            for variable in variables:
                for condition in conditions:
                    # Skip action with true_belief/false_belief that don't exist
                    condition_path = os.path.join(
                        CONDITION_DIR,
                        f'{init_belief}_{variable}_{condition}'
                    )
                    if not os.path.exists(condition_path):
                        pbar.update(1)
                        continue

                    result = evaluate_condition(
                        evaluator=evaluator,
                        init_belief=init_belief,
                        variable=variable,
                        condition=condition,
                        num_samples=num_samples,
                        verbose=verbose,
                        offset=offset
                    )
                    all_results.append(result)
                    pbar.update(1)

    # Aggregate results
    total_samples = sum(r['num_samples'] for r in all_results)
    total_correct = sum(r['num_correct'] for r in all_results)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Save summary
    model_name_safe = model_name.replace('/', '_')

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_file = os.path.join(
        RESULTS_DIR,
        f'summary_{model_name_safe}_{temperature}_{offset}_{num_samples or "all"}.json'
    )

    summary = {
        'model_name': model_name,
        'temperature': temperature,
        'overall_accuracy': overall_accuracy,
        'total_samples': total_samples,
        'total_correct': total_correct,
        'condition_results': all_results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print(" "*15 + "OVERALL SUMMARY")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Overall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_samples})")
    print(f"\nResults by condition:")
    print("-"*60)
    for result in all_results:
        print(f"  {result['condition']:45} {result['accuracy']:6.2%}")
    print("="*60)
    print(f"\nDetailed results saved to: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Gemma models on Theory of Mind tasks'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='google/gemma-3-4b-it',
        help='HuggingFace model name or local path'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--num_samples', '-n',
        type=int,
        default=100,
        help='Number of samples per condition (default: 100)'
    )
    parser.add_argument(
        '--offset', '-o',
        type=int,
        default=0,
        help='Offset to start from in dataset'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=100,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output for each sample'
    )
    parser.add_argument(
        '--condition',
        type=str,
        default=None,
        help='Evaluate single condition (format: init_belief_variable_condition)'
    )

    args = parser.parse_args()

    if args.condition:
        # Evaluate single condition
        parts = args.condition.split('_')
        if len(parts) != 4:
            print("Error: Condition format should be: init_belief_variable_condition")
            print("Example: 0_backward_belief_false_belief")
            return

        init_belief = f"{parts[0]}_{parts[1]}"
        variable = parts[2]
        condition = parts[3]

        # Initialize evaluator
        print("\n" + "="*60)
        print("Initializing model...")
        print("="*60)
        evaluator = GemmaEvaluator(
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )

        evaluate_condition(
            evaluator=evaluator,
            init_belief=init_belief,
            variable=variable,
            condition=condition,
            num_samples=args.num_samples,
            verbose=args.verbose,
            offset=args.offset
        )
    else:
        # Evaluate all conditions
        evaluate_all_conditions(
            model_name=args.model_name,
            temperature=args.temperature,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            offset=args.offset
        )


if __name__ == '__main__':
    main()
