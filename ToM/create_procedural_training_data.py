"""
Create training dataset for ToM steering vectors from procedural-evals-tom data.

This script:
1. Loads forward_belief examples (both true and false belief) from procedural-evals-tom
2. Samples 1,000 examples total (250 from each of 4 conditions)
3. Creates positive/negative training pairs with persona wrapping
4. Saves training data and metadata for tracking which examples were used
"""

import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


def load_csv_stories(csv_path: Path) -> List[Dict[str, str]]:
    """
    Load stories from a semicolon-delimited CSV file.

    Format: Story;Question;Correct_Answer;Incorrect_Answer

    Returns:
        List of dicts with keys: story, question, correct_answer, incorrect_answer
    """
    stories = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) == 4:
                stories.append({
                    'story': row[0].strip(),
                    'question': row[1].strip(),
                    'correct_answer': row[2].strip(),
                    'incorrect_answer': row[3].strip()
                })
    return stories


def create_training_pair(
    story: str,
    question: str,
    correct_answer: str,
    incorrect_answer: str,
    positive_persona: str,
    negative_persona: str
) -> Dict[str, str]:
    """
    Create a positive/negative training pair with persona wrapping.

    Following the approach from tom_steering_colab (1).ipynb:
    - Positive: Good ToM persona + story + question + correct answer
    - Negative: Poor ToM persona + story + question + incorrect answer
    """
    # Construct full text: story + question + answer
    positive_text = f"{positive_persona} {story} {question} {correct_answer}"
    negative_text = f"{negative_persona} {story} {question} {incorrect_answer}"

    return {
        'positive': positive_text,
        'negative': negative_text
    }


def sample_training_data(
    conditions_dir: Path,
    num_per_condition: int = 250,
    random_seed: int = 42
) -> Tuple[List[Dict], Dict]:
    """
    Sample training examples from all 4 forward_belief conditions.

    Args:
        conditions_dir: Path to procedural-evals-tom/data/conditions/
        num_per_condition: Number of examples to sample per condition (default 250)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (training_pairs, metadata)
    """
    random.seed(random_seed)

    # Define personas (following tom_steering_colab approach)
    positive_persona = "Pretend you're someone who is excellent at understanding minds, predicting behavior, and inferring what others think and feel."
    negative_persona = "Pretend you're someone who is terrible at understanding minds, predicting behavior, and inferring what others think and feel."

    # Define the 4 conditions to sample from
    conditions = [
        '0_forward_belief_false_belief',
        '0_forward_belief_true_belief',
        '1_forward_belief_false_belief',
        '1_forward_belief_true_belief'
    ]

    training_pairs = []
    metadata = {
        'num_training_examples': 0,
        'num_examples_per_condition': num_per_condition,
        'random_seed': random_seed,
        'conditions': {},
        'total_available_per_condition': 0,
        'remaining_for_eval': 0
    }

    for condition in conditions:
        csv_path = conditions_dir / condition / 'stories.csv'

        # Load all stories from this condition
        stories = load_csv_stories(csv_path)
        total_available = len(stories)

        # Update metadata
        if metadata['total_available_per_condition'] == 0:
            metadata['total_available_per_condition'] = total_available

        # Sample indices
        all_indices = list(range(total_available))
        sampled_indices = random.sample(all_indices, num_per_condition)
        sampled_indices.sort()  # Sort for easier debugging

        # Store metadata for this condition
        metadata['conditions'][condition] = {
            'indices': sampled_indices,
            'total': len(sampled_indices)
        }

        # Create training pairs from sampled examples
        for idx in sampled_indices:
            story_data = stories[idx]
            pair = create_training_pair(
                story_data['story'],
                story_data['question'],
                story_data['correct_answer'],
                story_data['incorrect_answer'],
                positive_persona,
                negative_persona
            )
            training_pairs.append(pair)

    # Update final metadata
    metadata['num_training_examples'] = len(training_pairs)
    metadata['remaining_for_eval'] = metadata['total_available_per_condition'] - num_per_condition

    return training_pairs, metadata


def main():
    """Main function to generate training data."""
    # Setup paths
    script_dir = Path(__file__).parent
    conditions_dir = script_dir / 'procedural-evals-tom' / 'data' / 'conditions'
    output_dir = script_dir / 'data' / 'datagen'

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if conditions directory exists
    if not conditions_dir.exists():
        raise FileNotFoundError(
            f"Conditions directory not found: {conditions_dir}\n"
            f"Please ensure procedural-evals-tom is in the ToM/ directory"
        )

    print("=" * 80)
    print("Creating ToM Training Dataset from Procedural-Evals-ToM")
    print("=" * 80)
    print(f"\nConditions directory: {conditions_dir}")
    print(f"Output directory: {output_dir}\n")

    # Sample training data
    print("Sampling training examples...")
    training_pairs, metadata = sample_training_data(
        conditions_dir=conditions_dir,
        num_per_condition=250,
        random_seed=42
    )

    print(f"✓ Created {len(training_pairs)} training pairs\n")

    # Print summary
    print("Summary:")
    print("-" * 80)
    for condition, info in metadata['conditions'].items():
        print(f"  {condition}: {info['total']} examples")
        print(f"    Sampled indices: {info['indices'][:5]} ... {info['indices'][-5:]}")
    print(f"\nTotal training examples: {metadata['num_training_examples']}")
    print(f"Remaining for evaluation per condition: {metadata['remaining_for_eval']}")
    print(f"Total remaining for evaluation: {metadata['remaining_for_eval'] * 4}")
    print("-" * 80)

    # Save training data
    training_data_path = output_dir / 'procedural_training_data.json'
    print(f"\nSaving training data to: {training_data_path}")
    with open(training_data_path, 'w', encoding='utf-8') as f:
        json.dump(training_pairs, f, indent=2, ensure_ascii=False)
    print("✓ Training data saved")

    # Save metadata
    metadata_path = output_dir / 'procedural_training_metadata.json'
    print(f"Saving metadata to: {metadata_path}")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("✓ Metadata saved")

    # Print example pair
    print("\n" + "=" * 80)
    print("Example Training Pair:")
    print("=" * 80)
    print("\nPositive (Good ToM):")
    print(training_pairs[0]['positive'][:300] + "...")
    print("\nNegative (Poor ToM):")
    print(training_pairs[0]['negative'][:300] + "...")
    print("=" * 80)

    print("\n✓ Dataset generation complete!")
    print(f"\nNext steps:")
    print(f"1. Use {training_data_path} to train steering vectors")
    print(f"2. Use {metadata_path} to exclude training examples during evaluation")
    print(f"3. Evaluate on the remaining {metadata['remaining_for_eval'] * 4} examples")


if __name__ == '__main__':
    main()
