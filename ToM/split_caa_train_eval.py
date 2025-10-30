"""
Split training and evaluation examples for CAA steering vectors from procedural-evals-tom.

This script:
1. Reads the CAA training metadata to identify which examples were used for training
2. Backs up the original stories.csv to stories_original.csv (if not already backed up)
3. Creates stories_train.csv with training examples
4. Replaces stories.csv with eval examples only

This ensures downstream evaluation code can use stories.csv without modification,
while guaranteeing no training examples are included.

Note: This operates on BOTH false_belief and true_belief conditions since CAA
uses matched pairs from both.
"""

import csv
import json
import shutil
from pathlib import Path
from typing import List, Dict, Set


def load_csv_rows(csv_path: Path) -> List[List[str]]:
    """Load all rows from a CSV file."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            rows.append(row)
    return rows


def write_csv_rows(csv_path: Path, rows: List[List[str]]):
    """Write rows to a CSV file."""
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerows(rows)


def split_condition_data(
    condition_dir: Path,
    training_indices: Set[int],
    backup: bool = True
) -> Dict[str, int]:
    """
    Split a condition's stories.csv into train and eval sets.
    Replaces stories.csv with eval examples only (for downstream use).

    Args:
        condition_dir: Path to the condition directory
        training_indices: Set of row indices used for training
        backup: Whether to backup the original file

    Returns:
        Dict with counts of train and eval examples
    """
    csv_path = condition_dir / 'stories.csv'

    # Check if original backup already exists
    backup_path = condition_dir / 'stories_original.csv'
    if backup and not backup_path.exists():
        shutil.copy(csv_path, backup_path)
        print(f"  ✓ Backed up original to: stories_original.csv")
    elif backup_path.exists():
        print(f"  ℹ Using existing backup: stories_original.csv")

    # Load all rows
    all_rows = load_csv_rows(csv_path)

    # Split into train and eval
    train_rows = []
    eval_rows = []

    for idx, row in enumerate(all_rows):
        if idx in training_indices:
            train_rows.append(row)
        else:
            eval_rows.append(row)

    # Write train file
    train_path = condition_dir / 'stories_train.csv'
    write_csv_rows(train_path, train_rows)
    print(f"  ✓ Created stories_train.csv with {len(train_rows)} examples")

    # Replace stories.csv with eval examples (so existing code works)
    write_csv_rows(csv_path, eval_rows)
    print(f"  ✓ Replaced stories.csv with {len(eval_rows)} eval examples")

    return {
        'total': len(all_rows),
        'train': len(train_rows),
        'eval': len(eval_rows)
    }


def main():
    """Main function to split train/eval examples."""
    # Setup paths
    script_dir = Path(__file__).parent
    conditions_dir = script_dir / 'procedural-evals-tom' / 'data' / 'conditions'
    metadata_path = script_dir / 'data' / 'datagen' / 'caa_training_metadata.json'

    print("=" * 80)
    print("Splitting CAA Training and Evaluation Examples")
    print("=" * 80)
    print(f"\nConditions directory: {conditions_dir}")
    print(f"Metadata file: {metadata_path}\n")

    # Check if metadata exists
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"Please run create_caa_training_data.py first"
        )

    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"Loaded metadata:")
    print(f"  Random seed: {metadata['random_seed']}")
    print(f"  Training examples per prefix: {metadata['num_examples_per_prefix']}")
    print(f"  Expected eval examples per prefix: {metadata['remaining_for_eval']}\n")

    # Process each prefix condition
    # For CAA, we need to split BOTH false_belief and true_belief conditions
    # since we use matched pairs
    results = {}

    for prefix_condition, condition_info in metadata['conditions'].items():
        training_indices = set(condition_info['indices'])

        # Process both false_belief and true_belief for this prefix
        for belief_type in ['false_belief', 'true_belief']:
            condition_name = f"{prefix_condition}_{belief_type}"

            print(f"\nProcessing: {condition_name}")
            print("-" * 80)

            condition_dir = conditions_dir / condition_name

            if not condition_dir.exists():
                print(f"  ✗ Directory not found: {condition_dir}")
                continue

            # Split the data
            counts = split_condition_data(
                condition_dir=condition_dir,
                training_indices=training_indices,
                backup=True
            )

            results[condition_name] = counts

            print(f"  Total: {counts['total']} | Train: {counts['train']} | Eval: {counts['eval']}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    total_train = sum(r['train'] for r in results.values())
    total_eval = sum(r['eval'] for r in results.values())
    total_examples = sum(r['total'] for r in results.values())

    print(f"\nTotal examples processed: {total_examples}")
    print(f"Total training examples: {total_train}")
    print(f"Total evaluation examples: {total_eval}")

    print(f"\nBreakdown by condition:")
    for condition_name, counts in results.items():
        print(f"  {condition_name}:")
        print(f"    Train: {counts['train']} | Eval: {counts['eval']}")

    print("\n" + "=" * 80)
    print("File Structure")
    print("=" * 80)
    print("\nEach condition directory now contains:")
    print("  - stories_original.csv  (backup of original data with all examples)")
    print("  - stories_train.csv     (examples used for CAA training)")
    print("  - stories.csv           (eval examples - training examples removed)")

    print("\n✓ Split complete!")
    print("\nNext steps:")
    print("  1. Use stories.csv files for downstream evaluation (existing code will work)")
    print("  2. Use stories_train.csv files for reference (already used in training)")
    print("  3. Keep stories_original.csv as backup of complete dataset")


if __name__ == '__main__':
    main()
