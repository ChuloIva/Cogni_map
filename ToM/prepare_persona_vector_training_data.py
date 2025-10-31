"""
Prepare training data for specialized ToM vectors with persona templates.

This script:
1. Selects 750 examples from each of 6 conditions (0_ prefix only)
2. For forward_belief conditions: reuses 250 from existing training + adds 500 new
3. For backward_belief and forward_action: selects fresh 750 examples
4. Creates metadata tracking which examples are used
5. Splits the condition CSV files into train/eval sets
"""

import csv
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict, Set, Tuple


# Configuration
RANDOM_SEED = 42
NUM_EXAMPLES_PER_CONDITION = 750

# Conditions to process (0_ prefix only - implicit belief reasoning)
CONDITIONS = [
    "0_forward_belief_true_belief",
    "0_forward_belief_false_belief",
    "0_backward_belief_true_belief",
    "0_backward_belief_false_belief",
    "0_forward_action_true_belief",
    "0_forward_action_false_belief",
]


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


def select_training_indices(
    condition_name: str,
    total_available: int,
    num_to_select: int,
    existing_indices: Set[int] = None
) -> Tuple[List[int], int]:
    """
    Select training indices for a condition.

    Args:
        condition_name: Name of the condition
        total_available: Total number of examples available
        num_to_select: Number of examples to select (750)
        existing_indices: Existing training indices if reusing (for forward_belief)

    Returns:
        Tuple of (list of indices, number of new indices added)
    """
    if existing_indices is not None:
        # Reuse existing + add more to reach 750
        existing_list = sorted(list(existing_indices))
        num_existing = len(existing_list)
        num_new_needed = num_to_select - num_existing

        print(f"  Reusing {num_existing} existing training examples")
        print(f"  Adding {num_new_needed} new examples to reach {num_to_select}")

        # Get available indices (not in existing)
        available_indices = [i for i in range(total_available) if i not in existing_indices]

        # Randomly select new indices
        random.seed(RANDOM_SEED)
        new_indices = random.sample(available_indices, num_new_needed)

        # Combine and sort
        all_indices = existing_list + new_indices
        return sorted(all_indices), num_new_needed
    else:
        # Fresh selection
        print(f"  Selecting {num_to_select} fresh examples")
        random.seed(RANDOM_SEED)
        indices = random.sample(range(total_available), num_to_select)
        return sorted(indices), num_to_select


def split_condition_data(
    condition_dir: Path,
    training_indices: Set[int],
    backup: bool = True
) -> Dict[str, int]:
    """
    Split a condition's stories.csv into train and eval sets.

    Args:
        condition_dir: Path to the condition directory
        training_indices: Set of row indices used for training
        backup: Whether to backup the original file

    Returns:
        Dict with counts of train and eval examples
    """
    csv_path = condition_dir / 'stories.csv'

    # Check if already split
    backup_path = condition_dir / 'stories_original.csv'
    if backup_path.exists():
        # Already backed up, load from backup
        print(f"  Found existing backup, loading from stories_original.csv")
        source_path = backup_path
    else:
        # First time, backup original
        if backup:
            shutil.copy(csv_path, backup_path)
            print(f"  ✓ Backed up original to: stories_original.csv")
        source_path = csv_path

    # Load all rows
    all_rows = load_csv_rows(source_path)

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

    # Replace stories.csv with eval examples
    write_csv_rows(csv_path, eval_rows)
    print(f"  ✓ Replaced stories.csv with {len(eval_rows)} eval examples")

    return {
        'total': len(all_rows),
        'train': len(train_rows),
        'eval': len(eval_rows)
    }


def main():
    """Main function to prepare training data."""
    script_dir = Path(__file__).parent
    conditions_dir = script_dir / 'procedural-evals-tom' / 'data' / 'conditions'

    # Load existing forward_belief training indices (if available)
    existing_metadata_path = script_dir / 'data' / 'datagen' / 'procedural_training_metadata.json'
    existing_indices = {}

    if existing_metadata_path.exists():
        with open(existing_metadata_path, 'r', encoding='utf-8') as f:
            existing_metadata = json.load(f)
            for condition_name, info in existing_metadata['conditions'].items():
                existing_indices[condition_name] = set(info['indices'])
            print(f"Loaded existing training indices for {len(existing_indices)} conditions\n")

    print("=" * 80)
    print("Preparing Training Data for Persona-Based ToM Vectors")
    print("=" * 80)
    print(f"\nConditions directory: {conditions_dir}")
    print(f"Training examples per condition: {NUM_EXAMPLES_PER_CONDITION}")
    print(f"Random seed: {RANDOM_SEED}\n")

    # Process each condition
    metadata = {
        'num_training_examples_per_condition': NUM_EXAMPLES_PER_CONDITION,
        'random_seed': RANDOM_SEED,
        'conditions': {}
    }

    results = {}

    for condition_name in CONDITIONS:
        print(f"\nProcessing: {condition_name}")
        print("-" * 80)

        condition_dir = conditions_dir / condition_name

        if not condition_dir.exists():
            print(f"  ✗ Directory not found: {condition_dir}")
            continue

        # Check if we have existing indices to reuse
        existing = existing_indices.get(condition_name)

        # Determine total available examples
        # Load from backup if it exists, otherwise from stories.csv
        backup_path = condition_dir / 'stories_original.csv'
        if backup_path.exists():
            source_path = backup_path
        else:
            source_path = condition_dir / 'stories.csv'

        all_rows = load_csv_rows(source_path)
        total_available = len(all_rows)

        print(f"  Total available examples: {total_available}")

        # Select training indices
        training_indices, num_new = select_training_indices(
            condition_name=condition_name,
            total_available=total_available,
            num_to_select=NUM_EXAMPLES_PER_CONDITION,
            existing_indices=existing
        )

        # Split the data
        counts = split_condition_data(
            condition_dir=condition_dir,
            training_indices=set(training_indices),
            backup=True
        )

        # Store metadata
        metadata['conditions'][condition_name] = {
            'indices': training_indices,
            'total': counts['total'],
            'train': counts['train'],
            'eval': counts['eval'],
            'reused_from_previous': len(existing) if existing else 0,
            'newly_added': num_new
        }

        results[condition_name] = counts

        print(f"  Total: {counts['total']} | Train: {counts['train']} | Eval: {counts['eval']}")

    # Save metadata
    output_metadata_path = script_dir / 'data' / 'datagen' / 'persona_vector_training_metadata.json'
    with open(output_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

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
    for condition_name in CONDITIONS:
        if condition_name in results:
            counts = results[condition_name]
            cond_meta = metadata['conditions'][condition_name]
            reused = cond_meta['reused_from_previous']
            new = cond_meta['newly_added']
            print(f"  {condition_name}:")
            print(f"    Train: {counts['train']} (reused: {reused}, new: {new}) | Eval: {counts['eval']}")

    print("\n" + "=" * 80)
    print("File Structure")
    print("=" * 80)
    print("\nEach condition directory now contains:")
    print(f"  - stories_original.csv  (backup of original data with all {metadata['conditions'][CONDITIONS[0]]['total']} examples)")
    print(f"  - stories_train.csv     ({NUM_EXAMPLES_PER_CONDITION} examples used for training)")
    print(f"  - stories.csv           (eval examples - training examples removed)")

    print(f"\n✓ Metadata saved to: {output_metadata_path}")
    print("\n✓ Data preparation complete!")
    print("\nNext steps:")
    print("  1. Use stories_train.csv files for vector training")
    print("  2. Use stories.csv files for downstream evaluation")
    print("  3. Training examples are tracked in persona_vector_training_metadata.json")


if __name__ == '__main__':
    main()
