"""
Split training and evaluation examples for IMPLICIT and EXPLICIT datasets.

This script:
1. Reads the training metadata for both implicit and explicit datasets
2. Creates stories_eval.csv with eval examples only for each condition
3. Can optionally backup original files if not already backed up

This ensures downstream evaluation code can use stories_eval.csv without modification,
while guaranteeing no training examples are included.
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
    dataset_type: str,
    backup: bool = True
) -> Dict[str, int]:
    """
    Create eval-only CSV from a condition's data.

    Args:
        condition_dir: Path to the condition directory
        training_indices: Set of row indices used for training
        dataset_type: "implicit" or "explicit" for naming
        backup: Whether to backup the original file

    Returns:
        Dict with counts of train and eval examples
    """
    # Use original CSV or regular CSV
    csv_path = condition_dir / 'stories_original.csv'
    if not csv_path.exists():
        csv_path = condition_dir / 'stories.csv'

    # Backup original file if needed
    if backup:
        backup_path = condition_dir / 'stories_original.csv'
        if not backup_path.exists() and csv_path.name == 'stories.csv':
            shutil.copy(csv_path, backup_path)
            print(f"  ✓ Backed up original to: stories_original.csv")

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

    # Write eval file with dataset type suffix
    eval_path = condition_dir / f'stories_eval_{dataset_type}.csv'
    write_csv_rows(eval_path, eval_rows)
    print(f"  ✓ Created stories_eval_{dataset_type}.csv with {len(eval_rows)} examples")

    return {
        'total': len(all_rows),
        'train': len(train_rows),
        'eval': len(eval_rows)
    }


def process_dataset(
    conditions_dir: Path,
    metadata_path: Path,
    dataset_type: str
) -> Dict:
    """
    Process a single dataset (implicit or explicit).

    Args:
        conditions_dir: Path to conditions directory
        metadata_path: Path to metadata JSON
        dataset_type: "implicit" or "explicit"

    Returns:
        Results dict with counts
    """
    print(f"\n{'='*80}")
    print(f"Processing {dataset_type.upper()} Dataset")
    print(f"{'='*80}")
    print(f"Metadata file: {metadata_path}\n")

    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"Loaded metadata:")
    print(f"  Condition type: {metadata['condition_type']}")
    print(f"  Prefix: {metadata['prefix']}")
    print(f"  Random seed: {metadata['random_seed']}")
    print(f"  Training examples per condition: {metadata['num_examples_per_condition']}")
    print(f"  Expected eval examples per condition: {metadata['remaining_for_eval']}\n")

    # Process each condition
    results = {}

    for condition_name, condition_info in metadata['conditions'].items():
        print(f"\nProcessing: {condition_name}")
        print("-" * 80)

        condition_dir = conditions_dir / condition_name

        if not condition_dir.exists():
            print(f"  ✗ Directory not found: {condition_dir}")
            continue

        # Convert indices list to set for faster lookup
        training_indices = set(condition_info['indices'])

        # Split the data
        counts = split_condition_data(
            condition_dir=condition_dir,
            training_indices=training_indices,
            dataset_type=dataset_type,
            backup=True
        )

        results[condition_name] = counts

        print(f"  Total: {counts['total']} | Train: {counts['train']} | Eval: {counts['eval']}")

    return results


def main():
    """Main function to split train/eval examples for both datasets."""
    # Setup paths
    script_dir = Path(__file__).parent
    conditions_dir = script_dir / 'procedural-evals-tom' / 'data' / 'conditions'
    datagen_dir = script_dir / 'data' / 'datagen'

    print("=" * 80)
    print("Splitting Training and Evaluation Examples")
    print("IMPLICIT and EXPLICIT Datasets")
    print("=" * 80)
    print(f"\nConditions directory: {conditions_dir}\n")

    # Process IMPLICIT dataset
    metadata_implicit = datagen_dir / 'procedural_training_metadata_implicit.json'
    results_implicit = process_dataset(
        conditions_dir=conditions_dir,
        metadata_path=metadata_implicit,
        dataset_type='implicit'
    )

    # Process EXPLICIT dataset
    metadata_explicit = datagen_dir / 'procedural_training_metadata_explicit.json'
    results_explicit = process_dataset(
        conditions_dir=conditions_dir,
        metadata_path=metadata_explicit,
        dataset_type='explicit'
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print("\n[IMPLICIT Dataset]")
    total_train_implicit = sum(r['train'] for r in results_implicit.values())
    total_eval_implicit = sum(r['eval'] for r in results_implicit.values())
    print(f"  Total training examples: {total_train_implicit}")
    print(f"  Total evaluation examples: {total_eval_implicit}")

    print("\n[EXPLICIT Dataset]")
    total_train_explicit = sum(r['train'] for r in results_explicit.values())
    total_eval_explicit = sum(r['eval'] for r in results_explicit.values())
    print(f"  Total training examples: {total_train_explicit}")
    print(f"  Total evaluation examples: {total_eval_explicit}")

    print("\n" + "=" * 80)
    print("File Structure")
    print("=" * 80)
    print("\nEach 0_ prefix condition directory now contains:")
    print("  - stories_original.csv       (backup of original data)")
    print("  - stories_eval_implicit.csv  (eval examples for implicit vector)")
    print("\nEach 1_ prefix condition directory now contains:")
    print("  - stories_original.csv       (backup of original data)")
    print("  - stories_eval_explicit.csv  (eval examples for explicit vector)")

    print("\n✓ Split complete!")
    print("\nNext steps:")
    print("  1. Use stories_eval_implicit.csv for evaluating the implicit vector")
    print("  2. Use stories_eval_explicit.csv for evaluating the explicit vector")
    print("  3. Keep stories_original.csv as backup of complete dataset")


if __name__ == '__main__':
    main()