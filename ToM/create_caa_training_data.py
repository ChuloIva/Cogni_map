"""
Create CAA-style training dataset for ToM steering vectors from procedural-evals-tom data.

This script replicates the BigToM CAA (Contrastive Activation Addition) approach:
- Extracts (prompt, cp, cn) triplets where:
  - p = story context (everything before the perceptual statement)
  - cp = positive completion (protagonist sees/witnesses event)
  - cn = negative completion (protagonist does not see/witness event)

Unlike the persona-wrapped approach, this uses ONLY the perceptual statements as
contrastive completions, following the paper's methodology.

This script:
1. Loads matched pairs from false_belief and true_belief conditions
2. Extracts the last sentence (perceptual statement) from each story
3. Validates that contexts match between pairs
4. Samples 800 examples total (400 from 0_ prefix, 400 from 1_ prefix)
5. Creates CAA triplets WITHOUT persona wrapping or question/answer pairs
6. Saves training data and metadata for tracking which examples were used
"""

import csv
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional


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


def split_story_and_perceptual_statement(story: str) -> Tuple[str, str]:
    """
    Split a story into context and perceptual statement.

    The perceptual statement is the last sentence, which describes whether
    the protagonist witnessed/noticed the critical event.

    Returns:
        Tuple of (context, perceptual_statement)
    """
    # Find the last sentence by looking for the last period followed by space or end
    # We need to be careful to not split on periods in abbreviations or names
    sentences = re.split(r'\. (?=[A-Z])', story)

    if len(sentences) < 2:
        # If we can't find multiple sentences, try a simpler split
        last_period = story.rfind('. ')
        if last_period == -1:
            # No period found, use the whole story as perceptual statement
            return "", story.strip()
        context = story[:last_period + 1].strip()
        perceptual_statement = story[last_period + 2:].strip()
    else:
        # Last sentence is the perceptual statement
        perceptual_statement = sentences[-1].strip()
        # Everything before is the context
        context = '. '.join(sentences[:-1]) + '.'

    return context, perceptual_statement


def create_caa_triplet(
    false_belief_story: Dict[str, str],
    true_belief_story: Dict[str, str],
    validate: bool = True
) -> Optional[Dict[str, str]]:
    """
    Create a CAA triplet (prompt, cp, cn) from matched story pairs.

    Args:
        false_belief_story: Story where protagonist doesn't witness event
        true_belief_story: Story where protagonist does witness event
        validate: Whether to validate that contexts match

    Returns:
        Dict with keys: prompt, positive_completion, negative_completion
        Returns None if validation fails
    """
    # Split both stories
    false_context, false_perceptual = split_story_and_perceptual_statement(
        false_belief_story['story']
    )
    true_context, true_perceptual = split_story_and_perceptual_statement(
        true_belief_story['story']
    )

    # Validate that contexts match (they should be identical)
    if validate:
        # Allow for minor whitespace differences
        false_context_normalized = ' '.join(false_context.split())
        true_context_normalized = ' '.join(true_context.split())

        if false_context_normalized != true_context_normalized:
            print(f"Warning: Context mismatch detected!")
            print(f"False: {false_context[:100]}...")
            print(f"True:  {true_context[:100]}...")
            return None

    # Use the context from true_belief (they should be identical)
    prompt = true_context

    # CAA triplet:
    # - cp (positive): protagonist witnesses event → true belief
    # - cn (negative): protagonist doesn't witness event → false belief
    return {
        'prompt': prompt,
        'positive_completion': true_perceptual,
        'negative_completion': false_perceptual
    }


def sample_caa_data(
    conditions_dir: Path,
    num_per_prefix: int = 400,
    random_seed: int = 42
) -> Tuple[List[Dict], Dict]:
    """
    Sample CAA training triplets from matched true/false belief pairs.

    Args:
        conditions_dir: Path to procedural-evals-tom/data/conditions/
        num_per_prefix: Number of examples to sample per prefix (0_ and 1_)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (training_triplets, metadata)
    """
    random.seed(random_seed)

    # Process both implicit (0_) and explicit (1_) conditions
    prefixes = ['0', '1']

    training_triplets = []
    metadata = {
        'num_training_examples': 0,
        'num_examples_per_prefix': num_per_prefix,
        'random_seed': random_seed,
        'conditions': {},
        'total_available_per_prefix': 0,
        'remaining_for_eval': 0,
        'validation_failures': 0,
        'match_failures': 0
    }

    for prefix in prefixes:
        # Load matched pairs
        false_belief_path = conditions_dir / f'{prefix}_forward_belief_false_belief' / 'stories.csv'
        true_belief_path = conditions_dir / f'{prefix}_forward_belief_true_belief' / 'stories.csv'

        print(f"\nProcessing prefix {prefix}_forward_belief...")
        print(f"  False belief: {false_belief_path}")
        print(f"  True belief:  {true_belief_path}")

        false_belief_stories = load_csv_stories(false_belief_path)
        true_belief_stories = load_csv_stories(true_belief_path)

        print(f"  Loaded {len(false_belief_stories)} false belief stories")
        print(f"  Loaded {len(true_belief_stories)} true belief stories")

        # Build a mapping from context to true_belief story
        # This allows us to match stories even if they're in different orders
        print(f"  Building context mapping...")
        true_belief_map = {}
        for tb_story in true_belief_stories:
            context, perceptual = split_story_and_perceptual_statement(tb_story['story'])
            # Normalize context for matching
            context_key = ' '.join(context.split())
            true_belief_map[context_key] = tb_story

        print(f"  Mapped {len(true_belief_map)} true belief stories by context")

        # Find matching pairs
        print(f"  Finding matching pairs...")
        matched_pairs = []
        for idx, fb_story in enumerate(false_belief_stories):
            context, perceptual = split_story_and_perceptual_statement(fb_story['story'])
            context_key = ' '.join(context.split())

            if context_key in true_belief_map:
                matched_pairs.append((idx, fb_story, true_belief_map[context_key]))

        total_available = len(matched_pairs)
        print(f"  Found {total_available} matching pairs")

        if metadata['total_available_per_prefix'] == 0:
            metadata['total_available_per_prefix'] = total_available

        if total_available < num_per_prefix:
            print(f"  Warning: Only {total_available} pairs available, but {num_per_prefix} requested")
            num_per_prefix = total_available

        # Sample from matched pairs
        sampled_pair_indices = random.sample(range(len(matched_pairs)), num_per_prefix)
        sampled_pair_indices.sort()

        # Extract the original false_belief indices for metadata
        sampled_fb_indices = [matched_pairs[i][0] for i in sampled_pair_indices]

        # Store metadata
        metadata['conditions'][f'{prefix}_forward_belief'] = {
            'indices': sampled_fb_indices,
            'total': len(sampled_fb_indices)
        }

        # Create CAA triplets from sampled pairs
        validation_failures = 0
        for pair_idx in sampled_pair_indices:
            fb_idx, fb_story, tb_story = matched_pairs[pair_idx]
            triplet = create_caa_triplet(
                fb_story,
                tb_story,
                validate=False  # We already validated by matching contexts
            )
            if triplet is not None:
                training_triplets.append(triplet)
            else:
                validation_failures += 1

        if validation_failures > 0:
            print(f"  Warning: {validation_failures} validation failures")
            metadata['validation_failures'] += validation_failures

        match_failures = len(false_belief_stories) - total_available
        if match_failures > 0:
            print(f"  Warning: {match_failures} stories couldn't be matched")
            metadata['match_failures'] += match_failures

        print(f"  Created {len(sampled_fb_indices) - validation_failures} triplets")

    # Update final metadata
    metadata['num_training_examples'] = len(training_triplets)
    metadata['remaining_for_eval'] = metadata['total_available_per_prefix'] - num_per_prefix

    return training_triplets, metadata


def main():
    """Main function to generate CAA training data."""
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
    print("Creating CAA-Style ToM Training Dataset (BigToM Replication)")
    print("=" * 80)
    print(f"\nConditions directory: {conditions_dir}")
    print(f"Output directory: {output_dir}\n")

    # Sample training data
    print("Sampling CAA training triplets...")
    training_triplets, metadata = sample_caa_data(
        conditions_dir=conditions_dir,
        num_per_prefix=400,
        random_seed=42
    )

    print(f"\n✓ Created {len(training_triplets)} CAA training triplets\n")

    # Print summary
    print("Summary:")
    print("-" * 80)
    for condition, info in metadata['conditions'].items():
        print(f"  {condition}: {info['total']} examples")
        print(f"    Sampled indices: {info['indices'][:5]} ... {info['indices'][-5:]}")
    print(f"\nTotal training examples: {metadata['num_training_examples']}")
    print(f"Validation failures: {metadata['validation_failures']}")
    print(f"Remaining for evaluation per prefix: {metadata['remaining_for_eval']}")
    print(f"Total remaining for evaluation: {metadata['remaining_for_eval'] * 2}")
    print("-" * 80)

    # Save training data
    training_data_path = output_dir / 'caa_training_data.json'
    print(f"\nSaving training data to: {training_data_path}")
    with open(training_data_path, 'w', encoding='utf-8') as f:
        json.dump(training_triplets, f, indent=2, ensure_ascii=False)
    print("✓ Training data saved")

    # Save metadata
    metadata_path = output_dir / 'caa_training_metadata.json'
    print(f"Saving metadata to: {metadata_path}")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("✓ Metadata saved")

    # Print example triplet
    print("\n" + "=" * 80)
    print("Example CAA Triplet:")
    print("=" * 80)
    example = training_triplets[0]
    print("\nPrompt (p):")
    print(f"  {example['prompt'][:200]}...")
    print("\nPositive Completion (cp) - Protagonist witnesses:")
    print(f"  {example['positive_completion']}")
    print("\nNegative Completion (cn) - Protagonist doesn't witness:")
    print(f"  {example['negative_completion']}")
    print("=" * 80)

    print("\n✓ Dataset generation complete!")
    print(f"\nNext steps:")
    print(f"1. Use {training_data_path} to train CAA steering vectors")
    print(f"2. Use {metadata_path} to exclude training examples during evaluation")
    print(f"3. Evaluate on the remaining {metadata['remaining_for_eval'] * 2} examples")


if __name__ == '__main__':
    main()
