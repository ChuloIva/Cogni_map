#!/usr/bin/env python3
"""
Stratified Batch Cognitive Action Data Generator

Generate any number of examples with consistent stratified distribution
across all 45 cognitive actions.

Features:
- Stratified sampling: equal examples per cognitive action
- Generate any batch size (100, 1000, 10000, etc.)
- First-person perspective only
- Simple complexity examples
- Rich variation: domains, triggers, emotional states, language styles, sentence starters
- Auto-checkpointing
- Uses gemma3:12b-it-q4_K_M model
- Concurrent requests for speed
"""

import json
import time
import random
import asyncio
import aiohttp
import argparse
import os
import sys
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import math
from pathlib import Path
from tqdm import tqdm

# Add datagen to Python path
script_dir = Path(__file__).parent
datagen_dir = script_dir / 'datagen'
if not datagen_dir.exists():
    # If datagen subdirectory doesn't exist, we're already in the datagen directory
    datagen_dir = script_dir
if str(datagen_dir) not in sys.path:
    sys.path.insert(0, str(datagen_dir))

# Import cognitive actions and variation pools
from variable_pools import (
    COGNITIVE_ACTIONS,
    DOMAINS,
    TRIGGERS,
    EMOTIONAL_STATES,
    LANGUAGE_STYLES
)

# Load sentence starters
SENTENCE_STARTERS_FILE = datagen_dir / 'all_truncated_outputs.json'
if SENTENCE_STARTERS_FILE.exists():
    with open(SENTENCE_STARTERS_FILE, 'r') as f:
        SENTENCE_STARTERS = [s for s in json.load(f) if s and len(s) > 2]
else:
    print(f"Warning: {SENTENCE_STARTERS_FILE} not found. Using minimal starters.")
    SENTENCE_STARTERS = ["I think", "I notice", "I realize", "I feel", "I see"]

# Set random seed for reproducibility
random.seed(42)

@dataclass
class VariedExample:
    text: str
    cognitive_action: str
    domain: str
    trigger: str
    emotional_state: str
    language_style: str
    sentence_starter: str


class StratifiedDataGenerator:
    def __init__(self, base_url="http://localhost:11434", max_parallel=4):
        self.base_url = base_url
        self.max_parallel = max_parallel

        # Store variation pools
        self.cognitive_actions = COGNITIVE_ACTIONS
        self.domains = DOMAINS
        self.triggers = TRIGGERS
        self.emotional_states = EMOTIONAL_STATES
        self.language_styles = LANGUAGE_STYLES
        self.sentence_starters = SENTENCE_STARTERS

    def calculate_stratified_distribution(self, total_examples: int) -> Dict[str, int]:
        """Calculate how many examples per cognitive action for stratified sampling."""
        num_actions = len(self.cognitive_actions)
        base_per_action = total_examples // num_actions
        remainder = total_examples % num_actions

        distribution = {}
        for idx, action in enumerate(self.cognitive_actions.keys()):
            # Distribute remainder across first few actions
            distribution[action] = base_per_action + (1 if idx < remainder else 0)

        return distribution

    def create_prompt(self, action: str, action_desc: str, domain: str,
                     trigger: str, emotional_state: str, language_style: str,
                     sentence_starter: str) -> str:
        """Create varied first-person prompt with rich context."""
        # Randomly decide whether to use sentence starter (50% of the time)
        use_starter = random.random() < 0.5

        starter_instruction = ""
        if use_starter:
            starter_instruction = f"\n- Start the example with: '{sentence_starter}'"

        return f"""Generate a simple, first-person example of someone {action}.

Action: {action}
Description: {action_desc}
Domain: {domain}
Trigger: {trigger}
Emotional state: {emotional_state}
Language style: {language_style}

Requirements:
- Write in first person (I, my, me)
- Keep it simple and realistic
- 2-5 sentences maximum
- Focus on the {action} cognitive action
- Use the {language_style} language style
- Incorporate the trigger and emotional state naturally{starter_instruction}
- Make it feel natural and relatable
- Show the cognitive process, not just state it

Example only (no explanation):"""

    async def generate_one(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                          action: str, action_desc: str, domain: str, trigger: str,
                          emotional_state: str, language_style: str,
                          sentence_starter: str, model: str) -> VariedExample:
        """Generate one varied example."""
        async with semaphore:
            prompt = self.create_prompt(action, action_desc, domain, trigger,
                                       emotional_state, language_style, sentence_starter)

            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    result = await response.json()
                    text = result.get('response', '').strip()

                    # Clean up the text
                    text = text.replace('"', '').strip()
                    if not text or len(text) < 20:
                        return None

                    return VariedExample(
                        text=text,
                        cognitive_action=action,
                        domain=domain,
                        trigger=trigger,
                        emotional_state=emotional_state,
                        language_style=language_style,
                        sentence_starter=sentence_starter
                    )
            except Exception as e:
                print(f"\nError generating example: {e}")
                return None

    async def generate_batch_async(self, count: int, action: str,
                                  action_desc: str, model: str, pbar: tqdm = None) -> List[VariedExample]:
        """Generate a batch of examples with rich variation."""
        semaphore = asyncio.Semaphore(self.max_parallel)
        async with aiohttp.ClientSession() as session:
            tasks = []
            for _ in range(count):
                # Randomly select variations for each example
                domain = random.choice(self.domains)
                trigger = random.choice(self.triggers)
                emotional_state = random.choice(self.emotional_states)
                language_style = random.choice(self.language_styles)
                sentence_starter = random.choice(self.sentence_starters)

                task = self.generate_one(session, semaphore, action, action_desc, domain,
                                       trigger, emotional_state, language_style,
                                       sentence_starter, model)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r is not None]

            if pbar:
                pbar.update(len(valid_results))

            return valid_results

    def generate_batch(self, count: int, action: str, action_desc: str,
                      model: str, pbar: tqdm = None) -> List[VariedExample]:
        """Synchronous wrapper for batch generation."""
        return asyncio.run(self.generate_batch_async(count, action, action_desc, model, pbar))

    def generate_stratified(self, total_examples: int, model: str,
                           checkpoint_dir: str) -> List[VariedExample]:
        """Generate stratified examples across all cognitive actions."""
        # Calculate distribution
        distribution = self.calculate_stratified_distribution(total_examples)

        print("\n" + "="*60)
        print("📊 STRATIFIED DISTRIBUTION")
        print("="*60)
        print(f"Total examples to generate: {total_examples:,}")
        print(f"Cognitive actions: {len(self.cognitive_actions)}")
        print(f"Examples per action: {min(distribution.values())} - {max(distribution.values())}")
        print("="*60 + "\n")

        all_examples = []
        start_time = time.time()

        # Generate in smaller batches for better progress tracking
        batch_size = 25  # Generate 25 examples at a time

        # Create progress bar
        with tqdm(total=total_examples, desc="Generating examples", unit="ex") as pbar:
            for action_idx, (action, action_desc) in enumerate(self.cognitive_actions.items(), 1):
                count = distribution[action]

                # Update progress bar description
                pbar.set_description(f"[{action_idx}/45] {action}")

                # Generate in smaller batches
                num_batches = (count + batch_size - 1) // batch_size
                action_examples = []

                for batch_idx in range(num_batches):
                    batch_count = min(batch_size, count - (batch_idx * batch_size))

                    # Generate batch
                    batch_examples = self.generate_batch(
                        count=batch_count,
                        action=action,
                        action_desc=action_desc,
                        model=model,
                        pbar=pbar
                    )

                    action_examples.extend(batch_examples)

                all_examples.extend(action_examples)

        # Save checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(
            checkpoint_dir,
            f"stratified_{total_examples}_{int(time.time())}.jsonl"
        )

        with open(checkpoint_file, 'w') as f:
            for ex in all_examples:
                f.write(json.dumps(asdict(ex)) + '\n')

        print(f"\n✅ Saved to: {checkpoint_file}")

        return all_examples


def main():
    parser = argparse.ArgumentParser(
        description='Generate stratified cognitive action training data'
    )
    parser.add_argument(
        '-n', '--num-examples',
        type=int,
        default=9000,
        help='Total number of examples to generate (default: 4500, which is 100 per action)'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='gemma3:4b',
        help='Ollama model to use (default: gemma3:12b-it-q4_K_M)'
    )
    parser.add_argument(
        '-p', '--parallel',
        type=int,
        default=8,
        help='Number of parallel requests (default: 8)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./generated_data',
        help='Output directory for checkpoints (default: ./generated_data)'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default='http://localhost:11434',
        help='Ollama base URL (default: http://localhost:11434)'
    )

    args = parser.parse_args()

    # Print configuration
    print("="*60)
    print("STRATIFIED GENERATION CONFIGURATION")
    print("="*60)
    print(f"Total examples: {args.num_examples:,}")
    print(f"Cognitive actions: {len(COGNITIVE_ACTIONS)}")
    print(f"Examples per action: ~{args.num_examples // len(COGNITIVE_ACTIONS)}")
    print(f"Model: {args.model}")
    print(f"Parallel requests: {args.parallel}")
    print(f"\nVariation dimensions:")
    print(f"  - Domains: {len(DOMAINS)}")
    print(f"  - Triggers: {len(TRIGGERS)}")
    print(f"  - Emotional states: {len(EMOTIONAL_STATES)}")
    print(f"  - Language styles: {len(LANGUAGE_STYLES)}")
    print(f"  - Sentence starters: {len(SENTENCE_STARTERS)}")
    print(f"\nPerspective: First-person only")
    print(f"Complexity: Simple only")
    print(f"\nOutput directory: {args.output_dir}")

    # Estimate time
    estimated_seconds_per_example = 20 / args.parallel
    estimated_minutes = args.num_examples * estimated_seconds_per_example / 60
    print(f"\nEstimated time: {estimated_minutes:.0f} minutes ({estimated_minutes/60:.1f} hours)")
    print("="*60)

    # Verify Ollama is running
    try:
        import requests
        response = requests.get(f'{args.base_url}/api/tags', timeout=5)
        if response.status_code != 200:
            print("\n❌ Error: Ollama is not responding correctly")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: Cannot connect to Ollama at {args.base_url}")
        print(f"   {e}")
        print("\nMake sure Ollama is running:")
        print("   ollama serve")
        sys.exit(1)

    # Initialize generator
    generator = StratifiedDataGenerator(
        base_url=args.base_url,
        max_parallel=args.parallel
    )

    print("\n" + "="*60)
    print(f"🚀 GENERATING {args.num_examples:,} STRATIFIED EXAMPLES")
    print("="*60)

    start_time = time.time()

    # Generate
    examples = generator.generate_stratified(
        total_examples=args.num_examples,
        model=args.model,
        checkpoint_dir=args.output_dir
    )

    # Calculate statistics
    elapsed = time.time() - start_time

    print("\n" + "="*60)
    print("🎉 GENERATION COMPLETE!")
    print("="*60)
    print(f"Total examples generated: {len(examples):,}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Average rate: {len(examples)/elapsed:.1f} examples/sec")
    print(f"\nData saved to: {args.output_dir}")
    print("="*60)

    # Verify stratification
    from collections import Counter
    distribution = Counter(ex.cognitive_action for ex in examples)

    print("\n📊 DISTRIBUTION VERIFICATION")
    print("="*60)
    print(f"Min examples per action: {min(distribution.values())}")
    print(f"Max examples per action: {max(distribution.values())}")
    print(f"Mean examples per action: {sum(distribution.values()) / len(distribution):.1f}")
    print(f"\nTop 10 cognitive actions:")
    for action, count in distribution.most_common(10):
        print(f"  {action}: {count}")
    print("="*60)


if __name__ == '__main__':
    main()