"""
Main Data Generation Engine for Cognitive Action Training Data
Integrates with Ollama for LLM generation
"""

import json
import time
import random
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from variable_pools import COGNITIVE_ACTIONS, get_random_selection
from prompt_templates import generate_prompt, generate_batch_prompts

@dataclass
class GeneratedExample:
    """Structure for a generated training example"""
    text: str
    primary_cognitive_action: str
    secondary_actions: List[str]
    domain: str
    complexity: str
    perspective: str
    format_type: str
    metadata: Dict[str, Any]

class CognitiveDataGenerator:
    """Main class for generating cognitive action training data"""

    def __init__(self, ollama_client=None, max_parallel=4):
        """Initialize the generator with optional Ollama client

        Args:
            ollama_client: Ollama client instance
            max_parallel: Maximum number of parallel requests (default: 4)
        """
        self.ollama_client = ollama_client
        self.generated_examples = []
        self.generation_stats = {
            'total_generated': 0,
            'by_cognitive_action': {},
            'by_domain': {},
            'by_complexity': {},
            'errors': []
        }
        self.max_parallel = max_parallel

    def generate_single_example(self,
                               cognitive_action: Optional[str] = None,
                               template_type: str = "single",
                               model: str = "gemma3:27b") -> Optional[GeneratedExample]:
        """Generate a single training example"""
        try:
            # Generate prompt
            prompt, params = generate_prompt(cognitive_action, template_type,
                                           self.generation_stats['total_generated'])

            # Generate with Ollama
            if self.ollama_client:
                response = self.ollama_client.generate(
                    model=model,
                    prompt=prompt
                )
                generated_text = response['response'].strip()
            else:
                # Fallback for testing without Ollama
                generated_text = f"[Generated example for {params['cognitive_action']} in {params['domain']}]"

            # Create example object
            example = GeneratedExample(
                text=generated_text,
                primary_cognitive_action=params['cognitive_action'],
                secondary_actions=[],  # Could be extracted with NLP
                domain=params['domain'],
                complexity=params['complexity_level'],
                perspective=params['perspective'],
                format_type=template_type,
                metadata={
                    'subject': params['subject'],
                    'emotional_state': params['emotional_state'],
                    'language_style': params['language_style'],
                    'unique_angle': params['unique_angle'],
                    'trigger': params['trigger'],
                    'generation_timestamp': time.time(),
                    'prompt_used': prompt
                }
            )

            # Update statistics
            self._update_stats(example)
            self.generated_examples.append(example)

            return example

        except Exception as e:
            error_info = {
                'error': str(e),
                'cognitive_action': cognitive_action,
                'template_type': template_type,
                'timestamp': time.time()
            }
            self.generation_stats['errors'].append(error_info)
            print(f"Error generating example: {e}")
            return None

    async def _generate_single_async(self,
                                     cognitive_action: Optional[str],
                                     template_type: str,
                                     model: str,
                                     session: aiohttp.ClientSession,
                                     iteration: int) -> Optional[GeneratedExample]:
        """Asynchronously generate a single example"""
        try:
            # Generate prompt
            prompt, params = generate_prompt(cognitive_action, template_type, iteration)

            # Generate with Ollama via aiohttp
            if self.ollama_client:
                url = f"{self.ollama_client.base_url}/api/generate"
                data = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }

                async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status == 200:
                        result = await response.json()
                        generated_text = result['response'].strip()
                    else:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")
            else:
                # Fallback for testing without Ollama
                generated_text = f"[Generated example for {params['cognitive_action']} in {params['domain']}]"

            # Create example object
            example = GeneratedExample(
                text=generated_text,
                primary_cognitive_action=params['cognitive_action'],
                secondary_actions=[],
                domain=params['domain'],
                complexity=params['complexity_level'],
                perspective=params['perspective'],
                format_type=template_type,
                metadata={
                    'subject': params['subject'],
                    'emotional_state': params['emotional_state'],
                    'language_style': params['language_style'],
                    'unique_angle': params['unique_angle'],
                    'trigger': params['trigger'],
                    'generation_timestamp': time.time(),
                    'prompt_used': prompt
                }
            )

            return example

        except Exception as e:
            error_info = {
                'error': str(e),
                'cognitive_action': cognitive_action,
                'template_type': template_type,
                'timestamp': time.time()
            }
            self.generation_stats['errors'].append(error_info)
            print(f"Error generating example: {e}")
            return None

    async def _generate_batch_async(self,
                                     batch_size: int,
                                     cognitive_action: Optional[str],
                                     template_type: str,
                                     model: str) -> List[GeneratedExample]:
        """Asynchronously generate a batch of examples using parallel requests"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(batch_size):
                task = self._generate_single_async(
                    cognitive_action,
                    template_type,
                    model,
                    session,
                    self.generation_stats['total_generated'] + i
                )
                tasks.append(task)

            # Process in parallel with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self.max_parallel)

            async def limited_task(task):
                async with semaphore:
                    return await task

            results = await asyncio.gather(*[limited_task(t) for t in tasks])

            # Filter out None results and update stats
            examples = []
            for example in results:
                if example:
                    self._update_stats(example)
                    self.generated_examples.append(example)
                    examples.append(example)

            return examples

    def generate_batch(self,
                      batch_size: int = 10,
                      cognitive_action: Optional[str] = None,
                      template_type: str = "single",
                      model: str = "gemma3:27b",
                      delay: float = 0.0) -> List[GeneratedExample]:
        """Generate a batch of examples using async parallel processing

        Args:
            batch_size: Number of examples to generate
            cognitive_action: Specific cognitive action to generate (None for random)
            template_type: Type of template to use
            model: Ollama model name
            delay: Legacy parameter, ignored when using async (kept for compatibility)

        Returns:
            List of generated examples
        """
        print(f"Generating {batch_size} examples in parallel (max {self.max_parallel} concurrent)...")

        # Run the async batch generation
        try:
            # Check if there's already an event loop (Jupyter/Colab)
            try:
                loop = asyncio.get_running_loop()
                # We're in an existing event loop (Jupyter/Colab), use nest_asyncio or await directly
                import nest_asyncio
                nest_asyncio.apply()
                examples = asyncio.run(
                    self._generate_batch_async(batch_size, cognitive_action, template_type, model)
                )
            except RuntimeError:
                # No event loop, create one (regular Python script)
                examples = asyncio.run(
                    self._generate_batch_async(batch_size, cognitive_action, template_type, model)
                )
            except ImportError:
                # nest_asyncio not available, use alternative approach
                print("nest_asyncio not found, using await in existing loop...")
                loop = asyncio.get_running_loop()
                examples = loop.run_until_complete(
                    self._generate_batch_async(batch_size, cognitive_action, template_type, model)
                )
        except Exception as e:
            print(f"Error in batch generation: {e}")
            print("Falling back to sequential generation...")
            # Fallback to sequential generation
            examples = []
            for i in range(batch_size):
                example = self.generate_single_example(cognitive_action, template_type, model)
                if example:
                    examples.append(example)
                    print(f"Generated example {i+1}/{batch_size}: {example.primary_cognitive_action}")
                if delay > 0:
                    time.sleep(delay)
            return examples

        print(f"✓ Generated {len(examples)} examples")
        return examples

    def generate_stratified_dataset(self,
                                   total_examples: int = 1000,
                                   model: str = "gemma3:27b") -> List[GeneratedExample]:
        """Generate a stratified dataset ensuring coverage across cognitive actions"""
        examples_per_action = total_examples // len(COGNITIVE_ACTIONS)
        all_examples = []

        for action in COGNITIVE_ACTIONS.keys():
            print(f"Generating {examples_per_action} examples for: {action}")

            # Mix template types for variety
            template_types = ["single"] * int(examples_per_action * 0.7) + \
                           ["chain"] * int(examples_per_action * 0.2) + \
                           ["dialogue"] * int(examples_per_action * 0.1)

            random.shuffle(template_types)

            for i, template_type in enumerate(template_types):
                example = self.generate_single_example(action, template_type, model)
                if example:
                    all_examples.append(example)

        # Shuffle final dataset
        random.shuffle(all_examples)
        return all_examples

    def generate_phase_dataset(self,
                              phase: str = "phase1",
                              target_examples: int = 20000,
                              model: str = "gemma3:27b") -> List[GeneratedExample]:
        """Generate dataset according to specific phases from instructions"""

        if phase == "phase1_round1":
            # Core cognitive actions - 500 examples each
            return self._generate_phase1_round1(target_examples, model)

        elif phase == "phase1_round2":
            # Cognitive action combinations
            return self._generate_phase1_round2(target_examples, model)

        elif phase == "phase2":
            # Domain variation
            return self._generate_phase2(target_examples, model)

        elif phase == "phase3":
            # Complexity and perspective variation
            return self._generate_phase3(target_examples, model)

        elif phase == "phase4":
            # Negative examples
            return self._generate_phase4(target_examples, model)

        elif phase == "phase5":
            # Dialogue format
            return self._generate_phase5(target_examples, model)

        elif phase == "phase6":
            # Thought-stream format
            return self._generate_phase6(target_examples, model)

        else:
            # Default to stratified
            return self.generate_stratified_dataset(target_examples, model)

    def _generate_phase1_round1(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 1 Round 1: Core cognitive actions"""
        examples_per_action = target_examples // len(COGNITIVE_ACTIONS)
        all_examples = []

        for action in COGNITIVE_ACTIONS.keys():
            examples = self.generate_batch(examples_per_action, action, "single", model)
            all_examples.extend(examples)

        return all_examples

    def _generate_phase1_round2(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 1 Round 2: Cognitive action combinations"""
        all_examples = []

        for i in range(target_examples):
            example = self.generate_single_example(None, "chain", model)
            if example:
                all_examples.append(example)

        return all_examples

    def _generate_phase2(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 2: Domain variation"""
        # Take seed examples and vary domains
        return self.generate_stratified_dataset(target_examples, model)

    def _generate_phase3(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 3: Complexity and perspective variation"""
        all_examples = []

        # Mix of complexity levels and perspectives
        for i in range(target_examples):
            template_type = random.choice(["single", "chain", "thought_stream"])
            example = self.generate_single_example(None, template_type, model)
            if example:
                all_examples.append(example)

        return all_examples

    def _generate_phase4(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 4: Negative examples"""
        all_examples = []

        for i in range(target_examples):
            example = self.generate_single_example(None, "negative", model)
            if example:
                all_examples.append(example)

        return all_examples

    def _generate_phase5(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 5: Dialogue format"""
        all_examples = []

        for i in range(target_examples):
            example = self.generate_single_example(None, "dialogue", model)
            if example:
                all_examples.append(example)

        return all_examples

    def _generate_phase6(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 6: Thought-stream format"""
        all_examples = []

        for i in range(target_examples):
            example = self.generate_single_example(None, "thought_stream", model)
            if example:
                all_examples.append(example)

        return all_examples

    def _update_stats(self, example: GeneratedExample):
        """Update generation statistics"""
        self.generation_stats['total_generated'] += 1

        # Update by cognitive action
        action = example.primary_cognitive_action
        if action not in self.generation_stats['by_cognitive_action']:
            self.generation_stats['by_cognitive_action'][action] = 0
        self.generation_stats['by_cognitive_action'][action] += 1

        # Update by domain
        domain = example.domain
        if domain not in self.generation_stats['by_domain']:
            self.generation_stats['by_domain'][domain] = 0
        self.generation_stats['by_domain'][domain] += 1

        # Update by complexity
        complexity = example.complexity
        if complexity not in self.generation_stats['by_complexity']:
            self.generation_stats['by_complexity'][complexity] = 0
        self.generation_stats['by_complexity'][complexity] += 1

    def export_dataset(self, filepath: str, format: str = "jsonl"):
        """Export generated dataset to file"""
        if format == "jsonl":
            with open(filepath, 'w') as f:
                for example in self.generated_examples:
                    json_obj = {
                        'text': example.text,
                        'primary_cognitive_action': example.primary_cognitive_action,
                        'secondary_actions': example.secondary_actions,
                        'domain': example.domain,
                        'complexity': example.complexity,
                        'perspective': example.perspective,
                        'format_type': example.format_type,
                        'metadata': example.metadata
                    }
                    f.write(json.dumps(json_obj) + '\n')

        elif format == "json":
            with open(filepath, 'w') as f:
                dataset = [
                    {
                        'text': example.text,
                        'primary_cognitive_action': example.primary_cognitive_action,
                        'secondary_actions': example.secondary_actions,
                        'domain': example.domain,
                        'complexity': example.complexity,
                        'perspective': example.perspective,
                        'format_type': example.format_type,
                        'metadata': example.metadata
                    }
                    for example in self.generated_examples
                ]
                json.dump(dataset, f, indent=2)

        print(f"Exported {len(self.generated_examples)} examples to {filepath}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return self.generation_stats

    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.generation_stats
        print(f"\nGeneration Statistics:")
        print(f"Total examples generated: {stats['total_generated']}")
        print(f"Errors encountered: {len(stats['errors'])}")

        print(f"\nBy Cognitive Action:")
        for action, count in sorted(stats['by_cognitive_action'].items()):
            print(f"  {action}: {count}")

        print(f"\nBy Domain:")
        for domain, count in sorted(stats['by_domain'].items()):
            print(f"  {domain}: {count}")

        print(f"\nBy Complexity:")
        for complexity, count in sorted(stats['by_complexity'].items()):
            print(f"  {complexity}: {count}")