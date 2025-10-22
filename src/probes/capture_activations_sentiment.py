"""
Activation capture script for Sentiment Probes using Gemma 3 4B

Key difference from cognitive action probes:
- Appends "The sentiment of this section is" instead of "The cognitive action..."
- This primes the model to think about sentiment rather than cognitive actions
- Captures the last token representation after processing this prompt
"""

from .gpu_utils import configure_amd_gpu

# Configure AMD GPU environment if detected (must be before torch import)
configure_amd_gpu()

import sys
from pathlib import Path
import torch
import json
import h5py
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import argparse

# Add nnsight to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "nnsight" / "src"))
from nnsight import LanguageModel


@dataclass
class SentimentExample:
    """Sentiment example from dataset"""
    text: str
    sentiment: str  # "positive" or "negative"
    emotion: str


def load_sentiment_dataset(dataset_path: str, limit: Optional[int] = None) -> List[SentimentExample]:
    """Load sentiment dataset from JSONL file"""
    examples = []

    with open(dataset_path, 'r') as f:
        for line in f:
            if limit and len(examples) >= limit:
                break
            data = json.loads(line)
            examples.append(SentimentExample(**data))

    return examples


class SentimentActivationCapture:
    """Captures activations from Gemma 3 4B for sentiment analysis"""

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        device: str = "auto",
        layers_to_capture: Optional[List[int]] = None
    ):
        """
        Initialize sentiment activation capture

        Args:
            model_name: HuggingFace model ID
            device: Device to run on ('cuda', 'cpu', or 'auto')
            layers_to_capture: Which transformer layers to capture
        """
        print(f"Loading model: {model_name}")

        # For VLMs like Gemma-3, skip loading vision tower
        from transformers import AutoModelForCausalLM, AutoConfig
        config = AutoConfig.from_pretrained(model_name)

        if hasattr(config, 'vision_config'):
            print("Detected vision-language model. Loading text-only (skipping vision tower)...")
            from transformers import Gemma3ForCausalLM, AutoTokenizer
            base_model = Gemma3ForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = LanguageModel(base_model, tokenizer=tokenizer)
        else:
            self.model = LanguageModel(model_name, device_map=device)

        self.model_name = model_name

        # Get number of layers in the model
        self.layers = self.model.model.layers
        self.num_layers = len(self.layers)
        print(f"Model has {self.num_layers} transformer layers")

        if layers_to_capture is None:
            self.layers_to_capture = [
                self.num_layers // 4,
                self.num_layers // 2,
                3 * self.num_layers // 4,
                self.num_layers - 1
            ]
        else:
            self.layers_to_capture = layers_to_capture

        print(f"Will capture activations from layers: {self.layers_to_capture}")

    def capture_single_example(
        self,
        text: str,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Capture activations for a single text at a specific layer

        Args:
            text: Input text
            layer_idx: Which layer to capture from

        Returns:
            Tensor of activations (hidden_size,) - last token representation
        """
        # ⭐ KEY DIFFERENCE: Use sentiment-specific prompt
        # This primes the model to think about sentiment instead of cognitive actions
        augmented_text = f"{text}\n\nThe sentiment of this section is"

        with self.model.trace(augmented_text) as tracer:
            hidden_states = self.layers[layer_idx].output[0].save()

        # Use last token representation (after the model has "thought about" the sentiment)
        activations = hidden_states[:, -1, :].squeeze(0)

        return activations

    def capture_single_example_all_layers(
        self,
        text: str
    ) -> Dict[int, torch.Tensor]:
        """
        Capture activations from ALL layers in a single forward pass

        Args:
            text: Input text

        Returns:
            Dict mapping layer_idx -> activations tensor (hidden_size,)
        """
        # ⭐ KEY DIFFERENCE: Use sentiment-specific prompt
        augmented_text = f"{text}\n\nThe sentiment of this section is"

        saved_states = {}

        with self.model.trace(augmented_text) as tracer:
            # Capture ALL layers simultaneously
            for layer_idx in self.layers_to_capture:
                saved_states[layer_idx] = self.layers[layer_idx].output[0].save()

        # Extract last token for each layer
        activations = {}
        for layer_idx, hidden_states in saved_states.items():
            activations[layer_idx] = hidden_states[:, -1, :].squeeze(0)

        return activations

    def capture_dataset_batched(
        self,
        examples: List[SentimentExample],
        layer_idx: int,
        batch_size: int = 1000,
        max_examples: Optional[int] = None,
        show_progress: bool = True
    ):
        """
        Generator that yields batches of activations for memory-efficient processing

        Args:
            examples: List of SentimentExample objects
            layer_idx: Which layer to capture from
            batch_size: Number of examples to process before yielding
            max_examples: Optional limit on number of examples
            show_progress: Whether to show progress bar

        Yields:
            Tuple of (activations, labels) for each batch
        """
        if max_examples:
            examples = examples[:max_examples]

        # Map sentiment to binary label (0=negative, 1=positive)
        sentiment_to_label = {"negative": 0, "positive": 1}

        iterator = tqdm(examples, desc=f"Layer {layer_idx}") if show_progress else examples

        activations_list = []
        labels_list = []

        for idx, example in enumerate(iterator):
            try:
                # Capture activations
                act = self.capture_single_example(example.text, layer_idx)
                activations_list.append(act.detach().cpu())

                # Get label
                label = sentiment_to_label[example.sentiment]
                labels_list.append(label)

                # Yield batch when we hit batch_size
                if len(activations_list) >= batch_size:
                    activations = torch.stack(activations_list)
                    labels = torch.tensor(labels_list, dtype=torch.long)
                    yield activations, labels

                    # Clear lists and GPU cache
                    activations_list = []
                    labels_list = []
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            except Exception as e:
                print(f"\nError processing example: {e}")
                print(f"Text: {example.text[:100]}...")
                continue

        # Yield remaining examples
        if activations_list:
            activations = torch.stack(activations_list)
            labels = torch.tensor(labels_list, dtype=torch.long)
            yield activations, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def capture_all_layers_optimized(
        self,
        examples: List[SentimentExample],
        batch_size: int = 300,
        max_examples: Optional[int] = None
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        🚀 OPTIMIZED: Capture ALL layers in a single pass

        Args:
            examples: List of SentimentExample objects
            batch_size: Batch size for processing
            max_examples: Optional limit on examples

        Returns:
            Tuple of (layer_activations_dict, labels_tensor)
        """
        if max_examples:
            examples = examples[:max_examples]

        # Map sentiment to binary label
        sentiment_to_label = {"negative": 0, "positive": 1}

        # Initialize storage for each layer
        layer_activations = {layer_idx: [] for layer_idx in self.layers_to_capture}
        labels_list = []

        print(f"🚀 Capturing {len(self.layers_to_capture)} layers in single pass...")
        print(f"   Memory per example: ~{len(self.layers_to_capture) * 4096 * 2 / 1024:.1f} KB")

        for example in tqdm(examples, desc="Processing examples"):
            try:
                # ONE forward pass captures ALL layers!
                all_layer_acts = self.capture_single_example_all_layers(example.text)

                # Store activations for each layer
                for layer_idx in self.layers_to_capture:
                    layer_activations[layer_idx].append(all_layer_acts[layer_idx].detach().cpu())

                # Store label once
                label = sentiment_to_label[example.sentiment]
                labels_list.append(label)

                # Periodic cleanup
                if len(labels_list) % batch_size == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            except Exception as e:
                print(f"\nError processing example: {e}")
                continue

        # Stack into tensors for each layer
        result = {}
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        for layer_idx in self.layers_to_capture:
            activations_tensor = torch.stack(layer_activations[layer_idx])
            result[layer_idx] = activations_tensor

        return result, labels_tensor


def create_splits(
    examples: List[SentimentExample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[SentimentExample], List[SentimentExample], List[SentimentExample]]:
    """Split examples into train/val/test sets with stratification"""
    import random
    random.seed(random_seed)

    # Separate by sentiment for stratification
    positive = [ex for ex in examples if ex.sentiment == "positive"]
    negative = [ex for ex in examples if ex.sentiment == "negative"]

    # Shuffle
    random.shuffle(positive)
    random.shuffle(negative)

    # Split each sentiment category
    def split_list(lst, train_r, val_r):
        n = len(lst)
        train_end = int(n * train_r)
        val_end = train_end + int(n * val_r)
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]

    pos_train, pos_val, pos_test = split_list(positive, train_ratio, val_ratio)
    neg_train, neg_val, neg_test = split_list(negative, train_ratio, val_ratio)

    # Combine and shuffle
    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Capture activations for sentiment probes")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL sentiment dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/activations/sentiment",
        help="Directory to save activations"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to capture (default: evenly spaced)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to process"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda/cpu/auto)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["hdf5"],
        default="hdf5",
        help="Output format"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="🚀 Capture all layers in single pass (faster!)"
    )

    args = parser.parse_args()

    # Load dataset
    print("Loading sentiment dataset...")
    examples = load_sentiment_dataset(args.dataset, limit=args.max_examples)

    print(f"\nDataset statistics:")
    print(f"  Total examples: {len(examples)}")
    positive_count = sum(1 for ex in examples if ex.sentiment == "positive")
    negative_count = len(examples) - positive_count
    print(f"  Positive: {positive_count}")
    print(f"  Negative: {negative_count}")

    # Create splits
    print("\nCreating train/val/test splits...")
    train_examples, val_examples, test_examples = create_splits(examples, random_seed=42)

    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")
    print(f"  Test: {len(test_examples)}")

    # Initialize capture
    capture = SentimentActivationCapture(
        model_name=args.model,
        device=args.device,
        layers_to_capture=args.layers
    )

    # Determine hidden size
    print("\nDetecting model hidden size...")
    test_activation = capture.capture_single_example(train_examples[0].text, capture.layers_to_capture[0])
    hidden_size = test_activation.shape[0]
    print(f"Hidden size: {hidden_size}")
    del test_activation

    # Capture activations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.single_pass:
        # 🚀 OPTIMIZED MODE: Capture all layers in ONE pass!
        print("\n" + "="*70)
        print("🚀 OPTIMIZED SINGLE-PASS MODE")
        print("="*70)
        print(f"Capturing ALL {len(capture.layers_to_capture)} layers simultaneously")
        print(f"Expected speedup: ~{len(capture.layers_to_capture)}x faster!\n")

        # Process each split
        for split_name, split_examples in [
            ('train', train_examples),
            ('val', val_examples),
            ('test', test_examples)
        ]:
            print(f"\n{'='*70}")
            print(f"Processing {split_name} split ({len(split_examples)} examples)")
            print(f"{'='*70}")

            layer_acts, labels = capture.capture_all_layers_optimized(
                split_examples,
                batch_size=args.batch_size,
                max_examples=None
            )

            # Save each layer's activations
            for layer_idx in capture.layers_to_capture:
                output_path = output_dir / f"layer_{layer_idx}_activations.h5"

                # Open in append mode for splits after train
                mode = 'a' if split_name != 'train' else 'w'

                with h5py.File(output_path, mode) as f:
                    if mode == 'w':
                        f.attrs['model_name'] = capture.model_name
                        f.attrs['layers'] = capture.layers_to_capture

                    grp = f.create_group(split_name)
                    grp.create_dataset('activations', data=layer_acts[layer_idx].float().numpy())
                    grp.create_dataset('labels', data=labels.numpy())

                print(f"  Saved layer {layer_idx} ({split_name})")

        print("\n" + "="*70)
        print("✅ SINGLE-PASS CAPTURE COMPLETE!")
        print("="*70)

    else:
        # Layer-by-layer mode with batched saving
        for layer_idx in capture.layers_to_capture:
            print(f"\n{'='*60}")
            print(f"Capturing activations from layer {layer_idx}")
            print(f"{'='*60}")

            output_path = output_dir / f"layer_{layer_idx}_activations.h5"

            # Process each split with batched saving
            for split_name, split_examples in [
                ('train', train_examples),
                ('val', val_examples),
                ('test', test_examples)
            ]:
                print(f"\nProcessing {split_name} split ({len(split_examples)} examples)...")

                # Open in append mode for splits after train
                mode = 'a' if split_name != 'train' else 'w'

                with h5py.File(output_path, mode) as f:
                    if mode == 'w':
                        f.attrs['model_name'] = capture.model_name
                        f.attrs['layers'] = capture.layers_to_capture

                    # Create group and pre-allocate datasets
                    grp = f.create_group(split_name)
                    grp.create_dataset(
                        'activations',
                        shape=(len(split_examples), hidden_size),
                        dtype='float32',
                        chunks=(min(1000, len(split_examples)), hidden_size)
                    )
                    grp.create_dataset(
                        'labels',
                        shape=(len(split_examples),),
                        dtype='int64',
                        chunks=(min(1000, len(split_examples)),)
                    )

                    # Process batches
                    offset = 0
                    batches_gen = capture.capture_dataset_batched(
                        split_examples,
                        layer_idx,
                        batch_size=args.batch_size
                    )

                    for batch_acts, batch_labels in batches_gen:
                        batch_size = len(batch_acts)
                        grp['activations'][offset:offset + batch_size] = batch_acts.float().numpy()
                        grp['labels'][offset:offset + batch_size] = batch_labels.numpy()
                        offset += batch_size

                    print(f"  Saved {offset} examples")

    print("\n" + "="*60)
    print("✅ SENTIMENT ACTIVATION CAPTURE COMPLETE")
    print("="*60)
    print(f"\nActivations saved to: {output_dir}")
    print(f"Layers captured: {capture.layers_to_capture}")
    print(f"Prompt used: '[text]\\n\\nThe sentiment of this section is'")


if __name__ == "__main__":
    main()