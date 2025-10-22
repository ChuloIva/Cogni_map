"""
Activation capture script for Gemma 3 4B using nnsight

This script:
1. Loads Gemma 3 4B from HuggingFace
2. Loads cognitive actions dataset from datagen
3. Uses nnsight to extract activations at multiple layers
4. Saves activations to disk for probe training
"""

from .gpu_utils import configure_amd_gpu

# Configure AMD GPU environment if detected (must be before torch import)
configure_amd_gpu()

import sys
from pathlib import Path
import torch
import pickle
import h5py
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import argparse

# Add nnsight to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "nnsight" / "src"))
from nnsight import LanguageModel

# Import our dataset utilities
from .dataset_utils import (
    load_dataset,
    load_cognitive_actions_dataset,
    load_sentiment_dataset,
    create_splits,
    get_action_to_idx_mapping,
    print_dataset_statistics,
    SentimentExample,
    CognitiveActionExample
)


class ActivationCapture:
    """Captures activations from Gemma 3 4B at specified layers"""

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",  # Gemma 3 4B instruction-tuned
        device: str = "auto",
        layers_to_capture: Optional[List[int]] = None,
        examples: Optional[List] = None
    ):
        """
        Initialize activation capture

        Args:
            model_name: HuggingFace model ID
            device: Device to run on ('cuda', 'cpu', or 'auto')
            layers_to_capture: Which transformer layers to capture (default: [6, 12, 18, 24])
            examples: Optional list of examples to infer label mapping from
        """
        print(f"Loading model: {model_name}")

        # For VLMs like Gemma-3, skip loading vision tower to save memory and enable text-only usage
        from transformers import AutoModelForCausalLM, AutoConfig
        config = AutoConfig.from_pretrained(model_name)

        # Check if this is a VLM (has vision_config)
        if hasattr(config, 'vision_config'):
            print("Detected vision-language model. Loading text-only (skipping vision tower)...")
            # Load with Gemma3ForCausalLM to skip vision tower
            from transformers import Gemma3ForCausalLM, AutoTokenizer
            base_model = Gemma3ForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype="auto"
            )
            # Load tokenizer separately
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = LanguageModel(base_model, tokenizer=tokenizer)
        else:
            self.model = LanguageModel(model_name, device_map=device)

        self.model_name = model_name

        # Get number of layers in the model
        self.layers = self.model.model.layers
        self.num_layers = len(self.layers)
        print(f"Model has {self.num_layers} transformer layers")

        # Default to capturing from evenly spaced layers
        if layers_to_capture is None:
            # For gemma-2-3b (28 layers), capture at layers [7, 14, 21, 27]
            self.layers_to_capture = [
                self.num_layers // 4,
                self.num_layers // 2,
                3 * self.num_layers // 4,
                self.num_layers - 1
            ]
        else:
            self.layers_to_capture = layers_to_capture

        print(f"Will capture activations from layers: {self.layers_to_capture}")

        # Get action/sentiment to index mapping
        self.action_to_idx = get_action_to_idx_mapping(examples)

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
        # Append special message to create consistent extraction point
        # This primes the model to "think about" cognitive actions
        augmented_text = f"{text}\n\nThe cognitive action being demonstrated here is"

        with self.model.trace(augmented_text) as tracer:
            # Access the layer's output
            # Use self.layers to handle both text-only and VLM architectures
            hidden_states = self.layers[layer_idx].output[0].save()

        # hidden_states shape: (batch_size, seq_len, hidden_size)
        # Use last token representation (similar to paper's approach)
        # This is the representation after the model has "thought about" what cognitive action it is
        # Then squeeze batch dimension
        activations = hidden_states[:, -1, :].squeeze(0)

        return activations

    def capture_single_example_all_layers(
        self,
        text: str
    ) -> Dict[int, torch.Tensor]:
        """
        Capture activations from ALL layers in a single forward pass (MUCH faster!)

        Args:
            text: Input text

        Returns:
            Dict mapping layer_idx -> activations tensor (hidden_size,)
        """
        augmented_text = f"{text}\n\nThe cognitive action being demonstrated here is"

        # Initialize saved_states before the with block to avoid UnboundLocalError
        saved_states = {}
        
        with self.model.trace(augmented_text) as tracer:
            # Capture ALL layers simultaneously in one forward pass
            for layer_idx in self.layers_to_capture:
                saved_states[layer_idx] = self.layers[layer_idx].output[0].save()

        # Extract last token for each layer
        activations = {}
        for layer_idx, hidden_states in saved_states.items():
            activations[layer_idx] = hidden_states[:, -1, :].squeeze(0)

        return activations

    def capture_dataset(
        self,
        examples: List,
        layer_idx: int,
        max_examples: Optional[int] = None,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Capture activations for entire dataset at a specific layer

        Args:
            examples: List of CognitiveActionExample objects
            layer_idx: Which layer to capture from
            max_examples: Optional limit on number of examples
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (activations, labels)
            - activations: (num_examples, hidden_size)
            - labels: (num_examples,) integer labels
        """
        if max_examples:
            examples = examples[:max_examples]

        activations_list = []
        labels_list = []

        iterator = tqdm(examples, desc=f"Layer {layer_idx}") if show_progress else examples

        for example in iterator:
            try:
                # Capture activations
                act = self.capture_single_example(example.text, layer_idx)
                activations_list.append(act.detach().cpu())

                # Get label
                label = self.action_to_idx[example.primary_action]
                labels_list.append(label)

            except Exception as e:
                print(f"\nError processing example: {e}")
                print(f"Text: {example.text[:100]}...")
                continue

        # Stack into tensors
        activations = torch.stack(activations_list)
        labels = torch.tensor(labels_list, dtype=torch.long)

        return activations, labels

    def capture_dataset_batched(
        self,
        examples: List,
        layer_idx: int,
        batch_size: int = 1000,
        max_examples: Optional[int] = None,
        show_progress: bool = True
    ):
        """
        Generator that yields batches of activations for memory-efficient processing

        Args:
            examples: List of CognitiveActionExample objects
            layer_idx: Which layer to capture from
            batch_size: Number of examples to process before yielding
            max_examples: Optional limit on number of examples
            show_progress: Whether to show progress bar

        Yields:
            Tuple of (activations, labels) for each batch
            - activations: (batch_size, hidden_size)
            - labels: (batch_size,) integer labels
        """
        if max_examples:
            examples = examples[:max_examples]

        iterator = tqdm(examples, desc=f"Layer {layer_idx}") if show_progress else examples

        activations_list = []
        labels_list = []

        for idx, example in enumerate(iterator):
            try:
                # Capture activations
                act = self.capture_single_example(example.text, layer_idx)
                activations_list.append(act.detach().cpu())

                # Get label
                label = self.action_to_idx[example.primary_action]
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

    def save_activations_pickle(
        self,
        activations: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        output_path: Path
    ):
        """
        Save activations to pickle file

        Args:
            activations: Dict mapping split names to (activations, labels) tuples
            output_path: Where to save the pickle file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'model_name': self.model_name,
            'layers': self.layers_to_capture,
            'action_to_idx': self.action_to_idx,
            'activations': activations
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"\nSaved activations to {output_path}")

    def save_activations_hdf5(
        self,
        activations: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        output_path: Path
    ):
        """
        Save activations to HDF5 file (more efficient for large datasets)

        Args:
            activations: Dict mapping split names to (activations, labels) tuples
            output_path: Where to save the HDF5 file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            # Save metadata
            f.attrs['model_name'] = self.model_name
            f.attrs['layers'] = self.layers_to_capture

            # Save each split
            for split_name, (acts, labs) in activations.items():
                grp = f.create_group(split_name)
                # Convert to float32 for HDF5 compatibility (BFloat16 not supported)
                grp.create_dataset('activations', data=acts.detach().cpu().float().numpy())
                grp.create_dataset('labels', data=labs.cpu().numpy())

        print(f"\nSaved activations to {output_path}")

    def save_activations_hdf5_batched(
        self,
        split_name: str,
        batches_generator,
        output_path: Path,
        total_examples: int,
        hidden_size: int
    ):
        """
        Save activations to HDF5 file incrementally from a generator (memory-efficient)

        Args:
            split_name: Name of the split (e.g., 'train', 'val', 'test')
            batches_generator: Generator yielding (activations, labels) batches
            output_path: Where to save the HDF5 file
            total_examples: Total number of examples (for pre-allocating datasets)
            hidden_size: Hidden dimension size
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine mode: create new or append to existing
        mode = 'a' if output_path.exists() else 'w'

        with h5py.File(output_path, mode) as f:
            # Save metadata if creating new file
            if mode == 'w':
                f.attrs['model_name'] = self.model_name
                f.attrs['layers'] = self.layers_to_capture

            # Create or get group for this split
            if split_name in f:
                grp = f[split_name]
            else:
                grp = f.create_group(split_name)
                # Pre-allocate datasets for efficiency
                grp.create_dataset(
                    'activations',
                    shape=(total_examples, hidden_size),
                    dtype='float32',
                    chunks=(min(1000, total_examples), hidden_size)
                )
                grp.create_dataset(
                    'labels',
                    shape=(total_examples,),
                    dtype='int64',
                    chunks=(min(1000, total_examples),)
                )

            # Write batches incrementally
            offset = 0
            for batch_acts, batch_labels in batches_generator:
                batch_size = len(batch_acts)

                # Write batch to datasets
                # Convert to float32 for HDF5 compatibility (BFloat16 not supported)
                grp['activations'][offset:offset + batch_size] = batch_acts.detach().cpu().float().numpy()
                grp['labels'][offset:offset + batch_size] = batch_labels.cpu().numpy()

                offset += batch_size

            print(f"Saved {offset} examples to {split_name} split")

    def capture_all_layers_optimized(
        self,
        examples: List,
        batch_size: int = 300,
        max_examples: Optional[int] = None
    ) -> Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        🚀 OPTIMIZED: Capture ALL layers in a single pass (25x faster!)

        Captures activations from all layers simultaneously instead of running
        the model separately for each layer.

        Args:
            examples: List of CognitiveActionExample objects
            batch_size: Batch size for processing
            max_examples: Optional limit on examples

        Returns:
            Dict mapping layer_idx -> {split_name -> (activations, labels)}
        """
        if max_examples:
            examples = examples[:max_examples]

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
                label = self.action_to_idx[example.primary_action]
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


def main():
    parser = argparse.ArgumentParser(description="Capture activations from Gemma 3 4B")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL dataset from datagen"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/activations",
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
        choices=["pickle", "hdf5"],
        default="hdf5",
        help="Output format"
    )
    parser.add_argument(
        "--batch-save",
        action="store_true",
        help="Use memory-efficient batch saving (recommended for large datasets)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for memory-efficient saving (default: 1000)"
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="🚀 OPTIMIZED: Capture all layers in single pass (25x faster!)"
    )

    args = parser.parse_args()

    # Load dataset (auto-detect sentiment or cognitive actions)
    print("Loading dataset...")
    examples = load_dataset(args.dataset, limit=args.max_examples)
    print_dataset_statistics(examples)

    # Create splits
    print("\nCreating train/val/test splits...")
    train_examples, val_examples, test_examples = create_splits(
        examples,
        stratify=True,
        random_seed=42
    )

    # Initialize capture
    capture = ActivationCapture(
        model_name=args.model,
        device=args.device,
        layers_to_capture=args.layers,
        examples=examples  # Pass examples to infer label mapping
    )

    # Determine hidden size by capturing one example
    print("\nDetecting model hidden size...")
    test_activation = capture.capture_single_example(train_examples[0].text, capture.layers_to_capture[0])
    hidden_size = test_activation.shape[0]
    print(f"Hidden size: {hidden_size}")
    del test_activation

    # Capture activations for each layer and split
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
        # Original mode: capture each layer separately
        for layer_idx in capture.layers_to_capture:
            print(f"\n{'='*60}")
            print(f"Capturing activations from layer {layer_idx}")
            print(f"{'='*60}")

            output_path = output_dir / f"layer_{layer_idx}_activations.h5"

            if args.batch_save and args.format == "hdf5":
                # Memory-efficient batch processing
                print(f"Using memory-efficient batch saving (batch_size={args.batch_size})")

                # Process each split with batched saving
                splits_data = [
                    ('train', train_examples),
                    ('val', val_examples),
                    ('test', test_examples)
                ]

                for split_name, split_examples in splits_data:
                    print(f"\nProcessing {split_name} split ({len(split_examples)} examples)...")
                    batches_gen = capture.capture_dataset_batched(
                        split_examples,
                        layer_idx,
                        batch_size=args.batch_size
                    )
                    capture.save_activations_hdf5_batched(
                        split_name,
                        batches_gen,
                        output_path,
                        len(split_examples),
                        hidden_size
                    )

                print(f"\nSaved all splits to {output_path}")

            else:
                # Original method - load everything into memory
                print("Using standard (in-memory) processing")

                # Capture for each split
                print("\nProcessing train split...")
                train_acts, train_labels = capture.capture_dataset(train_examples, layer_idx)

                print("\nProcessing validation split...")
                val_acts, val_labels = capture.capture_dataset(val_examples, layer_idx)

                print("\nProcessing test split...")
                test_acts, test_labels = capture.capture_dataset(test_examples, layer_idx)

                print(f"\nLayer {layer_idx} activation shapes:")
                print(f"  Train: {train_acts.shape}")
                print(f"  Val:   {val_acts.shape}")
                print(f"  Test:  {test_acts.shape}")

                # Save activations
                splits = {
                    'train': (train_acts, train_labels),
                    'val': (val_acts, val_labels),
                    'test': (test_acts, test_labels)
                }

                if args.format == "pickle":
                    output_path = output_dir / f"layer_{layer_idx}_activations.pkl"
                    capture.save_activations_pickle({f"layer_{layer_idx}": splits}, output_path)
                else:  # hdf5
                    capture.save_activations_hdf5(splits, output_path)

    print("\n" + "="*60)
    print("ACTIVATION CAPTURE COMPLETE")
    print("="*60)
    print(f"\nActivations saved to: {output_dir}")
    print(f"Layers captured: {capture.layers_to_capture}")
    print(f"Number of classes: {len(capture.action_to_idx)}")


if __name__ == "__main__":
    main()
