"""
Parallel training pipeline for binary cognitive action probes
Trains multiple probes simultaneously to maximize GPU utilization
Optimized for high-VRAM GPUs (e.g., 40GB+)
"""

from .gpu_utils import configure_amd_gpu

# Configure AMD GPU environment if detected (must be before torch import)
configure_amd_gpu()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import argparse
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from .probe_models import BinaryLinearProbe, BinaryMultiHeadProbe, save_probe
from .dataset_utils import get_idx_to_action_mapping, create_binary_labels


class BinaryActivationDataset(Dataset):
    """Dataset for loading pre-computed activations with binary labels"""

    def __init__(self, activations: torch.Tensor, labels: torch.Tensor):
        """
        Initialize dataset

        Args:
            activations: Tensor of shape (num_examples, hidden_dim)
            labels: Binary tensor of shape (num_examples,) with values 0.0 or 1.0
        """
        self.activations = activations
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]


def load_activations_from_hdf5(
    file_path: Path
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load activations from HDF5 file

    Args:
        file_path: Path to HDF5 file

    Returns:
        Tuple of (train_acts, train_labels, val_acts, val_labels, test_acts, test_labels)
    """
    with h5py.File(file_path, 'r') as f:
        # Load activations (saved as float32 for HDF5 compatibility)
        # Convert back to bfloat16 to match original model dtype
        train_acts = torch.from_numpy(f['train']['activations'][:]).bfloat16()
        train_labels = torch.from_numpy(f['train']['labels'][:])

        val_acts = torch.from_numpy(f['val']['activations'][:]).bfloat16()
        val_labels = torch.from_numpy(f['val']['labels'][:])

        test_acts = torch.from_numpy(f['test']['activations'][:]).bfloat16()
        test_labels = torch.from_numpy(f['test']['labels'][:])

    return train_acts, train_labels, val_acts, val_labels, test_acts, test_labels


class BinaryProbeTrainer:
    """Trainer for binary cognitive action probes"""

    def __init__(
        self,
        probe: nn.Module,
        device: str = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-3,
        use_scheduler: bool = True,
        num_epochs: int = 50
    ):
        """
        Initialize trainer

        Args:
            probe: Binary probe model to train
            device: Device to train on (auto-detects if None)
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization strength
            use_scheduler: Whether to use learning rate scheduler
            num_epochs: Total number of epochs (for scheduler)
        """
        # Auto-detect device if not provided
        if device is None:
            from gpu_utils import get_optimal_device
            device = get_optimal_device()

        self.device = device

        # Convert probe to bfloat16 to match activation dtype, then move to device
        # Note: MPS doesn't support bfloat16, so keep as float32 for MPS
        if device == "mps":
            self.probe = probe.to(device)
        else:
            self.probe = probe.to(torch.bfloat16).to(device)

        self.optimizer = optim.AdamW(
            self.probe.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Binary cross-entropy with logits (more numerically stable)
        self.criterion = nn.BCEWithLogitsLoss()

        # Learning rate scheduler (cosine annealing)
        self.scheduler = None
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=1e-5
            )

        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        show_progress: bool = False
    ) -> float:
        """
        Train for one epoch

        Args:
            train_loader: DataLoader for training data
            show_progress: Whether to show progress bar

        Returns:
            Average training loss
        """
        self.probe.train()
        total_loss = 0.0
        num_batches = 0

        iterator = tqdm(train_loader, desc="Training", leave=False) if show_progress else train_loader

        for batch_acts, batch_labels in iterator:
            batch_acts = batch_acts.to(self.device)
            batch_labels = batch_labels.to(self.device).unsqueeze(1)  # (batch, 1)

            # Forward pass
            logits = self.probe(batch_acts)  # (batch, 1)
            loss = self.criterion(logits, batch_labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if show_progress:
                iterator.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(
        self,
        val_loader: DataLoader,
        show_progress: bool = False
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Evaluate on validation/test set

        Args:
            val_loader: DataLoader for validation data
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (loss, auc_roc, accuracy, predictions, true_labels)
        """
        self.probe.eval()
        total_loss = 0.0
        num_batches = 0
        all_probs = []
        all_labels = []

        iterator = tqdm(val_loader, desc="Evaluating", leave=False) if show_progress else val_loader

        with torch.no_grad():
            for batch_acts, batch_labels in iterator:
                batch_acts = batch_acts.to(self.device)
                batch_labels = batch_labels.to(self.device).unsqueeze(1)

                logits = self.probe(batch_acts)
                loss = self.criterion(logits, batch_labels)

                probs = torch.sigmoid(logits)

                total_loss += loss.item()
                num_batches += 1

                all_probs.append(probs.float().cpu().numpy())
                all_labels.append(batch_labels.float().cpu().numpy())

        avg_loss = total_loss / num_batches
        all_probs = np.concatenate(all_probs).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        # Compute metrics
        auc_roc = roc_auc_score(all_labels, all_probs)
        preds = (all_probs >= 0.5).astype(int)
        accuracy = accuracy_score(all_labels, preds)

        return avg_loss, auc_roc, accuracy, all_probs, all_labels

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: Optional[Path] = None,
        action_name: str = "",
        save_best: bool = True,
        early_stopping_patience: int = 10,
        verbose: bool = False
    ) -> Dict:
        """
        Full training loop with early stopping

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            action_name: Name of the action being trained
            save_best: Whether to save best model based on validation AUC
            early_stopping_patience: Stop if no improvement for N epochs
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        best_val_auc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_accuracy': [],
            'learning_rates': []
        }

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, show_progress=False)
            history['train_loss'].append(train_loss)

            # Validate
            val_loss, val_auc, val_acc, _, _ = self.evaluate(val_loader, show_progress=False)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['val_accuracy'].append(val_acc)

            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Print progress every 5 epochs if verbose
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"  Epoch {epoch + 1:2d}/{num_epochs}: "
                      f"Loss={train_loss:.4f}, Val AUC={val_auc:.4f}, Val Acc={val_acc:.4f}")

            # Save best model and check for improvement
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                epochs_without_improvement = 0

                if save_best and save_dir and action_name:
                    save_path = save_dir / f"probe_{action_name}.pth"
                    save_probe(
                        self.probe,
                        save_path,
                        metadata={
                            'action': action_name,
                            'epoch': epoch + 1,
                            'val_auc': val_auc,
                            'val_accuracy': val_acc,
                            'val_loss': val_loss
                        }
                    )
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1} (best: epoch {best_epoch + 1}, AUC={best_val_auc:.4f})")
                break

        # Add final summary to history
        history['best_epoch'] = best_epoch + 1
        history['best_val_auc'] = best_val_auc
        history['stopped_early'] = epochs_without_improvement >= early_stopping_patience

        return history


def compute_binary_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    action_name: str
) -> Dict:
    """
    Compute comprehensive binary classification metrics

    Args:
        true_labels: Ground truth binary labels
        predictions: Binary predictions (0 or 1)
        probabilities: Predicted probabilities
        action_name: Name of the cognitive action

    Returns:
        Dictionary of metrics
    """
    # Binary classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0
    )

    auc_roc = roc_auc_score(true_labels, probabilities)
    auc_pr = average_precision_score(true_labels, probabilities)
    accuracy = accuracy_score(true_labels, predictions)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

    metrics = {
        'action': action_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
    }

    return metrics


# Thread-safe progress tracking
class ProgressTracker:
    """Thread-safe progress tracker for parallel training"""
    
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def update(self, action_name: str, metrics: Dict):
        """Update progress and print status"""
        with self.lock:
            self.completed += 1
            elapsed = time.time() - self.start_time
            avg_time = elapsed / self.completed
            remaining = avg_time * (self.total - self.completed)
            
            print(f"[{self.completed:2d}/{self.total}] ✓ {action_name:30s} | "
                  f"AUC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f} | "
                  f"Acc: {metrics['accuracy']:.4f} | "
                  f"ETA: {remaining/60:.1f}m")


def train_single_probe_worker(
    action_idx: int,
    action_name: str,
    train_acts: torch.Tensor,
    train_labels: torch.Tensor,
    val_acts: torch.Tensor,
    val_labels: torch.Tensor,
    test_acts: torch.Tensor,
    test_labels: torch.Tensor,
    output_dir: Path,
    args,
    progress_tracker: ProgressTracker
) -> Tuple[Dict, Dict]:
    """
    Worker function to train a single probe
    
    Returns:
        Tuple of (metrics, history)
    """
    try:
        input_dim = train_acts.shape[1]
        
        # Create binary labels for this action
        train_binary_labels = create_binary_labels(train_labels, action_idx)
        val_binary_labels = create_binary_labels(val_labels, action_idx)
        test_binary_labels = create_binary_labels(test_labels, action_idx)
        
        # Create datasets
        train_dataset = BinaryActivationDataset(train_acts, train_binary_labels)
        val_dataset = BinaryActivationDataset(val_acts, val_binary_labels)
        test_dataset = BinaryActivationDataset(test_acts, test_binary_labels)
        
        # Only use pin_memory if tensors are on CPU (pin_memory doesn't work with CUDA tensors)
        use_pin_memory = not train_acts.is_cuda
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0,  # 0 for GPU training with threads
            pin_memory=use_pin_memory
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=use_pin_memory
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=use_pin_memory
        )
        
        # Create probe
        if args.model_type == "linear":
            probe = BinaryLinearProbe(input_dim)
        else:
            probe = BinaryMultiHeadProbe(input_dim, hidden_dim=args.hidden_dim)
        
        # Train
        trainer = BinaryProbeTrainer(
            probe,
            device=args.device,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            use_scheduler=args.use_scheduler,
            num_epochs=args.epochs
        )
        
        history = trainer.train(
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            save_dir=output_dir,
            action_name=action_name,
            save_best=True,
            early_stopping_patience=args.early_stopping_patience,
            verbose=False  # Don't print per-epoch progress in parallel mode
        )
        
        # Evaluate on test set
        test_loss, test_auc, test_acc, test_probs, test_labels_np = trainer.evaluate(
            test_loader,
            show_progress=False
        )
        
        test_preds = (test_probs >= 0.5).astype(int)
        metrics = compute_binary_metrics(
            test_labels_np,
            test_preds,
            test_probs,
            action_name
        )
        
        # Save per-action metrics
        metrics_path = output_dir / f"metrics_{action_name}.json"
        with open(metrics_path, 'w') as f:
            json.dump({**metrics, 'history': history}, f, indent=2)
        
        # Update progress
        progress_tracker.update(action_name, metrics)
        
        return metrics, history
        
    except Exception as e:
        print(f"ERROR training {action_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def train_all_binary_probes_parallel(
    train_acts: torch.Tensor,
    train_labels: torch.Tensor,
    val_acts: torch.Tensor,
    val_labels: torch.Tensor,
    test_acts: torch.Tensor,
    test_labels: torch.Tensor,
    output_dir: Path,
    args
):
    """
    Train 45 separate binary probes in parallel using multiple workers
    
    Args:
        train_acts, train_labels: Training activations and labels
        val_acts, val_labels: Validation activations and labels
        test_acts, test_labels: Test activations and labels
        output_dir: Directory to save trained probes
        args: Training arguments
    """
    idx_to_action = get_idx_to_action_mapping()
    num_actions = len(idx_to_action)
    input_dim = train_acts.shape[1]
    
    print(f"\n{'='*70}")
    print(f"PARALLEL TRAINING: {num_actions} binary probes using {args.num_workers} workers")
    print(f"{'='*70}")
    print(f"Input dim: {input_dim}")
    print(f"Model type: {args.model_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Parallel workers: {args.num_workers}")
    print(f"Device: {args.device}")
    
    # Calculate memory usage estimate
    if args.model_type == "linear":
        params_per_probe = input_dim + 1  # weights + bias
        size_mb = (params_per_probe * 2 * args.num_workers) / 1e6  # bfloat16 = 2 bytes
    else:
        params_per_probe = (input_dim * args.hidden_dim + args.hidden_dim * args.hidden_dim + 
                           args.hidden_dim + 1)
        size_mb = (params_per_probe * 2 * args.num_workers) / 1e6
    
    print(f"Est. model memory: ~{size_mb:.1f} MB for {args.num_workers} probes")
    print(f"{'='*70}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Move activations to GPU if they fit (faster training)
    if args.pin_activations_to_gpu:
        print("Pinning activations to GPU memory for faster training...")
        train_acts = train_acts.to(args.device)
        train_labels = train_labels.to(args.device)
        val_acts = val_acts.to(args.device)
        val_labels = val_labels.to(args.device)
        test_acts = test_acts.to(args.device)
        test_labels = test_labels.to(args.device)
        print("✓ Activations pinned to GPU\n")
    
    # Create progress tracker
    progress_tracker = ProgressTracker(num_actions)
    
    all_metrics = []
    start_time = time.time()
    
    # Train probes in parallel using ThreadPoolExecutor
    # Threads work better than processes for GPU work (PyTorch releases GIL)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        futures = {}
        for action_idx in range(num_actions):
            action_name = idx_to_action[action_idx]
            future = executor.submit(
                train_single_probe_worker,
                action_idx, action_name,
                train_acts, train_labels,
                val_acts, val_labels,
                test_acts, test_labels,
                output_dir, args, progress_tracker
            )
            futures[future] = action_name
        
        # Collect results as they complete
        for future in as_completed(futures):
            action_name = futures[future]
            try:
                metrics, history = future.result()
                all_metrics.append(metrics)
            except Exception as e:
                print(f"✗ Failed to train {action_name}: {str(e)}")
    
    elapsed = time.time() - start_time
    
    # Compute aggregate metrics
    print(f"\n{'='*70}")
    print("AGGREGATE METRICS ACROSS ALL PROBES")
    print(f"{'='*70}")
    
    avg_auc = np.mean([m['auc_roc'] for m in all_metrics])
    avg_f1 = np.mean([m['f1'] for m in all_metrics])
    avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    
    print(f"Average AUC-ROC:  {avg_auc:.4f}")
    print(f"Average F1:       {avg_f1:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:   {avg_recall:.4f}")
    print(f"\nTotal training time: {elapsed/60:.1f} minutes")
    print(f"Average time per probe: {elapsed/num_actions:.1f} seconds")
    
    # Save aggregate metrics
    aggregate_metrics = {
        'average_auc_roc': float(avg_auc),
        'average_f1': float(avg_f1),
        'average_accuracy': float(avg_accuracy),
        'average_precision': float(avg_precision),
        'average_recall': float(avg_recall),
        'total_training_time_seconds': float(elapsed),
        'num_workers': args.num_workers,
        'per_action_metrics': all_metrics
    }
    
    aggregate_path = output_dir / "aggregate_metrics.json"
    with open(aggregate_path, 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)
    
    print(f"\nSaved {num_actions} probes to {output_dir}")
    print(f"Saved aggregate metrics to {aggregate_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Train binary cognitive action probes in parallel")
    parser.add_argument(
        "--activations",
        type=str,
        required=True,
        help="Path to HDF5 file with activations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/probes_binary",
        help="Directory to save trained probes"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["linear", "multihead"],
        default="linear",
        help="Type of probe model"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension for MultiHeadProbe"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=18,
        help="Batch size for training (default: 128, balanced for small datasets)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (default: 50 with early stopping)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="Weight decay / L2 regularization (default: 1e-3)"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience (stop if no improvement for N epochs)"
    )
    parser.add_argument(
        "--use-scheduler",
        action="store_true",
        default=True,
        help="Use cosine annealing LR scheduler"
    )
    parser.add_argument(
        "--no-scheduler",
        dest="use_scheduler",
        action="store_false",
        help="Disable LR scheduler"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8 for 40GB GPU)"
    )
    parser.add_argument(
        "--pin-activations-to-gpu",
        action="store_true",
        default=True,
        help="Pin activations to GPU memory for faster training (recommended for 40GB+ VRAM)"
    )
    parser.add_argument(
        "--no-pin-activations",
        dest="pin_activations_to_gpu",
        action="store_false",
        help="Don't pin activations to GPU (use if you have limited VRAM)"
    )

    args = parser.parse_args()

    # Handle "auto" device by converting to cuda or cpu
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {args.device}")

    # Load activations
    print("Loading activations...")
    train_acts, train_labels, val_acts, val_labels, test_acts, test_labels = \
        load_activations_from_hdf5(Path(args.activations))

    print(f"Train: {train_acts.shape}, Val: {val_acts.shape}, Test: {test_acts.shape}")
    
    # Estimate memory usage
    acts_memory_mb = (train_acts.numel() + val_acts.numel() + test_acts.numel()) * 2 / 1e6
    print(f"Activation memory: ~{acts_memory_mb:.1f} MB")

    # Train all binary probes in parallel
    train_all_binary_probes_parallel(
        train_acts, train_labels,
        val_acts, val_labels,
        test_acts, test_labels,
        Path(args.output_dir),
        args
    )

    print("\n✓ Parallel training complete!")


if __name__ == "__main__":
    main()

