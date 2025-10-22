"""
Regression-based sentiment probes for continuous sentiment scoring

Instead of binary classification (0/1), these probes output continuous
sentiment scores (e.g., -3 to +3) for smoother, more nuanced detection.
"""

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys


class SentimentRegressionProbe(nn.Module):
    """
    Linear regression probe for continuous sentiment prediction

    Outputs unbounded continuous scores instead of 0-1 probabilities.
    Positive scores = positive sentiment, negative scores = negative sentiment.
    """

    def __init__(self, input_dim: int, dropout: float = 0.1):
        """
        Initialize regression probe

        Args:
            input_dim: Dimension of input activations
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, 1)  # Single output for regression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input activations (batch_size, input_dim)

        Returns:
            Sentiment scores (batch_size, 1) - unbounded continuous values
        """
        x = self.dropout(x)
        scores = self.linear(x)  # No sigmoid/softmax, raw linear output
        return scores

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get sentiment predictions

        Args:
            x: Input activations (batch_size, input_dim)

        Returns:
            Sentiment scores (batch_size, 1)
        """
        return self.forward(x)


class SentimentRegressionDataset(Dataset):
    """Dataset for regression-based sentiment training"""

    def __init__(self, activations: torch.Tensor, labels: torch.Tensor):
        """
        Initialize dataset

        Args:
            activations: Tensor of shape (num_examples, hidden_dim)
            labels: Sentiment labels (0 = negative, 1 = positive)
                    Will be converted to regression targets (-1 to +1)
        """
        self.activations = activations
        # Convert binary labels to regression targets: 0 -> -1, 1 -> +1
        # Match dtype with activations (bfloat16)
        targets = (labels.float() * 2.0) - 1.0
        self.targets = targets.to(activations.dtype)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.activations[idx], self.targets[idx]


class SentimentRegressionTrainer:
    """Trainer for regression-based sentiment probes"""

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
            probe: Regression probe model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: L2 regularization
            use_scheduler: Whether to use LR scheduler
            num_epochs: Total epochs for scheduler
        """
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        # Convert to bfloat16 to match activations
        if device == "mps":
            self.probe = probe.to(device)
        else:
            self.probe = probe.to(torch.bfloat16).to(device)

        self.optimizer = optim.AdamW(
            self.probe.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # MSE loss for regression
        self.criterion = nn.MSELoss()

        # Learning rate scheduler
        self.scheduler = None
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=1e-5
            )

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader, show_progress: bool = False) -> float:
        """Train for one epoch"""
        self.probe.train()
        total_loss = 0.0
        num_batches = 0

        iterator = tqdm(train_loader, desc="Training", leave=False) if show_progress else train_loader

        for batch_acts, batch_targets in iterator:
            batch_acts = batch_acts.to(self.device)
            batch_targets = batch_targets.to(self.device).unsqueeze(1)

            # Forward pass
            predictions = self.probe(batch_acts)
            loss = self.criterion(predictions, batch_targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if show_progress:
                iterator.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def evaluate(self, val_loader: DataLoader, show_progress: bool = False) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Evaluate on validation/test set

        Returns:
            Tuple of (mse, mae, r2, predictions, true_targets)
        """
        self.probe.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []

        iterator = tqdm(val_loader, desc="Evaluating", leave=False) if show_progress else val_loader

        with torch.no_grad():
            for batch_acts, batch_targets in iterator:
                batch_acts = batch_acts.to(self.device)
                batch_targets = batch_targets.to(self.device).unsqueeze(1)

                predictions = self.probe(batch_acts)
                loss = self.criterion(predictions, batch_targets)

                total_loss += loss.item()
                num_batches += 1

                all_preds.append(predictions.float().cpu().numpy())
                all_targets.append(batch_targets.float().cpu().numpy())

        mse = total_loss / num_batches
        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        # Compute metrics
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        return mse, mae, r2, all_preds, all_targets

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: Optional[Path] = None,
        save_best: bool = True,
        early_stopping_patience: int = 10,
        verbose: bool = False
    ) -> Dict:
        """
        Full training loop with early stopping

        Returns:
            Training history dictionary
        """
        best_val_mse = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0

        history = {
            'train_loss': [],
            'val_mse': [],
            'val_mae': [],
            'val_r2': [],
            'learning_rates': []
        }

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, show_progress=False)
            history['train_loss'].append(train_loss)

            # Validate
            val_mse, val_mae, val_r2, _, _ = self.evaluate(val_loader, show_progress=False)
            history['val_mse'].append(val_mse)
            history['val_mae'].append(val_mae)
            history['val_r2'].append(val_r2)

            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Print progress
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"  Epoch {epoch + 1:2d}/{num_epochs}: "
                      f"Loss={train_loss:.4f}, Val MSE={val_mse:.4f}, "
                      f"Val MAE={val_mae:.4f}, Val R²={val_r2:.4f}")

            # Save best model
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                epochs_without_improvement = 0

                if save_best and save_dir:
                    save_path = save_dir / "sentiment_regression_probe.pth"
                    self.save_probe(
                        save_path,
                        metadata={
                            'epoch': epoch + 1,
                            'val_mse': val_mse,
                            'val_mae': val_mae,
                            'val_r2': val_r2
                        }
                    )
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1} "
                          f"(best: epoch {best_epoch + 1}, MSE={best_val_mse:.4f})")
                break

        history['best_epoch'] = best_epoch + 1
        history['best_val_mse'] = best_val_mse
        history['stopped_early'] = epochs_without_improvement >= early_stopping_patience

        return history

    def save_probe(self, save_path: Path, metadata: Optional[Dict] = None):
        """Save probe model"""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'model_state_dict': self.probe.state_dict(),
            'model_class': 'SentimentRegressionProbe',
            'model_config': {
                'input_dim': self.probe.input_dim
            }
        }

        if metadata:
            state['metadata'] = metadata

        torch.save(state, save_path)


def load_activations_from_hdf5(file_path: Path) -> Tuple:
    """Load activations from HDF5 file"""
    with h5py.File(file_path, 'r') as f:
        train_acts = torch.from_numpy(f['train']['activations'][:]).bfloat16()
        train_labels = torch.from_numpy(f['train']['labels'][:])

        val_acts = torch.from_numpy(f['val']['activations'][:]).bfloat16()
        val_labels = torch.from_numpy(f['val']['labels'][:])

        test_acts = torch.from_numpy(f['test']['activations'][:]).bfloat16()
        test_labels = torch.from_numpy(f['test']['labels'][:])

    return train_acts, train_labels, val_acts, val_labels, test_acts, test_labels


def compute_regression_metrics(
    true_targets: np.ndarray,
    predictions: np.ndarray
) -> Dict:
    """
    Compute regression metrics

    Args:
        true_targets: Ground truth targets (-1 to +1)
        predictions: Predicted scores

    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(true_targets, predictions)
    mae = mean_absolute_error(true_targets, predictions)
    r2 = r2_score(true_targets, predictions)

    # Classification accuracy (using 0 as threshold)
    true_classes = (true_targets > 0).astype(int)
    pred_classes = (predictions > 0).astype(int)
    accuracy = np.mean(true_classes == pred_classes)

    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'accuracy': float(accuracy),  # Binary accuracy at 0 threshold
        'mean_prediction': float(np.mean(predictions)),
        'std_prediction': float(np.std(predictions)),
        'min_prediction': float(np.min(predictions)),
        'max_prediction': float(np.max(predictions))
    }

    return metrics


def train_sentiment_regression_probe(
    train_acts: torch.Tensor,
    train_labels: torch.Tensor,
    val_acts: torch.Tensor,
    val_labels: torch.Tensor,
    test_acts: torch.Tensor,
    test_labels: torch.Tensor,
    output_dir: Path,
    args
) -> Dict:
    """
    Train sentiment regression probe

    Returns:
        Dictionary with metrics and history
    """
    input_dim = train_acts.shape[1]

    print(f"\n{'='*70}")
    print("TRAINING REGRESSION-BASED SENTIMENT PROBE")
    print(f"{'='*70}")
    print(f"Input dim: {input_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    # Create datasets
    train_dataset = SentimentRegressionDataset(train_acts, train_labels)
    val_dataset = SentimentRegressionDataset(val_acts, val_labels)
    test_dataset = SentimentRegressionDataset(test_acts, test_labels)

    # Create data loaders
    use_pin_memory = not train_acts.is_cuda

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
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
    probe = SentimentRegressionProbe(input_dim)

    # Train
    trainer = SentimentRegressionTrainer(
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
        save_best=True,
        early_stopping_patience=args.early_stopping_patience,
        verbose=True
    )

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_mse, test_mae, test_r2, test_preds, test_targets = trainer.evaluate(
        test_loader,
        show_progress=False
    )

    metrics = compute_regression_metrics(test_targets, test_preds)

    print(f"\nTest Results:")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Accuracy (at 0): {metrics['accuracy']:.4f}")
    print(f"  Score range: [{metrics['min_prediction']:.2f}, {metrics['max_prediction']:.2f}]")

    # Save metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({**metrics, 'history': history}, f, indent=2)

    print(f"\n✅ Saved metrics to {metrics_path}")

    return metrics, history


def main():
    parser = argparse.ArgumentParser(
        description="Train regression-based sentiment probe"
    )
    parser.add_argument(
        "--activations",
        type=str,
        required=True,
        help="Path to HDF5 file with activations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/probes_regression/sentiment",
        help="Directory to save trained probe"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="Weight decay / L2 regularization"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience"
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

    args = parser.parse_args()

    # Handle "auto" device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load activations
    print("Loading activations...")
    train_acts, train_labels, val_acts, val_labels, test_acts, test_labels = \
        load_activations_from_hdf5(Path(args.activations))

    print(f"Train: {train_acts.shape}, Val: {val_acts.shape}, Test: {test_acts.shape}")

    # Train probe
    train_sentiment_regression_probe(
        train_acts, train_labels,
        val_acts, val_labels,
        test_acts, test_labels,
        Path(args.output_dir),
        args
    )

    print("\n✅ Regression probe training complete!")


if __name__ == "__main__":
    main()
