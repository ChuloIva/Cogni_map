"""
Probe model architectures for cognitive action detection
"""

# FOR AMD GPU
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["AMD_SERIALIZE_KERNEL"] = "3"
os.environ["TORCH_USE_HIP_DSA"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, List, Tuple


class LinearProbe(nn.Module):
    """Simple linear probe for cognitive action classification"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1
    ):
        """
        Initialize linear probe

        Args:
            input_dim: Dimension of input activations
            num_classes: Number of cognitive action classes
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input activations (batch_size, input_dim)

        Returns:
            Logits (batch_size, num_classes)
        """
        x = self.dropout(x)
        logits = self.linear(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions

        Args:
            x: Input activations (batch_size, input_dim)

        Returns:
            Probabilities (batch_size, num_classes)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions

        Args:
            x: Input activations (batch_size, input_dim)

        Returns:
            Class indices (batch_size,)
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)


class MultiHeadProbe(nn.Module):
    """
    Multi-head probe with shared encoder and per-action prediction heads
    More powerful than LinearProbe, useful for learning better representations
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head probe

        Args:
            input_dim: Dimension of input activations
            num_classes: Number of cognitive action classes
            hidden_dim: Dimension of hidden layer
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-head attention for feature refinement
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input activations (batch_size, input_dim)

        Returns:
            Logits (batch_size, num_classes)
        """
        # Encode
        h = self.encoder(x)  # (batch_size, hidden_dim)

        # Add sequence dimension for attention
        h = h.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Self-attention
        h_attn, _ = self.attention(h, h, h)  # (batch_size, 1, hidden_dim)

        # Remove sequence dimension
        h_attn = h_attn.squeeze(1)  # (batch_size, hidden_dim)

        # Classify
        logits = self.classifier(h_attn)  # (batch_size, num_classes)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions"""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions"""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)


class BinaryLinearProbe(nn.Module):
    """Binary linear probe for one-vs-rest classification"""

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize binary linear probe

        Args:
            input_dim: Dimension of input activations
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input activations (batch_size, input_dim)

        Returns:
            Logits (batch_size, 1) - raw logits for BCEWithLogitsLoss
        """
        x = self.dropout(x)
        logits = self.linear(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions

        Args:
            x: Input activations (batch_size, input_dim)

        Returns:
            Probabilities (batch_size, 1)
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions

        Args:
            x: Input activations (batch_size, input_dim)
            threshold: Decision threshold

        Returns:
            Binary predictions (batch_size, 1)
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).float()


class BinaryMultiHeadProbe(nn.Module):
    """
    Binary multi-head probe for one-vs-rest classification
    More powerful than BinaryLinearProbe
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize binary multi-head probe

        Args:
            input_dim: Dimension of input activations
            hidden_dim: Dimension of hidden layer
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-head attention for feature refinement
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Binary classification head
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input activations (batch_size, input_dim)

        Returns:
            Logits (batch_size, 1)
        """
        # Encode
        h = self.encoder(x)  # (batch_size, hidden_dim)

        # Add sequence dimension for attention
        h = h.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Self-attention
        h_attn, _ = self.attention(h, h, h)  # (batch_size, 1, hidden_dim)

        # Remove sequence dimension
        h_attn = h_attn.squeeze(1)  # (batch_size, hidden_dim)

        # Binary classify
        logits = self.classifier(h_attn)  # (batch_size, 1)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions"""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions"""
        probs = self.predict_proba(x)
        return (probs >= threshold).float()


class CalibratedProbe(nn.Module):
    """
    Wrapper that adds temperature scaling for probability calibration
    Useful for getting better-calibrated confidence scores
    """

    def __init__(
        self,
        base_probe: nn.Module,
        temperature: float = 1.0
    ):
        """
        Initialize calibrated probe

        Args:
            base_probe: Base probe model (LinearProbe or MultiHeadProbe)
            temperature: Temperature for scaling logits
        """
        super().__init__()
        self.base_probe = base_probe
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temperature scaling"""
        logits = self.base_probe(x)
        return logits / self.temperature

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get calibrated probability predictions"""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions"""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)


def save_probe(
    probe: nn.Module,
    save_path: Path,
    metadata: Optional[Dict] = None
):
    """
    Save probe model to disk

    Args:
        probe: Probe model to save
        save_path: Where to save the model
        metadata: Optional metadata dict to save
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        'model_state_dict': probe.state_dict(),
        'model_class': probe.__class__.__name__,
        'model_config': {
            'input_dim': probe.input_dim
        }
    }

    # Add num_classes for multi-class probes
    if hasattr(probe, 'num_classes'):
        state['model_config']['num_classes'] = probe.num_classes

    # Add hidden_dim for multi-head probes
    if hasattr(probe, 'hidden_dim'):
        state['model_config']['hidden_dim'] = probe.hidden_dim

    if metadata:
        state['metadata'] = metadata

    torch.save(state, save_path)
    print(f"Saved probe to {save_path}")


def load_probe(
    load_path: Path,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict]:
    """
    Load probe model from disk

    Args:
        load_path: Path to saved model
        device: Device to load model on

    Returns:
        Tuple of (probe_model, metadata)
    """
    # Handle "auto" device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    state = torch.load(load_path, map_location=device)

    model_class = state['model_class']
    model_config = state['model_config']

    # Instantiate the appropriate model class
    if model_class == 'LinearProbe':
        probe = LinearProbe(**model_config)
    elif model_class == 'MultiHeadProbe':
        probe = MultiHeadProbe(**model_config)
    elif model_class == 'BinaryLinearProbe':
        probe = BinaryLinearProbe(**model_config)
    elif model_class == 'BinaryMultiHeadProbe':
        probe = BinaryMultiHeadProbe(**model_config)
    elif model_class == 'CalibratedProbe':
        # Need to load base probe first
        base_config = model_config.copy()
        base_class = state.get('base_probe_class', 'LinearProbe')
        if base_class == 'LinearProbe':
            base_probe = LinearProbe(**base_config)
        else:
            base_probe = MultiHeadProbe(**base_config)
        probe = CalibratedProbe(base_probe)
    elif model_class == 'SentimentRegressionProbe':
        # Import SentimentRegressionProbe from the same package
        from .sentiment_regression_probe import SentimentRegressionProbe
        probe = SentimentRegressionProbe(**model_config)
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    # Detect dtype from saved weights (probes are trained in bfloat16)
    # Get dtype from the first weight tensor in state_dict
    first_weight_key = next(iter(state['model_state_dict'].keys()))
    saved_dtype = state['model_state_dict'][first_weight_key].dtype

    # Convert model to same dtype as saved weights BEFORE loading state_dict
    # This prevents automatic dtype conversion during load_state_dict
    probe = probe.to(dtype=saved_dtype)

    # Load state dict (now dtypes match, no conversion happens)
    probe.load_state_dict(state['model_state_dict'])

    # Move to device while preserving dtype
    probe.to(device=device)
    probe.eval()

    metadata = state.get('metadata', {})

    print(f"Loaded probe from {load_path}")
    return probe, metadata


def get_probe_predictions(
    probe: nn.Module,
    activations: torch.Tensor,
    top_k: int = 5,
    threshold: float = 0.1
) -> List[List[Tuple[int, float]]]:
    """
    Get top-k predictions with confidence scores

    Args:
        probe: Trained probe model
        activations: Input activations (batch_size, input_dim)
        top_k: Number of top predictions to return
        threshold: Minimum confidence threshold

    Returns:
        List of lists of (class_idx, confidence) tuples, one per batch item
    """
    probe.eval()
    with torch.no_grad():
        probs = probe.predict_proba(activations)  # (batch_size, num_classes)

    results = []
    for prob in probs:
        # Get top-k predictions
        top_probs, top_indices = torch.topk(prob, min(top_k, len(prob)))

        # Filter by threshold
        predictions = [
            (idx.item(), p.item())
            for idx, p in zip(top_indices, top_probs)
            if p.item() >= threshold
        ]

        results.append(predictions)

    return results


if __name__ == "__main__":
    # Test the models
    print("Testing probe models...")

    batch_size = 4
    input_dim = 768  # Typical for smaller transformers
    num_classes = 45

    # Create dummy input
    x = torch.randn(batch_size, input_dim)

    # Test LinearProbe
    print("\n1. Testing LinearProbe...")
    linear_probe = LinearProbe(input_dim, num_classes)
    logits = linear_probe(x)
    probs = linear_probe.predict_proba(x)
    preds = linear_probe.predict(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Probs shape: {probs.shape}")
    print(f"   Predictions shape: {preds.shape}")
    print(f"   Predictions: {preds}")

    # Test MultiHeadProbe
    print("\n2. Testing MultiHeadProbe...")
    multi_probe = MultiHeadProbe(input_dim, num_classes, hidden_dim=512)
    logits = multi_probe(x)
    probs = multi_probe.predict_proba(x)
    preds = multi_probe.predict(x)

    print(f"   Logits shape: {logits.shape}")
    print(f"   Probs shape: {probs.shape}")
    print(f"   Predictions: {preds}")

    # Test CalibratedProbe
    print("\n3. Testing CalibratedProbe...")
    calibrated_probe = CalibratedProbe(linear_probe, temperature=1.5)
    probs = calibrated_probe.predict_proba(x)

    print(f"   Calibrated probs shape: {probs.shape}")

    # Test save/load
    print("\n4. Testing save/load...")
    save_path = Path("test_probe.pth")
    save_probe(linear_probe, save_path, metadata={'test': True})
    loaded_probe, metadata = load_probe(save_path, device='cpu')
    print(f"   Metadata: {metadata}")
    save_path.unlink()  # Clean up

    # Test get_probe_predictions
    print("\n5. Testing get_probe_predictions...")
    predictions = get_probe_predictions(linear_probe, x, top_k=3, threshold=0.01)
    print(f"   Number of batch items: {len(predictions)}")
    print(f"   Predictions for first item: {predictions[0]}")

    print("\n✓ All tests passed!")
