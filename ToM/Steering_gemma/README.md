# Gemma 3 Steering Adapter

This directory contains an improved adapter for steering Gemma 3 4B models using the repeng library.

## Problem

The original repeng implementation doesn't normalize steering vectors after training, leading to:
- **Huge coefficients needed** (300-500 instead of 0.5-5.0)
- **Inconsistent behavior** across different layers
- **Unpredictable steering strength**

## Solution

The `gemma3_adapter.py` provides:

1. **Automatic vector normalization** - Normalizes all direction vectors to unit norm
2. **Activation magnitude tracking** - Measures baseline activation magnitudes per layer
3. **Adaptive coefficient scaling** - Adjusts coefficients based on layer characteristics
4. **Per-layer control** - Supports different coefficients for different layers

## Key Findings from gemma_specifics.md

Based on research and testing:
- Gemma 3 uses **QK-norm instead of soft-capping** (affects activation magnitudes)
- Alternating **local/global attention layers** have different activation scales (5:1 ratio)
- Optimal layers are typically in the **latter half of the model** where representations "converge"
- **Unit-normalized vectors** should work with coefficients in the **0.5-5.0 range**

## Files

- `gemma3_adapter.py` - The main adapter module
- `test_gemma3_adapter.ipynb` - Test notebook with sentiment steering
- `gemma_specifics.md` - Detailed technical notes on Gemma 3 architecture

## Quick Start

### Basic Usage

```python
from gemma3_adapter import (
    Gemma3ControlVector,
    create_gemma3_model,
    make_dataset_with_truncation
)
from repeng import DatasetEntry

# 1. Load model
model, tokenizer = create_gemma3_model(
    model_name="google/gemma-3-4b-it",
    layer_range=(-4, -20),  # Layers -4 to -20
    use_bfloat16=True
)

# 2. Create training dataset
dataset = make_dataset_with_truncation(
    template="Act as if you're extremely {persona}.",
    pos_personas=["happy", "joyful"],
    neg_personas=["sad", "depressed"],
    suffixes=your_suffix_list,
    truncate_suffixes=True
)

# 3. Train normalized vector
vector = Gemma3ControlVector.train(
    model,
    tokenizer,
    dataset,
    method='pca_center',
    measure_activations=True,  # Measure activation magnitudes
    normalize_vectors=True,     # Normalize to unit norm
    batch_size=16
)

# 4. Apply steering with reasonable coefficients
model.set_control(vector, coeff=2.0)  # Not 300-500!

# 5. Generate
output = model.generate(...)
```

### Advanced: Adaptive Scaling

```python
from gemma3_adapter import Gemma3ControlModel

# Wrap as Gemma3ControlModel for advanced features
model = Gemma3ControlModel(base_model, layer_ids)

# Train with activation measurement
vector = Gemma3ControlVector.train(
    model, tokenizer, dataset,
    measure_activations=True
)

# Apply with adaptive per-layer scaling
model.set_control_per_layer(
    vector,
    base_coeff=2.0,
    use_adaptive_scaling=True  # Automatically adjusts per layer
)
```

## Testing

Run the test notebook to verify the adapter works:

```bash
jupyter notebook test_gemma3_adapter.ipynb
```

The notebook will:
1. Train normalized and unnormalized vectors
2. Compare coefficient requirements
3. Test adaptive scaling
4. Show that normalized vectors work with coefficients ~1-5 instead of 300-500

## Expected Results

### With Normalization (New)
- Vector norms: ~1.0 (unit normalized)
- Working coefficients: **0.5 - 5.0**
- Behavior: Consistent, predictable steering

### Without Normalization (Old)
- Vector norms: ~0.001 - 0.01 (unnormalized)
- Required coefficients: **300 - 500**
- Behavior: Requires guesswork to find right scale

## Integration with ToM Training

To use this adapter in your ToM steering notebook:

1. **Import the adapter:**
   ```python
   from gemma3_adapter import Gemma3ControlVector, create_gemma3_model
   ```

2. **Replace model loading:**
   ```python
   # OLD:
   # model = ControlModel(base_model, layer_ids)

   # NEW:
   model, tokenizer = create_gemma3_model(
       layer_range=(-4, -20),
       use_bfloat16=True
   )
   ```

3. **Replace vector training:**
   ```python
   # OLD:
   # tom_vector = ControlVector.train(model, tokenizer, dataset, method='pca_center')

   # NEW:
   tom_vector = Gemma3ControlVector.train(
       model, tokenizer, dataset,
       method='pca_center',
       measure_activations=True,
       normalize_vectors=True
   )
   ```

4. **Use normal coefficients:**
   ```python
   # OLD:
   # model.set_control(tom_vector, coeff=300)

   # NEW:
   model.set_control(tom_vector, coeff=2.0)
   ```

## Technical Details

### Normalization Method

Each direction vector is normalized to unit norm:
```python
normalized_direction = direction / np.linalg.norm(direction)
```

### Activation Measurement

For each layer, we measure the median L2 norm of activations across sample inputs:
```python
activation_norm = np.median(np.linalg.norm(hidden_states, axis=1))
```

### Adaptive Scaling

Coefficients are scaled inversely with activation magnitude:
```python
scaled_coeff = base_coeff * (target_norm / layer_activation_norm)
```

This accounts for varying activation scales across layers.

## Best Practices

1. **Always normalize vectors** - Set `normalize_vectors=True` (default)
2. **Measure activations** - Set `measure_activations=True` for adaptive scaling
3. **Start with coeff=1.0-2.0** - Then tune up or down
4. **Use bfloat16** - Better numerical stability for Gemma 3
5. **Test per-layer** - Do a layer sweep to find optimal layers (typically -8 to -16)

## References

- `gemma_specifics.md` - Detailed analysis of Gemma 3 architecture and steering
- repeng library: https://github.com/vgel/repeng
- CAA paper: Contrastive Activation Addition for steering LLMs