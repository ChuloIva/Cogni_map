"""
Debug layer wrapping and steering application
"""
import sys
import torch
import numpy as np

sys.path.insert(0, '../repeng')
sys.path.insert(0, '.')

from gemma3_adapter import create_gemma3_model, Gemma3ControlVector
from repeng import DatasetEntry, ControlVector
from repeng.control import model_layer_list, ControlModule

print("Loading model...")
model, tokenizer = create_gemma3_model(
    model_name="google/gemma-3-4b-it",
    layer_range=(-4, -8),
    use_bfloat16=True
)

print("\n" + "="*80)
print("LAYER WRAPPING INSPECTION")
print("="*80)

# Check if layers are properly wrapped
layers = model_layer_list(model)
print(f"\nTotal layers in model: {len(layers)}")
print(f"Layer IDs to control: {model.layer_ids}")

# Check actual layer indices
n_layers = len(layers)
actual_indices = [i if i >= 0 else n_layers + i for i in model.layer_ids]
print(f"Actual layer indices: {actual_indices}")

# Check if layers are wrapped with ControlModule
print("\nChecking if layers are wrapped:")
for idx in actual_indices:
    layer = layers[idx]
    is_wrapped = isinstance(layer, ControlModule)
    print(f"  Layer {idx}: {'✓ ControlModule' if is_wrapped else '✗ Not wrapped'}")
    if is_wrapped:
        has_control = layer.params.control is not None
        print(f"    Has control: {'Yes' if has_control else 'No'}")

# Create a simple test vector manually
print("\n" + "="*80)
print("CREATING MANUAL TEST VECTOR")
print("="*80)

# Create a dummy vector with known values
test_directions = {}
for layer_id in actual_indices:
    # Create a vector with large values to see clear effect
    test_directions[layer_id] = np.ones(4096, dtype=np.float32) * 0.1

test_vector = ControlVector(
    model_type=model.model.config.model_type,
    directions=test_directions
)

print(f"Created test vector for layers: {list(test_directions.keys())}")
print(f"Vector magnitude: {np.linalg.norm(test_directions[actual_indices[0]]):.4f}")

# Apply the vector
print("\nApplying test vector with coeff=1.0...")
model.set_control(test_vector, coeff=1.0)

# Check if it was applied
print("\nChecking if control was applied to layers:")
for idx in actual_indices:
    layer = layers[idx]
    if isinstance(layer, ControlModule):
        has_control = layer.params.control is not None
        if has_control:
            control_norm = torch.norm(layer.params.control).item()
            print(f"  Layer {idx}: ✓ Control applied (norm={control_norm:.4f})")
        else:
            print(f"  Layer {idx}: ✗ No control")
    else:
        print(f"  Layer {idx}: ✗ Not a ControlModule")

model.reset()

# Now test with actual training
print("\n" + "="*80)
print("TRAINING ACTUAL VECTOR")
print("="*80)

test_dataset = [
    DatasetEntry(
        positive="Act as if you're extremely happy. That's great!",
        negative="Act as if you're extremely sad. That's terrible."
    )
] * 10

print(f"Training with {len(test_dataset)} pairs...")
model.reset()

# Use Gemma3ControlVector instead of base ControlVector!
trained_vector = Gemma3ControlVector.train(
    model, tokenizer, test_dataset,
    method='pca_center', batch_size=4,
    measure_activations=False,
    normalize_vectors=False
)

print(f"\nTrained vector layers: {list(trained_vector.directions.keys())}")
for layer_id in list(trained_vector.directions.keys())[:3]:
    norm = np.linalg.norm(trained_vector.directions[layer_id])
    print(f"  Layer {layer_id} norm: {norm:.8f}")
    print(f"  First 10 values: {trained_vector.directions[layer_id][:10]}")

# Apply trained vector
print("\nApplying trained vector with coeff=2.0...")
model.set_control(trained_vector, coeff=2.0)

# Check if applied
print("\nVerifying control application:")
for idx in actual_indices:
    layer = layers[idx]
    if isinstance(layer, ControlModule):
        if layer.params.control is not None:
            control_norm = torch.norm(layer.params.control).item()
            print(f"  Layer {idx}: ✓ Applied (norm={control_norm:.4f})")
        else:
            print(f"  Layer {idx}: ✗ No control")

# Test generation
print("\n" + "="*80)
print("GENERATION TEST")
print("="*80)

def generate_simple(prompt, max_tokens=30):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids.input_ids,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

test_prompt = "I am feeling"

print("\n[No steering]")
model.reset()
output1 = generate_simple(test_prompt)
print(output1)

print("\n[With steering, coeff=2.0]")
model.set_control(trained_vector, coeff=2.0)
output2 = generate_simple(test_prompt)
print(output2)

print("\n[With steering, coeff=10.0]")
model.set_control(trained_vector, coeff=10.0)
output3 = generate_simple(test_prompt)
print(output3)

model.reset()