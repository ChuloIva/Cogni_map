#!/usr/bin/env python3
"""
Step-by-step diagnostic to find why steering vectors aren't working.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, 'repeng')
from repeng import ControlModel, ControlVector, DatasetEntry
from repeng.control import ControlModule

print("="*80)
print("STEERING VECTOR PIPELINE DIAGNOSTIC")
print("="*80)

# Load model
model_name = "google/gemma-3-4b-it"
print(f"\n1. Loading {model_name}...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = 0

print(f"   Model class: {base_model.__class__.__name__}")

# Set up repeng_layers
print("\n2. Setting up repeng_layers override...")
if hasattr(base_model, 'language_model') and hasattr(base_model.language_model, 'layers'):
    print("   Found multimodal architecture")
    base_model.repeng_layers = base_model.language_model.layers
    layers = base_model.language_model.layers
    num_layers = len(layers)
    # CRITICAL FIX: Add num_hidden_layers to config
    base_model.config.num_hidden_layers = num_layers
    print(f"   Set config.num_hidden_layers = {num_layers}")
elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
    print("   Found standard architecture")
    base_model.repeng_layers = base_model.model.layers
    layers = base_model.model.layers
    num_layers = len(layers)

print(f"   Total layers: {num_layers}")
print(f"   repeng_layers set: {hasattr(base_model, 'repeng_layers')}")

# Create ControlModel
print("\n3. Creating ControlModel...")
layer_ids = list(range(-5, -10, -1))  # Just 5 layers for faster testing
print(f"   Layer IDs: {layer_ids}")

model = ControlModel(base_model, layer_ids)
print(f"   ✓ ControlModel created")
print(f"   Wrapped layer IDs: {model.layer_ids}")

# Verify layers are wrapped
print("\n4. Verifying layers are wrapped with ControlModule...")
wrapped_count = 0
for i in model.layer_ids:
    layer = layers[i]
    is_wrapped = isinstance(layer, ControlModule)
    print(f"   Layer {i}: {is_wrapped}")
    if is_wrapped:
        wrapped_count += 1

if wrapped_count == 0:
    print(f"   ✗ ERROR: No layers were wrapped!")
    print(f"   Checking actual layer references...")
    # Check if the layers object is the same
    if hasattr(base_model, 'repeng_layers'):
        print(f"   repeng_layers is layers: {base_model.repeng_layers is layers}")
else:
    print(f"   ✓ {wrapped_count}/{len(model.layer_ids)} layers wrapped")

# Create a tiny training dataset
print("\n5. Creating minimal training dataset...")
train_data = [
    DatasetEntry(positive="Happy: That", negative="Sad: That"),
    DatasetEntry(positive="Happy: I", negative="Sad: I"),
    DatasetEntry(positive="Happy: This", negative="Sad: This"),
]
print(f"   Created {len(train_data)} training pairs")

# Train vector
print("\n6. Training control vector...")
model.reset()
try:
    control_vector = ControlVector.train(model, tokenizer, train_data)
    print(f"   ✓ Vector trained successfully")
    print(f"   Vector has {len(control_vector.directions)} layer directions")

    # Check vector magnitudes
    print("\n7. Checking vector magnitudes...")
    for layer_id, direction in control_vector.directions.items():
        magnitude = torch.norm(torch.tensor(direction)).item()
        print(f"   Layer {layer_id}: magnitude = {magnitude:.4f}")

except Exception as e:
    print(f"   ✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test applying control
print("\n8. Testing control application...")
model.reset()

# Get a simple input
test_text = "Hello"
messages = [{"role": "user", "content": test_text}]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

print(f"   Input: {repr(input_text[:50])}")

# Forward pass without control
print("\n9. Forward pass WITHOUT control...")
with torch.no_grad():
    outputs_baseline = model(**input_ids, output_hidden_states=True)
    hidden_baseline = outputs_baseline.hidden_states[-1][0, -1, :5]  # Last token, first 5 dims
    print(f"   Last hidden state (first 5 dims): {hidden_baseline.float().cpu().numpy()}")

# Apply control
print("\n10. Applying control vector (coeff=2.0)...")
model.set_control(control_vector, coeff=2.0)
print("   Control vector applied to model")

# Forward pass with control
print("\n12. Forward pass WITH control...")
with torch.no_grad():
    outputs_steered = model(**input_ids, output_hidden_states=True)
    hidden_steered = outputs_steered.hidden_states[-1][0, -1, :5]
    print(f"   Last hidden state (first 5 dims): {hidden_steered.float().cpu().numpy()}")

# Compare
print("\n13. Comparing hidden states...")
diff = torch.norm(hidden_steered - hidden_baseline).item()
print(f"   Difference magnitude: {diff:.4f}")

if diff < 0.001:
    print("   ✗ ERROR: Hidden states are nearly identical - steering not working!")
else:
    print(f"   ✓ Hidden states differ - steering is affecting activations")

# Test generation
print("\n14. Testing generation...")
model.reset()

gen_kwargs = {
    "max_new_tokens": 20,
    "do_sample": False,
    "pad_token_id": tokenizer.pad_token_id,
}

baseline_output = model.generate(input_ids.input_ids, **gen_kwargs)
baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)

model.set_control(control_vector, coeff=2.0)
steered_output = model.generate(input_ids.input_ids, **gen_kwargs)
steered_text = tokenizer.decode(steered_output[0], skip_special_tokens=True)

print(f"\n   Baseline:  {baseline_text}")
print(f"   Steered:   {steered_text}")
print(f"   Same: {baseline_text == steered_text}")

if baseline_text == steered_text:
    print("\n   ✗ ERROR: Generated text is identical - steering not affecting generation!")
else:
    print("\n   ✓ Generated text differs - steering is working!")

model.reset()

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)