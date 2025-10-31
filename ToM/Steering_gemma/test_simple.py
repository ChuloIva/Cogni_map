"""
Simple test to verify steering is working
"""
import sys
import json
import torch

sys.path.insert(0, '../repeng')
sys.path.insert(0, '.')

from gemma3_adapter import (
    Gemma3ControlVector,
    create_gemma3_model,
)
from repeng import DatasetEntry

print("="*80)
print("SIMPLE STEERING TEST")
print("="*80)

# Load model
print("\n1. Loading model...")
model, tokenizer = create_gemma3_model(
    model_name="google/gemma-3-4b-it",
    layer_range=(-4, -20),  # 17 layers
    use_bfloat16=True
)
print(f"✓ Model loaded, controlling {len(model.layer_ids)} layers")

# Load suffixes
print("\n2. Loading training data...")
try:
    with open("/Users/ivanculo/Desktop/Projects/Cogni_map/brije/ToM/repeng/notebooks/data/all_truncated_outputs.json") as f:
        suffixes = json.load(f)
    print(f"✓ Loaded {len(suffixes)} suffixes")
except:
    suffixes = ["That's interesting.", "I see.", "Makes sense."] * 30
    print(f"✓ Using {len(suffixes)} fallback suffixes")

# Create simple dataset
print("\n3. Creating training dataset...")
dataset = []
for suffix in suffixes[:300]:  # Use 50 suffixes
    dataset.append(DatasetEntry(
        positive=f"Act as if you're extremely happy. {suffix}",
        negative=f"Act as if you're extremely sad. {suffix}"
    ))
print(f"✓ Created {len(dataset)} training pairs")

# Train vector
print("\n4. Training steering vector...")
model.reset()
vector = Gemma3ControlVector.train(
    model,
    tokenizer,
    dataset,
    method='pca_center',
    measure_activations=False,
    normalize_vectors=True,
    batch_size=16
)
print(f"✓ Vector trained for layers: {list(vector.directions.keys())}")

# Check vector norms
first_layer = list(vector.directions.keys())[0]
norm = torch.norm(torch.tensor(vector.directions[first_layer])).item()
print(f"  First layer norm: {norm:.6f} (should be ~1.0)")

# Test generation
print("\n" + "="*80)
print("GENERATION TESTS")
print("="*80)

def generate(prompt, max_tokens=60):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

    output = model.generate(
        input_ids.input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

test_prompt = "How was your day?"

print(f"\nPrompt: '{test_prompt}'\n")

# Baseline
print("[BASELINE - No steering]")
print("-"*80)
model.reset()
baseline = generate(test_prompt)
print(baseline)
print()

# Positive steering with different coefficients
for coeff in [1.0, 2.0, 5.0]:
    print(f"[POSITIVE STEERING - coeff={coeff}]")
    print("-"*80)
    model.set_control(vector, coeff=coeff)
    positive = generate(test_prompt)
    print(positive)
    print()

# Negative steering
for coeff in [1.0, 2.0, 5.0]:
    print(f"[NEGATIVE STEERING - coeff=-{coeff}]")
    print("-"*80)
    model.set_control(vector, coeff=-coeff)
    negative = generate(test_prompt)
    print(negative)
    print()

model.reset()

print("="*80)
print("TEST COMPLETE")
print("="*80)
print("\nIf you see different responses above (happy vs sad tone), steering is working!")
print("If all responses are the same, there's still an issue.")