"""
Debug script to diagnose steering issues
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, '../repeng')
sys.path.insert(0, '.')

from gemma3_adapter import (
    Gemma3ControlVector,
    create_gemma3_model,
    make_dataset_with_truncation,
    print_vector_analysis
)
from repeng import ControlVector, DatasetEntry

print("Loading model...")
model, tokenizer = create_gemma3_model(
    model_name="google/gemma-3-4b-it",
    layer_range=(-4, -8),  # Just a few layers for quick testing
    use_bfloat16=True
)

print("\n" + "="*80)
print("Creating test dataset...")
print("="*80)

# Create a simple test dataset
test_suffixes = ["That's interesting.", "I understand.", "Makes sense."] * 5

test_dataset = make_dataset_with_truncation(
    template="Act as if you're extremely {persona}.",
    pos_personas=["happy", "joyful"],
    neg_personas=["sad", "depressed"],
    suffixes=test_suffixes,
    truncate_suffixes=False,  # No truncation for simplicity
    max_truncations=1
)

print(f"Created {len(test_dataset)} training pairs")

print("\n" + "="*80)
print("Training THREE vectors for comparison:")
print("1. Base repeng (no normalization)")
print("2. Gemma3 with normalization")
print("3. Gemma3 without normalization")
print("="*80)

# Train base vector (original repeng)
print("\n--- Training BASE vector (original repeng) ---")
model.reset()
base_vector = ControlVector.train(
    model, tokenizer, test_dataset,
    method='pca_center', batch_size=8
)

# Train normalized vector
print("\n--- Training NORMALIZED vector ---")
model.reset()
normalized_vector = Gemma3ControlVector.train(
    model, tokenizer, test_dataset,
    method='pca_center',
    measure_activations=False,
    normalize_vectors=True,
    batch_size=8
)

# Train unnormalized Gemma3 vector
print("\n--- Training UNNORMALIZED Gemma3 vector ---")
model.reset()
unnormalized_gemma_vector = Gemma3ControlVector.train(
    model, tokenizer, test_dataset,
    method='pca_center',
    measure_activations=False,
    normalize_vectors=False,
    batch_size=8
)

print("\n" + "="*80)
print("VECTOR COMPARISON")
print("="*80)

# Compare vector properties
for name, vec in [("Base", base_vector), ("Normalized", normalized_vector), ("Unnormalized", unnormalized_gemma_vector)]:
    print(f"\n{name} Vector:")
    first_layer = list(vec.directions.keys())[0]
    norm = np.linalg.norm(vec.directions[first_layer])
    print(f"  First layer ({first_layer}) norm: {norm:.8f}")
    print(f"  First 5 values: {vec.directions[first_layer][:5]}")

# Test generation function
def generate_text(prompt, model, tokenizer, max_new_tokens=60):
    """Generate text using chat template format."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

    output = model.generate(
        input_ids.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Deterministic for debugging
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

test_prompt = "How are you feeling today?"

print("\n" + "="*80)
print("GENERATION TESTS")
print("="*80)

# Baseline
print("\n[BASELINE - No steering]")
model.reset()
baseline = generate_text(test_prompt, model, tokenizer)
print(baseline[:200])

# Test base vector with different coefficients
print("\n" + "="*80)
print("BASE VECTOR (original repeng)")
print("="*80)

for coeff in [0.5, 1.0, 2.0, 100, 300]:
    print(f"\n--- Coefficient: {coeff} ---")
    model.set_control(base_vector, coeff=coeff)
    output = generate_text(test_prompt, model, tokenizer)
    print(output[:150])
    model.reset()

# Test normalized vector
print("\n" + "="*80)
print("NORMALIZED VECTOR")
print("="*80)

for coeff in [0.5, 1.0, 2.0, 100, 300]:
    print(f"\n--- Coefficient: {coeff} ---")
    model.set_control(normalized_vector, coeff=coeff)
    output = generate_text(test_prompt, model, tokenizer)
    print(output[:150])
    model.reset()

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Check if vectors are actually different after normalization
first_layer = list(base_vector.directions.keys())[0]
base_norm = np.linalg.norm(base_vector.directions[first_layer])
normalized_norm = np.linalg.norm(normalized_vector.directions[first_layer])

print(f"\nBase vector norm: {base_norm:.8f}")
print(f"Normalized vector norm: {normalized_norm:.8f}")
print(f"Scaling factor needed: {base_norm / normalized_norm:.2f}x")

# Check if direction is the same (just scaled)
base_unit = base_vector.directions[first_layer] / base_norm
normalized_unit = normalized_vector.directions[first_layer] / normalized_norm
cosine_sim = np.dot(base_unit, normalized_unit)
print(f"\nCosine similarity: {cosine_sim:.8f}")
print("(Should be ~1.0 if they point in the same direction)")

if abs(cosine_sim) < 0.99:
    print("\n⚠️  WARNING: Vectors are pointing in different directions!")
    print("This suggests the normalization is changing the vector direction, not just scale.")