#!/usr/bin/env python3
"""
Diagnostic script to inspect Gemma model structure and ControlModel wrapping.
Run this to see what layers are actually being wrapped.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import sys
sys.path.insert(0, 'repeng')
from repeng import ControlModel, ControlVector, DatasetEntry

def inspect_model_structure():
    """Inspect the Gemma model structure before wrapping."""
    model_name = "google/gemma-3-4b-it"

    print("="*80)
    print("GEMMA-3-4B-IT MODEL STRUCTURE INSPECTION")
    print("="*80)

    # Load config
    print("\n1. Loading model config...")
    config = AutoConfig.from_pretrained(model_name)

    print(f"   Model type: {config.model_type}")
    print(f"   Config class: {config.__class__.__name__}")

    # Try to get common attributes (Gemma3 might use different names)
    for attr_name in ['num_hidden_layers', 'num_layers', 'n_layers']:
        if hasattr(config, attr_name):
            print(f"   Num hidden layers ({attr_name}): {getattr(config, attr_name)}")
            break

    for attr_name in ['hidden_size', 'n_embd', 'd_model']:
        if hasattr(config, attr_name):
            print(f"   Hidden size ({attr_name}): {getattr(config, attr_name)}")
            break

    for attr_name in ['num_attention_heads', 'n_head', 'num_heads']:
        if hasattr(config, attr_name):
            print(f"   Num attention heads ({attr_name}): {getattr(config, attr_name)}")
            break

    print(f"   Has vision_config: {hasattr(config, 'vision_config')}")

    # Show all config attributes
    print("\n   All config attributes:")
    for attr in sorted(dir(config)):
        if not attr.startswith('_') and not callable(getattr(config, attr)):
            try:
                val = getattr(config, attr)
                if isinstance(val, (int, str, bool, float)):
                    print(f"     {attr}: {val}")
            except:
                pass

    # Load model (small footprint - just structure)
    print("\n2. Loading model structure (this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    print(f"   Model class: {model.__class__.__name__}")
    print(f"   Model dtype: {model.dtype}")

    # Inspect named modules
    print("\n3. Model module hierarchy:")
    for name, module in model.named_modules():
        if 'layers' in name and len(name.split('.')) <= 3:  # Show top-level structure
            print(f"   {name}: {module.__class__.__name__}")

    # Check if model.layers exists
    print("\n4. Checking for model.layers attribute:")
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        print(f"   ✓ Found model.model.layers")
        print(f"   Number of layers: {len(layers)}")
        print(f"   Layer type: {type(layers[0]).__name__}")
    else:
        print("   ✗ model.model.layers not found!")
        print("   Available attributes:")
        for attr in dir(model):
            if not attr.startswith('_'):
                print(f"     - {attr}")

    # Test ControlModel wrapping
    print("\n5. Testing ControlModel wrapping:")
    layer_ids = list(range(-5, -18, -1))
    print(f"   Layer IDs to wrap: {layer_ids}")

    try:
        control_model = ControlModel(model, layer_ids)
        print(f"   ✓ ControlModel created successfully")
        print(f"   Wrapped layer IDs: {control_model.layer_ids}")

        # Check if layers are actually wrapped
        layers = model.model.layers
        print("\n6. Verifying layer wrapping:")
        for i in control_model.layer_ids:
            layer = layers[i]
            from repeng.control import ControlModule
            is_wrapped = isinstance(layer, ControlModule)
            print(f"   Layer {i}: {type(layer).__name__} - {'✓ Wrapped' if is_wrapped else '✗ Not wrapped'}")

    except Exception as e:
        print(f"   ✗ Failed to create ControlModel: {e}")
        import traceback
        traceback.print_exc()

    # Test tokenizer
    print("\n7. Testing tokenizer:")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"   Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Pad token ID: {tokenizer.pad_token_id}")

    # Test chat template
    print("\n8. Testing chat template:")
    test_messages = [{"role": "user", "content": "Hello"}]
    try:
        chat_text = tokenizer.apply_chat_template(
            test_messages,
            add_generation_prompt=True,
            tokenize=False
        )
        print(f"   ✓ Chat template works")
        print(f"   Example output:\n{repr(chat_text[:100])}...")
    except Exception as e:
        print(f"   ✗ Chat template failed: {e}")

    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    inspect_model_structure()
