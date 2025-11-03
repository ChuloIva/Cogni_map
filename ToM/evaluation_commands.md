# Cognitive Action Evaluation Commands

Quick reference for running evaluations with different steering vector configurations.

## Basic Usage

### Single Vector Evaluation
```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vector steering_vectors/tom_caa_forward_belief_new.gguf \
  --steering-coeff -400 \
  --num-samples 20 \
  --output-prefix single_vector_test \
  --condition 0_forward_belief_false_belief
  
```

python evaluate_cognitive_actions_with_steering.py \
  --steering-vector steering_vectors/tom_caa_forward_belief_new.gguf \
  --steering-coeff -400 \
  --num-samples 100 \
  --output-prefix backward_100_1 \
  --condition 1_backward_belief_false_belief


## Combined Vector Evaluations

### All Three ToM Vectors (Equal Weight)
```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_order_init_chat.gguf \
    steering_vectors/tom_direction_chat.gguf \
    steering_vectors/tom_core_capabilities_chat.gguf \
  --steering-coeffs -200 -200 -200 \
  --num-samples 30 \
  --output-prefix combined_all_equal
```
python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_direction.gguf \
    steering_vectors/tom_core_capabilities.gguf \
    steering_vectors/tom_backward_belief.gguf \
  --steering-coeffs 700 700 300 \
  --num-samples 15 \
  --output-prefix three_vecs

### Core + Direction (No Order Init)
```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_core_capabilities_chat.gguf \
    steering_vectors/tom_direction_chat.gguf \
  --steering-coeffs 1000 1000 \
  --num-samples 20 \
  --output-prefix core_direction
```

### Core Only (High Strength)
```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vector steering_vectors/tom_core_capabilities.gguf \
  --steering-coeff 600 \
  --num-samples 10 \
  --output-prefix core_high_strength
```

### Weighted Combination (Core=2x, Direction=1x)
```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_core_capabilities_chat.gguf \
    steering_vectors/tom_direction_chat.gguf \
  --steering-coeffs 2000 1000 \
  --num-samples 20 \
  --output-prefix core_2x_direction_1x
```

### Subtraction Example (Core - Order Init)
```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_forward_belief_false_persona_all_layers.gguf \
  --steering-coeff 400 \
  --num-samples 20 \
  --output-prefix forward_belief \
  --condition 0_forward_belief_false_belief \
```

## Full Dataset Evaluation

### Large Scale Test (100 samples)
```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_order_init.gguf \
    steering_vectors/tom_direction_chat.gguf \
    steering_vectors/tom_core_capabilities_chat.gguf \
  --steering-coeffs 1000 1000 1000 \
  --num-samples 100 \
  --output-prefix combined_large_test
```

### Test Different ToM Conditions
```bash
# False belief condition
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_core_capabilities_chat.gguf \
    steering_vectors/tom_direction_chat.gguf \
  --steering-coeffs 1000 1000 \
  --condition 0_backward_belief_false_belief \
  --num-samples 20 \
  --output-prefix false_belief_combined

# True belief condition
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_core_capabilities_chat.gguf \
    steering_vectors/tom_direction_chat.gguf \
  --steering-coeffs 1000 1000 \
  --condition 1_backward_belief_true_belief \
  --num-samples 20 \
  --output-prefix true_belief_combined
```

python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_cca_forward_belief_new.gguf \
    steering_vectors/tom_core_capabilites.gguf \
  --steering-coeffs 300 500 \
  --condition 1_forward_belief_true_belief \
  --num-samples 20 \
  --output-prefix true_belief_combined

## Additional Options

```bash
# Specify different probe directory
--probes-dir data/probes_binary

# Use different model
--model google/gemma-3-4b-it

# Start from offset in dataset
--offset 50

# Custom output prefix
--output-prefix my_experiment_name
```

## Output Files

Results are saved in `ToM/results/` directory:
- `{prefix}_raw.csv` - Raw results for each sample
- `{prefix}_summary.json` - Summary statistics
- `{prefix}_differences.png` - Cognitive action differences plot
- `{prefix}_accuracy.png` - Accuracy comparison plot