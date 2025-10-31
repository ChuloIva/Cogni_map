# Optimized Vector Combinations for Specific ToM Conditions

Based on analysis of trained vectors and condition requirements, here are optimized evaluation commands for each condition.

## Trained Vectors Summary

From `tom_steering_colab (1).ipynb`, we have these specialized vectors:

1. **tom_order_init** - Implicit vs explicit belief (inferring from context vs stated beliefs)
2. **tom_direction** - Forward vs backward reasoning (predicting future vs reconstructing past)
3. **tom_variable** - Belief vs action (what they think vs what they do)
4. **tom_belief_type** - True vs false belief handling
5. **tom_forward_belief** - Tracking beliefs as events unfold (including false beliefs)
6. **tom_backward_belief** - Inferring prior beliefs from outcomes
7. **tom_forward_action** - Predicting actions based on beliefs
8. **tom_core_capabilities** - General ToM (perspective-taking, counterfactual reasoning)

---

## Condition-Specific Evaluations

### 1. Backward Belief + False Belief (0_backward_belief_false_belief)
**Task**: Infer what someone believed when they didn't witness a change
**Key Challenge**: Reconstruct past false beliefs from current state

```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_backward_belief.gguf \
    steering_vectors/tom_belief_type.gguf \
    steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 800 600 400 \
  --condition 0_backward_belief_false_belief \
  --num-samples 20 \
  --output-prefix backward_false_optimized
```

---

### 2. Backward Belief + True Belief (0_backward_belief_true_belief)
**Task**: Infer what someone believed when they witnessed the change
**Key Challenge**: Reconstruct updated beliefs after observation

```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_backward_belief.gguf \
    steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 800 500 \
  --condition 0_backward_belief_true_belief \
  --num-samples 20 \
  --output-prefix backward_true_optimized
```

---

### 3. Forward Action + False Belief (0_forward_action_false_belief)
**Task**: Predict what someone will do based on their false belief
**Key Challenge**: Simulate action from outdated/incorrect belief

```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_forward_action.gguf \
    steering_vectors/tom_belief_type.gguf \
    steering_vectors/tom_variable.gguf \
  --steering-coeffs 900 600 400 \
  --condition 0_forward_action_false_belief \
  --num-samples 20 \
  --output-prefix forward_action_false_optimized
```

---

### 4. Forward Action + True Belief (0_forward_action_true_belief)
**Task**: Predict what someone will do based on their updated true belief
**Key Challenge**: Predict action from accurate belief

```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_forward_action.gguf \
    steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 800 500 \
  --condition 0_forward_action_true_belief \
  --num-samples 20 \
  --output-prefix forward_action_true_optimized
```

---

### 5. Forward Belief + False Belief (0_forward_belief_false_belief)
**Task**: Track what someone believes when they miss critical information
**Key Challenge**: Recognize outdated belief formation

```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_forward_belief.gguf \
    steering_vectors/tom_belief_type.gguf \
    steering_vectors/tom_order_init.gguf \
  --steering-coeffs 800 600 400 \
  --condition 0_forward_belief_false_belief \
  --num-samples 20 \
  --output-prefix forward_belief_false_optimized
```

---

### 6. Forward Belief + True Belief (0_forward_belief_true_belief)
**Task**: Track what someone believes when they observe the change
**Key Challenge**: Track belief updates in real-time

```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_forward_belief.gguf \
    steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 100 100 \
  --condition 0_forward_belief_true_belief \
  --num-samples 10 \
  --output-prefix forward_belief_true_optimized
```

---

### 7. Backward Belief + False Belief (1_backward_belief_false_belief)
**Task**: Same as #1 but explicit belief statement included
**Key Challenge**: Use explicit belief to reconstruct false belief

```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_backward_belief.gguf \
    steering_vectors/tom_belief_type.gguf \
    steering_vectors/tom_order_init.gguf \
  --steering-coeffs 700 600 500 \
  --condition 1_backward_belief_false_belief \
  --num-samples 20 \
  --output-prefix backward_false_explicit_optimized
```

---

### 8. Backward Belief + True Belief (1_backward_belief_true_belief)
**Task**: Same as #2 but with explicit belief statement
**Key Challenge**: Use explicit belief + observation to reconstruct

```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_backward_belief.gguf \
    steering_vectors/tom_order_init.gguf \
    steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 700 500 400 \
  --condition 1_backward_belief_true_belief \
  --num-samples 20 \
  --output-prefix backward_true_explicit_optimized
```

---

### 9. Forward Action + False Belief (1_forward_action_false_belief)
**Task**: Same as #3 but with explicit belief statement
**Key Challenge**: Predict action using explicit false belief

```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_forward_action.gguf \
    steering_vectors/tom_belief_type.gguf \
    steering_vectors/tom_order_init.gguf \
  --steering-coeffs 800 600 500 \
  --condition 1_forward_action_false_belief \
  --num-samples 20 \
  --output-prefix forward_action_false_explicit_optimized
```

---

### 10. Forward Action + True Belief (1_forward_action_true_belief)
**Task**: Same as #4 but with explicit belief statement
**Key Challenge**: Predict action using explicit true belief

```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_forward_action.gguf \
    steering_vectors/tom_order_init.gguf \
    steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 700 500 400 \
  --condition 1_forward_action_true_belief \
  --num-samples 20 \
  --output-prefix forward_action_true_explicit_optimized
```

---

### 11. Forward Belief + False Belief (1_forward_belief_false_belief)
**Task**: Same as #5 but with explicit belief statement
**Key Challenge**: Track explicit + implicit false belief

```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_forward_belief.gguf \
    steering_vectors/tom_belief_type.gguf \
    steering_vectors/tom_order_init.gguf \
  --steering-coeffs 700 600 500 \
  --condition 1_forward_belief_false_belief \
  --num-samples 20 \
  --output-prefix forward_belief_false_explicit_optimized
```

---

### 12. Forward Belief + True Belief (1_forward_belief_true_belief)
**Task**: Same as #6 but with explicit belief statement
**Key Challenge**: Track explicit + observed true belief

```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_forward_belief.gguf \
    steering_vectors/tom_order_init.gguf \
    steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 700 500 400 \
  --condition 1_forward_belief_true_belief \
  --num-samples 20 \
  --output-prefix forward_belief_true_explicit_optimized
```

---

## Batch Evaluation Scripts

### Run All False Belief Conditions (Implicit - 0 prefix)
```bash
cd ToM

python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors steering_vectors/tom_backward_belief.gguf steering_vectors/tom_belief_type.gguf steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 800 600 400 \
  --condition 0_backward_belief_false_belief \
  --num-samples 20 \
  --output-prefix backward_false_optimized

python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors steering_vectors/tom_forward_action.gguf steering_vectors/tom_belief_type.gguf steering_vectors/tom_variable.gguf \
  --steering-coeffs 900 600 400 \
  --condition 0_forward_action_false_belief \
  --num-samples 20 \
  --output-prefix forward_action_false_optimized

python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors steering_vectors/tom_forward_belief.gguf steering_vectors/tom_belief_type.gguf steering_vectors/tom_order_init.gguf \
  --steering-coeffs 800 600 400 \
  --condition 0_forward_belief_false_belief \
  --num-samples 20 \
  --output-prefix forward_belief_false_optimized
```

### Run All True Belief Conditions (Implicit - 0 prefix)
```bash
cd ToM

python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors steering_vectors/tom_backward_belief.gguf steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 800 500 \
  --condition 0_backward_belief_true_belief \
  --num-samples 20 \
  --output-prefix backward_true_optimized

python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors steering_vectors/tom_forward_action.gguf steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 800 500 \
  --condition 0_forward_action_true_belief \
  --num-samples 20 \
  --output-prefix forward_action_true_optimized

python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors steering_vectors/tom_forward_belief.gguf steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 800 500 \
  --condition 0_forward_belief_true_belief \
  --num-samples 20 \
  --output-prefix forward_belief_true_optimized
```

### Run All Explicit Belief Conditions (1 prefix)
```bash
cd ToM

# False beliefs with explicit statements
python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors steering_vectors/tom_backward_belief.gguf steering_vectors/tom_belief_type.gguf steering_vectors/tom_order_init.gguf \
  --steering-coeffs 700 600 500 \
  --condition 1_backward_belief_false_belief \
  --num-samples 20 \
  --output-prefix backward_false_explicit_optimized

python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors steering_vectors/tom_forward_action.gguf steering_vectors/tom_belief_type.gguf steering_vectors/tom_order_init.gguf \
  --steering-coeffs 800 600 500 \
  --condition 1_forward_action_false_belief \
  --num-samples 20 \
  --output-prefix forward_action_false_explicit_optimized

python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors steering_vectors/tom_forward_belief.gguf steering_vectors/tom_belief_type.gguf steering_vectors/tom_order_init.gguf \
  --steering-coeffs 700 600 500 \
  --condition 1_forward_belief_false_belief \
  --num-samples 20 \
  --output-prefix forward_belief_false_explicit_optimized

# True beliefs with explicit statements
python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors steering_vectors/tom_backward_belief.gguf steering_vectors/tom_order_init.gguf steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 700 500 400 \
  --condition 1_backward_belief_true_belief \
  --num-samples 20 \
  --output-prefix backward_true_explicit_optimized

python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors steering_vectors/tom_forward_action.gguf steering_vectors/tom_order_init.gguf steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 700 500 400 \
  --condition 1_forward_action_true_belief \
  --num-samples 20 \
  --output-prefix forward_action_true_explicit_optimized

python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors steering_vectors/tom_forward_belief.gguf steering_vectors/tom_order_init.gguf steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 700 500 400 \
  --condition 1_forward_belief_true_belief \
  --num-samples 20 \
  --output-prefix forward_belief_true_explicit_optimized
```

---

## Experimental: Alternative Vector Combinations

### High Intensity False Belief Focus
```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_belief_type.gguf \
    steering_vectors/tom_core_capabilities.gguf \
  --steering-coeffs 1200 800 \
  --condition 0_forward_belief_false_belief \
  --num-samples 20 \
  --output-prefix high_intensity_false_belief
```

### Minimal Vector Approach (Core Only)
```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vector steering_vectors/tom_core_capabilities.gguf \
  --steering-coeff 1000 \
  --condition 0_backward_belief_false_belief \
  --num-samples 20 \
  --output-prefix core_only_test
```

### Kitchen Sink (All Relevant Vectors)
```bash
cd ToM && python evaluate_cognitive_actions_with_steering.py \
  --steering-vectors \
    steering_vectors/tom_backward_belief.gguf \
    steering_vectors/tom_belief_type.gguf \
    steering_vectors/tom_core_capabilities.gguf \
    steering_vectors/tom_order_init.gguf \
    steering_vectors/tom_direction.gguf \
  --steering-coeffs 500 500 400 300 200 \
  --condition 0_backward_belief_false_belief \
  --num-samples 20 \
  --output-prefix kitchen_sink_backward_false
```

---

## Notes

**Vector Selection Logic:**
- **0_ prefix** = Implicit (no explicit belief statement) → less order_init weight needed
- **1_ prefix** = Explicit (includes belief statement) → order_init is crucial
- **false_belief** = Always include tom_belief_type.gguf at high weight
- **true_belief** = Focus on core_capabilities, less need for belief_type
- **backward_** = tom_backward_belief.gguf is primary
- **forward_action** = tom_forward_action.gguf is primary
- **forward_belief** = tom_forward_belief.gguf is primary

**Coefficient Ranges:**
- Primary vector: 700-900
- Secondary vector: 500-600
- Tertiary vector: 300-500

**Output Location:**
All results saved to `ToM/results/` with files:
- `{prefix}_raw.csv`
- `{prefix}_summary.json`
- `{prefix}_differences.png`
- `{prefix}_accuracy.png`
- `{prefix}_heatmap.png`