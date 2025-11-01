# Technical Report: Cognitive Action Activation Patterns in Theory of Mind Steering

## 1. Model Architecture

**Base Model:** `google/gemma-3-4b-it`
- Vision-Language Model (VLM) architecture
- Text-only tower used for this research: `model.language_model.layers`
- **Total layers:** 34 (text pathway)
- Precision: bfloat16
- Parameters: ~4 billion

**Hardware Configuration:**
```python
# AMD GPU optimization
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["AMD_SERIALIZE_KERNEL"] = "3"
os.environ["TORCH_USE_HIP_DSA"] = "1"
```

---

## 2. Steering Vector Training (CAA)

### 2.1 Framework
- **Library:** `repeng` (representation engineering)
- **Method:** Contrastive Activation Addition (CAA) with PCA centering
- **Target Layers:** -4 to -20 (last 17 layers)
  - Actual indices: layers 14-30 (in 34-layer model)

### 2.2 Training Data

**Dataset:** 752 CAA triplets from BigToM benchmark
- Source condition: `0_forward_belief`
- Random seed: 42
- Generated: 1000 triplets total (248 excluded for validation/evaluation)

**Triplet Structure:**
```python
{
  "prompt": "<situation context ending at crucial event>",
  "positive_completion": "<Protagonist> sees <event/change>.",
  "negative_completion": "<Protagonist> does not see <event/change>."
}
```

**Example:**
```
Prompt: "Carlos is a farmer in a small village in Mexico, tending to
his cornfield. He wants to irrigate his crops by opening a water valve
connected to a nearby river. Carlos closed the valve the previous evening.
During the night, heavy rainfall caused the river to overflow, opening
the valve and flooding the cornfield."

Positive: "Carlos sees the flooded cornfield."
Negative: "Carlos does not see the flooded cornfield."
```

**Key Design Principles:**
- NO persona wrapping ("acting as a ToM expert")
- NO question-answer pairs in training
- ONLY perceptual contrasts (seeing vs. not seeing events/changes)
- Captures internal representation of perspective-taking

### 2.3 Training Procedure

```python
from repeng import ControlVector, ControlModel, DatasetEntry

# Prepare dataset
dataset = [
    DatasetEntry(
        positive=f"{triplet['prompt']} {triplet['positive_completion']}",
        negative=f"{triplet['prompt']} {triplet['negative_completion']}"
    )
    for triplet in training_data
]

# Train vector
model.reset()
tom_vector = ControlVector.train(
    model,
    tokenizer,
    dataset,
    method='pca_center'
)

# Export
tom_vector.export_gguf("tom_caa_forward_belief.gguf")
```

**Output:** 17-layer directional updates stored in `.gguf` format

---

## 3. Cognitive Action Probes

### 3.1 Architecture

**Binary Linear Probes:**
```python
class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=1, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, num_classes)
```

**Specifications:**
- One probe per cognitive action per layer
- Layers 10-20 (11 layers total)
- Binary classification (action present/absent)
- Sigmoid activation for confidence scores
- Threshold: 0.001

### 3.2 Cognitive Action Taxonomy

**44 cognitive actions across 5 categories:**

| Category | Tag | Count | Key Actions |
|----------|-----|-------|-------------|
| **Metacognitive** | META | 7 | reconsidering, updating_beliefs, suspending_judgment |
| **Analytical** | ANL | 16 | noticing, questioning, understanding, analyzing |
| **Creative** | CRV | 6 | hypothesis_generation, reframing, counterfactual_reasoning |
| **Emotional** | EMO | 14 | emotion_perception, emotional_reappraisal, emotion_understanding |
| **Memory** | MEM | 1 | remembering |

### 3.3 Activation Measurement

**Metric:** Layer count (number of layers where action is detected above threshold)

```python
def run_probes_on_activations(layer_activations, threshold=0.001):
    for (action_name, layer_idx), probe in probes.items():
        logits = probe(layer_activations[layer_idx])
        confidence = torch.sigmoid(logits).item()

        if confidence > threshold:
            activated_layers[action_name].append(layer_idx)

    return {
        action: {
            'layer_count': len(activated_layers[action]),
            'activated_layers': activated_layers[action]
        }
        for action in actions
    }
```

---

## 4. Evaluation Methodology

### 4.1 Benchmark Dataset

**Source:** BigToM procedural evaluation benchmark
- **Path:** `procedural-evals-tom/data/conditions/`
- **Conditions:** 12 total (forward/backward × belief/action × true/false belief)
- **Format:** CSV with semicolon delimiter

**CSV Structure:**
```
story;question;true_answer;wrong_answer
```

**Example (0_forward_belief_true_belief):**
```
Story: "Carlos is a farmer in a small village in Mexico, tending to his
cornfield. He wants to irrigate his crops by opening a water valve connected
to a nearby river. Carlos closed the valve the previous evening. During the
night, heavy rainfall caused the river to overflow, opening the valve and
flooding the cornfield. Carlos sees the flooded cornfield."

Question: "Does Carlos believe the water valve is closed or open?"
True Answer: "Carlos believes the water valve is open."
Wrong Answer: "Carlos believes the water valve is closed."
```

### 4.2 Experimental Design

**Two Conditions:**
1. **Baseline:** No steering (coefficient = 0.0)
2. **Steered:** ToM vector applied (coefficient = 1.5)

**Three Activation Capture Points:**

```python
# 1. After story + question (before answer)
prompt = f"Story: {story}\n\nQuestion: {question}\n\nAnswer:"
augmented = f"{prompt}\n\nThe cognitive action being demonstrated here is"
# Capture activations at final token position

# 2. After story + question + TRUE answer
full_text = f"{prompt} {true_answer}"
augmented = f"{full_text}\n\nThe cognitive action being demonstrated here is"
# Capture activations

# 3. After story + question + WRONG answer
full_text = f"{prompt} {wrong_answer}"
augmented = f"{full_text}\n\nThe cognitive action being demonstrated here is"
# Capture activations
```

### 4.3 Answer Selection (Probability Ranking)

**Method:** Multiple-choice probability ranking (BigToM approach)

```python
# Format as multiple choice
mcq_prompt = (
    f"Story: {story}\n\n"
    f"Question: {question}\n"
    f"Choose one of the following:\n"
    f"a) {option_a}\n"
    f"b) {option_b}\n\n"
    f"Please answer with the letter of your choice (a or b).\n"
    f"Answer:"
)

# Get logits for letter tokens
def calculate_letter_probability(prompt, letter):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = base_model(**inputs)
    logits = outputs.logits[0, -1, :]  # Last position

    # Get token IDs for letter (try variants: 'a', ' a', 'a)', ' a)')
    letter_tokens = [
        tokenizer.encode(letter, add_special_tokens=False),
        tokenizer.encode(f" {letter}", add_special_tokens=False),
        tokenizer.encode(f"{letter})", add_special_tokens=False),
        tokenizer.encode(f" {letter})", add_special_tokens=False),
    ]

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Return max probability across variants
    return max(probs[token_id[0]].item() for token_id in letter_tokens if token_id)

# Select answer
prob_a = calculate_letter_probability(mcq_prompt, 'a')
prob_b = calculate_letter_probability(mcq_prompt, 'b')
selected = 'a' if prob_a > prob_b else 'b'
```

**Grading:**
```python
is_correct = (prob_true > prob_wrong)
```

### 4.4 Activation Extraction

**Framework:** `nnsight` for interventional model analysis

```python
from nnsight import LanguageModel

model = LanguageModel(base_model, tokenizer=tokenizer)

# Extract activations
with model.trace(augmented_prompt) as tracer:
    for layer_idx in range(10, 21):  # Layers 10-20
        hidden_states = model.model.layers[layer_idx].output[0]
        saved_activations[layer_idx] = hidden_states[:, -1, :].save()

# Convert to CPU tensors for probe inference
layer_activations = {
    layer_idx: act.squeeze(0).cpu()
    for layer_idx, act in saved_activations.items()
}

# Run probes
action_predictions = run_probes_on_activations(layer_activations)
```

---

## 5. Output Data Structure

### 5.1 Per-Sample Results (CSV)

**Columns:**
```
story, question, true_answer, wrong_answer,

# Baseline condition
baseline_selected, baseline_correct, baseline_prob_true, baseline_prob_wrong,
baseline_activations_at_question,      # Dict[action_name -> layer_count]
baseline_activations_after_true,       # Dict[action_name -> layer_count]
baseline_activations_after_wrong,      # Dict[action_name -> layer_count]

# Steered condition
steered_selected, steered_correct, steered_prob_true, steered_prob_wrong,
steered_activations_at_question,       # Dict[action_name -> layer_count]
steered_activations_after_true,        # Dict[action_name -> layer_count]
steered_activations_after_wrong,       # Dict[action_name -> layer_count]

# Differences (steered - baseline)
diff_at_question,                      # Dict[action_name -> int]
diff_after_true,                       # Dict[action_name -> int]
diff_after_wrong,                      # Dict[action_name -> int]

accuracy_improvement                   # Bool
```

### 5.2 Summary Statistics (JSON)

```json
{
  "num_samples": 100,
  "baseline_accuracy": 0.45,
  "steered_accuracy": 0.56,
  "accuracy_improvement": 0.11,
  "num_improved": 15,

  "baseline_prob_true_avg": 0.446,
  "baseline_prob_wrong_avg": 0.554,
  "steered_prob_true_avg": 0.560,
  "steered_prob_wrong_avg": 0.440,

  "top_differences_at_question": [
    ["emotion_perception", 2.35],
    ["noticing", 1.59],
    ["hypothesis_generation", 1.57],
    ...
  ],

  "mean_diff_at_question": {
    "emotion_perception": 2.35,
    "noticing": 1.59,
    ...
  },

  "mean_diff_after_true": {...},
  "mean_diff_after_wrong": {...}
}
```

---

## 6. Implementation Files

### 6.1 Training
- `train_caa_tom_vector.ipynb` - CAA training pipeline
- `create_caa_training_data.py` - Generate triplets from BigToM
- `data/datagen/caa_training_data.json` - 752 triplets
- `data/datagen/caa_training_metadata.json` - Dataset metadata

### 6.2 Evaluation
- `evaluate_cognitive_actions_with_steering.py` - Baseline vs steered comparison
- `evaluate_cognitive_actions_baseline.py` - Baseline-only analysis
- `batch_evaluate_steering_intensities.py` - Coefficient sweep

### 6.3 Probes
- `src/probes/probe_models.py` - LinearProbe architecture
- `src/probes/action_categories.py` - 44-action taxonomy
- `src/probes/train_binary_probes.py` - Probe training pipeline
- `data/probes_binary/layer_{10-20}/probe_{action}.pth` - Trained probes

### 6.4 Visualization
- `visualize_cognitive_actions.py` - Generate plots from results

---

## 7. Dependencies

```
torch>=2.0.0
transformers>=4.35.0
nnsight>=0.2.0
repeng>=0.1.0
numpy>=1.26.4
scikit-learn>=1.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.66.1
```

---

## 8. Computational Requirements

**Training:**
- GPU: 24GB+ VRAM
- Time: ~30-60 minutes (752 triplets)
- Memory: ~20GB peak

**Evaluation:**
- GPU: 16GB+ VRAM
- Time: ~2-3 hours per 100 samples (2 conditions × 3 timepoints × 11 layers)
- Memory: ~18GB peak

---

## 9. Key Design Decisions

### 9.1 CAA vs Traditional Steering

**Traditional:**
```
Positive: "As a ToM expert, I would answer..."
Negative: "As someone poor at ToM, I would answer..."
```

**CAA (This Work):**
```
Positive: "[Context] Protagonist sees the event."
Negative: "[Context] Protagonist does not see the event."
```

**Advantages:**
- Purer representation (no persona contamination)
- More generalizable (no reasoning templates)
- Captures perceptual grounding of ToM

### 9.2 Probability Ranking vs Text Generation

**Why probability ranking:**
- Avoids text parsing errors
- More reliable evaluation metric
- Follows BigToM benchmark methodology
- Deterministic (no sampling variance)

### 9.3 Layer Count as Cognitive Marker

**Hypothesis:**
- More layers activated → stronger/more distributed cognitive process
- Layer count correlates with depth of engagement
- Changes in layer counts reveal mechanistic shifts under steering

---

## 10. Reproducibility

**Random seeds:** 42 (all components)

**Fixed parameters:**
- CAA method: `pca_center`
- Steering layers: -4 to -20 (layers 14-30)
- Probe layers: 10-20
- Activation threshold: 0.001
- Steering coefficient: 1.5 (default)

**Data availability:**
- Training triplets: `data/datagen/caa_training_data.json`
- Benchmark: BigToM procedural-evals
- Probe weights: `data/probes_binary/`
- Steering vector: `.gguf` format

---

## Appendix: Cognitive Action Definitions

**Metacognitive (7):** reconsidering, updating_beliefs, suspending_judgment, meta_awareness, metacognitive_monitoring, metacognitive_regulation, self_questioning

**Analytical (16):** noticing, pattern_recognition, zooming_out, zooming_in, questioning, abstracting, concretizing, connecting, distinguishing, perspective_taking, convergent_thinking, understanding, applying, analyzing, evaluating, cognition_awareness

**Creative (6):** creating, divergent_thinking, hypothesis_generation, counterfactual_reasoning, analogical_thinking, reframing

**Emotional (14):** emotional_reappraisal, emotion_receiving, emotion_responding, emotion_valuing, emotion_organizing, emotion_characterizing, situation_selection, situation_modification, attentional_deployment, response_modulation, emotion_perception, emotion_facilitation, emotion_understanding, emotion_management, accepting

**Memory (1):** remembering
