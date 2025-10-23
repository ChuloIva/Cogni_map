# Probe Training Methodology: Complete Technical Guide

Based on the paper "Brije: A General Framework for Real-Time Detection of Cognitive Constructs in Language Models" and the implementation codebase.

## Table of Contents

1. [Overview](#overview)
2. [Data Generation Pipeline](#data-generation-pipeline)
3. [Activation Capture Methodology](#activation-capture-methodology)
4. [Probe Architecture](#probe-architecture)
5. [Training Procedure](#training-procedure)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Performance Results](#performance-results)
8. [Implementation Details](#implementation-details)

---

## Overview

The Brije framework implements a complete pipeline for detecting cognitive constructs in language models through three integrated components:

1. **Synthetic Data Generation** - LLM-based generation of diverse, labeled training examples
2. **Activation Capture** - augmented prompting technique for extracting discriminative representations
3. **Binary Probe Training** - One-vs-rest classification using linear/multi-head probes

### Key Innovation: Generalizability

Unlike traditional mechanistic interpretability work focused on specific phenomena, Brije provides a **general methodology** for detecting any cognitive construct for which sufficient quality training data can be generated. The framework is demonstrated with 45 cognitive actions but can be adapted to detect any construct of interest (e.g., deceptive reasoning, theory-of-mind, domain expertise).

---

## Data Generation Pipeline

### Scientific Foundation

The 45 cognitive actions are organized into five scientifically-grounded categories:

1. **Core Reasoning** (19 actions): reconsidering, reframing, noticing, perspective-taking, questioning, abstracting, connecting, distinguishing, updating beliefs, pattern recognition, analogical thinking, counterfactual reasoning, hypothesis generation, meta-awareness, etc.

2. **Bloom's Taxonomy** (6 actions): remembering, understanding, applying, analyzing, evaluating, creating

3. **Guilford's Operations** (3 actions): divergent thinking, convergent thinking, cognition awareness

4. **Metacognitive** (3 actions): metacognitive monitoring, metacognitive regulation, self-questioning

5. **Affective Regulation** (14 actions): emotional reappraisal, emotion receiving, responding, valuing, organizing, characterizing, situation selection/modification, attentional deployment, response modulation, emotion perception, facilitation, understanding, management

### Architecture

The data generation system consists of three modular components:

#### 1. Variable Pools (`variable_pools.py`)

Defines the taxonomy and contextual variables:

- **45 Cognitive Actions** with descriptions and examples
- **36 Domains** for contextual diversity (personal life, education, science, healthcare, business, etc.)
- **50+ Subjects** per domain for specific scenarios
- **Emotional states** (calm, anxious, excited, conflicted, etc.)
- **Triggers** (events that initiate cognitive processes)
- **Perspectives** (1st, 2nd, 3rd person)
- **Complexity levels** (simple, moderate, complex)
- **Language styles** (conversational, formal, reflective, academic)

#### 2. Prompt Templates (`prompt_templates.py`)

Four template types ensure format diversity:

- **Single Action** (70%): One clear cognitive action demonstration
  - Example: *"After reviewing the data, she began reconsidering her initial hypothesis."*

- **Action Chains** (20%): Multiple sequential actions
  - Example: *"He started by recalling past events, then analyzing patterns, before finally inferring the cause."*

- **Dialogue Format** (5%): Conversational examples
  - Example: *"I'm reconsidering my approach." "What made you question it?"*

- **Thought Stream** (5%): Internal monologue
  - Example: *"Analyzing the situation... comparing options... deciding on the best path..."*

Each template includes variable substitution for:
- Domain and subject context
- Complexity control
- Perspective variation
- Style flexibility

#### 3. Data Generator (`data_generator.py`)

Core engine with production-grade features:

**Async Parallel Processing:**
```python
class CognitiveDataGenerator:
    def __init__(self, ollama_client, max_parallel=16):
        # Supports 16 concurrent requests via asyncio
        # Uses semaphores for controlled parallelism
        # Achieves 16x speedup over sequential generation
```

**Key Features:**
- **Async/await** with `aiohttp` for concurrent API calls
- **Stratified sampling** ensures even distribution across actions (~155 examples per action)
- **Automatic checkpointing** every 100 examples to prevent data loss
- **Error handling** with fallback to sequential generation
- **Real-time statistics** tracking (actions, domains, complexity)
- **Multiple export formats** (JSONL, JSON, CSV)

### Generation Process

**Step-by-step workflow:**

1. **Stratified Sampling**: Divide target examples evenly across 45 cognitive actions

2. **Template Mixing**: For each action, mix formats:
   - 70% single action templates
   - 20% action chain templates
   - 5% dialogue templates
   - 5% thought-stream templates

3. **Variable Substitution**: For each template, randomly sample:
   - Domain and specific subject
   - Complexity level
   - Emotional state
   - Perspective
   - Language style
   - Unique angle/trigger

4. **Parallel Generation**:
   - Create async tasks for each example
   - Semaphore limits concurrency to `max_parallel`
   - Ollama processes requests in parallel via context buffers

5. **Quality Validation**: 
   - Length filtering
   - Uniqueness checking
   - Format verification

6. **Checkpointing**: Save progress every 100 examples

7. **Export**: Final dataset in JSONL format with full metadata

### Performance Metrics

Using **Ollama with gemma3:27b** on A100 40GB GPU:

- **Parallel requests**: 16 concurrent
- **Generation time**: ~3.7 hours for 7,000 examples
- **Throughput**: ~1,890 examples/hour
- **Configuration**:
  ```bash
  export OLLAMA_NUM_PARALLEL=16      # Max parallel context buffers
  export OLLAMA_MAX_QUEUE=512        # Request queue size
  export OLLAMA_MAX_LOADED_MODELS=1  # Keep 1 model in memory
  ```

### Dataset Quality Metrics

The generated dataset achieves:

- **Stratification**: 155 examples per action (±5)
- **Diversity**: 36 domains, 50+ subjects each
- **Complexity**: Balanced across simple/moderate/complex
- **Average length**: ~709 characters (108 words)
- **Uniqueness**: >99% unique texts
- **Format variety**: 70/20/5/5 distribution maintained

---

## Activation Capture Methodology

### Model Configuration

- **Base Model**: Gemma 3 4B (instruction-tuned) - `google/gemma-3-4b-it`
- **Target Layers**: Layers 21-30 (75-95% depth) where high-level reasoning emerges
- **Framework**: Uses `nnsight` for activation extraction
- **Precision**: bfloat16 for efficiency (converted to float32 for HDF5 storage)

### Novel Augmented Prompting Technique

**The key innovation**: Instead of extracting activations from raw text, Brije appends a specific prompt to prime the model to "think about" cognitive actions, producing more discriminative representations.

**Augmentation Process:**

Original text:
```
"After reviewing the experimental results, I began questioning my initial assumptions about the correlation."
```

Augmented text:
```
"After reviewing the experimental results, I began questioning my initial assumptions about the correlation.

The cognitive action being demonstrated here is"
```

**Why this works:**

1. **Priming Effect**: The appended prompt causes the model to activate neurons related to cognitive action reasoning
2. **Consistent Extraction Point**: The last token position provides a standardized representation
3. **Discriminative Features**: Model representations become more separable in activation space
4. **Task Alignment**: The probe's task (detecting cognitive actions) aligns with what the model is "thinking about"

### Activation Extraction Implementation

```python
def capture_single_example(self, text: str, layer_idx: int) -> torch.Tensor:
    """
    Capture activations for a single text at a specific layer
    """
    # Append special message to create consistent extraction point
    augmented_text = f"{text}\n\nThe cognitive action being demonstrated here is"
    
    with self.model.trace(augmented_text) as tracer:
        # Access the layer's output
        hidden_states = self.layers[layer_idx].output[0].save()
    
    # hidden_states shape: (batch_size, seq_len, hidden_size)
    # Use last token representation (after the augmented prompt)
    activations = hidden_states[:, -1, :].squeeze(0)
    
    return activations
```

### Multi-Layer Capture Optimization

**🚀 Optimized approach** (25x faster): Capture ALL layers in a single forward pass

```python
def capture_single_example_all_layers(self, text: str) -> Dict[int, torch.Tensor]:
    """
    Capture activations from ALL layers in a single forward pass
    """
    augmented_text = f"{text}\n\nThe cognitive action being demonstrated here is"
    
    saved_states = {}
    with self.model.trace(augmented_text) as tracer:
        # Capture ALL layers simultaneously in one forward pass
        for layer_idx in self.layers_to_capture:
            saved_states[layer_idx] = self.layers[layer_idx].output[0].save()
    
    # Extract last token for each layer
    activations = {}
    for layer_idx, hidden_states in saved_states.items():
        activations[layer_idx] = hidden_states[:, -1, :].squeeze(0)
    
    return activations
```

**Performance comparison:**
- **Sequential**: 10 layers × 7,000 examples = 70,000 forward passes
- **Optimized**: 7,000 forward passes capturing 10 layers each
- **Speedup**: ~10x faster (linear with number of layers)

### Data Splits

Standard stratified splits ensuring class balance:

- **Training**: 70% of data (~4,900 examples)
- **Validation**: 15% of data (~1,050 examples)
- **Test**: 15% of data (~1,050 examples)

Stratification ensures each cognitive action appears proportionally in all splits.

### Storage Format

Activations saved in HDF5 format for efficiency:

```python
with h5py.File(output_path, 'w') as f:
    # Metadata
    f.attrs['model_name'] = 'google/gemma-3-4b-it'
    f.attrs['layers'] = [21, 22, ..., 30]
    
    # Each split
    for split_name in ['train', 'val', 'test']:
        grp = f.create_group(split_name)
        # Convert bfloat16 to float32 for HDF5 compatibility
        grp.create_dataset('activations', data=acts.float().numpy())
        grp.create_dataset('labels', data=labels.numpy())
```

**File structure:**
```
data/activations/
├── layer_21_activations.h5  # Shape: (N, 4096) for Gemma 3 4B
├── layer_22_activations.h5
├── ...
└── layer_30_activations.h5
```

---

## Probe Architecture

### Design Philosophy

Brije uses **binary classification** with a **one-vs-rest strategy** rather than multi-class classification:

**Advantages:**
1. **Independent Predictions**: Each action can be detected independently
2. **Multi-Action Detection**: Multiple actions can activate simultaneously
3. **Threshold Control**: Per-action confidence thresholds
4. **Easier Training**: Binary classification is simpler than 45-way classification
5. **Extensibility**: Add new actions without retraining existing probes

**Trade-off**: Requires training 45 probes per layer instead of 1 multi-class probe (450 total for 10 layers vs 10 multi-class probes).

### Model Types

#### 1. Binary Linear Probe (Default)

Simple linear classifier with dropout:

```python
class BinaryLinearProbe(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, 1)  # Single output for binary
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        logits = self.linear(x)
        return logits  # Raw logits for BCEWithLogitsLoss
```

**Parameters:**
- Input dim: 4096 (Gemma 3 4B hidden size)
- Output dim: 1 (binary classification)
- Dropout: 0.1
- Total parameters: ~4,097 per probe

**Advantages:**
- Fast training (<1 minute per probe)
- No overfitting on small datasets
- High interpretability
- Low memory footprint

#### 2. Binary Multi-Head Probe (Optional)

More powerful model with attention:

```python
class BinaryMultiHeadProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Shared encoder (2-layer MLP)
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
        # Encode
        h = self.encoder(x)  # (batch, hidden_dim)
        
        # Self-attention
        h = h.unsqueeze(1)  # (batch, 1, hidden_dim)
        h_attn, _ = self.attention(h, h, h)
        h_attn = h_attn.squeeze(1)
        
        # Classify
        logits = self.classifier(h_attn)
        return logits
```

**Parameters:**
- Input dim: 4096
- Hidden dim: 512
- Attention heads: 8
- Total parameters: ~2.7M per probe

**When to use:**
- Large datasets (>10k examples per action)
- Complex cognitive constructs requiring non-linear separation
- When computational resources allow

**Trade-off**: Higher capacity but risk of overfitting on small datasets (~150 examples per action).

---

## Training Procedure

### Training Configuration

**Hyperparameters** (optimized for small datasets):

```python
# Optimizer
optimizer = AdamW(
    probe.parameters(),
    lr=5e-4,              # Learning rate
    weight_decay=1e-3     # L2 regularization (strong for small data)
)

# Loss function
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits

# Learning rate scheduler
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,     # Cosine annealing over epochs
    eta_min=1e-5          # Minimum learning rate
)

# Training parameters
batch_size = 16           # Small batch for small dataset
num_epochs = 50           # Max epochs
early_stopping_patience = 10  # Stop if no improvement for 10 epochs
```

**Design choices:**

1. **Learning Rate (5e-4)**: Higher than typical fine-tuning (1e-5) because:
   - Small model (linear probe)
   - Training from scratch (not fine-tuning)
   - Small dataset requires faster convergence

2. **Weight Decay (1e-3)**: Strong regularization because:
   - Small dataset (~150 positive examples per action)
   - High risk of overfitting
   - Linear model benefits from L2 penalty

3. **Batch Size (16)**: Small because:
   - Small dataset (~5k training examples)
   - More gradient updates per epoch
   - Better generalization on small data

4. **Early Stopping (10 epochs)**: Prevents overfitting by:
   - Monitoring validation AUC-ROC
   - Saving best checkpoint
   - Stopping when no improvement

5. **Cosine Annealing**: Smooth learning rate decay:
   - Starts at 5e-4
   - Decays to 1e-5
   - Prevents getting stuck in local minima

### Binary Label Creation

One-vs-rest strategy:

```python
def create_binary_labels(labels: torch.Tensor, target_action_idx: int) -> torch.Tensor:
    """
    Create binary labels for one-vs-rest classification
    
    Args:
        labels: Multi-class labels (N,) with values [0, 44]
        target_action_idx: Index of the action to classify (e.g., 15 for "reconsidering")
    
    Returns:
        Binary labels (N,) with values 0.0 or 1.0
    """
    binary_labels = (labels == target_action_idx).float()
    return binary_labels
```

**Class imbalance:**
- Positive examples: ~155 (1/45 = 2.2%)
- Negative examples: ~6,845 (44/45 = 97.8%)

**Handling imbalance:**
- Validation metric: AUC-ROC (robust to imbalance)
- No class weights (would harm calibration)
- Early stopping on AUC prevents overfitting to majority class

### Training Loop

```python
class BinaryProbeTrainer:
    def train(self, train_loader, val_loader, num_epochs, save_dir, action_name):
        best_val_auc = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.probe.train()
            for batch_acts, batch_labels in train_loader:
                batch_acts = batch_acts.to(self.device)
                batch_labels = batch_labels.to(self.device).unsqueeze(1)
                
                # Forward pass
                logits = self.probe(batch_acts)
                loss = self.criterion(logits, batch_labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Validation phase
            self.probe.eval()
            val_loss, val_auc, val_acc = self.evaluate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save best model based on validation AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_without_improvement = 0
                save_probe(self.probe, save_dir / f"probe_{action_name}.pth")
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return best_val_auc
```

### Training All Probes

For 45 cognitive actions × 10 layers = **450 total probes**:

```python
def train_all_binary_probes(train_acts, train_labels, val_acts, val_labels, 
                            test_acts, test_labels, output_dir, args):
    """
    Train 45 separate binary probes (one per cognitive action)
    """
    idx_to_action = get_idx_to_action_mapping()
    num_actions = 45
    
    all_metrics = []
    
    for action_idx in range(num_actions):
        action_name = idx_to_action[action_idx]
        print(f"\n[{action_idx+1}/{num_actions}] Training probe: {action_name}")
        
        # Create binary labels for this action
        train_binary = create_binary_labels(train_labels, action_idx)
        val_binary = create_binary_labels(val_labels, action_idx)
        test_binary = create_binary_labels(test_labels, action_idx)
        
        # Create probe
        probe = BinaryLinearProbe(input_dim=4096)
        trainer = BinaryProbeTrainer(probe, device='cuda', lr=5e-4)
        
        # Train
        history = trainer.train(train_loader, val_loader, num_epochs=50)
        
        # Evaluate on test set
        test_loss, test_auc, test_acc, test_probs, test_labels = trainer.evaluate(test_loader)
        
        metrics = compute_binary_metrics(test_labels, test_probs, action_name)
        all_metrics.append(metrics)
        
        # Save probe and metrics
        save_probe(probe, output_dir / f"probe_{action_name}.pth")
        save_json(metrics, output_dir / f"metrics_{action_name}.json")
    
    # Compute aggregate metrics
    avg_auc = np.mean([m['auc_roc'] for m in all_metrics])
    avg_f1 = np.mean([m['f1'] for m in all_metrics])
    
    print(f"\nAverage AUC-ROC across 45 probes: {avg_auc:.4f}")
    print(f"Average F1 across 45 probes: {avg_f1:.4f}")
```

### Training Time

**Per probe** (BinaryLinearProbe on GPU):
- Training: ~30-60 seconds
- 45 probes: ~45 minutes per layer
- 10 layers: ~7.5 hours total

**Optimization**: Train layers in parallel on multi-GPU systems or sequentially on single GPU.

---

## Evaluation Metrics

### Binary Classification Metrics

For each probe, compute:

```python
def compute_binary_metrics(true_labels, predictions, probabilities, action_name):
    """
    Compute comprehensive binary classification metrics
    """
    # Core metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    
    auc_roc = roc_auc_score(true_labels, probabilities)
    auc_pr = average_precision_score(true_labels, probabilities)
    accuracy = accuracy_score(true_labels, predictions)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    
    return {
        'action': action_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,      # Primary metric
        'auc_pr': auc_pr,
        'confusion_matrix': {
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp
        }
    }
```

### Primary Metric: AUC-ROC

**Why AUC-ROC is used as the primary metric:**

1. **Threshold-independent**: Measures discrimination ability across all thresholds
2. **Robust to class imbalance**: Works well with 2.2% positive class
3. **Interpretability**: 
   - 0.5 = random guessing
   - 0.7 = acceptable discrimination
   - 0.8 = good discrimination
   - 0.9 = excellent discrimination

4. **Comparable across actions**: Normalized metric for comparing probe performance

### Secondary Metrics

- **F1 Score**: Harmonic mean of precision/recall (threshold=0.5)
- **AUC-PR**: Area under precision-recall curve (better for extreme imbalance)
- **Accuracy**: Overall correctness (less meaningful due to imbalance)

### Aggregate Metrics

Across all 45 cognitive actions:

```python
aggregate_metrics = {
    'average_auc_roc': np.mean([m['auc_roc'] for m in all_metrics]),
    'average_f1': np.mean([m['f1'] for m in all_metrics]),
    'average_accuracy': np.mean([m['accuracy'] for m in all_metrics]),
    'std_auc_roc': np.std([m['auc_roc'] for m in all_metrics]),
    'min_auc_roc': np.min([m['auc_roc'] for m in all_metrics]),
    'max_auc_roc': np.max([m['auc_roc'] for m in all_metrics]),
    'per_action_metrics': all_metrics
}
```

---

## Performance Results

### Overall Performance (45 Cognitive Actions)

**Average across all probes (10 layers, 450 total probes):**

- **Average AUC-ROC**: 0.78
- **Average F1 Score**: 0.68
- **Average Accuracy**: Not reported (misleading due to class imbalance)

### Performance by Action Category

**Best-performing actions** (AUC-ROC > 0.85):
- **Noticing**: 0.87 - Detecting attention to patterns/details
- **Analyzing**: 0.86 - Breaking down information systematically
- **Hypothesis generation**: 0.85 - Proposing explanatory theories

**Good performance** (AUC-ROC 0.75-0.85):
- Most core reasoning actions (reconsidering, reframing, perspective-taking)
- Bloom's taxonomy actions (understanding, applying, evaluating)
- Metacognitive actions (monitoring, regulation)

**Challenging actions** (AUC-ROC 0.65-0.75):
- Fine-grained emotional distinctions (emotion perception vs. emotion understanding)
- Subtle cognitive differences (recognizing vs. recalling)
- Actions with overlapping semantics

### Layer Specialization

**Key finding**: Different layers specialize in different types of cognitive actions.

**Early layers (21-23)** - ~75% depth:
- Basic pattern recognition
- Noticing and attention
- Simple cognitive operations

**Middle layers (24-27)** - ~80-90% depth:
- Perspective-taking
- Distinguishing and comparing
- Intermediate reasoning

**Late layers (28-30)** - ~90-100% depth:
- Metacognitive monitoring
- Hypothesis generation
- Abstract reasoning
- Complex emotional regulation

**Practical implication**: Use layer-specific probes or ensemble across layers for best detection.

### Validation Study: Therapy Transcripts

Applied to **AnnoMI corpus** (133 therapy sessions):
- **1,935 utterances** (860 therapist, 1,075 client)
- **Carl Rogers' therapy sessions** analyzed

**Therapist cognitive patterns:**
1. Response modulation (managing emotional expression)
2. Noticing (detecting client states)
3. Hypothesis generation (inferring issues)
4. Emotion perception (recognizing emotions)

**Client cognitive patterns:**
1. Self-questioning (examining beliefs)
2. Reconsidering (updating perspectives)
3. Emotional reappraisal (reinterpreting emotions)
4. Perspective-taking (seeing alternative views)

**Temporal dynamics:**
- Cognitive actions often activate in sequences
- Common pattern: noticing → hypothesis generation → perspective-taking
- Mirrors therapeutic reasoning progressions

### Performance Factors

**What makes a cognitive action easier to detect:**
1. **Distinctiveness**: Unique linguistic markers (e.g., "I didn't realize..." for noticing)
2. **Coarse-grained**: Broad conceptual categories (analyzing vs. evaluating)
3. **Frequent in training data**: Well-represented in synthetic examples
4. **Clear manifestation**: Explicit expression in text

**What makes detection harder:**
1. **Semantic overlap**: Similar meanings (emotion perception vs. emotion understanding)
2. **Fine-grained distinctions**: Subtle differences within a category
3. **Implicit actions**: Cognitive processes not explicitly stated
4. **Data quality**: Noisy or ambiguous training examples

---

## Implementation Details

### Complete Training Pipeline

**Step 1: Generate Training Data**

```bash
# Run the data generation pipeline (see data/datagen/)
# Generates 7,000 examples stratified across 45 actions
# Time: ~3.7 hours on A100 40GB with 16 parallel requests
cd data/datagen
jupyter notebook Cognitive_Action_Data_Generator_Colab.ipynb
```

**Step 2: Capture Activations**

```bash
# Extract activations from Gemma 3 4B for layers 21-30
# Use optimized single-pass mode for 10x speedup
python src/probes/capture_activations.py \
  --dataset data/datagen/generated_data/cognitive_actions_7k_final.jsonl \
  --output-dir data/activations \
  --model google/gemma-3-4b-it \
  --layers 21 22 23 24 25 26 27 28 29 30 \
  --device auto \
  --format hdf5 \
  --single-pass  # 🚀 10x faster!

# Output: data/activations/layer_21_activations.h5, ..., layer_30_activations.h5
```

**Step 3: Train Binary Probes (Per Layer)**

```bash
# Train 45 binary probes for each layer
for layer in {21..30}; do
  echo "Training layer ${layer}..."
  python src/probes/train_binary_probes.py \
    --activations data/activations/layer_${layer}_activations.h5 \
    --output-dir data/probes_binary/layer_${layer} \
    --model-type linear \
    --batch-size 16 \
    --epochs 50 \
    --lr 5e-4 \
    --weight-decay 1e-3 \
    --early-stopping-patience 10 \
    --device auto
done

# Output: data/probes_binary/layer_21/probe_reconsidering.pth, ...
```

**Step 4: Aggregate Results**

```bash
# Collect metrics from all layers and actions
python scripts/aggregate_probe_metrics.py \
  --probe-dir data/probes_binary \
  --output data/probes_binary/training_summary.json
```

### Parallel Training (Multi-GPU)

For faster training on multi-GPU systems:

```bash
# Use the parallel training script
./train_probe_pipeline_parallel.sh
```

This script:
1. Distributes layer training across GPUs
2. Trains 45 probes per layer in sequence
3. Aggregates results after completion

### Training Script (`train_probe_pipeline.sh`)

Example complete pipeline script:

```bash
#!/bin/bash
# Complete training pipeline for all layers

# Configuration
MODEL_NAME="google/gemma-3-4b-it"
DATASET="data/datagen/generated_data/cognitive_actions_7k_final.jsonl"
LAYERS=(21 22 23 24 25 26 27 28 29 30)
OUTPUT_DIR="data/probes_binary"

# Step 1: Capture activations (if not already done)
if [ ! -d "data/activations" ]; then
    echo "Capturing activations..."
    python src/probes/capture_activations.py \
        --dataset $DATASET \
        --output-dir data/activations \
        --model $MODEL_NAME \
        --layers ${LAYERS[@]} \
        --device auto \
        --format hdf5 \
        --single-pass
fi

# Step 2: Train probes for each layer
for layer in ${LAYERS[@]}; do
    echo "========================================="
    echo "Training layer ${layer}..."
    echo "========================================="
    
    python src/probes/train_binary_probes.py \
        --activations data/activations/layer_${layer}_activations.h5 \
        --output-dir $OUTPUT_DIR/layer_${layer} \
        --model-type linear \
        --batch-size 16 \
        --epochs 50 \
        --lr 5e-4 \
        --weight-decay 1e-3 \
        --early-stopping-patience 10 \
        --device auto
    
    echo "✓ Completed layer ${layer}"
done

# Step 3: Generate summary
echo "Generating training summary..."
python scripts/aggregate_metrics.py \
    --probe-dir $OUTPUT_DIR \
    --output $OUTPUT_DIR/training_summary.json

echo "✓ Pipeline complete!"
```

### File Structure After Training

```
brije/
├── data/
│   ├── datagen/
│   │   └── generated_data/
│   │       └── cognitive_actions_7k_final.jsonl  # Training data
│   ├── activations/
│   │   ├── layer_21_activations.h5  # (7000, 4096)
│   │   ├── layer_22_activations.h5
│   │   ├── ...
│   │   └── layer_30_activations.h5
│   └── probes_binary/
│       ├── layer_21/
│       │   ├── probe_reconsidering.pth
│       │   ├── probe_noticing.pth
│       │   ├── ...  # 45 probes
│       │   ├── metrics_reconsidering.json
│       │   └── aggregate_metrics.json
│       ├── layer_22/
│       │   └── ...  # 45 probes
│       ├── ...
│       ├── layer_30/
│       │   └── ...  # 45 probes
│       ├── training_summary.json  # Overall results
│       └── per_action_layer_analysis.json  # Best layer per action
```

### Loading and Using Trained Probes

```python
from src.probes.probe_models import load_probe
from src.probes.capture_activations import ActivationCapture

# Load a trained probe
probe, metadata = load_probe(
    'data/probes_binary/layer_28/probe_reconsidering.pth',
    device='cuda'
)

print(f"Probe for: {metadata['action']}")
print(f"Best validation AUC: {metadata['val_auc']:.4f}")

# Capture activations for new text
capture = ActivationCapture(model_name="google/gemma-3-4b-it")
text = "After reviewing the evidence, I began reconsidering my stance."
activation = capture.capture_single_example(text, layer_idx=28)

# Get prediction
probe.eval()
with torch.no_grad():
    logit = probe(activation.unsqueeze(0).to('cuda'))
    prob = torch.sigmoid(logit).item()

print(f"Probability of 'reconsidering': {prob:.3f}")
```

### Multi-Layer Inference (Best Practice)

Instead of using a single layer, aggregate predictions across multiple layers:

```python
from src.probes.universal_multi_layer_inference import UniversalMultiLayerInference

# Initialize multi-layer inference engine
inference = UniversalMultiLayerInference(
    probe_dir='data/probes_binary',
    layers=[28, 29, 30],  # Use top 3 layers
    model_name='google/gemma-3-4b-it',
    device='cuda'
)

# Analyze text
text = "I'm starting to question my assumptions about this problem."
predictions = inference.predict_text(text, threshold=0.5)

for action, confidence, layer in predictions:
    print(f"{action}: {confidence:.3f} (layer {layer})")
```

### Real-Time Streaming Inference

For token-by-token analysis during generation:

```python
from src.probes.streaming_probe_inference import StreamingProbeInference

# Initialize streaming inference
streaming = StreamingProbeInference(
    probe_dir='data/probes_binary',
    layers=[28, 29, 30],
    model_name='google/gemma-3-4b-it',
    threshold=0.5
)

# Stream analysis
prompt = "Analyze the effectiveness of cognitive behavioral therapy."
for token_idx, token, active_actions in streaming.stream_generation(prompt):
    print(f"Token {token_idx}: {token}")
    if active_actions:
        print(f"  Active: {', '.join([f'{a}({c:.2f})' for a, c in active_actions])}")
```

---

## Additional Resources

### Code Files

- **Data Generation**: `data/datagen/`
  - `data_generator.py` - Core generation engine
  - `variable_pools.py` - Taxonomies and variables
  - `prompt_templates.py` - Template system
  - `Cognitive_Action_Data_Generator_Colab.ipynb` - Production notebook

- **Activation Capture**: `src/probes/`
  - `capture_activations.py` - Activation extraction
  - `dataset_utils.py` - Data handling utilities

- **Probe Training**: `src/probes/`
  - `train_binary_probes.py` - Training pipeline
  - `probe_models.py` - Model architectures
  - `best_probe_loader.py` - Loading best probes per layer

- **Inference**: `src/probes/`
  - `universal_multi_layer_inference.py` - Multi-layer inference
  - `streaming_probe_inference.py` - Real-time streaming
  - `Interactive_TUI.py` - Terminal UI for visualization

- **Experiments**: `src/experiments/`
  - `visualization_suite.py` - Plotting and analysis
  - `tom_analysis.py` - Theory-of-mind case study

### Shell Scripts

- `train_probe_pipeline.sh` - Sequential training pipeline
- `train_probe_pipeline_parallel.sh` - Multi-GPU parallel training
- `train_all_layers.sh` - Complete automation script

### Notebooks

- `notebooks/Streaming_Token_Probe_Demo.ipynb` - Interactive demo
- `notebooks/showcase_all_features.ipynb` - Feature showcase

### Related Work

**Papers cited in methodology:**

1. **Mechanistic Interpretability**: Rai et al. (2025) - Survey of transformer interpretability
2. **Probing Classifiers**: Belinkov & Glass (2021) - Probing as diagnostic technique
3. **Edge Probing**: Tenney et al. (2019) - Comparing contextualized representations
4. **Metacognitive Space**: Recent work on internal state monitoring in LLMs
5. **INSIDE**: Exploiting internal states for hallucination detection

### Citation

If you use this methodology, please cite:

```bibtex
@article{brije2025,
  title={Brije: A General Framework for Real-Time Detection of Cognitive Constructs in Language Models},
  author={Anonymous},
  journal={AAAI 2026 (under review)},
  year={2025}
}
```

---

## Summary: Key Takeaways

### Innovation Points

1. **Augmented Prompting**: Novel technique for extracting discriminative activations by priming the model to "think about" cognitive actions

2. **Scalable Data Generation**: LLM-based synthetic generation eliminates manual labeling bottleneck (7,000 examples in ~3.7 hours)

3. **One-vs-Rest Binary Classification**: Enables independent multi-action detection and extensibility

4. **Layer Specialization**: Different layers detect different types of cognitive constructs

5. **General Framework**: Not limited to 45 actions—can detect any construct with sufficient training data

### Best Practices

1. **Use augmented prompting** when extracting activations
2. **Stratify data** to ensure balanced training
3. **Train on multiple layers** and aggregate predictions
4. **Use AUC-ROC** as primary metric for imbalanced classes
5. **Apply early stopping** to prevent overfitting on small datasets
6. **Start with linear probes** before trying more complex models
7. **Validate on real-world data** (e.g., therapy transcripts) to ensure practical utility

### Limitations and Future Work

**Current limitations:**
- Detected "constructs" are statistical patterns, not necessarily ground truth about model internals
- Probe accuracy varies by construct complexity (0.65-0.87 AUC)
- Quality depends critically on synthetic training data quality
- Demonstrated primarily on one model family (Gemma)

**Future directions:**
1. Causal validation through activation steering
2. Expansion to other models and modalities
3. Automated data quality assessment
4. Detection of cognitive construct transitions/sequences
5. Applications to AI safety (deception detection, alignment verification)
6. Integration with human cognitive modeling research

---

**Document Version**: 1.0  
**Last Updated**: October 23, 2025  
**Based on**: Paper draft + codebase implementation

