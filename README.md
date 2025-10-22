# Brije: Watching Minds Think

This repository provides a toolkit to detect 45 cognitive actions in real-time as a language model (Gemma-3-4B) generates text. Think of it as an fMRI for an AI's thought process.

## Core Features

### 1. Universal Probes (Cognitive Action Detection)
Detect 45 cognitive actions like `analyzing`, `reconsidering`, `divergent_thinking`, and `self_questioning` using trained probes on the model's internal states across layers 1-28.

### 2. Sentiment Probes (Regression-Based)
Capture continuous sentiment scores (-3 to +3) using regression probes trained on layers 1-11, providing nuanced emotional context beyond binary classification.

### 3. Interactive TUI (Token-Level Visualization)
Explore cognitive actions and sentiment at the token level with a full-screen terminal interface that shows:
- Color-coded token streams with activation highlighting
- Real-time cognitive action predictions
- Layer-by-layer activation heatmaps
- Per-action confidence scores and distributions

---

## Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/brije.git
cd brije

# Install dependencies
pip install torch transformers h5py scikit-learn tqdm textual rich matplotlib seaborn pandas nnsight
```

### Quick Start: Use Pre-Trained Probes

**All 45 cognitive action probes and sentiment probes are already trained!** You can immediately run inference without training:

#### Example 1: Universal Probes (Cognitive Actions)
```bash
python src/probes/test_universal_inference.py
```

This demonstrates three output modes:
- **Mode 1**: Top predictions across all layers (flat ranked list)
- **Mode 2**: Predictions grouped by action (shows which layers activate for each cognitive process)
- **Mode 3**: Predictions grouped by layer (shows which processes activate at each layer)

#### Example 2: Interactive TUI
```bash
python src/probes/Interactive_TUI.py
```

Navigate with arrow keys to explore:
- Token-by-token cognitive activations
- Sentiment scores per token
- Layer distribution for each detected action
- Real-time confidence metrics

#### Example 3: Custom Text Analysis
```bash
python src/probes/universal_multi_layer_inference.py \
    --text "After reconsidering my approach, I began analyzing the problem differently."
```

---

## Example Outputs

### Universal Probes: Action Detection

**Input Text:**
```
"What if we completely flipped the script? Instead of chasing the same customers
everyone else wants, what about targeting the segment nobody's paying attention to?"
```

**Output (Grouped by Action):**
```
Predictions grouped by action:
  ✓ 1. divergent_thinking             (Layers 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)  Count: 10  Max: 1.0000
  ✓ 2. questioning                    (Layers 21, 29, 30     )  Count:  3  Max: 1.0000
  ✓ 3. convergent_thinking            (Layers 24, 27, 28     )  Count:  3  Max: 1.0000
  ✓ 4. noticing                       (Layers 28, 30         )  Count:  2  Max: 1.0000
  ✓ 5. reframing                      (Layers 26             )  Count:  1  Max: 0.9922
  ✓ 6. creating                       (Layers 22             )  Count:  1  Max: 0.9492
```

### End-to-End Example: Therapy Session Analysis

The `Keep_viz/` directory contains a complete example analyzing a Carl Rogers therapy session:

**What's Included:**
- Annotated transcript with cognitive actions and sentiment
- Visualization generation script (`analyze_carl_rogers.py`)
- Publication-quality comparison visualizations

**Visualizations Generated:**
1. **Cognitive Action Comparison** - Side-by-side therapist vs client action frequencies
2. **Cognitive Action Bias** - Log2 ratio showing therapist-dominant vs client-dominant patterns
3. **Sentiment Analysis** - Mean sentiment associations with different cognitive actions

**Run the Example:**
```bash
cd Keep_viz
python analyze_carl_rogers.py
```

This demonstrates how to:
- Parse probe-annotated transcripts
- Aggregate data by speaker
- Create comparative visualizations
- Analyze cognitive patterns in conversations

---

## Training Your Own Probes (Optional)

While pre-trained probes are provided, you can train custom probes on your own data:

### Train Universal Probes
```bash
# 1. Capture model activations (2-3 hours on GPU)
python src/probes/capture_activations.py \
    --model google/gemma-3-4b-it \
    --layer 27 \
    --device auto

# 2. Train all 45 probes in parallel (1-2 hours)
python src/probes/train_all_layers.py \
    --activations data/activations/ \
    --output-dir data/probes_binary \
    --device auto
```

### Train Sentiment Probes
```bash
# 1. Capture sentiment activations
python src/probes/capture_activations_sentiment.py \
    --model google/gemma-3-4b-it \
    --device auto

# 2. Train regression probes
python src/probes/sentiment_regression_probe.py \
    --activations data/sentiment_activations/ \
    --output-dir data/sentiment \
    --device auto
```

---

## Notebooks

Interactive Jupyter notebooks are provided for hands-on exploration:

1. **`notebooks/test_universal_inference.ipynb`** - Universal probes demo with visualizations
2. **`notebooks/Sentiment_Full_Pipeline_Colab.ipynb`** - Complete sentiment probe pipeline (Google Colab compatible)
3. **`notebooks/Streaming_Token_Probe_Demo.ipynb`** - Interactive TUI demonstration

---

## Repository Structure

```
brije/
├── src/probes/              # Core probe functionality
│   ├── probe_models.py      # Probe architectures (Linear, MultiHead)
│   ├── best_probe_loader.py # Intelligent layer selection
│   ├── capture_activations.py  # Activation extraction
│   ├── train_all_layers.py  # Parallel training orchestration
│   ├── streaming_probe_inference.py  # Real-time inference engine
│   ├── interactive_probe_viewer.py   # TUI interface
│   ├── Interactive_TUI.py   # Main TUI entry point
│   └── sentiment_regression_probe.py  # Sentiment probe training
│
├── notebooks/               # Interactive examples
│   ├── test_universal_inference.ipynb
│   ├── Sentiment_Full_Pipeline_Colab.ipynb
│   └── Streaming_Token_Probe_Demo.ipynb
│
├── Keep_viz/                # End-to-end example
│   ├── analyze_carl_rogers.py  # Visualization generation script
│   ├── data/                # Example session transcript
│   └── *.png                # Generated visualizations
│
├── paper/                   # Research paper materials
├── data/                    # Trained probes & datasets
└── third_party/nnsight/     # Model intervention library
```

---

## How It Works

### Architecture Overview

```
TEXT INPUT
    │
    ├──────────────────────┬──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
Gemma-3-4B           Gemma-3-4B           Gemma-3-4B
(Cognitive)          (Sentiment)          (Base Model)
Layers 1-28          Layers 1-11
    │                      │
    ▼                      ▼
Cognitive Probes     Sentiment Probes
(45 actions × 28)    (Regression)
    │                      │
    └──────────────────────┴──────────────────────┘
                           │
                           ▼
                  Interactive TUI
            (Token visualization + metrics)
```

### Cognitive Action Categories

The 45 cognitive actions are organized into 5 categories:

- **Metacognitive** (13): Monitoring, planning, self-reflection
- **Analytical** (12): Logical reasoning, pattern recognition, evaluation
- **Creative** (8): Divergent thinking, imagination, ideation
- **Emotional** (7): Emotion perception, empathy, regulation
- **Memory** (5): Encoding, retrieval, consolidation

---

## Hardware Support

Brije supports multiple hardware backends:
- **NVIDIA GPUs** (CUDA)
- **Apple Silicon** (MPS)
- **AMD GPUs** (ROCm)
- **CPU** (fallback)

The system automatically detects available hardware via `gpu_utils.py`.

---

## Technical Details

### Universal Probes
- **Architecture**: Binary linear probes (one-vs-rest classification)
- **Training**: BCEWithLogitsLoss with early stopping
- **Inference**: Best layer per action (automatic selection based on AUC/F1)
- **Output**: Confidence scores (0-1) with threshold filtering

### Sentiment Probes
- **Architecture**: Linear regression (single output, unbounded)
- **Training**: MSE loss on normalized targets (-1 to +1)
- **Inference**: Z-score normalized sentiment scores (typically -3 to +3)
- **Output**: Continuous scores (not probabilities)

### Interactive TUI
- **Framework**: Textual (rich-based terminal UI)
- **Features**:
  - Token-by-token navigation (arrow keys)
  - Real-time activation updates
  - Layer distribution heatmaps
  - Color-coded confidence indicators
  - Multi-layer aggregation

---

## Citation

If you use Brije in your research, please cite:

```bibtex
@article{brije2025,
  title={Brije: Watching Minds Think},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## The Big Idea

By making an AI's cognitive processes visible, we can:
- Better understand how language models "think"
- Analyze reasoning patterns for safety and alignment
- Apply insights to human domains (therapy, education, collaboration)
- Build more interpretable and controllable AI systems

---

## License

MIT License - See LICENSE file for details

---

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Support

For issues, questions, or feature requests, please open an issue on GitHub.
