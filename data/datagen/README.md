# Cognitive Action Dataset Generator 🧠

A production-ready pipeline for generating high-quality training data for cognitive action recognition using LLMs.

**Generated Dataset:** [Hugging Face Hub - cognitive-actions-7k](https://huggingface.co/datasets/Koalacrown/cognitive-actions-7k)

## 🎯 What This Is

This repository contains the complete data generation pipeline used to create a stratified dataset of **6,975+ examples** across **45 cognitive actions**. The system uses scientifically-grounded taxonomies and parallel async processing to generate diverse, high-quality training examples.

### Scientific Foundation

The cognitive actions are based on established psychological frameworks:
- **Bloom's Taxonomy** - Cognitive process dimensions
- **Guilford's Structure of Intellect** - Divergent/convergent thinking
- **Krathwohl's Affective Domain** - Emotional awareness and regulation
- **Gross's Emotion Regulation Model** - Emotional processing strategies
- **Metacognitive Process Frameworks** - Self-awareness and monitoring

## 🏗️ Architecture

The system is built from modular, reusable components:

```
datagen/
├── variable_pools.py        # Taxonomies & variable definitions
├── prompt_templates.py      # Template system for diversity
├── data_generator.py        # Core generation engine with async
├── Cognitive_Action_Data_Generator_Colab.ipynb  # Production notebook
```

## 📦 Core Components

### 1. Variable Pools (`variable_pools.py`)

Defines the cognitive action taxonomy and contextual variables:

- **45 Cognitive Actions** organized by category:
  - Metacognitive (reconsidering, self-questioning, monitoring)
  - Analytical (comparing, deconstructing, inferring)
  - Synthetic (integrating, synthesizing, connecting)
  - Creative (imagining, hypothesizing, brainstorming)
  - Evaluative (critiquing, judging, assessing)
  - Memory (recalling, recognizing, memorizing)
  - Emotional (suppressing, reappraising, accepting)

- **36 Domains** for contextual diversity (personal life, education, science, etc.)
- **50+ Subjects** per domain for specific scenarios
- **Emotional states, triggers, perspectives** for rich variation

### 2. Prompt Templates (`prompt_templates.py`)

Four template types for format diversity:

- **Single Action** (70%): One clear cognitive action
  - Example: *"After reviewing the data, she began [reconsidering] her initial hypothesis."*

- **Action Chains** (20%): Multiple sequential actions
  - Example: *"He started by [recalling] past events, then [analyzing] patterns, before finally [inferring] the cause."*

- **Dialogue Format** (5%): Conversational examples
  - Example: *"I'm [reconsidering] my approach." "What made you [question] it?"*

- **Thought Stream** (5%): Internal monologue
  - Example: *"[Analyzing] the situation... [comparing] options... [deciding] on the best path..."*

Each template includes:
- Complexity control (simple, moderate, complex)
- Perspective variation (1st, 2nd, 3rd person)
- Style flexibility (conversational, formal, reflective)

### 3. Data Generator (`data_generator.py`)

The core engine with production-grade features:

#### Async Parallel Processing
```python
class CognitiveDataGenerator:
    def __init__(self, ollama_client, max_parallel=16):
        # Supports 16 concurrent requests via asyncio
        # Uses semaphores for controlled parallelism
        # Achieves 16x speedup over sequential generation
```

#### Key Features:
- **Async/await** with `aiohttp` for concurrent API calls
- **Stratified sampling** ensures even distribution across actions
- **Automatic checkpointing** every N examples to prevent data loss
- **Error handling** with fallback to sequential generation
- **Real-time statistics** tracking (actions, domains, complexity)
- **Multiple export formats** (JSONL, JSON, CSV)

#### Performance:
- **16 parallel requests** on 40GB GPU (A100)
- ~**3.7 hours** for 7,000 examples
- Ollama configuration: `OLLAMA_NUM_PARALLEL=16`

## 🚀 Usage

### Quick Start (Google Colab)

1. **Open the production notebook:**
   - `Cognitive_Action_Data_Generator_Colab.ipynb`

2. **Run cells 1-8 sequentially:**
   - Cell 1: Install dependencies
   - Cell 2: Install & configure Ollama
   - Cell 3: Pull gemma3:27b model
   - Cell 4: Load generation modules
   - Cell 5: Create Ollama client
   - Cell 6: Mount Google Drive
   - Cell 7: Review configuration
   - Cell 8: Generate 7,000 examples (~3.7 hours)

3. **Checkpoints saved automatically:**
   - Every 100 examples → Google Drive
   - Final dataset: `cognitive_actions_7k_final_[timestamp].jsonl`

### Configuration

Edit the config in Cell 7:

```python
CONFIG = {
    'total_examples': 7000,        # Target size
    'model': 'gemma3:27b',         # Ollama model
    'max_parallel': 16,            # Concurrent requests
    'checkpoint_interval': 100,    # Checkpoint frequency
    'checkpoint_dir': '/content/drive/MyDrive/...'
}
```

### Hardware Requirements

| Hardware | Recommended `max_parallel` | Time for 7K |
|----------|---------------------------|-------------|
| CPU 16GB RAM | 2-4 | ~24-48 hours |
| CPU 32GB+ RAM | 4-8 | ~12-24 hours |
| GPU 24GB VRAM | 8-12 | ~5-7 hours |
| GPU 40GB VRAM | 12-16 | ~3-5 hours |

## 📊 Generation Process

### How It Works

1. **Stratified Sampling**: Divides target evenly across 45 actions (~155 each)

2. **Template Mixing**: For each action, mixes formats:
   - 70% single action
   - 20% action chains
   - 5% dialogue
   - 5% thought-stream

3. **Batch Generation**: Groups by template type for efficient processing

4. **Parallel Execution**:
   - Creates async tasks for each example
   - Semaphore limits to `max_parallel`
   - Ollama processes in parallel via context buffers

5. **Checkpointing**: Saves progress every 100 examples

6. **Export**: Final dataset in JSONL format with metadata

### Example Generation Flow

```
Action: "reconsidering"
  ├─ Generate 108 "single" templates
  ├─ Generate 31 "chain" templates
  ├─ Generate 8 "dialogue" templates
  └─ Generate 8 "thought_stream" templates
       ↓
  [Parallel batch processing with 16 concurrent requests]
       ↓
  Save checkpoint → Continue to next action
```

## 🔧 Ollama Configuration

The system requires proper Ollama setup for maximum performance:

### Environment Variables

```bash
export OLLAMA_NUM_PARALLEL=16      # Max parallel context buffers
export OLLAMA_MAX_QUEUE=512        # Request queue size
export OLLAMA_MAX_LOADED_MODELS=1  # Keep 1 model in memory
export OLLAMA_HOST=0.0.0.0:11434   # Server address
```

### VRAM Usage

- **Idle** (model loaded): ~18GB
- **Active** (16 parallel): ~38GB
- **Per context buffer**: ~1-2GB

Model loads **once** (18GB), parallel contexts add memory dynamically.


## 📈 Quality Metrics

The generated dataset achieves:

- **Stratification**: 155 examples per action (±5)
- **Diversity**: 36 domains, 50+ subjects each
- **Complexity**: Balanced across simple/moderate/complex
- **Length**: ~709 chars avg (108 words)
- **Uniqueness**: >99% unique texts
- **Format variety**: 70/20/5/5 distribution maintained

## 🔄 Extending the System

### Adding New Cognitive Actions

Edit `variable_pools.py`:

```python
COGNITIVE_ACTIONS = {
    "your_new_action": {
        "category": "metacognitive",
        "description": "The act of...",
        "examples": ["example1", "example2"]
    }
}
```

### Adding New Templates

Edit `prompt_templates.py`:

```python
SINGLE_ACTION_TEMPLATES.append(
    "Your new template with {cognitive_action} and {domain}"
)
```

### Adjusting Generation Parameters

Modify in notebook Cell 7 or directly in code:

- `max_parallel`: Increase for faster generation (if hardware allows)
- `checkpoint_interval`: Reduce for more frequent saves
- `template_distribution`: Change format ratios
- `complexity_distribution`: Adjust difficulty levels

## 🐛 Troubleshooting

### Common Issues

**1. Ollama not utilizing GPU**
- Set env vars **before** starting Ollama
- Restart: `pkill ollama && ollama serve`
- Check with: `nvidia-smi` (should show ~38GB during generation)

**2. Generation is slow**
- Increase `max_parallel` if you have spare VRAM
- Check Ollama is configured: `curl http://localhost:11434/api/tags`

**3. Out of Memory (OOM)**
- Reduce `max_parallel`
- Use smaller model (e.g., gemma3:9b)

**4. Checkpoint recovery**
- Load checkpoint files from Drive
- Combine with: `cat checkpoint_*.jsonl > combined.jsonl`

## 📚 Citation

If you use this generation system, please cite:

```bibtex
@software{cognitive_datagen,
  title={Cognitive Action Dataset Generator},
  author={Ivan Culo},
  year={2025},
  url={https://github.com/ChuloIva/datagen/},
  note={Pipeline for generating cognitive action recognition datasets}
}
```

## 🔗 Links

- **Generated Dataset**: [Hugging Face Hub](https://huggingface.co/datasets/Koalacrown/cognitive-actions-7k)
- **Issues**: [GitHub Issues](https://github.com/ChuloIva/datagen/issues)

## 📄 License

MIT License - Free for commercial and research use.

---

**Built with:** Python, Ollama, gemma3:27b, asyncio, aiohttp, Hugging Face Datasets