# Sentiment Probe Pipeline - Complete Guide

This guide covers the complete end-to-end pipeline for generating sentiment data, capturing activations, and training binary sentiment probes on **all 34 layers** of Gemma 3 4B.

## 📚 Overview

### What You'll Get
- **700 positive sentiment examples** (joy, gratitude, hope, excitement, etc.)
- **700 negative sentiment examples** (sadness, anger, fear, anxiety, etc.)
- **Activations from all 34 Gemma layers** (including missing layers 1-20)
- **Binary sentiment probes trained on all layers**
- **Performance visualization** showing which layers are best for sentiment detection

### Time & Resources
- **Total runtime**: ~4-6 hours on Google Colab (T4 GPU)
- **GPU requirement**: T4 or better (15GB+ VRAM)
- **Storage**: ~5-8 GB total (data + activations + probes)

---

## 🚀 Quick Start

### Option 1: Full Pipeline (Recommended)

Use the comprehensive notebook that does everything:

1. **Open in Google Colab**: Upload `Sentiment_Full_Pipeline_Colab.ipynb` to Colab
2. **Select GPU runtime**: Runtime → Change runtime type → GPU (T4)
3. **Run all cells**: Runtime → Run all
4. **Wait ~4-6 hours**: The pipeline will:
   - Generate 1,400 sentiment examples
   - Capture activations from all 34 layers (in batches of 10)
   - Train binary probes for each layer
   - Visualize performance
   - Save everything to Google Drive

### Option 2: Separate Steps

If you want more control, use the individual notebooks:

#### Step 1: Generate Sentiment Data
- **Notebook**: `third_party/datagen/Sentiment_Data_Generator_Colab.ipynb`
- **Runtime**: ~30-60 minutes
- **Requirements**: Ollama running locally OR use Gemma on Colab
- **Output**:
  - `positive_sentiment_700.jsonl`
  - `negative_sentiment_700.jsonl`
  - `sentiment_combined_1400.jsonl`

#### Step 2: Capture Activations & Train Probes
- **Notebook**: `Sentiment_Full_Pipeline_Colab.ipynb`
- **Runtime**: ~4-5 hours
- **Skip**: Data generation section (use your existing data)

---

## 📁 File Locations

### Generated Files (in Google Drive)

After completion, you'll have:

```
/content/drive/MyDrive/brije_sentiment_outputs/
├── data/
│   ├── positive_sentiment_700.jsonl
│   ├── negative_sentiment_700.jsonl
│   └── sentiment_combined_1400.jsonl
├── activations/
│   ├── layer_1_activations.h5
│   ├── layer_2_activations.h5
│   ├── ...
│   └── layer_34_activations.h5
├── probes/
│   ├── layer_1/
│   │   ├── probe_positive.pth
│   │   ├── probe_negative.pth
│   │   └── aggregate_metrics.json
│   ├── layer_2/
│   ├── ...
│   └── layer_34/
└── sentiment_layer_comparison.png
```

### Local Repository Structure

```
brije/
├── notebooks/
│   ├── Sentiment_Full_Pipeline_Colab.ipynb        # Complete pipeline
│   └── SENTIMENT_PIPELINE_README.md               # This file
├── third_party/datagen/
│   ├── Sentiment_Data_Generator_Colab.ipynb       # Data generation only
│   └── generated_data/
│       ├── positive_sentiment_700.jsonl
│       ├── negative_sentiment_700.jsonl
│       └── sentiment_combined_1400.jsonl
├── data/
│   ├── activations/sentiment/                      # Layer activations
│   └── probes_binary/sentiment/                    # Trained probes
└── src/probes/
    ├── capture_activations.py                      # Used by pipeline
    ├── train_binary_probes_parallel.py             # Used by pipeline
    └── universal_multi_layer_inference.py          # For using probes
```

---

## 🔧 Configuration Options

### Memory Management (OOM Prevention)

The pipeline captures layers in **batches of 10** to avoid out-of-memory errors:

```python
# In Sentiment_Full_Pipeline_Colab.ipynb
CAPTURE_CONFIG = {
    'layer_batches': [
        list(range(1, 11)),   # Batch 1: Layers 1-10
        list(range(11, 21)),  # Batch 2: Layers 11-20
        list(range(21, 31)),  # Batch 3: Layers 21-30 (skip if already captured)
        list(range(31, 35))   # Batch 4: Layers 31-34
    ]
}
```

**If you already have layers 21-30** from cognitive action training:
- Uncomment the line that removes batch 3:
  ```python
  CAPTURE_CONFIG['layer_batches'] = [b for i, b in enumerate(CAPTURE_CONFIG['layer_batches']) if i != 2]
  ```

### Training Parameters

```python
TRAIN_CONFIG = {
    'probe_type': 'linear',              # 'linear' or 'multihead'
    'batch_size': 32,                    # Increase if you have more VRAM
    'epochs': 50,                        # Max epochs (early stopping active)
    'learning_rate': 0.0005,             # 5e-4
    'weight_decay': 0.001,               # L2 regularization
    'early_stopping_patience': 10,       # Stop if no improvement
    'num_workers': 2,                    # For binary task (pos/neg)
    'pin_activations_to_gpu': True       # Faster training, needs more VRAM
}
```

**For lower VRAM (e.g., T4 15GB)**:
- Set `batch_size: 16`
- Set `pin_activations_to_gpu: False`

**For higher VRAM (e.g., A100 40GB)**:
- Set `batch_size: 64`
- Keep `pin_activations_to_gpu: True`

---

## 📊 Expected Performance

Based on similar binary classification tasks:

### Typical AUC-ROC Scores by Layer Range
- **Early layers (1-10)**: 0.70-0.80 (basic features)
- **Middle layers (11-20)**: 0.80-0.90 (semantic features)
- **Late layers (21-30)**: 0.85-0.95 (high-level sentiment)
- **Final layers (31-34)**: 0.80-0.90 (task-specific)

### Best Performing Layers
Usually layers **24-28** perform best for sentiment detection, as they capture high-level semantic features before task-specific adaptation.

---

## 🎯 Using the Trained Probes

### Option 1: Use Best Layer Only

```python
# After finding best layer from visualization (e.g., layer 26)
from src.probes.multi_probe_inference import MultiProbeInference

inference = MultiProbeInference(
    probes_dir='data/probes_binary/sentiment/layer_26',
    model_name='google/gemma-3-4b-it',
    layer=26
)

text = "I'm so excited about this new opportunity! It feels like everything is finally coming together."
results = inference.predict(text, top_k=2)

# Expected output:
# [
#   {'action': 'positive', 'probability': 0.95},
#   {'action': 'negative', 'probability': 0.05}
# ]
```

### Option 2: Multi-Layer Ensemble

Use probes from multiple layers for more robust predictions:

```python
from src.probes.universal_multi_layer_inference import UniversalMultiLayerInference

inference = UniversalMultiLayerInference(
    model_name='google/gemma-3-4b-it',
    probe_dirs={
        24: 'data/probes_binary/sentiment/layer_24',
        26: 'data/probes_binary/sentiment/layer_26',
        28: 'data/probes_binary/sentiment/layer_28'
    }
)

text = "I'm feeling really frustrated and overwhelmed by everything."
results = inference.predict_ensemble(text)

# Ensemble uses voting or averaging across layers
```

---

## 🔍 Troubleshooting

### OOM Errors During Capture

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in layer batches (capture 5 layers at a time instead of 10):
   ```python
   'layer_batches': [
       list(range(1, 6)),    # Batch 1: 5 layers
       list(range(6, 11)),   # Batch 2: 5 layers
       # etc...
   ]
   ```

2. Reduce batch_size for capture:
   ```python
   '--batch-size', '500'  # Instead of 1000
   ```

### OOM Errors During Training

**Symptom**: Memory error during probe training

**Solutions**:
1. Reduce batch size:
   ```python
   'batch_size': 16  # Instead of 32
   ```

2. Disable GPU pinning:
   ```python
   'pin_activations_to_gpu': False
   ```

3. Use linear probes instead of multihead:
   ```python
   'probe_type': 'linear'  # Smaller model
   ```

### Data Generation Takes Too Long

**Symptom**: Generating 1,400 examples takes hours

**Solutions**:
1. Increase parallel requests (if using Ollama):
   ```python
   'parallel_requests': 16  # Instead of 8
   ```

2. Use smaller batches with more frequent checkpoints:
   ```python
   batch_size = 25  # Generate 25 at a time
   ```

3. Use pre-generated data (if available)

### Hugging Face Authentication Fails

**Symptom**: Cannot download Gemma model

**Solutions**:
1. Ensure you've accepted Gemma license on HuggingFace
2. Generate access token at https://huggingface.co/settings/tokens
3. Run `notebook_login()` again with valid token

---

## 🔗 Integration with Existing Probes

### Combining with Cognitive Action Probes

You can use both cognitive action probes (45 actions) and sentiment probes (2 sentiments) together:

```python
# Load both probe systems
cognitive_inference = MultiProbeInference(
    probes_dir='data/probes_binary/layer_26',  # Cognitive actions
    model_name='google/gemma-3-4b-it',
    layer=26
)

sentiment_inference = MultiProbeInference(
    probes_dir='data/probes_binary/sentiment/layer_26',  # Sentiment
    model_name='google/gemma-3-4b-it',
    layer=26
)

# Get both predictions
text = "I'm reconsidering my approach, and honestly, it feels liberating!"

cognitive_results = cognitive_inference.predict(text, top_k=3)
sentiment_results = sentiment_inference.predict(text, top_k=1)

print(f"Cognitive actions: {cognitive_results}")
# [{'action': 'reconsidering', 'probability': 0.92}, ...]

print(f"Sentiment: {sentiment_results}")
# [{'action': 'positive', 'probability': 0.88}]
```

### Shared Layer Analysis

Since you're capturing all 34 layers, you can:

1. **Compare cognitive vs sentiment detection across layers**
   - See which layers specialize in sentiment vs cognitive processing
   - Identify if there's layer separation between emotion and reasoning

2. **Use best layer per task**
   - Layer 26 might be best for cognitive actions
   - Layer 24 might be best for sentiment
   - Use task-specific layers for optimal performance

3. **Build multi-task probes**
   - Train joint probes that detect both simultaneously
   - Useful for understanding emotion-cognition interaction

---

## 📈 Performance Optimization

### For Faster Inference

1. **Use linear probes**: 10-20x faster than multihead
2. **Pin probes to GPU memory**: Preload all probes at startup
3. **Batch processing**: Process multiple texts together
4. **Cache activations**: Reuse activations if analyzing same text multiple times

### For Better Accuracy

1. **Ensemble multiple layers**: Average predictions from layers 24, 26, 28
2. **Use calibrated thresholds**: Find optimal decision boundaries on validation set
3. **Context augmentation**: Add task-specific context to prompts
4. **Fine-tune on domain data**: If working in specific domain (e.g., therapy)

---

## 📝 Citation & Attribution

If you use this sentiment probe pipeline in your research:

```bibtex
@software{brije_sentiment_probes,
  title={Sentiment Probes for Gemma 3 4B},
  author={Brije Project},
  year={2025},
  url={https://github.com/ChuloIva/brije}
}
```

---

## 🆘 Support

- **Issues**: Open an issue on GitHub
- **Questions**: Check existing issues or discussions
- **Documentation**: See main README.md for general Brije usage

---

## ✅ Checklist

Before running the pipeline:

- [ ] Google Colab account with GPU access
- [ ] Hugging Face account with Gemma access approved
- [ ] Google Drive with 8+ GB free space
- [ ] Hugging Face access token ready
- [ ] 4-6 hours of runtime available (or plan to run in stages)

During pipeline:

- [ ] GPU is detected (check cell 1 output)
- [ ] Repository cloned successfully
- [ ] Dependencies installed without errors
- [ ] Google Drive mounted
- [ ] Hugging Face login successful
- [ ] Data generation complete (1,400 examples)
- [ ] All layer batches captured successfully
- [ ] All layers trained successfully
- [ ] Visualization generated
- [ ] Files backed up to Google Drive

After completion:

- [ ] Downloaded best layer probes
- [ ] Verified probe files are complete
- [ ] Tested inference on sample text
- [ ] (Optional) Integrated with cognitive action probes

---

**Good luck with your sentiment probe training! 🎉**
