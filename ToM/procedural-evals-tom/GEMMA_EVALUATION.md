# Gemma 3 4B IT - Theory of Mind Evaluation

This guide explains how to evaluate Gemma 3 4B IT (or other HuggingFace models) on the Theory of Mind (ToM) tasks.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure HuggingFace Token

Create a `.env` file in the root directory:

```bash
echo "HUGGINGFACE_TOKEN=hf_your_token_here" > .env
```

Get your HuggingFace token from: https://huggingface.co/settings/tokens

## Running Evaluations

### Quick Start - Evaluate All Conditions (100 samples each)

```bash
cd code/src
python evaluate_gemma.py
```

This will:
- Load Gemma 3 4B IT from your local HuggingFace cache
- Evaluate on all 12 conditions in `data/conditions/`
- Process 100 stories per condition (1200 total)
- Show progress bars for each step
- Save results to `data/results/`

### Evaluate All Samples

To evaluate all ~1200 samples per condition:

```bash
python evaluate_gemma.py --num_samples 0
```

(Note: `num_samples=0` or negative values will use all available samples)

### Evaluate Specific Conditions

Evaluate a single condition:

```bash
python evaluate_gemma.py --condition 0_backward_belief_false_belief
```

### Custom Model

Use a different model:

```bash
python evaluate_gemma.py --model_name google/gemma-2-9b-it
```

### Advanced Options

```bash
python evaluate_gemma.py \
  --model_name google/gemma-3-4b-it \
  --num_samples 200 \
  --temperature 0.0 \
  --max_tokens 100 \
  --offset 0 \
  --verbose
```

**Parameters:**
- `--model_name`: HuggingFace model name (default: google/gemma-3-4b-it)
- `--num_samples` or `-n`: Stories per condition (default: 100)
- `--temperature`: Sampling temperature (default: 0.0)
- `--max_tokens`: Maximum generation length (default: 100)
- `--offset` or `-o`: Start from this sample index (default: 0)
- `--verbose` or `-v`: Print detailed output for each sample
- `--condition`: Evaluate single condition only

## Understanding the Conditions

The evaluation covers 12 conditions organized by:

1. **Order Init**: `0` or `1` (order of information presentation)
2. **Direction**: `backward` or `forward` (temporal reasoning direction)
3. **Variable**: `belief` or `action` (what's being tested)
4. **Belief Type**: `true_belief` or `false_belief` (ground truth)

Example conditions:
- `0_backward_belief_false_belief`
- `1_forward_action_true_belief`

## Results

### Output Files

Results are saved to `data/results/`:

```
data/results/
├── summary_google_gemma-3-4b-it_0.0_0_100.json  # Overall summary
├── 0_backward_belief_false_belief/
│   ├── prediction_google_gemma-3-4b-it_0.0_belief_false_belief_0_100.csv
│   └── accuracy_google_gemma-3-4b-it_0.0_belief_false_belief_0_100.csv
├── 0_backward_belief_true_belief/
│   └── ...
└── ...
```

### Summary JSON Structure

```json
{
  "model_name": "google/gemma-3-4b-it",
  "temperature": 0.0,
  "overall_accuracy": 0.75,
  "total_samples": 1200,
  "total_correct": 900,
  "condition_results": [
    {
      "condition": "0_backward_belief_false_belief",
      "accuracy": 0.72,
      "num_samples": 100,
      "num_correct": 72,
      "uncertain_grades": 3
    },
    ...
  ]
}
```

## Visualizing Results

### Generate All Visualizations

```bash
cd code/src
python visualize_results.py --summary "../../data/results/summary_google_gemma-3-4b-it_*.json"
```

This creates:
- `*_breakdown.png`: Bar chart of accuracy by condition
- `*_heatmap.png`: Heatmaps showing accuracy patterns
- `*_table.csv`: Formatted results table

### Compare Multiple Models

```bash
python visualize_results.py \
  --summary "../../data/results/summary_*.json" \
  --compare \
  --output_dir "../../data/results/comparisons"
```

### Specify Output Directory

```bash
python visualize_results.py \
  --summary "../../data/results/summary_google_gemma-3-4b-it_0.0_0_100.json" \
  --output_dir "./visualizations"
```

## Example Workflow

```bash
# 1. Navigate to source directory
cd ToM/procedural-evals-tom/code/src

# 2. Run evaluation (100 samples per condition)
python evaluate_gemma.py

# 3. View results
cat ../../data/results/summary_google_gemma-3-4b-it_0.0_0_100.json

# 4. Generate visualizations
python visualize_results.py --summary "../../data/results/summary_*.json"

# 5. Check the plots
ls ../../data/results/*.png
```

## Progress Tracking

The evaluation includes comprehensive progress tracking:

1. **Model Loading**: Shows tokenizer and model loading progress
2. **Per-Condition Progress**: Individual progress bar for each condition
3. **Overall Progress**: Tracks completion across all conditions
4. **Detailed Results**: Shows accuracy after each condition completes

## GPU Acceleration

The script automatically detects and uses GPU if available. Model is loaded with:
- `device_map="auto"` for automatic multi-GPU distribution
- `torch.bfloat16` precision on GPU for efficiency
- `torch.float32` precision on CPU

## Troubleshooting

### Out of Memory

Reduce batch size or use a smaller model:
```bash
python evaluate_gemma.py --model_name google/gemma-2-2b-it
```

### Model Not Found in Cache

Ensure the model is downloaded:
```python
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", token="your_token")
```

### Missing HUGGINGFACE_TOKEN

Check your `.env` file exists and contains:
```
HUGGINGFACE_TOKEN=hf_your_actual_token
```

## Implementation Details

### Evaluation Method

1. **Story Presentation**: Full story context provided
2. **MCQ Format**: Questions with two options (a/b)
3. **Answer Grading**:
   - Explicit key matching (a) or b))
   - Fuzzy content matching
   - Keyword overlap scoring
4. **Random Seed**: Fixed at 0 for reproducibility

### Grading Strategy

The grading system uses multiple strategies:
1. Check for explicit answer keys (a) or b))
2. Compare keyword overlap with true vs. wrong answers
3. Mark as "uncertain" if no clear match (counted as incorrect)

## Citation

If you use this evaluation code, please cite the original paper:

```
@article{bigtom2024,
  title={Understanding Social Reasoning in LLMs with LLMs},
  year={2024}
}
```