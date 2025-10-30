# Batch Steering Vector Evaluation

This directory contains scripts for systematically evaluating steering vectors across different intensities and ToM benchmark conditions.

## Scripts

### 1. `batch_evaluate_steering_intensities.py`

Main script that runs batch evaluations with varying steering vector intensities.

**Features:**
- Tests steering vectors from intensity 500 to 2000 (configurable)
- Automatically matches steering vectors to appropriate conditions
- Evaluates 100 samples per condition (configurable)
- Saves results with descriptive filenames

**Usage:**

```bash
# Run all evaluations (CAUTION: This will take many hours!)
python batch_evaluate_steering_intensities.py

# Dry run to see what will be executed
python batch_evaluate_steering_intensities.py --dry-run

# Custom intensity range
python batch_evaluate_steering_intensities.py \
    --intensity-min 500 \
    --intensity-max 2000 \
    --intensity-step 250 \
    --num-samples 100

# Test specific vector only
python batch_evaluate_steering_intensities.py \
    --vector tom_backward_belief.gguf \
    --num-samples 50

# Test specific condition only
python batch_evaluate_steering_intensities.py \
    --condition 0_backward_belief_false_belief \
    --num-samples 50

# Custom probes directory
python batch_evaluate_steering_intensities.py \
    --probes-dir ../trained_probes
```

**Options:**
- `--model`: HuggingFace model name (default: google/gemma-3-4b-it)
- `--probes-dir`: Directory with trained probes (default: data/probes_binary)
- `--num-samples`, `-n`: Samples per evaluation (default: 100)
- `--intensity-min`: Minimum steering intensity (default: 500)
- `--intensity-max`: Maximum steering intensity (default: 2000)
- `--intensity-step`: Step size for intensities (default: 250)
- `--dry-run`: Print commands without executing
- `--vector`: Run only specific steering vector
- `--condition`: Run only specific condition

**Output:**
Results are saved to `batch_results/batch_TIMESTAMP/` with files:
- `{vector}__{condition}__coeff{intensity}_raw.csv`: Raw results
- `{vector}__{condition}__coeff{intensity}_summary.json`: Summary statistics
- `{vector}__{condition}__coeff{intensity}_differences.png`: Activation differences plot
- `{vector}__{condition}__coeff{intensity}_accuracy.png`: Accuracy comparison
- `batch_metadata.json`: Experiment tracking metadata

### 2. `analyze_batch_results.py`

Analyzes batch results and generates visualizations and reports.

**Features:**
- Intensity vs accuracy curves for each vector/condition
- Heatmaps of accuracy improvements
- Optimal intensity recommendations
- Comprehensive summary reports

**Usage:**

```bash
# Analyze results from a batch run
python analyze_batch_results.py batch_results/batch_20240101_120000/

# Or find the latest batch directory
python analyze_batch_results.py $(ls -td batch_results/batch_* | head -1)
```

**Output:**
Creates the following files in the batch results directory:
- `intensity_curves.png`: Accuracy vs intensity curves
- `improvement_heatmap.png`: Heatmap of improvements by vector/intensity
- `optimal_intensities.csv`: Best intensity for each vector/condition pair
- `summary_report.txt`: Comprehensive text report

## Steering Vector to Condition Mappings

The script automatically matches steering vectors to appropriate conditions:

| Steering Vector | Conditions |
|----------------|------------|
| `tom_backward_belief.gguf` | All backward_belief conditions |
| `tom_forward_belief.gguf` | All forward_belief conditions |
| `tom_forward_action.gguf` | All forward_action conditions |
| `tom_general.gguf` | Sample of all condition types |
| `tom_core_capabilities.gguf` | Select conditions |
| `tom_direction.gguf` | Forward and backward conditions |
| `tom_belief_type.gguf` | Belief-related conditions |

## Example Workflow

```bash
# 1. Test a small subset first (recommended)
python batch_evaluate_steering_intensities.py \
    --vector tom_backward_belief.gguf \
    --num-samples 10 \
    --intensity-min 500 \
    --intensity-max 1000 \
    --intensity-step 250

# 2. Analyze the test results
LATEST_BATCH=$(ls -td batch_results/batch_* | head -1)
python analyze_batch_results.py "$LATEST_BATCH"

# 3. If satisfied, run full evaluation
python batch_evaluate_steering_intensities.py --num-samples 100

# 4. Analyze full results
LATEST_BATCH=$(ls -td batch_results/batch_* | head -1)
python analyze_batch_results.py "$LATEST_BATCH"
```

## Estimated Runtime

- **Per experiment**: ~5 minutes (depends on GPU, model size, samples)
- **Full batch** (all vectors, all intensities):
  - ~7 vectors × 2-4 conditions each × 7 intensity values = ~140-196 experiments
  - Estimated total: **12-16 hours**

**Recommendation**: Start with a subset or lower sample count to verify everything works!

## Configuration Details

### Default Intensity Range
- Minimum: 500
- Maximum: 2000
- Step: 250
- Values tested: [500, 750, 1000, 1250, 1500, 1750, 2000]

### Default Settings
- Samples: 100 per condition
- Probe layers: 10-20
- Steering layers: -4 to -19 (last 16 layers)

## Troubleshooting

**Out of memory errors:**
```bash
# Reduce samples or test one vector at a time
python batch_evaluate_steering_intensities.py \
    --vector tom_backward_belief.gguf \
    --num-samples 50
```

**Resume failed experiments:**
Check `batch_metadata.json` in results directory to see which experiments failed, then run them individually:
```bash
python batch_evaluate_steering_intensities.py \
    --vector <failed_vector> \
    --condition <failed_condition>
```

**Check progress:**
```bash
# Monitor the latest batch directory
watch -n 5 "ls -lh batch_results/$(ls -t batch_results/ | head -1)/ | wc -l"
```

## Notes

- Results accumulate in `batch_results/` directory
- Each batch run creates a timestamped subdirectory
- The evaluation script automatically moves results to the batch directory
- You can safely interrupt and resume by running with `--vector` and `--condition` filters
- Use `--dry-run` to preview without executing

## Questions?

See the main evaluation script documentation:
```bash
python evaluate_cognitive_actions_with_steering.py --help
```