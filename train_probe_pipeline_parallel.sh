#!/bin/bash
# Parallel Binary Probe Training Pipeline - Optimized for High-VRAM GPUs (40GB+)
# Trains 45 binary probes in parallel to maximize GPU utilization

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default configuration - Optimized for 40GB VRAM
DATASET_PATH="./third_party/datagen/generated_data/stratified_combined_31500.jsonl"
ACTIVATIONS_DIR="./data/activations"
PROBES_DIR="./data/probes_binary"
MODEL="google/gemma-3-4b-it"
LAYER=27
PROBE_TYPE="linear"
BATCH_SIZE=128        # Balanced batch size (good gradient updates + GPU utilization)
EPOCHS=50
DEVICE="auto"
NUM_WORKERS=8         # Train 8 probes simultaneously
PIN_ACTIVATIONS=true  # Pin activations to GPU for faster training

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --probe-type)
            PROBE_TYPE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --no-pin-activations)
            PIN_ACTIVATIONS=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Parallel training optimized for high-VRAM GPUs (40GB+)"
            echo ""
            echo "Options:"
            echo "  --dataset PATH          Path to dataset JSONL file"
            echo "  --layer NUM             Layer to extract activations from (default: 27)"
            echo "  --probe-type TYPE       Probe type: linear or multihead (default: linear)"
            echo "  --epochs NUM            Training epochs per probe (default: 50)"
            echo "  --batch-size NUM        Batch size for training (default: 512)"
            echo "  --device DEVICE         Device: auto, cuda, or cpu (default: auto)"
            echo "  --num-workers NUM       Number of parallel workers (default: 8)"
            echo "  --no-pin-activations    Don't pin activations to GPU memory"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Use defaults (8 workers, batch size 512)"
            echo "  $0"
            echo ""
            echo "  # Max out 40GB GPU (12 workers, batch size 1024)"
            echo "  $0 --num-workers 12 --batch-size 1024"
            echo ""
            echo "  # Conservative settings for lower VRAM"
            echo "  $0 --num-workers 4 --batch-size 256 --no-pin-activations"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${BLUE}"
echo "==================================================================="
echo "  PARALLEL BINARY PROBE TRAINING PIPELINE"
echo "  High-Performance Training for 40GB+ GPUs"
echo "==================================================================="
echo -e "${NC}"

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Dataset: $DATASET_PATH"
echo "  Model: $MODEL"
echo "  Layer: $LAYER"
echo "  Probe Type: $PROBE_TYPE (45 binary probes)"
echo "  Epochs per Probe: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo "  Parallel Workers: $NUM_WORKERS"
echo "  Pin Activations to GPU: $PIN_ACTIVATIONS"
echo "  Output:"
echo "    - Activations: $ACTIVATIONS_DIR"
echo "    - Binary Probes: $PROBES_DIR"
echo ""

# Performance estimate
echo -e "${YELLOW}Performance Estimate:${NC}"
if [ $NUM_WORKERS -ge 8 ]; then
    echo "  Expected speedup: ${NUM_WORKERS}x faster than sequential"
    echo "  Estimated total time: 10-15 minutes (vs 1-2 hours sequential)"
else
    echo "  Expected speedup: ${NUM_WORKERS}x faster than sequential"
    echo "  Estimated total time: 20-30 minutes"
fi
echo ""

# Confirm to proceed
read -p "Start parallel training pipeline? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Change to probe directory
cd src/probes

# Step 1: Check if activations already exist
ACTIVATION_FILE="../../$ACTIVATIONS_DIR/layer_${LAYER}_activations.h5"

if [ -f "$ACTIVATION_FILE" ]; then
    echo -e "\n${YELLOW}Step 1: Activations${NC}"
    echo -e "${GREEN}✓ Activations already exist at: $ACTIVATION_FILE${NC}"
    echo "  Skipping activation capture..."

    read -p "Re-capture activations? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SKIP_CAPTURE=false
    else
        SKIP_CAPTURE=true
    fi
else
    SKIP_CAPTURE=false
fi

# Step 1: Capture Activations
if [ "$SKIP_CAPTURE" = false ]; then
    echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Step 1: Capturing Activations from Layer $LAYER${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

    # Check if dataset exists
    if [ ! -f "../../$DATASET_PATH" ]; then
        echo -e "${RED}ERROR: Dataset not found at ../../$DATASET_PATH${NC}"
        exit 1
    fi

    DATASET_SIZE=$(wc -l < "../../$DATASET_PATH")
    echo "  Dataset size: $DATASET_SIZE examples"
    echo "  Estimated time: 2-3 hours on GPU"
    echo ""

    python capture_activations.py \
        --dataset "../../$DATASET_PATH" \
        --output-dir "../../$ACTIVATIONS_DIR" \
        --model "$MODEL" \
        --layers $LAYER \
        --format hdf5 \
        --batch-save \
        --batch-size 1000 \
        --device "$DEVICE"

    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ Activation capture complete!${NC}"
    else
        echo -e "\n${RED}✗ Activation capture failed!${NC}"
        exit 1
    fi
fi

# Step 2: Train Binary Probes in Parallel
echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Step 2: Parallel Training of 45 Binary Probes${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

if [ ! -f "$ACTIVATION_FILE" ]; then
    echo -e "${RED}ERROR: Activations not found at $ACTIVATION_FILE${NC}"
    exit 1
fi

echo "  Input: $ACTIVATION_FILE"
echo "  Strategy: Parallel One-vs-Rest (${NUM_WORKERS} workers)"
echo "  Epochs per probe: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Estimated time: 10-20 minutes per layer (${NUM_WORKERS}x speedup)"
echo ""

# Build training command for parallel binary probes
TRAIN_CMD="python train_binary_probes_parallel.py \
    --activations $ACTIVATION_FILE \
    --output-dir ../../$PROBES_DIR \
    --model-type $PROBE_TYPE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --device $DEVICE \
    --num-workers $NUM_WORKERS"

# Add hidden dim for multihead probe
if [ "$PROBE_TYPE" = "multihead" ]; then
    TRAIN_CMD="$TRAIN_CMD --hidden-dim 512"
fi

# Add pin activations flag
if [ "$PIN_ACTIVATIONS" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --pin-activations-to-gpu"
else
    TRAIN_CMD="$TRAIN_CMD --no-pin-activations"
fi

echo -e "${BLUE}Starting parallel training...${NC}\n"

$TRAIN_CMD

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Parallel probe training complete!${NC}"
else
    echo -e "\n${RED}✗ Parallel probe training failed!${NC}"
    exit 1
fi

# Step 3: Test Multi-Probe Inference
echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Step 3: Testing Multi-Probe Inference${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Check if probes directory exists
if [ ! -d "../../$PROBES_DIR" ]; then
    echo -e "${RED}ERROR: Probes directory not found at ../../$PROBES_DIR${NC}"
    exit 1
fi

# Count number of probes
PROBE_COUNT=$(find "../../$PROBES_DIR" -name "probe_*.pth" | wc -l)
if [ $PROBE_COUNT -eq 0 ]; then
    echo -e "${RED}ERROR: No trained probes found in ../../$PROBES_DIR${NC}"
    exit 1
fi

echo "  Found $PROBE_COUNT trained binary probes"
echo "  Running inference on test examples..."
echo ""

# Test example
TEST_TEXT="After receiving feedback from her colleague, Sarah began reconsidering her initial approach to the project."

echo -e "${BLUE}Test Example:${NC}"
echo "  \"$TEST_TEXT\""
echo ""

python multi_probe_inference.py \
    --probes-dir "../../$PROBES_DIR" \
    --model "$MODEL" \
    --layer $LAYER \
    --text "$TEST_TEXT" \
    --top-k 5 \
    --threshold 0.1

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Inference testing complete!${NC}"
else
    echo -e "\n${RED}✗ Inference testing failed!${NC}"
    exit 1
fi

# Step 4: Summary
echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   PIPELINE COMPLETE! 🎉${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"

echo ""
echo "Files created:"
echo "  📊 Activations:       $ACTIVATION_FILE"
echo "  🧠 Binary Probes:     ../../$PROBES_DIR/probe_*.pth ($PROBE_COUNT files)"
echo "  📈 Aggregate Metrics: ../../$PROBES_DIR/aggregate_metrics.json"
echo "  📝 Per-Action Metrics: ../../$PROBES_DIR/metrics_*.json"
echo ""

# Display aggregate metrics if available
AGGREGATE_METRICS_FILE="../../$PROBES_DIR/aggregate_metrics.json"
if [ -f "$AGGREGATE_METRICS_FILE" ]; then
    echo -e "${BLUE}Aggregate Test Performance (across all 45 probes):${NC}"

    # Extract key metrics using python
    python3 << EOF
import json
import sys

try:
    with open("$AGGREGATE_METRICS_FILE", 'r') as f:
        metrics = json.load(f)

    print(f"  Average AUC-ROC:   {metrics['average_auc_roc']:.3f}")
    print(f"  Average F1:        {metrics['average_f1']:.3f}")
    print(f"  Average Accuracy:  {metrics['average_accuracy']:.3f}")
    print(f"  Average Precision: {metrics['average_precision']:.3f}")
    print(f"  Average Recall:    {metrics['average_recall']:.3f}")
    
    if 'total_training_time_seconds' in metrics:
        time_mins = metrics['total_training_time_seconds'] / 60
        print(f"\n  Total training time: {time_mins:.1f} minutes")
        print(f"  Speedup factor:     {metrics['num_workers']}x")
except Exception as e:
    print(f"  Could not parse metrics: {e}", file=sys.stderr)
EOF

    echo ""
fi

echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Review aggregate metrics: cat ../../$PROBES_DIR/aggregate_metrics.json"
echo "  2. Test with custom text:"
echo "     cd src/probes"
echo "     python multi_probe_inference.py --probes-dir ../../$PROBES_DIR --text 'Your text here'"
echo "  3. Use in Liminal Backrooms:"
echo "     cd third_party/liminal_backrooms"
echo "     python main.py"
echo "     (Select model with probes in the GUI)"
echo ""

echo -e "${GREEN}All done! 🚀${NC}"
echo ""
echo -e "${YELLOW}Performance Tips:${NC}"
echo "  - For maximum speed on 40GB GPU: --num-workers 12 --batch-size 1024"
echo "  - For stability: --num-workers 6 --batch-size 512"
echo "  - Monitor GPU usage with: watch -n 1 nvidia-smi"

