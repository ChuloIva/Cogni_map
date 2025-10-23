#!/bin/bash
# Train Binary Probes Across All Layers (4-28) with PARALLEL TRAINING
# Assumes activations are already captured
# Replicates Colab notebook workflow for local execution

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default configuration (matching Colab notebook)
ACTIVATIONS_DIR="./data/activations"
PROBES_DIR="./data/probes_binary"
LAYER_START=21
LAYER_END=30
PROBE_TYPE="linear"
BATCH_SIZE=32           # Optimized for parallel training
EPOCHS=50
LR=0.0005              # 5e-4
WEIGHT_DECAY=0.001     # 1e-3
EARLY_STOPPING=10
DEVICE="auto"
USE_SCHEDULER=true
NUM_WORKERS=45          # PARALLEL TRAINING: 45 workers = 45x speedup!
PIN_ACTIVATIONS=true    # Pin to GPU memory for faster training

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --layer-start)
            LAYER_START="$2"
            shift 2
            ;;
        --layer-end)
            LAYER_END="$2"
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
        --lr)
            LR="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
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
        --no-scheduler)
            USE_SCHEDULER=false
            shift
            ;;
        --no-pin-activations)
            PIN_ACTIVATIONS=false
            shift
            ;;
        --activations-dir)
            ACTIVATIONS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            PROBES_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Train binary probes across all layers with PARALLEL training"
            echo "(Replicates Colab notebook workflow for local execution)"
            echo ""
            echo "Options:"
            echo "  --layer-start NUM        Start layer (default: 4)"
            echo "  --layer-end NUM          End layer (default: 28)"
            echo "  --probe-type TYPE        Probe type: linear or multihead (default: linear)"
            echo "  --epochs NUM             Max epochs with early stopping (default: 50)"
            echo "  --batch-size NUM         Batch size (default: 32)"
            echo "  --lr FLOAT               Learning rate (default: 0.0005)"
            echo "  --weight-decay FLOAT     Weight decay (default: 0.001)"
            echo "  --device DEVICE          Device: auto, cuda, or cpu (default: auto)"
            echo "  --num-workers NUM        Parallel workers (default: 45 for 45x speedup!)"
            echo "  --no-scheduler           Disable learning rate scheduler"
            echo "  --no-pin-activations     Don't pin activations to GPU (use if limited VRAM)"
            echo "  --activations-dir PATH   Activations directory (default: ./data/activations)"
            echo "  --output-dir PATH        Output directory (default: ./data/probes_binary)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Train all layers with defaults (45 workers, like Colab)"
            echo "  $0"
            echo ""
            echo "  # Quick test with fewer layers and workers"
            echo "  $0 --layer-start 20 --layer-end 22 --epochs 10 --num-workers 8"
            echo ""
            echo "  # Conservative settings for lower VRAM (~16GB)"
            echo "  $0 --num-workers 8 --batch-size 16 --no-pin-activations"
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
echo "  🚀 PARALLEL TRAINING: BINARY PROBES ACROSS ALL LAYERS"
echo "  (Replicates Colab notebook workflow)"
echo "==================================================================="
echo -e "${NC}"

# Print configuration
NUM_LAYERS=$((LAYER_END - LAYER_START + 1))
TOTAL_PROBES=$((NUM_LAYERS * 45))

echo -e "${YELLOW}Configuration:${NC}"
echo "  Activations:  $ACTIVATIONS_DIR"
echo "  Output:       $PROBES_DIR"
echo "  Layers:       $LAYER_START-$LAYER_END ($NUM_LAYERS layers)"
echo "  Total Probes: $TOTAL_PROBES (45 per layer)"
echo "  Probe Type:   $PROBE_TYPE"
echo "  Epochs:       $EPOCHS (with early stopping)"
echo "  Batch Size:   $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Weight Decay:  $WEIGHT_DECAY"
echo "  Early Stopping: $EARLY_STOPPING epochs"
echo "  Scheduler:    $USE_SCHEDULER"
echo "  Device:       $DEVICE"
echo ""
echo -e "${YELLOW}🚀 Parallel Training Settings:${NC}"
echo "  Workers:      $NUM_WORKERS (train $NUM_WORKERS probes simultaneously)"
echo "  Pin to GPU:   $PIN_ACTIVATIONS"
echo "  Expected speedup: ~${NUM_WORKERS}x faster than sequential!"
echo ""

# Check if activations exist
if [ ! -d "$ACTIVATIONS_DIR" ]; then
    echo -e "${RED}ERROR: Activations directory not found: $ACTIVATIONS_DIR${NC}"
    exit 1
fi

NUM_ACTIVATION_FILES=$(find "$ACTIVATIONS_DIR" -name "layer_*_activations.h5" 2>/dev/null | wc -l)
if [ $NUM_ACTIVATION_FILES -eq 0 ]; then
    echo -e "${RED}ERROR: No activation files found in $ACTIVATIONS_DIR${NC}"
    echo "Please run activation capture first."
    exit 1
fi

echo -e "${GREEN}✓ Found $NUM_ACTIVATION_FILES activation files${NC}"
echo ""

# Estimate time with parallel training
EST_TIME_MIN=$((NUM_LAYERS * 3))  # ~2-5 minutes per layer with parallel training
EST_TIME_HOURS=$((EST_TIME_MIN / 60))

echo -e "${YELLOW}⏰ Estimated time with parallel training: $EST_TIME_HOURS hours ${NC}"
echo -e "${YELLOW}   (vs ~$((NUM_LAYERS * 25 / 60)) hours sequential - ${NUM_WORKERS}x speedup!)${NC}"
echo ""

# Confirm to proceed
read -p "Start parallel training for all layers? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Change to probe directory
cd src/probes

# Build command
CMD="python train_all_layers.py \
    --activations-dir ../../$ACTIVATIONS_DIR \
    --output-dir ../../$PROBES_DIR \
    --layer-start $LAYER_START \
    --layer-end $LAYER_END \
    --probe-type $PROBE_TYPE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight-decay $WEIGHT_DECAY \
    --early-stopping-patience $EARLY_STOPPING \
    --device $DEVICE \
    --num-workers $NUM_WORKERS"

if [ "$USE_SCHEDULER" = false ]; then
    CMD="$CMD --no-scheduler"
fi

if [ "$PIN_ACTIVATIONS" = false ]; then
    CMD="$CMD --no-pin-activations"
fi

# Run training
echo -e "\n${BLUE}Starting parallel training...${NC}\n"

$CMD

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ All layers training complete!${NC}"
else
    echo -e "\n${RED}✗ Training failed!${NC}"
    exit 1
fi

# Display summary
cd ../..

SUMMARY_FILE="$PROBES_DIR/training_summary.json"
if [ -f "$SUMMARY_FILE" ]; then
    echo -e "\n${BLUE}Training Summary:${NC}"

    python3 << EOF
import json

with open("$SUMMARY_FILE", 'r') as f:
    summary = json.load(f)

stats = summary['statistics']
print(f"  Layers trained:   {summary['total_layers']}")
print(f"  Total probes:     {summary['total_probes']}")
print(f"  Total time:       {summary['total_time_hours']:.2f} hours")
print(f"  Parallel workers: {summary['num_workers']}")
print(f"\n  Average AUC-ROC:  {stats['avg_auc']:.4f}")
print(f"  Best AUC-ROC:     {stats['best_auc']:.4f} (Layer {stats['best_layer']})")
print(f"  Worst AUC-ROC:    {stats['worst_auc']:.4f}")
EOF

    echo ""
fi

# Display per-action analysis if available
PER_ACTION_FILE="$PROBES_DIR/per_action_layer_analysis.json"
if [ -f "$PER_ACTION_FILE" ]; then
    echo -e "${BLUE}Per-Action Layer Analysis:${NC}"

    python3 << EOF
import json

with open("$PER_ACTION_FILE", 'r') as f:
    analysis = json.load(f)

summary = analysis['summary']
print(f"  Analyzed {summary['total_actions']} cognitive actions across {summary['total_layers_tested']} layers")
if summary.get('most_common_best_layer'):
    print(f"  Most effective layer: {summary['most_common_best_layer']}")

print(f"\n  See $PER_ACTION_FILE for detailed analysis")
EOF

    echo ""
fi

echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Review summary:"
echo "     cat $PROBES_DIR/training_summary.json"
echo ""
echo "  2. Review per-action analysis:"
echo "     cat $PROBES_DIR/per_action_layer_analysis.json"
echo ""
echo "  3. Test with best layer:"
echo "     cd src/probes"
echo "     python multi_probe_inference.py --probes-dir ../../$PROBES_DIR/layer_<best> --text 'Your text'"
echo ""

echo -e "${GREEN}All done! 🚀${NC}"
echo ""
echo -e "${YELLOW}Performance Summary:${NC}"
echo "  • Parallel training with $NUM_WORKERS workers"
echo "  • ~${NUM_WORKERS}x faster than sequential training"
echo "  • Trained $TOTAL_PROBES probes across $NUM_LAYERS layers"
echo "  • Per-action layer analysis completed"
