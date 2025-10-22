#!/bin/bash
# Complete workflow for cognitive action probe system
# This script demonstrates the full pipeline from data to inference

set -e  # Exit on error

echo "==================================================================="
echo "BRIJE: Cognitive Action Detection System - Complete Workflow"
echo "==================================================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DATASET_PATH="./third_party/datagen/generated_data/stratified_combined_31500.jsonl"
ACTIVATIONS_DIR="./data/activations"
PROBES_DIR="./data/probes"
MODEL="google/gemma-2-3b-it"

echo -e "\n${BLUE}Configuration:${NC}"
echo "  Dataset: $DATASET_PATH"
echo "  Model: $MODEL"
echo "  Output directories:"
echo "    - Activations: $ACTIVATIONS_DIR"
echo "    - Probes: $PROBES_DIR"

# Step 1: Check dataset
echo -e "\n${YELLOW}Step 1: Checking dataset...${NC}"
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Please ensure you have the cognitive actions dataset in third_party/datagen/generated_data/"
    exit 1
fi

DATASET_SIZE=$(wc -l < "$DATASET_PATH")
echo -e "${GREEN}✓ Dataset found: $DATASET_SIZE examples${NC}"

# Step 2: Capture activations
echo -e "\n${YELLOW}Step 2: Capturing activations from Gemma 3 4B...${NC}"
echo "This may take 30-60 minutes on GPU..."

cd src/probes

if [ ! -f "$ACTIVATIONS_DIR/layer_27_activations.h5" ]; then
    python capture_activations.py \
        --dataset ../../$DATASET_PATH \
        --output-dir ../../$ACTIVATIONS_DIR \
        --model $MODEL \
        --layers 27 \
        --format hdf5 \
        --device auto

    echo -e "${GREEN}✓ Activations captured${NC}"
else
    echo -e "${GREEN}✓ Activations already exist, skipping...${NC}"
fi

# Step 3: Train probe
echo -e "\n${YELLOW}Step 3: Training linear probe...${NC}"
echo "This may take 10-20 minutes on GPU..."

if [ ! -f "../../$PROBES_DIR/best_probe.pth" ]; then
    python train_probes.py \
        --activations ../../$ACTIVATIONS_DIR/layer_27_activations.h5 \
        --output-dir ../../$PROBES_DIR \
        --model-type linear \
        --batch-size 32 \
        --epochs 20 \
        --lr 0.001 \
        --device auto

    echo -e "${GREEN}✓ Probe trained${NC}"
else
    echo -e "${GREEN}✓ Probe already trained, skipping...${NC}"
fi

# Step 4: Test inference
echo -e "\n${YELLOW}Step 4: Testing probe inference...${NC}"

TEST_TEXT="After receiving feedback from her colleague, Sarah began reconsidering her initial approach to the project. She realized that she had been making assumptions without fully understanding the constraints."

echo "Test text:"
echo "  \"$TEST_TEXT\""
echo ""

python probe_inference.py \
    --probe ../../$PROBES_DIR/best_probe.pth \
    --model $MODEL \
    --layer 27 \
    --text "$TEST_TEXT" \
    --top-k 5

echo -e "${GREEN}✓ Inference test complete${NC}"

# Step 5: Instructions for GUI
cd ../../third_party/liminal_backrooms

echo -e "\n${YELLOW}Step 5: Ready to launch GUI!${NC}"
echo ""
echo "To run the liminal backrooms interface with probes:"
echo ""
echo -e "  ${BLUE}cd third_party/liminal_backrooms${NC}"
echo -e "  ${BLUE}python main.py${NC}"
echo ""
echo "In the GUI:"
echo "  1. Select 'Gemma 3 4B (with Probes)' for AI-1 and/or AI-2"
echo "  2. Choose a prompt style (e.g., 'Cognitive Roles - Analyst vs Creative')"
echo "  3. Set number of turns (5-10 recommended)"
echo "  4. Enter a prompt or click 'Propagate'"
echo "  5. Watch the cognitive actions activate in real-time!"
echo ""
echo -e "${GREEN}==================================================================="
echo "SETUP COMPLETE!"
echo "===================================================================${NC}"
echo ""
echo "Quick test command:"
echo -e "  ${BLUE}python ../../src/probes/probe_inference.py --probe ../../data/probes/best_probe.pth --model $MODEL --layer 27${NC}"
echo ""
