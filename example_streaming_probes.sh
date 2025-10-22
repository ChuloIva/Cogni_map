#!/bin/bash

# Example: Streaming probe inference with token-level tracking

echo "=========================================="
echo "Streaming Probe Inference Examples"
echo "=========================================="
echo ""

# Example 1: Basic streaming with real-time output
echo "1. Basic streaming inference with real-time output:"
echo "---"
python src/probes/streaming_probe_inference.py \
    --probes-dir data \
    --model google/gemma-2-3b-it \
    --text "After analyzing the data, I began reconsidering my assumptions and questioning my initial approach." \
    --top-k 5 \
    --threshold 0.1

echo ""
echo ""

# Example 2: Visualize specific action
echo "2. Visualize token activations for specific action:"
echo "---"
python src/probes/streaming_probe_inference.py \
    --probes-dir data \
    --text "She was comparing different solutions to find the best approach." \
    --top-k 10 \
    --threshold 0.05 \
    --visualize "Comparing" \
    --no-realtime

echo ""
echo ""

# Example 3: Export to CSV for analysis
echo "3. Export token activations to CSV:"
echo "---"
python src/probes/streaming_probe_inference.py \
    --probes-dir data \
    --text "The quarterly numbers look concerning. Revenue is up but margins are down." \
    --export-csv output/token_activations.csv \
    --no-realtime

echo ""
echo ""
echo "=========================================="
echo "Examples complete!"
echo "=========================================="
