#!/bin/bash

# Script to generate 2000 stories using OpenAI Batch API
# Benefits:
# - 50% cost reduction (approximately $0.30 instead of $0.60)
# - No rate limit concerns
# - Automatic retry handling
# - Up to 24 hour turnaround

cd ../src

echo "=========================================="
echo "OpenAI Batch API - Story Generation"
echo "=========================================="
echo "Creating batch requests for 2000 stories..."
echo "Model: gpt-4o-mini"
echo "Cost: ~$0.30 (50% off standard pricing)"
echo "Turnaround: Up to 24 hours"
echo "=========================================="
echo ""

# Step 1: Create and submit batch job
uv run bigtom_batch_create.py \
  --model gpt-4o-mini \
  --temperature 0.5 \
  --max_tokens 450 \
  --num_shots 3 \
  --num_stories 2000

echo ""
echo "=========================================="
echo "Batch job submitted successfully!"
echo "=========================================="
echo ""
echo "The batch job will complete within 24 hours."
echo "To check status, run:"
echo "  ./batch_check_status.sh"
echo ""
echo "To retrieve results when completed, run:"
echo "  ./batch_retrieve_results.sh"
echo ""
echo "=========================================="
