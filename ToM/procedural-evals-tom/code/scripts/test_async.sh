#!/bin/bash

# Quick test script to verify async generation works
# Generates only 5 stories to test the system

cd ../src

echo "=========================================="
echo "Testing async generation with 5 stories"
echo "=========================================="

uv run bigtom_async.py \
  --model gpt-4o \
  --temperature 0.5 \
  --max_tokens 450 \
  --num_completions 1 \
  --num_shots 3 \
  --num_stories 5 \
  --rpm_limit 450 \
  --tpm_limit 27000 \
  --max_retries 5 \
  --max_concurrent 3 \
  --verbose

echo "=========================================="
echo "Test complete!"
echo "=========================================="