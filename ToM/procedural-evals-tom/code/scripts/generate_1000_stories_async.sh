#!/bin/bash

# Script to generate 1000 stories using gpt-4o with concurrent workers
# This uses async execution with multiple concurrent requests while respecting rate limits
# - No 24-hour wait like batch API
# - Automatic retry on rate limit errors
# - Real-time progress tracking

cd ../src

echo "=========================================="
echo "Generating 1000 stories with gpt-4o-mini (Async)"
echo "Rate limits: 450 RPM, 180,000 TPM (Tier 1 - 90% safety margin)"
echo "Max concurrent requests: 20"
echo "=========================================="

# Generate 1000 stories using UV with async workers
# Tier 1 limits: 500 RPM, 200,000 TPM
# Using 90% of limits for safety: 450 RPM, 180,000 TPM
uv run bigtom_async.py \
  --model gpt-4o-mini-2024-07-18 \
  --temperature 0.5 \
  --max_tokens 450 \
  --num_completions 1 \
  --num_shots 3 \
  --num_stories 1000 \
  --rpm_limit 450 \
  --tpm_limit 180000 \
  --max_retries 5 \
  --max_concurrent 20 \
  --verbose

echo "=========================================="
echo "Story generation complete!"
echo "=========================================="

# Generate conditions from the stories
echo "Generating conditions for the 6 required tasks..."
uv run generate_conditions.py

echo "=========================================="
echo "All done! Check the following directories:"
echo "  - data/bigtom/bigtom.csv (raw stories)"
echo "  - data/conditions/ (6 condition directories)"
echo "=========================================="