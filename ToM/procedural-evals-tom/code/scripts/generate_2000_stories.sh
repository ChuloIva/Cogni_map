#!/bin/bash

# Script to generate 2000 stories using gpt-4o with rate limiting
# This will create stories for 6 conditions:
# - Forward Belief (true_belief, false_belief)
# - Forward Action (true_belief, false_belief)
# - Backward Belief (true_belief, false_belief)

cd ../src

echo "=========================================="
echo "Generating 2000 stories with gpt-4o"
echo "Rate limits: 450 RPM, 27000 TPM"
echo "=========================================="

# Generate 2000 stories using UV
uv run bigtom.py \
  --model gpt-4o \
  --temperature 0.5 \
  --max_tokens 450 \
  --num_completions 1 \
  --num_shots 3 \
  --num_stories 2000 \
  --rpm_limit 450 \
  --tpm_limit 27000 \
  --max_retries 5 \
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