#!/bin/bash

# Check the status of the batch job

cd ../src

echo "=========================================="
echo "Checking Batch Job Status"
echo "=========================================="
echo ""

uv run bigtom_batch_retrieve.py --check_status

echo ""
