#!/bin/bash

# Retrieve and process batch results

cd ../src

echo "=========================================="
echo "Retrieving Batch Job Results"
echo "=========================================="
echo ""

uv run bigtom_batch_retrieve.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Generating conditions for the 6 required tasks..."
    echo "=========================================="
    uv run generate_conditions.py

    echo ""
    echo "=========================================="
    echo "All done! Check the following directories:"
    echo "  - data/bigtom/bigtom.csv (raw stories)"
    echo "  - data/conditions/ (6 condition directories)"
    echo "=========================================="
fi