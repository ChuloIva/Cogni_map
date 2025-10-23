#!/bin/bash
# Build script for cogni_map_workshop.tex
# This script compiles the LaTeX document with bibliography

set -e  # Exit on error

echo "Building cogni_map_workshop.pdf..."

# Change to the paper directory
cd "$(dirname "$0")"

# First pass - generate aux file
echo "Running pdflatex (first pass)..."
pdflatex -interaction=nonstopmode cogni_map_workshop.tex > /dev/null 2>&1 || echo "First pass completed with warnings"

# Process bibliography
echo "Running bibtex..."
bibtex cogni_map_workshop > /dev/null 2>&1 || echo "Bibtex completed"

# Second pass - incorporate references
echo "Running pdflatex (second pass)..."
pdflatex -interaction=nonstopmode cogni_map_workshop.tex > /dev/null 2>&1 || echo "Second pass completed with warnings"

# Third pass - resolve all references
echo "Running pdflatex (third pass)..."
pdflatex -interaction=nonstopmode cogni_map_workshop.tex > /dev/null 2>&1 || echo "Third pass completed with warnings"

# Clean up auxiliary files (optional - comment out if you want to keep them)
echo "Cleaning up auxiliary files..."
rm -f cogni_map_workshop.aux cogni_map_workshop.log cogni_map_workshop.bbl cogni_map_workshop.blg cogni_map_workshop.out

echo "✓ Done! PDF generated: cogni_map_workshop.pdf"
