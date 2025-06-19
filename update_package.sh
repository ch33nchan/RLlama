#!/bin/bash

# Script to update the RLlama package on PyPI

# Make sure we're in the right directory
cd /Users/cheencheen/Desktop/git/rl/rllama

# Check version in __init__.py
echo "Current package version:"
grep "__version__" rllama/__init__.py

# Install build tools if needed
pip install --upgrade build twine

# Clean up previous builds
echo "Cleaning up previous builds..."
rm -rf dist/
rm -rf build/
rm -rf *.egg-info

# Build the package
echo "Building package..."
python -m build

# List the generated distributions
echo "Built distributions:"
ls -l dist/

# Upload to PyPI
echo "Ready to upload to PyPI. Proceed? (y/n)"
read confirm

if [ "$confirm" = "y" ]; then
    python -m twine upload dist/*
    echo "Package published successfully!"
else
    echo "Upload canceled. You can upload later with: python -m twine upload dist/*"
fi
