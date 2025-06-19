#!/bin/bash

# Build and publish RLlama to PyPI

# Clean up previous builds
rm -rf dist/
rm -rf build/
rm -rf *.egg-info

# Build the package
echo "Building package..."
python -m build

# Check if twine is installed
if ! command -v twine &> /dev/null; then
    echo "twine not found. Installing..."
    pip install twine
fi

# Upload to PyPI
echo "Uploading to PyPI..."
python -m twine upload dist/*

echo "Package published successfully!"
