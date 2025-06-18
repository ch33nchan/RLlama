#!/bin/bash

# Build and publish RLlama to PyPI
# Usage: ./publish.sh [test|prod]

# Default to test PyPI
TARGET="test"

# Check command line argument
if [ "$1" == "prod" ]; then
    TARGET="prod"
    echo "WARNING: Publishing to production PyPI!"
    read -p "Are you sure? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
fi

# Install build tools if needed
pip install --upgrade pip build twine

# Clean old build artifacts
rm -rf dist build *.egg-info

# Build package
echo "Building package..."
python -m build

# Check the package
echo "Checking package..."
twine check dist/*

if [ "$TARGET" == "test" ]; then
    echo "Uploading to Test PyPI..."
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
else
    echo "Uploading to PyPI..."
    twine upload dist/*
fi

echo "Done!"
