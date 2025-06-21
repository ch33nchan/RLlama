#!/bin/bash

# Build the website
npm run build

# Check if build directory exists, some Docusaurus configs use "build" others use "rllama-docs/build"
if [ -d "build" ]; then
  cd build
elif [ -d "../build" ]; then
  cd ../build
else
  echo "Error: Build directory not found!"
  exit 1
fi

# Create a git repository
git init
git add -A
git commit -m "Deploy documentation"

# Push to the gh-pages branch of your repository
# Using main instead of master
git push -f git@github.com:ch33nchan/RLlama.git main:gh-pages

cd -
