#!/bin/bash

# Build the website
npm run build

# Navigate to the build directory
if [ -d "build" ]; then
  cd build
else
  echo "Error: Build directory not found!"
  exit 1
fi

# Create a git repository
git init
git add -A
git commit -m "Deploy documentation"

# Push to the gh-pages branch of your repository
git push -f git@github.com:ch33nchan/RLlama.git main:gh-pages

cd -
echo "Deployment complete! Your site should be available at: https://ch33nchan.github.io/RLlama/"
