---
id: methods
title: Installation Methods
sidebar_label: Installation
slug: /installation/methods
---

# Installing RLlama

RLlama can be installed using various methods, depending on your needs and preferences.

## Using pip (Recommended)

The simplest way to install RLlama is using pip:

```bash
pip install rllama
```

This will install the core RLlama package with basic dependencies.

## Installation with Optional Features

RLlama offers several optional feature sets that can be installed as needed:

```bash
# With Gym integration
pip install "rllama[gym]"

# With Stable Baselines3 integration
pip install "rllama[sb3]"

# With visualization tools
pip install "rllama[vis]"

# With optimization tools
pip install "rllama[optim]"

# With all optional dependencies
pip install "rllama[all]"
```

## Installation from Source

For the latest development version or to contribute to RLlama, you can install from source:

```bash
git clone https://github.com/ch33nchan/RLlama.git
cd RLlama
pip install -e .
```

The `-e` flag installs the package in "editable" mode, allowing you to modify the source code and immediately see the effects.

## Docker Installation

RLlama is also available as a Docker image with all dependencies pre-installed:

```bash
docker pull ch33nchan/rllama:latest
```

To run the Docker container:

```bash
docker run -it ch33nchan/rllama:latest
```

## Verifying Your Installation

After installing RLlama, you can verify that it's working correctly with:

```python
import rllama
print(rllama.__version__)

# Create a simple reward engine
from rllama import RewardEngine
engine = RewardEngine()
print("RLlama is installed correctly!")
```

If this runs without errors, your installation is working properly.
