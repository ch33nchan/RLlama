# Installation Guide

## Basic Installation

RLlama is available on PyPI, making it easy to install with pip:

```bash
pip install rllama
```

This installs the core functionality of RLlama, including the reward engine, basic reward components, and fundamental tools.

## Installation with Optional Dependencies

RLlama offers several optional dependency groups for specific use cases:

### Gym Integration

For working with OpenAI Gym environments:

```bash
pip install "rllama[gym]"
```

This installs:
- gym (≥0.17.0)

### Stable Baselines Integration 

For working with Stable Baselines3:

```bash
pip install "rllama[sb3]"
```

This installs:
- stable-baselines3 (≥1.5.0)

### Visualization Tools

For advanced visualization and dashboard features:

```bash
pip install "rllama[vis]"
```

This installs:
- streamlit (≥1.10.0)

### RLHF Support

For Reinforcement Learning from Human Feedback:

```bash
pip install "rllama[rlhf]"
```

This installs:
- tqdm (≥4.45.0)

### Development Tools

For contributing to RLlama:

```bash
pip install "rllama[dev]"
```

This installs:
- pytest (≥6.0.0)
- black (≥22.3.0)
- isort (≥5.10.0)
- build
- twine

### Complete Installation

To install all optional dependencies:

```bash
pip install "rllama[all]"
```

## Development Installation

For development work, install RLlama in editable mode:

```bash
git clone https://github.com/ch33nchanyes/rllama.git
cd rllama
pip install -e ".[dev]"
```

This allows you to modify the code and see changes without reinstalling.

## Requirements

RLlama requires:

- Python ≥ 3.7
- numpy ≥ 1.20.0
- pyyaml ≥ 5.1
- torch ≥ 1.9.0
- matplotlib ≥ 3.3.0
- optuna ≥ 3.0.0

## Verifying Installation

After installation, verify that RLlama is installed correctly by running:

```python
import rllama
print(f"RLlama version: {rllama.__version__}")
```

