import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rllama",
    version="0.1.0",
    author="Ch33nchan",
    author_email="your.email@example.com",
    description="A lightweight reinforcement learning library with focus on negative reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ch33nchan/rllama",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "gymnasium>=0.26.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "wandb>=0.12.0",
    ],
    extras_require={
        "mujoco": ["mujoco>=2.2.0", "dm_control>=1.0.0"],
        "sb3": ["stable-baselines3>=1.6.0"],
        "lerobot": ["lerobot>=0.1.0"],
        "unity": ["mlagents>=0.28.0"],
        "dev": [
            "pytest>=6.2.5",
            "black>=21.9b0",
            "isort>=5.9.3",
            "flake8>=3.9.2",
            "sphinx>=4.2.0",
        ],
    },
)