from setuptools import setup, find_packages
import os

# Read README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Composable reward engineering framework for reinforcement learning"

# Read requirements
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
if os.path.exists(requirements_path):
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh 
                       if line.strip() and not line.startswith("#") and not line.startswith("-")]
else:
    requirements = [
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ]

setup(
    name="rllama",
    version="0.1.1",  # Increment version
    author="RLlama Team",  # Update author
    author_email="contact@rllama.dev",  # Update email
    description="Advanced composable reward engineering framework for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ch33nchan/RLlama",  # Update URL
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",  # Update status
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",  # Add 3.11
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "numpy>=1.21.0", 
        "pandas>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "trl": ["trl>=0.4.0", "accelerate>=0.20.0"],
        "optimization": ["optuna>=3.0.0", "scipy>=1.9.0"],
        "dashboard": ["streamlit>=1.25.0", "plotly>=5.15.0"],
        "sb3": ["stable-baselines3>=2.0.0"],
        "all": [
            "trl>=0.4.0", "accelerate>=0.20.0",
            "optuna>=3.0.0", "scipy>=1.9.0", 
            "streamlit>=1.25.0", "plotly>=5.15.0",
            "stable-baselines3>=2.0.0"
        ],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "isort>=5.10.0", "flake8>=4.0.0"],
        "docs": ["mkdocs>=1.4.0", "mkdocs-material>=8.5.0"],
    },
    entry_points={
        "console_scripts": [
            "rllama-dashboard=run_dashboard:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="reinforcement-learning,reward-engineering,llm,rlhf,machine-learning,ai",
    project_urls={
        "Bug Reports": "https://github.com/ch33nchan/RLlama/issues",
        "Source": "https://github.com/ch33nchan/RLlama",
        "Documentation": "https://github.com/ch33nchan/RLlama/blob/main/docs1.md",
        "PyPI": "https://pypi.org/project/rllama/",
    },
)