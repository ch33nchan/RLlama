from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rllama",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Composable reward engineering framework for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rllama",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "isort>=5.10.0", "flake8>=4.0.0"],
        "docs": ["mkdocs>=1.4.0", "mkdocs-material>=8.5.0"],
        "dashboard": ["streamlit>=1.25.0", "plotly>=5.15.0"],
        "optimization": ["optuna>=3.0.0"],
        "all": ["streamlit>=1.25.0", "plotly>=5.15.0", "optuna>=3.0.0"]
    },
    entry_points={
        "console_scripts": [
            "rllama-dashboard=rllama.dashboard.streamlit_app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)