from setuptools import setup, find_packages

setup(
    name="rllama",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.28.1",
        "trl>=0.4.1",
        "peft>=0.3.0",
        "gymnasium",
        "accelerate>=0.19.0",
        "bitsandbytes>=0.37.2",
        "tqdm",
        "numpy"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A reinforcement learning framework using LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ch33nchan/RLlama",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)