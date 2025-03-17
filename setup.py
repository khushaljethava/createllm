from setuptools import setup, find_packages

setup(
    name="createllm",
    version="0.1.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "dataclasses>=0.6",
        "typing-extensions>=4.5.0"
    ],
    python_requires=">=3.7",
) 