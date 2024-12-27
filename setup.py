#!/usr/bin/env python
"""The setup script."""
from setuptools import find_packages
from setuptools import setup
from createllm import __version__

with open("README.md", encoding="utf8") as readme_file:
    readme = readme_file.read()

with open("requirements.txt", "r") as file:
    requirements = [r for r in file.readlines() if len(r) > 0]

setup_requirements = []

test_requirements = ["pytest"].extend(requirements)

setup(
    author="Khushal Jethava",
    author_email="khushaljethwa14@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: End Users/Desktop ",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="A Python package that enables users to create and train their own Language Learning Models (LLMs) from scratch using custom datasets. This package provides a simplified approach to building, training, and deploying custom language models tailored to specific domains or use cases.",
    include_package_data=True,
    install_requires=requirements,
    keywords="createllm",
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    name="createllm",
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/khushaljethava/createllm",
    version=__version__,
    zip_safe=False,
)