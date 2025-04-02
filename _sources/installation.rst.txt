============
Installation
============

Requirements
-----------

createllm requires Python 3.7 or higher and the following dependencies:

* PyTorch >= 2.0.0
* tqdm >= 4.65.0
* numpy >= 1.24.0
* dataclasses >= 0.6
* typing-extensions >= 4.5.0

Installation Methods
------------------

1. Install from PyPI
~~~~~~~~~~~~~~~~~~~

The recommended way to install createllm is using pip:

.. code-block:: bash

    pip install createllm

2. Install from Source
~~~~~~~~~~~~~~~~~~~~~

To install the latest development version:

.. code-block:: bash

    git clone https://github.com/khushaljethava/createllm.git
    cd createllm
    pip install -e .

3. Install with Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development or testing purposes, install with additional dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

This will install additional packages:
* coverage - for testing coverage
* mypy - for type checking
* pytest - for running tests
* ruff - for linting

Verifying Installation
--------------------

After installation, you can verify that createllm is installed correctly:

.. code-block:: python

    import createllm
    print(createllm.__version__)  # Should print the installed version

GPU Support
----------

createllm automatically detects and uses CUDA if available. To ensure GPU support:

1. Install PyTorch with CUDA support:

.. code-block:: bash

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

2. Verify CUDA availability:

.. code-block:: python

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~

1. ImportError: No module named 'createllm'
   - Ensure you're using the correct Python environment
   - Try reinstalling the package

2. CUDA not available
   - Check if PyTorch is installed with CUDA support
   - Verify your NVIDIA drivers are up to date

3. Memory Issues
   - Reduce batch size in GPTTrainer
   - Use a smaller model configuration
   - Enable gradient checkpointing

Getting Help
~~~~~~~~~~~

If you encounter any issues:

* Check the `GitHub issues <https://github.com/khushaljethava/createllm/issues>`_
* Open a new issue with:
  - Your Python version
  - PyTorch version
  - CUDA version (if applicable)
  - Full error message
  - Steps to reproduce