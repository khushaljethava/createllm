.. include:: ../CONTRIBUTING.rst

Contributing
============

Thank you for your interest in contributing to createllm! This document provides guidelines and instructions for contributing to the project.

Development Setup
---------------

1. Fork and Clone
~~~~~~~~~~~~~~~

First, fork the repository and clone your fork:

.. code-block:: bash

    git clone https://github.com/your-username/createllm.git
    cd createllm

2. Install Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the package in development mode with all dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

This will install:
* coverage - for testing coverage
* mypy - for type checking
* pytest - for running tests
* ruff - for linting

3. Set Up Pre-commit Hooks
~~~~~~~~~~~~~~~~~~~~~~~

Install pre-commit hooks to ensure code quality:

.. code-block:: bash

    pre-commit install

Development Guidelines
-------------------

1. Code Style
~~~~~~~~~~~

* Follow PEP 8 guidelines
* Use type hints for all function parameters and return values
* Keep functions focused and single-purpose
* Write clear and descriptive variable names
* Add docstrings for all public functions and classes

2. Testing
~~~~~~~~

* Write tests for new features
* Ensure all tests pass before submitting PR
* Maintain or improve test coverage
* Include both unit and integration tests

3. Documentation
~~~~~~~~~~~~~

* Update documentation for new features
* Add docstrings following Google style
* Include examples in docstrings
* Update README.md and README.rst if needed

4. Git Workflow
~~~~~~~~~~~~

* Create a new branch for each feature/fix
* Use descriptive branch names
* Write clear commit messages
* Keep commits focused and atomic

Making Changes
------------

1. Create a New Branch
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git checkout -b feature/your-feature-name
    # or
    git checkout -b fix/your-fix-name

2. Make Your Changes
~~~~~~~~~~~~~~~~~

* Write your code
* Add tests
* Update documentation
* Run tests and linting

3. Commit Your Changes
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git add .
    git commit -m "Description of your changes"

4. Push to Your Fork
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git push origin feature/your-feature-name

5. Create a Pull Request
~~~~~~~~~~~~~~~~~~~~~

* Go to the GitHub repository
* Click "New Pull Request"
* Select your branch
* Fill in the PR template
* Submit the PR

Pull Request Guidelines
--------------------

1. Title and Description
~~~~~~~~~~~~~~~~~~~~~

* Use clear and descriptive titles
* Fill in the PR template completely
* Include:
  - Purpose of changes
  - Implementation details
  - Testing done
  - Screenshots (if applicable)

2. Code Review
~~~~~~~~~~~

* Address all review comments
* Keep the PR focused and manageable
* Update documentation as needed
* Ensure CI checks pass

3. Merging
~~~~~~~~~

* Get approval from maintainers
* Ensure all checks pass
* Keep commits clean and organized
* Update version if needed

Testing Guidelines
---------------

1. Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

    # Run all tests
    pytest

    # Run with coverage
    pytest --cov=createllm

    # Run specific test file
    pytest tests/test_model.py

2. Writing Tests
~~~~~~~~~~~~~

* Test both success and failure cases
* Use appropriate fixtures
* Mock external dependencies
* Test edge cases

3. Test Coverage
~~~~~~~~~~~~~

* Maintain or improve coverage
* Focus on critical paths
* Test public API thoroughly
* Include integration tests

Documentation Guidelines
--------------------

1. Code Documentation
~~~~~~~~~~~~~~~~~~

* Add docstrings to all public functions/classes
* Include type hints
* Provide usage examples
* Document exceptions

2. API Documentation
~~~~~~~~~~~~~~~~~

* Update API reference
* Add new examples
* Document breaking changes
* Keep documentation up to date

3. User Documentation
~~~~~~~~~~~~~~~~~~

* Update user guides
* Add new tutorials
* Document new features
* Include troubleshooting guides

Release Process
-------------

1. Version Bumping
~~~~~~~~~~~~~~~

* Update version in:
  - pyproject.toml
  - setup.py
  - __init__.py
* Follow semantic versioning

2. Changelog
~~~~~~~~~~

* Update CHANGELOG.md
* Document breaking changes
* List new features
* Note bug fixes

3. Release Notes
~~~~~~~~~~~~~

* Write clear release notes
* Highlight major changes
* Include migration guides
* Document deprecations

Getting Help
----------

If you need help:

* Check existing issues
* Ask in discussions
* Contact maintainers
* Join the community

Thank You
--------

Thank you for contributing to createllm! Your contributions help make the project better for everyone.
