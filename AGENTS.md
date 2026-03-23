# Agent Instructions

This document provides instructions for AI agents when working with code in this repository.

## Project Overview

DEPENDENT-CENSORING-DETECTION contains methods for detecting dependent censoring in survival analysis datasets. The repository includes:

- An under development Python package for dependent censoring detection
- Synthetic data generation scripts
- Semi-synthetic data generation scripts
- Real data processing scripts
- Experiment scripts for running detection on various datasets

## Key Commands

### Environment Setup and Package Management

Environment is managed with `uv` and dependencies are listed in `requirements.txt`, `env.yaml` and `pyproject.toml`. 

To activate the environment:

```bash
source .venv/bin/activate
```

### Running Experiments

```bash
python -m experiments.run_exp --dataset <DATASET_NAME> --n-trials <NUM_TRIALS> --seed <RANDOM_SEED>
```

## Development Workflow

1. Add new tests for any new features in the `tests/` directory. Or update existing tests if modifying existing functionality.
2. Run tests: `pytest --all` (includes integration tests)
3. Format code: `isort .` then `black .`
4. If adding new features, update both `requirements.txt` and `env.yaml`
5. Ensure sufficient test coverage for new features

Other style guidelines:

- Type hints are used throughout the codebase. Ensure that all new code includes appropriate type annotations.
- Public APIs must have docstrings following the Google style guide. Internal functions should have docstrings as needed for clarity.
- Line length limit: 120 characters (Black config)

## Development Philosophy

1. **Simplicity**: Write simple, straightforward code
2. **Readability**: Make code easy to understand
3. **Performance**: Consider performance without sacrificing readability
4. **Maintainability**: Write code that's easy to update
5. **Testability**: Ensure code is testable
6. **Reusability**: Create reusable components and functions
7. **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

1. **Early Returns**: Use to avoid nested conditions
2. **DRY Code**: Don't repeat yourself
3. **Minimal Changes**: Only modify code related to the task at hand
4. **Function Ordering**: Define composing functions before their components
5. **TODO Comments**: Mark issues in existing code with "TODO:" prefix
6. **Build Iteratively**: Start with minimal functionality and verify it works before adding complexity
7. **Run Tests**: Test your code frequently with realistic inputs and validate outputs
8. **Build Test Environments**: Create testing environments for components that are difficult to validate directly
9. **Functional Code**: Use functional and stateless approaches where they improve clarity
10. **Clean logic**: Keep core logic clean and push implementation details to the edges
11. **File Organization**: Balance file organization with simplicity - use an appropriate number of files for the project scale

## First Principles

Please think using first principles. You can't always assume I know exactly what I want and how to get it. Be deliberate, start with the original needs and problems, and if the motivations and goals are unclear, stop and discuss them with me

## Solution Guidelines

When you are required to provide a modification or refactoring solution, it must comply with the following guidelines:

- No compatibility or patching solutions are allowed.
- No over-design is allowed; maintain the shortest path to implementation and do not violate the first requirement.
- No solutions beyond the reqirements I have provided are allowed, such as fallback or degradtion solutions.
- You must ensure the logical correctness of the solution; it must undergo end-to-end logical verification.
