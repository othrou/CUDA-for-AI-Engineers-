# Contributing to CUDA for AI Engineers

Thanks for your interest in contributing to this project. The goal of this repository is to make CUDA and PyTorch optimization more accessible through practical examples, benchmarks, and learning materials.

## Ways to contribute

You can help by:

- improving or fixing notebook examples
- adding new CUDA/PyTorch optimization examples
- improving benchmark scripts and performance analysis
- fixing documentation or setup instructions
- reporting bugs or unclear examples
- proposing new learning modules or exercises

## Before you start

Please read the repository documentation and inspect the existing notebooks and examples to understand the current structure and style before proposing a change.

When opening an issue or PR, include:

- a clear description of the problem or idea
- relevant context, such as the notebook, script, or benchmark involved
- environment details (OS, CUDA version, PyTorch version, GPU model)
- steps to reproduce if reporting a bug

## Local setup

1. Fork the repository.
2. Clone your fork locally.
3. Create a feature branch:

   ```bash
   git checkout -b feature/my-improvement
   ```

4. Install the project dependencies required for your area of work.

5. Run the relevant examples or checks before submitting a PR.

## Contribution guidelines

### Notebooks

- Keep notebooks readable and well-structured.
- Add concise explanations near key code cells.
- Include comments where the CUDA or optimization logic is non-obvious.
- Avoid large outputs in committed notebooks unless they are intentionally part of the example.
- If a notebook depends on specific hardware or drivers, document that clearly.

### Benchmarks and performance work

- Include reproducible setup instructions.
- Mention the environment used for benchmarking.
- Explain what metric is being measured and why.
- Prefer clear, interpretable results and benchmarking methodology.

### Documentation

- Use clear headings and examples.
- Keep instructions practical and copy-paste friendly.
- Add notes for common issues, especially around CUDA setup and compatibility.

## Pull request checklist

Before submitting a PR, please confirm:

- the change is scoped and focused
- the affected notebook, script, or documentation is clearly described
- relevant instructions or examples were updated
- the code or notebook runs as expected in a clean environment
- the PR includes a concise summary and testing notes

## Code of conduct

Please be respectful and constructive in discussions, issues, and pull requests. We welcome contributions from beginners and experienced developers alike.

## Questions?

If you are unsure whether a contribution fits the project, open an issue first and start a discussion before writing a large patch.

Thank you for helping improve CUDA for AI Engineers.
