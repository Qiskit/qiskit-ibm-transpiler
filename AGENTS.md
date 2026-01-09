# AGENTS.md

This file provides context for AI coding agents working on the `qiskit-ibm-transpiler` project.

## Project Overview

A Python library for IBM Quantum's AI-powered transpiler service, enabling quantum circuit optimization through reinforcement learning. Supports both local and cloud-based transpilation modes.

## Setup Commands

```bash
# Install dependencies (Python 3.10–3.13 supported, 3.11 recommended)
uv sync --group dev

# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Build documentation
uv sync --group docs
uv run scripts/docs
```

## Code Style

- **Formatter**: Black with line length 88
- **Linter**: Ruff with rules `E` (errors) and `I` (imports)
- **Target Python**: 3.11
- **Docstrings**: reStructuredText format for Sphinx
- **Imports**: Use absolute imports, sorted by isort rules

## Testing

- Tests are in `tests/` directory using pytest
- Run specific tests: `uv run pytest tests/test_file.py -k "test_name"`
- Markers available:
  - `disable_monkeypatch`: Disable env var mocking
  - `e2e`: End-to-end tests (requires `RUN_E2E=1`)
- Always add or update tests for code changes

## Project Structure

```
qiskit_ibm_transpiler/     # Main package
├── ai/                    # AI transpiler passes (routing, synthesis, collection)
├── wrappers/              # Base classes and API wrappers
└── transpiler_service.py  # Main TranspilerService class

tests/                     # Test suite
docs/                      # Sphinx documentation
examples/                  # Jupyter notebook examples
release-notes/             # Towncrier release notes
└── unreleased/           # Pending release notes
```

## Release Notes

User-facing changes require a release note in `release-notes/unreleased/`:
- File format: `<github-number>.<type>.rst`
- Types: `feat`, `upgrade`, `deprecation`, `bug`, `other`
- Use RST syntax with cross-references

## PR Guidelines

- Run tests and linting before committing
- Include release notes for user-facing changes
- Follow existing code patterns and naming conventions
