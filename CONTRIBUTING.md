# Contributing

Thank you for helping improve GaussWeave.

## Development setup

Use Linux or WSL 2 with Python 3.12 and uv 0.11.x:

```bash
git clone https://github.com/asguinea/gaussweave.git
cd gaussweave
uv sync --frozen --extra dev
```

Before opening a pull request, run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

Run `uv run ruff format .` to apply formatting. GPU changes should also run the
relevant marked tests with the `gpu` extra installed.

## Changes

- Keep changes focused and include regression tests for behavior changes.
- Preserve deterministic seeds, canonical serialization, and artifact provenance.
- Update schemas and examples together when changing a file format.
- Do not commit datasets, model weights, generated results, credentials, or files
  whose redistribution terms are unclear.
- Use clear commit messages and describe verification performed in the pull request.

By contributing, you agree that your contributions are licensed under Apache-2.0.
All participants must follow the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
