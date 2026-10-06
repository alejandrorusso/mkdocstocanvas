# Contributing

Contributions are welcome!

## Development setup

```bash
git clone <your-repo-url>
cd mkdocstocanvas
uv sync
cp mkdocs.example.yml mkdocs.yml    # local working config (gitignored)
cp -r docs-example docs             # local course content (gitignored)
cp .env.example .env                # Canvas credentials (gitignored)
uv run mkdocstocanvas --help        # verify the install
```

`uv sync` installs the CLI into the project venv (`.venv/`, editable — it always runs your working-tree code). That venv is not on your `PATH`, so prefix commands with `uv run` or `source .venv/bin/activate` first. For a bare `mkdocstocanvas` that works anywhere, install it as a standalone tool instead:

```bash
uv tool install --editable .   # follows your working-tree code while developing
# frozen snapshot instead:     uv tool install .
```

## Commands

```bash
make check       # lint + typecheck + test (run before committing)
make lint        # ruff check
make format      # ruff check --fix + ruff format
make typecheck   # pyright
make test        # pytest
```

Everything runs through `uv run` under the hood.

## Rules

- **Tests must stay offline.** Never add tests that call the Canvas API — all network access must be mocked (`requests` or the existing fakes in `tests/test_canvas.py`). The real upload/delete commands operate on a live Canvas course.
- **Don't break the marker contract.** Pages, labs and the syllabus carry an invisible ownership marker that powers scoped deletes and `rebuild-cache`. If you change the marker format, update both sides together (see `documentation/HOW_IT_WORKS.md`).
- **Broad `except Exception` in per-item loops is deliberate** — failures are reported and the run continues.

## Submitting

1. Fork the repository
2. Create a feature branch
3. Make your changes and run `make check`
4. Submit a pull request

## License

By contributing, you agree that your contributions will be licensed under the Mozilla Public License 2.0 (MPL-2.0).
