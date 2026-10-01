# AGENTS.md

## Commands

```bash
make check            # lint + typecheck + test (run before committing)
make lint             # ruff check
make format           # ruff check --fix + ruff format
make typecheck        # pyright
make test             # pytest
uv run pytest tests/test_math.py -k roundtrip   # single test
```

- Everything runs through `uv run` (uv-managed project; deps in `pyproject.toml`, locked in `uv.lock`).
- The CLI entrypoint is `mkdocstocanvas` (Typer, defined in `src/mkdocstocanvas/cli.py`). The Makefile wraps it via `$(call canvas, ...)` — make targets are dev shortcuts only; user-facing docs (README) show the CLI commands directly.

## Live API caution

`mkdocstocanvas upload-*` / `delete-*` operate on a **real Canvas course** using the token in `.env` (gitignored; template in `.env.example`). Never add automated tests that call the Canvas API — all tests must stay offline (mock `requests` / use the existing fakes in `tests/test_canvas.py`).

## Architecture

```
cli.py (Typer commands)
  → api/canvas.py      CanvasUploader: thin requests wrapper, one method per endpoint
  → uploaders/         base.py = ContentUploader (shared cache/assets/link-resolution,
                       console + progress + delete-flow helpers); pages.py, modules.py,
                       labs.py subclass or delegate to it
  → processing/        markdown → html pipeline; math.py LaTeX sentinel protection;
                       excel.py {{ render_excel_sheet(...) }} macro
  → models/page.py     MarkdownPage (title/content/md5)
  → utils/cache.py     .canvas_upload_state.json (gitignored, atomic writes)
  → utils/marker.py    ownership marker (make/parse)
```

- Incremental uploads are driven by `.canvas_upload_state.json` (MD5 + Canvas slug per file). Delete it to force a full re-upload — or better, `mkdocstocanvas rebuild-cache` / let the auto-rebuild recover it from Canvas markers.
- Internal `.md` links and images are rewritten to Canvas URLs during `_prepare_content` — the cache must be in sync or links break.

## Ownership markers & scoped deletes (important contract)

- Every page/lab/syllabus body gets an invisible marker `<span class="mkdocstocanvas-marker" data-rel="…" data-md5="…"></span>` (prepended in `ContentUploader._prepare_content`). Canvas **strips HTML comments** from page bodies, so a comment cannot be used — data attributes on a span survive. Changing the marker format breaks `rebuild-cache` and scoped deletes — update both sides together; old-format markers are simply not recognized (a full re-upload re-stamps them).
- `delete-pages` / `delete-modules` are **managed-only** by default (marker or cache/nav-section match); `--all` deletes everything. Don't "simplify" this back to delete-all.
- Modules have no body to carry a marker: they are matched by cached `canvas_module_id` (stored under the cache's `modules` key) or by exact nav-section name.
- Assets are deduplicated against Canvas files by content hash, falling back to filename+size (`find_existing_file`, lazily indexed per client; Canvas does not report md5 for course files) — **regardless of `--force`**. `--force` only skips the page-level cache and the local asset cache entry; it can never create duplicate files.

## Conventions that differ from defaults

- **Broad `except Exception` is deliberate** in per-item loops (uploads, cell colors, PDF matching): failures are reported and the run continues. Ruff ignores `BLE001` for this reason (see `[tool.ruff.lint]` in `pyproject.toml` — also `DTZ005`, `PLW1510`).
- **Auth/404 errors must not be conflated**: `get_existing_page` returns `None` only for 404 and *raises* on anything else (e.g. 401), because `create_or_update_page` would otherwise POST duplicate pages. Keep that contract.
- All HTTP goes through `_TimeoutSession` (30s default timeout) — never call `requests` directly.
- Python floor is 3.10 (`requires-python`); ruff/pyright target `py310`.

## Known quirks / issues

- **Lab detection has exactly two rules** (consolidated; don't add more): `uploaders/labs.py::is_lab_rel_path` decides which files are labs (`labs/` dir + name starts with `lab`, case-insensitive); module *sections* are skipped by section name (`startswith("lab")` in `modules.py`, intentional). `delete-labs` matches cached assignment IDs first, `_LAB_ASSIGNMENT_PATTERN` only as fallback.
- `utils/config.py` raises `typer.Exit` from library code (works only inside the CLI app).
- The math protection uses `§...§` sentinels (e.g. `§UNDERSCORE§`); when touching `processing/math.py`, verify with the round-trip tests in `tests/test_math.py`.
