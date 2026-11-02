# Refactoring TODO

Organized by priority (leverage × risk-of-regression). Check items off as you go;
commit after each phase so each is independently shippable.

See `AGENTS.md` for the full architectural rationale behind each item.

## Phase 1 — Fix broken workflows & clean stale artifacts (low risk, unblocking)

- [ ] Rewrite `Makefile` to call the `mkdocstocanvas` CLI instead of the deleted
      `scripts/upload_*.py` files. Every target currently references scripts that
      don't exist.
- [ ] Rewrite `src/rebuild_canvas.sh` to call `mkdocstocanvas upload-all` (or
      delete it — the CLI's `upload-all` command already does the same thing).
- [ ] Decide on the PDF pipeline: either implement the missing
      `scripts/rename_pdfs_with_order.py` or remove references to it from the
      Makefile, README, and `ModuleUploader._normalize_stem`. Currently two
      divergent PDF pipelines exist and neither fully works.
- [ ] Delete `scripts/__pycache__/canvas_processing.cpython-311.pyc` (orphaned
      bytecode from a deleted module).
- [ ] Delete committed build artifacts: `dist/` (74 MB sdist + wheel).
- [ ] Fix README "Project Structure" section — still lists `scripts/upload_*.py`
      which no longer exist.
- [ ] Fix README Python version: says 3.8, `pyproject.toml` says `>=3.10` and
      uses `str | None`. README is wrong.
- [ ] Pick one config mechanism: `tokens.sh` (sourced) vs `.env` (python-dotenv).
      Document the chosen one; remove or deprecate the other.
- [ ] Make `create_client` raise a clear error when `CANVAS_BASE_URL` is unset
      instead of silently defaulting to `canvas.instructure.com` (causes
      confusing 404s at the wrong host).
- [ ] Rotate the `CANVAS_API_TOKEN` in `.env` if it has ever been shared, and
      consider documenting an `XDG_CONFIG_HOME`-based location for secrets.

## Phase 2 — Add tests, lint, typecheck (before refactoring)

- [ ] Add `[tool.ruff]` config to `pyproject.toml`.
- [ ] Add `mypy` or tighten `pyright` config; remove the `# pyright: ignore`
      sprinkled in `pages.py` once types are honest.
- [ ] Add `pytest` + dev extras (`[project.optional-dependencies].dev`).
- [ ] Add a CI skeleton (GitHub Actions: ruff + mypy + pytest on push).
- [ ] Write golden/master tests for `process_markdown_to_html` on sample
      inputs — captures current behaviour so refactors don't silently regress.
- [ ] Write unit tests for `_resolve_page_links` and `_process_markdown_assets`
      (regex-heavy, regression-prone).
- [ ] Write unit tests for `robust_math_protection` / `restore_protected_math_content`
      with edge cases: nested `$$`, escaped `\$`, content containing `§`.
- [ ] Write property-based tests (Hypothesis) for the markdown rewriting layer.

## Phase 3 — Split `CanvasUploader` + pagination + typed errors

- [ ] Extract a `_paginate(url, params)` generator; remove the 3× copy-pasted
      Link-header loop in `list_pages`/`list_modules`/`list_assignments`.
- [ ] Define typed exceptions: `CanvasAuthError`, `CanvasNotFoundError`,
      `CanvasAPIError`, `CanvasRateLimitError`.
- [ ] Split `api/canvas.py` into focused clients:
  - [ ] `api/pages.py`       → `PagesAPI`
  - [ ] `api/modules.py`     → `ModulesAPI`
  - [ ] `api/files.py`       → `FilesAPI`
  - [ ] `api/assignments.py` → `AssignmentsAPI`
  - [ ] `api/courses.py`     → `CoursesAPI` (syllabus + connection test)
- [ ] Replace the `try/except → err_console.print → return None/False/[]`
      pattern: raise typed exceptions from the API layer.
- [ ] Return `None` from API methods only for genuine "not found" cases,
      never for failures.
- [ ] Add retry-with-backoff (exponential + jitter) for 429/5xx/timeouts in
      the API layer.
- [ ] Catch domain exceptions at the CLI top level; map to exit codes and
      user-facing messages.
- [ ] Make `ContentUploader`/`PageUploader`/etc. take the new per-resource
      clients instead of the god class.

## Phase 4 — Split `pages_cache` + fix lab/page collision

- [ ] Define typed dataclasses for cache entries (`PageCacheEntry`,
      `LabCacheEntry`, `FileCacheEntry`) instead of `dict[str, dict]`.
- [ ] Split `pages_cache` into `pages` / `labs` / `files` sub-caches.
- [ ] Fix `labs.py:67` writing a partial `{"canvas_url": url}` entry that can
      clobber a real page entry on path collision.
- [ ] Stop `_collect_mentions` from nuking `mentioned_by` on every entry
      before rebuilding (`pages.py:329`) — if the run aborts, the index is gone.
- [ ] Prune `mentioned_by` entries for deleted files (no pruning currently).

## Phase 5 — Replace network-per-page existence check

- [ ] Replace the per-page `get_existing_page` call in `_needs_upload`
      (`pages.py:434`) with a single `list_pages()` call + a slug-set lookup
      (the pattern already used in `modules.py`).

## Phase 6 — Non-destructive module upload

- [ ] Re-architect `ModuleUploader.upload_all` to diff modules by name and
      update in place instead of delete-all-then-recreate.
- [ ] Only delete modules that have vanished from `mkdocs.yml`.
- [ ] Preserve manually-added module items (per `todo.md`: "make sure to not
      delete manually added stuff").

## Phase 7 — Pure content pipeline

- [ ] Make `_process_markdown_assets(md_page, content) -> str` pure — no
      mutation of `md_page.content`.
- [ ] Make `_resolve_page_links(md_page, content) -> str` pure.
- [ ] Make `_prepare_content` compose the pure functions and return HTML.
- [ ] Remove the "reset content on each upload" hack in `base.py:90`.

## Phase 8 — Unify lab predicate + extract `SyllabusUploader`

- [ ] Define one canonical "is this a lab?" predicate; replace
      `find_lab_files`, `_LAB_PATTERN`, and the `parent.name == "labs"` check
      in `_pages_to_upload`.
- [ ] Extract `SyllabusUploader` from `PageUploader` — it's a different
      Canvas resource (`upload_syllabus`, not `create_or_update_page`).
- [ ] Remove the scattered `_is_syllabus_page` branches from `pages.py`.

## Phase 9 — Safer markdown & HTML processing

- [ ] Rewrite `_MD_LINK_PATTERN` / `_ASSET_PATTERN` as a python-markdown
      treeprocessor/preprocessor, or at minimum skip fenced code spans,
      inline code, and HTML comments before substituting.
- [ ] Replace `add_pygments_inline_styles` (string replacement on the whole
      document) with `HtmlFormatter(noclasses=True)` or an HTML parser
      (BeautifulSoup).
- [ ] Same for `style_admonitions` — use an HTML parser, not string replace.
- [ ] Delete dead math functions (`process_math_block`, `wrap_standalone_align`
      in `math.py`, `protect_math_content`/`restore_math_content`,
      `preserve_linebreaks_in_math`) — only `robust_math_protection` is called.
- [ ] Replace `§UNDERSCORE§`/`§ASTERISK§`/`§TILDE§` sentinels in math
      protection with a token table (placeholder ID → content dict, restore by ID).
- [ ] Unify the two `wrap_standalone_align` definitions (one in `markdown.py:27`,
      one in `math.py:31`).

## Phase 10 — Layering / dedup / hygiene

- [ ] Extract one `delete_resources(client, list_fn, delete_fn, id_key,
      name_key, *, force, label)` helper; collapse
      `delete_all_pages`/`delete_all_modules`/`delete_all_labs`.
- [ ] Move `_print_summary` variants into a `reporting.py` util.
- [ ] Parse `mkdocs.yml` once and pass the dict; remove the triple-parse in
      `parse_mkdocs_nav` / `parse_mkdocs_nav_sections` / `parse_syllabus_nav_file`.
- [ ] Make `MarkdownPage` construction a classmethod factory that can fail;
      make `title`/`md5` lazy or `Optional` with honest types.
- [ ] Fix `labs.py:50` assuming `root_path=lab_file.parent.parent` — breaks
      if labs aren't exactly one level under docs.
- [ ] Build a normalized title→page lookup dict once in `ModuleUploader`
      instead of the O(n) per-call regex in `_find_page`.
- [ ] Make `ModuleUploader` extend `ContentUploader`; reuse `pages_cache`
      instead of resolving module→page links by title via `list_pages()`.
- [ ] Replace `upload_all`'s 3-pass labs→pages→labs with a topological sort
      over the mention graph (each page uploaded once).
- [ ] Introduce `logging` + `RichHandler` instead of module-level `Console`
      instances everywhere.
