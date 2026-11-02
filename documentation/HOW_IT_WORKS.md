# How mkdocstocanvas Works

This document explains the mechanics behind the tool: the upload cache, incremental uploads, ownership markers, safe deletes, asset deduplication and PDF generation.

For a plain list of what your markdown can contain (math, Excel, admonitions…), see [FEATURES.md](FEATURES.md). For setup instructions, see the [README](../README.md).

## Table of Contents

- [The upload cache](#the-upload-cache)
- [Incremental page uploads](#incremental-page-uploads)
- [Ownership markers and safe deletes](#ownership-markers-and-safe-deletes)
- [Asset deduplication](#asset-deduplication)
- [Recovering a lost cache](#recovering-a-lost-cache)
- [PDF generation](#pdf-generation)
- [How labs are recognized](#how-labs-are-recognized)

## The upload cache

All upload state lives in `.canvas_upload_state.json` (gitignored) in the project root:

- **Pages**: for each uploaded markdown file — page title, MD5 hash of the source, the Canvas URL slug, the Canvas page URL, a `mentioned_by` index (which other files link to it) and a snapshot of the Canvas URLs its `.md` links resolved to at upload time.
- **Files**: for each uploaded image/asset — content hash and Canvas URL, so unchanged assets are never re-uploaded.
- **Labs**: the Canvas assignment URL for each lab file.

The cache is what makes re-runs cheap and keeps links between pages pointing at the right Canvas URLs. It is updated after every upload run.

## Incremental page uploads

`upload-pages` only sends what actually changed:

1. **Changed files** (MD5 differs from the cache, or new) are uploaded.
2. **Unchanged files** are skipped and reported as "Skipped".
3. A page is also re-uploaded when any of its resolved `.md` links now point to a different Canvas URL than at its last upload (e.g. a link target changed its slug).
4. If an uploaded page's slug changed, pages that mention it are re-uploaded in a second pass so their links stay correct.

Updates go through Canvas's *create-or-update* semantics: existing pages are matched by slug (falling back to a unique title match), so Canvas page URLs never churn between runs.

## Ownership markers and safe deletes

Every page, lab assignment and the syllabus body uploaded by the tool carries an invisible marker at the top:

```html
<span class="mkdocstocanvas-marker" data-rel="lectures/01-intro.md" data-md5="ab12cd…"></span>
```

(Canvas strips HTML comments from page bodies, so a plain comment cannot be used — data attributes on a span survive.)

The marker identifies content as **tool-managed** and powers:

- **Scoped deletes** — `delete-pages` / `delete-modules` only remove marker-carrying (or cache-matched) content. Anything you created manually in Canvas (quizzes, pages, modules) is left untouched unless you pass `--all`.
- **Cache rebuilding** — `rebuild-cache` re-derives the cache from Canvas by matching markers back to local files, so a lost cache never leads to duplicate pages.
- **Re-upload detection** — the marker's `data-md5` lets the tool verify the Canvas body still matches the local file.

Modules have no body to carry a marker; they are matched by the cached `canvas_module_id`, falling back to exact nav-section names. Lab assignments are matched via the cache first, then by the `Lab <number>` name pattern.

> **Note**: content uploaded with older versions of the tool has no marker. `delete-pages` treats it as foreign until it has been re-uploaded once (`mkdocstocanvas upload-pages --force` re-stamps it).

## Asset deduplication

Before uploading an image or file, the tool checks Canvas for an identical file (by content hash, falling back to filename + size) and reuses it instead of creating a duplicate. This happens **regardless of `--force`** — `--force` only skips the page-level cache and the local asset cache entry; it can never create duplicate files.

## Recovering a lost cache

If `.canvas_upload_state.json` is lost or deleted (new machine, cleaned repo):

```bash
mkdocstocanvas rebuild-cache
```

matches pages, labs and the syllabus back to local files via their ownership markers, and assets via content hashes. The command also runs automatically at the start of `upload-pages` / `upload-labs` whenever the cache is empty — losing the cache never causes duplicate pages or files.

## PDF generation

PDFs are produced by the `mkdocs-page-pdf` plugin during `mkdocs build` (one PDF per page) and collected into `pdf/` by the `pdf` command. When uploading modules with `--add-pdf`, each PDF is matched to its page by filename, so keep markdown and PDF filenames consistent (they are generated from the same pages automatically).

## How labs are recognized

A markdown file is treated as a **lab** (uploaded as a Canvas *assignment*) when both are true:

- it is directly inside `docs/labs/`, and
- its filename starts with `lab` (case-insensitive), e.g. `lab1.md`, `Lab2-intro.md`.

Any other file inside `docs/labs/` (e.g. `notes.md`) is uploaded as a **regular page** and included in modules like any other page — nothing is silently skipped.

Two related rules:

- A top-level nav section whose name starts with "Lab" (e.g. "Labs") is **skipped when creating modules** — labs are managed as assignments, not module items.
- `delete-labs` identifies Canvas assignments primarily via the upload cache, falling back to the `Lab <number>` name pattern for assignments uploaded without the cache. Labs renamed on Canvas are still found and deleted as long as they were uploaded from this machine.
