# Content & Features Reference

What your markdown can contain when publishing through mkdocstocanvas, and how to organise it.

For how the tool behaves internally (cache, markers, deletes, PDFs), see [HOW_IT_WORKS.md](HOW_IT_WORKS.md). For setup, see the [README](../README.md).

## Table of Contents

- [Supported content types](#supported-content-types)
- [Labs](#labs)
- [Mathematical formulas](#mathematical-formulas)
- [Code blocks](#code-blocks)
- [Admonitions](#admonitions)
- [Images](#images)
- [Tables](#tables)
- [Excel spreadsheets](#excel-spreadsheets)
- [Adding new content](#adding-new-content)
- [Customizing the theme](#customizing-the-theme)

## Supported content types

| Content | Becomes in Canvas |
|---|---|
| Regular pages | Canvas wiki pages (nav → modules) |
| `docs/labs/lab*.md` | Canvas assignments (see [Labs](#labs)) |
| Syllabus nav section | The course syllabus body |
| Images / files in markdown | Uploaded to Canvas, links rewritten |
| `--add-pdf` | One PDF per page, attached inside modules |

Math, code highlighting, admonitions, tables and Excel rendering below are all converted to **inline-styled HTML**, so they render correctly inside Canvas (which ignores custom stylesheets).

## Labs

Files **directly under `docs/labs/`** whose filename starts with `lab` (case-insensitive) are published as **Canvas assignments** instead of wiki pages — e.g. `lab1.md`, `Lab2-intro.md`, or a numberless `lab-intro.md`. Everything else is a regular page: `notes.md` inside `labs/`, or a `lab1.md` outside `labs/`, and files in `labs/` subdirectories are ignored. Detection is filename-based, so `upload-labs` finds labs even when they are not listed in the navigation.

### Naming and title

```markdown
# Lab 1: Python Basics for Machine Learning
```

- The first-level heading becomes the **Canvas assignment name** (fallback: the file name).
- Numbered names (`Lab 1`, `Lab 12`) also let `delete-labs` find assignments that were renamed in Canvas or are missing from the cache — see [How labs are recognized](HOW_IT_WORKS.md#how-labs-are-recognized).

### What a lab looks like

See the ready-made examples in `docs-example/labs/`. Recommended structure:

```markdown
# Lab 1: Python Basics for Machine Learning

**Due Date**: Week 3
**Points**: 100
**Submission**: Submit your `.py` file and a PDF report

## Overview

Some text.

!!! important "Learning Objectives"
    - Objective one
    - Objective two
```

- Lab bodies support everything else on this page: math, code blocks, admonitions, images, tables and Excel sheets.
- Lines like **Due Date** / **Points** / **Submission** are plain text — they render into the assignment description but are **not** parsed into Canvas settings.

### What Canvas gets

- A published **assignment** (never a wiki page), named after the H1.
- On creation, Canvas assigns 100 points, points-based grading, and accepts `py`, `ipynb`, `txt`, `pdf` and `zip` file uploads plus online text entry.
- Re-uploading a lab rewrites the **description only** — due dates, points and other settings you changed in Canvas are preserved. Set deadlines in Canvas, not in the markdown.
- Labs are never module items, and a nav section named `Lab …` does not become a module. Still list labs in your `mkdocs.yml` nav so they appear in the local preview and in generated PDFs.

## Mathematical formulas

Inline math: `$E = mc^2$`

Display math:

```markdown
$$
\frac{\partial \mathcal{L}}{\partial \theta} = \frac{1}{n} \sum_{i=1}^{n} (h_\theta(x_i) - y_i)x_i
$$
```

LaTeX `align` environments are automatically converted to Canvas-compatible `array` form.

## Code blocks

Fenced code blocks get Pygments syntax highlighting with inline styles:

````markdown
```python
def hello_world():
    """A simple example."""
    print("Hello, World!")
```
````

Collapsible sections work via `pymdownx.details` / `superfences` (enabled extensions also include footnotes, definition lists, attribute lists and metadata).

## Admonitions

```markdown
!!! note "Important Information"
    This is a note admonition.

!!! warning "Be Careful"
    This is a warning.

!!! important "Critical"
    This is very important!

!!! tip "Pro Tip"
    Here's a helpful tip.

!!! danger "Well Done"
    Danger blocks are supported too.
```

## Images

```markdown
![Alt text](../assets/images/diagram.png)
```

Images are automatically uploaded to Canvas and the markdown link is rewritten to the hosted URL. Unchanged images are never re-uploaded (see [Asset deduplication](HOW_IT_WORKS.md#asset-deduplication)).

## Tables

```markdown
| Method | Time Complexity | Space Complexity |
|--------|----------------|------------------|
| Method A | O(n) | O(1) |
| Method B | O(n²) | O(n) |
```

## Excel spreadsheets

Embed Excel spreadsheets directly in your markdown with full color preservation:

```markdown
{{ render_excel_sheet('./path/to/spreadsheet.xlsx', 'SheetName') }}
```

**Example:**

```markdown
## Course Schedule

{{ render_excel_sheet('./schedule.xlsx', 'Schedule') }}
```

Features:

- Preserves cell background colors (including theme colors with tints)
- Preserves text colors and bold formatting
- Extracts colors from custom Excel themes
- Renders as responsive HTML tables
- Works in both local preview and Canvas

**Requirements:**

- `openpyxl` (already included in the project dependencies)
- Place Excel files in your `docs/` directory

## Adding new content

1. **Add a new lecture:**
   - Create `docs/lectures/new-lecture.md`
   - Add it to the `mkdocs.yml` navigation:
     ```yaml
     - Lecture 4 - New Topic:
       - lectures/04-new-topic.md
     ```
   - Run `mkdocstocanvas upload-modules --add-pdf`

2. **Add a new lab:**
   - Create `docs/labs/lab-new.md` — the filename must **start with `lab`** (see [Labs](#labs) for the expected format)
   - Add it to the `mkdocs.yml` navigation
   - Run `mkdocstocanvas upload-labs`

## Customizing the theme

Edit your `mkdocs.yml` to change colors, fonts, and features:

```yaml
theme:
  name: material
  palette:
    primary: indigo
    accent: indigo
  features:
    - navigation.tabs
    - navigation.sections
```
