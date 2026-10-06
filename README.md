# Canvas Course Publisher

A complete system for publishing MkDocs-based course content to Canvas LMS, with automatic PDF generation and module organization.

## Features

- 📚 **Automatic module creation** — converts the `mkdocs.yml` nav structure into Canvas modules
- 📄 **PDF generation** — one PDF per page, attached inside modules with `--add-pdf`
- 🔗 **Smart linking** — internal `.md` links and images are rewritten to Canvas URLs
- 🚀 **Incremental updates** — MD5-based caching; only changed files are uploaded
- 🛡️ **Safe deletes** — uploaded content carries an invisible ownership marker; manually created Canvas content is never touched
- 🧪 **Lab management** — `docs/labs/lab*.md` files become Canvas assignments
- 🧮 **Rich content** — LaTeX math, Pygments syntax highlighting, admonitions, tables
- 📊 **Excel rendering** — embed spreadsheets with full color preservation
- ♻️ **Cache recovery** — a lost upload cache can be rebuilt from Canvas

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.14+ to install the current TestPyPI dev build (`3.0.3.dev1`); upcoming releases built from this repository support Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Canvas API token with course management permissions
- A Playwright/Chromium browser for PDF generation (only needed for `pdf`/`upload-modules`; install once with `mkdocstocanvas pdf --install-browser`)

## Installation

### 1. Install the CLI as a tool

Install `mkdocstocanvas` once, as a standalone tool — it then works from any directory, each course directory keeps its own credentials and upload cache, and tool upgrades never touch your course files. Dev builds are published on [TestPyPI](https://test.pypi.org/project/mkdocstocanvas/):

```bash
# uv (recommended)
uv tool install --index "https://test.pypi.org/simple/" "mkdocstocanvas==3.0.3.dev1"

# or pip (inside a virtualenv):
python -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://test.pypi.org/simple/ "mkdocstocanvas==3.0.3.dev1"
```

`--extra-index-url` / `--index` add TestPyPI *alongside* normal PyPI — the tool comes from TestPyPI, all dependencies from PyPI. Don't use `-i`/`--index-url` alone: pip would look for the dependencies on TestPyPI and fail (see [Troubleshooting](documentation/TROUBLESHOOTING.md)).

Verify with `mkdocstocanvas --help`. Upgrade later with `uv tool upgrade mkdocstocanvas` (or repeat the `pip` command with a newer pin). Once a stable release reaches normal PyPI, a plain `uv tool install mkdocstocanvas` / `pip install mkdocstocanvas` will work.

### 2. Create a course directory from the examples

Give every course its own directory — course content, configuration, credentials and the upload cache all live together, and nothing is tied to a code checkout:

```bash
mkdir my-course && cd my-course

# grab the templates from the repository (shallow clone, then discard it)
git clone --depth 1 <your-repo-url> /tmp/mkdocstocanvas
cp /tmp/mkdocstocanvas/mkdocs.example.yml mkdocs.yml    # course configuration
cp -r /tmp/mkdocstocanvas/docs-example docs             # course content
cp /tmp/mkdocstocanvas/.env.example .env                # Canvas credentials
rm -rf /tmp/mkdocstocanvas
```

Run every `mkdocstocanvas` command from inside this directory: the `.env` is loaded from here and the upload cache (`.canvas_upload_state.json`) is written here — so multiple courses each get their own isolated state, and your content never mixes with the tool's code.

```
my-course/
├── mkdocs.yml                  # your course configuration
├── docs/                       # your course content
├── .env                        # Canvas credentials — never commit
├── .canvas_upload_state.json   # upload cache (auto-generated, one per course)
├── pdf/                        # generated PDFs
└── site/                       # built site preview
```

If you keep your course directory under git, ignore credentials and generated files:

```bash
cat > .gitignore <<'EOF'
.env
.canvas_upload_state.json
site/
pdf/
__pycache__/
EOF
```

### 3. Configure Canvas API access

Edit the `.env` you copied into your course directory:

```bash
CANVAS_API_TOKEN="your_canvas_api_token_here"
CANVAS_BASE_URL="https://canvas.instructure.com"  # or your institution's URL
CANVAS_COURSE_ID="your_course_id_here"
```

The file is loaded automatically by the CLI (`.env` is gitignored — never commit real credentials).

**Getting your Canvas API token:** Canvas → Account → Settings → Approved Integrations → "+ New Access Token".

**Finding your course ID:** it is the number in the URL when viewing your course:
```
https://canvas.instructure.com/courses/12345
                                         ^^^^^ this is your course ID
```

## Configuration

The `mkdocs.yml` file (your local copy of `mkdocs.example.yml`) defines your course structure:

```yaml
nav:
  - Home:
    - index.md
  - Lecture 1 - Introduction:
    - lectures/01-introduction.md
  - Lecture 2 - Linear Regression:
    - lectures/02-linear-regression.md
  - Lab 1 - Python Basics:
    - labs/lab1.md
```

**Important**: the navigation structure determines how modules are created in Canvas.

### Content organization

```
docs/
├── index.md              # Course homepage
├── lectures/             # Lecture content
├── labs/                 # Lab assignments (lab*.md → Canvas assignments)
└── assets/
    └── images/           # Images used in content
```

A file is treated as a **lab** when it is directly inside `docs/labs/` and its filename starts with `lab` (case-insensitive); everything else is a regular page. See [How labs are recognized](documentation/HOW_IT_WORKS.md#how-labs-are-recognized) for the full rules.

## Project Structure

The layout below is the **development repository**. As an end user you only need the course directory created in Installation step 2 — you never have to clone this repo unless you're developing (see [CONTRIBUTING.md](CONTRIBUTING.md)).

```
mkdocstocanvas/
├── docs-example/                # Example course content (tracked template)
├── docs/                        # Your live course content (gitignored — copy from docs-example/)
├── src/mkdocstocanvas/          # The CLI and Canvas integration package
│   ├── api/                     # Canvas API client
│   ├── uploaders/               # Page, module and lab uploaders
│   ├── processing/              # Markdown, math and Excel processing
│   ├── models/                  # Data models
│   └── utils/                   # Config, cache and marker helpers
├── tests/                       # Test suite
├── Makefile                     # Shortcuts for the CLI commands
├── mkdocs.example.yml           # Example MkDocs configuration (tracked template)
├── mkdocs.yml                   # Your live MkDocs configuration (gitignored)
├── .env.example                 # Example Canvas credentials file (tracked)
├── .env                         # Your Canvas API credentials (gitignored)
├── pyproject.toml               # Project metadata and dependencies
├── .canvas_upload_state.json    # Upload cache (auto-generated, gitignored)
├── pdf/                         # Generated PDFs (auto-created)
└── site/                        # Built site (auto-created)
```

**Important files:**
- `mkdocs.example.yml`, `docs-example/`, `.env.example` — tracked templates; copy them when setting up (see Installation)
- `mkdocs.yml`, `docs/`, `.env` — your local working copies (gitignored — never commit credentials)
- `.canvas_upload_state.json` — tracks upload state; if lost or out of sync, run `mkdocstocanvas rebuild-cache`

## Usage

Everything is done through the `mkdocstocanvas` CLI, run from inside your course directory (Installation step 2). Run `mkdocstocanvas --help` for an overview, or `mkdocstocanvas <command> --help` for a specific command. If you installed from source instead, prefix with `uv run` (see [CONTRIBUTING.md](CONTRIBUTING.md)).

### Local preview

```bash
mkdocstocanvas serve               # fast mode, live reload
mkdocstocanvas serve --all-plugins # full plugin support (slower)
```

### Generating PDFs

```bash
mkdocstocanvas pdf
# first run, or if no PDFs are produced:
mkdocstocanvas pdf --install-browser
```

### Publishing to Canvas

Complete rebuild (recommended for initial setup):

```bash
mkdocstocanvas delete-all --force
mkdocstocanvas upload-all --force
```

⚠️ **Warning**: this deletes existing tool-managed content! Use only for fresh setup or complete updates.

Individual workflows:

```bash
mkdocstocanvas upload-pages                # pages only, incremental
mkdocstocanvas upload-labs                 # lab assignments only
mkdocstocanvas upload-modules --add-pdf    # builds PDFs, uploads pages, creates + publishes modules
mkdocstocanvas upload-all                  # labs → pages → labs → modules
```

### Deleting content

⚠️ **Warning**: these commands permanently delete content from Canvas!

By default, deletes only remove content **uploaded by mkdocstocanvas** (matched via ownership markers and the cache). Manually created Canvas content is left untouched; add `--all` to include it:

```bash
mkdocstocanvas delete-pages          # tool-managed pages only
mkdocstocanvas delete-modules        # tool-managed modules only
mkdocstocanvas delete-labs           # lab assignments
mkdocstocanvas delete-all            # everything uploaded by the tool
mkdocstocanvas delete-pages --all    # everything, including manually created content
```

### Updating existing content

1. Edit your markdown files
2. Upload changes:

```bash
mkdocstocanvas upload-pages             # only changed files
mkdocstocanvas upload-modules --add-pdf # recreate modules with PDFs
```

The upload detects changed files by MD5 hash, updates existing Canvas pages (preserving URLs), and skips unchanged files. To re-upload everything regardless of the cache:

```bash
mkdocstocanvas upload-pages --force     # also overwrites manual edits made in Canvas
```

If the upload cache is lost or deleted, run `mkdocstocanvas rebuild-cache` — see [Recovering a lost cache](documentation/HOW_IT_WORKS.md#recovering-a-lost-cache).

## Documentation

- [Content & features reference](documentation/FEATURES.md) — math, code, admonitions, images, Excel, adding content
- [How it works](documentation/HOW_IT_WORKS.md) — upload cache, incremental uploads, ownership markers, safe deletes, PDFs, lab rules
- [Troubleshooting](documentation/TROUBLESHOOTING.md) — full problem/solution list
- [Changelog](documentation/CHANGELOG.md) — release history
- [Contributing](CONTRIBUTING.md) — development setup, testing rules, PR checklist

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and the rules (offline tests only, marker contract).

## License

This project is licensed under the Mozilla Public License 2.0 (MPL-2.0). See the [LICENSE](LICENSE) file for details.
