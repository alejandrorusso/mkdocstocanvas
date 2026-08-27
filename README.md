# Canvas Course Publisher

A complete system for publishing MkDocs-based course content to Canvas LMS, with automatic PDF generation and module organization.

## Features

- 📚 **Automatic Module Creation**: Converts MkDocs navigation structure into Canvas modules
- 📄 **PDF Generation**: Automatically generates PDFs from markdown content
- 🔗 **Smart Linking**: Maintains internal links between course pages
- 🚀 **Incremental Updates**: Smart caching system only uploads changed files
- 🔄 **Page URL Preservation**: Updates existing Canvas pages without creating duplicates
- 🧪 **Lab Management**: Separate workflow for managing lab assignments
- 📊 **Excel Sheet Rendering**: Embed Excel spreadsheets with full color preservation
- 🎨 **Rich Content**: Full support for:
  - Mathematical formulas (LaTeX/MathJax)
  - Code syntax highlighting with Pygments
  - Admonitions (Note, Warning, Important, etc.)
  - Images and diagrams
  - Tables and lists
  - Excel spreadsheets with theme colors

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Canvas API token with course management permissions
- A Playwright/Chromium browser for PDF generation (only needed for `pdf`/`upload-modules`; install once with `mkdocstocanvas pdf --install-browser`)

## Installation

### 1. Clone and install

```bash
git clone <your-repo-url>
cd mkdocstocanvas
uv sync
```

Or without uv:

```bash
pip install -e .
```

All dependencies (MkDocs, Material theme, plugins, openpyxl, etc.) are declared in `pyproject.toml` and installed automatically.

### 2. Configure Canvas API Access

Copy the example environment file and fill in your Canvas credentials:

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

Edit `.env`:

```bash
CANVAS_API_TOKEN="your_canvas_api_token_here"
CANVAS_BASE_URL="https://canvas.instructure.com"  # or your institution's URL
CANVAS_COURSE_ID="your_course_id_here"
```

The file is loaded automatically by the CLI (`.env` is gitignored — never commit real credentials).

**Getting your Canvas API Token:**

1. Log in to Canvas
2. Go to Account → Settings
3. Scroll to "Approved Integrations"
4. Click "+ New Access Token"
5. Copy the generated token

**Finding your Course ID:**

The course ID is in the URL when viewing your course:
```
https://canvas.instructure.com/courses/12345
                                         ^^^^^ this is your course ID
```

## Configuration

### MkDocs Configuration

The `mkdocs.yml` file defines your course structure. Key sections:

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

**Important**: The navigation structure determines how modules are created in Canvas.

### Content Organization

```
docs/
├── index.md              # Course homepage
├── lectures/             # Lecture content
│   ├── 01-introduction.md
│   ├── 02-linear-regression.md
│   └── 03-neural-networks.md
├── labs/                 # Lab assignments
│   ├── lab1.md
│   └── lab2.md
└── assets/
    └── images/           # Images used in content
```

## Project Structure

```
mkdocstocanvas/
├── docs/                      # Course content (markdown files)
│   ├── index.md
│   ├── lectures/
│   ├── labs/
│   └── assets/
├── src/mkdocstocanvas/        # The CLI and Canvas integration package
│   ├── api/                   # Canvas API client
│   ├── uploaders/             # Page, module and lab uploaders
│   ├── processing/            # Markdown, math and Excel processing
│   ├── models/                # Data models
│   └── utils/                 # Config and cache helpers
├── tests/                     # Test suite (run with `uv run pytest`)
├── Makefile                   # Shortcuts for the CLI commands
├── mkdocs.yml                 # MkDocs configuration
├── pyproject.toml             # Project metadata and dependencies
├── .env                       # Canvas API credentials (gitignored)
├── .canvas_upload_state.json  # Upload cache (auto-generated)
├── pdf/                       # Generated PDFs (auto-created)
└── site/                      # Built site (auto-created)
```

**Important Files:**
- `.canvas_upload_state.json` - Tracks upload state; delete to force re-upload all files
- `.env` - Must be created with your Canvas API credentials (not tracked in git)

## Usage

Everything is done through the `mkdocstocanvas` CLI. Run `mkdocstocanvas --help` for an overview of all commands, or `mkdocstocanvas <command> --help` for the options of a specific command.

If you installed with uv, prefix commands with `uv run`, e.g. `uv run mkdocstocanvas serve`.

### Local Development

Preview your content locally:

```bash
mkdocstocanvas serve
```

This starts a local server at `http://localhost:8000` with live reload enabled.

**Note**: After editing markdown files, mkdocs will automatically rebuild. You may need to manually refresh your browser (F5) if auto-refresh doesn't work.

For full plugin support (slower):

```bash
mkdocstocanvas serve --all-plugins
```

### Building PDFs

Generate PDFs from your markdown content:

```bash
mkdocstocanvas pdf
```

PDFs are generated per page by the `mkdocs-page-pdf` plugin and collected into the `pdf/` directory. If no PDFs are produced, run once with the browser installer:

```bash
mkdocstocanvas pdf --install-browser
```

### Publishing to Canvas

#### Complete Rebuild (Recommended for Initial Setup)

Completely rebuild your Canvas course from scratch:

```bash
mkdocstocanvas delete-all --force
mkdocstocanvas upload-all --force
```

**⚠️ Warning**: This deletes all existing content! Use only for fresh setup or complete updates.

#### Individual Workflows

Upload only pages (incremental — only changed files):

```bash
mkdocstocanvas upload-pages
```

Upload only lab assignments:

```bash
mkdocstocanvas upload-labs
```

Upload modules (includes PDF generation, page upload and module creation):

```bash
mkdocstocanvas upload-modules --add-pdf
```

This command:
1. Builds the site and generates PDFs
2. Uploads all pages to Canvas
3. Creates modules with pages and PDFs
4. Publishes all the modules

Upload everything at once:

```bash
mkdocstocanvas upload-all
```

### Deleting Content

**⚠️ Warning**: These commands permanently delete content from Canvas! They ask for confirmation first.

Delete all modules (keeps pages):

```bash
mkdocstocanvas delete-modules
```

Delete all pages:

```bash
mkdocstocanvas delete-pages
```

Delete all lab assignments:

```bash
mkdocstocanvas delete-labs
```

Delete everything:

```bash
mkdocstocanvas delete-all
```

### Cleaning Local Files

Remove generated files (the upload cache is kept):

```bash
rm -rf site/ pdf/
```

### Running the Tests

```bash
uv run pytest
```

### Updating Existing Content

The system uses incremental updates for efficiency:

1. **Edit your markdown files**
2. **Upload changes:**
   ```bash
   mkdocstocanvas upload-pages        # Only uploads changed files
   mkdocstocanvas upload-modules --add-pdf   # Recreates modules with PDFs
   ```

The upload will:
- Automatically detect which files changed (MD5 hash comparison)
- Update existing Canvas pages (preserves URLs)
- Skip unchanged files for faster uploads
- Regenerate and upload PDFs

**Force full upload if needed:**
```bash
mkdocstocanvas upload-pages --force  # Uploads all files, ignoring cache
```

## How It Works

### Incremental Upload System

The page upload system uses intelligent caching to only upload changed files:

**Upload State Tracking:**
- Maintains `.canvas_upload_state.json` to track uploaded files
- Stores MD5 hash of each file's content
- Stores Canvas page URL slug for updates
- Only uploads files that have changed since last upload

**Update Behavior:**
- **Changed files**: Updates the existing Canvas page (preserves URL)
- **New files**: Creates new Canvas pages
- **Unchanged files**: Skips upload (reports as "Skipped")

**Force Upload:**
```bash
mkdocstocanvas upload-pages --force    # Ignores cache, uploads everything
# OR manually:
rm .canvas_upload_state.json && mkdocstocanvas upload-pages
```

### PDF Generation

PDFs are produced by the `mkdocs-page-pdf` plugin during `mkdocs build` (one PDF per page) and copied into `pdf/` by the `pdf` command. When uploading modules, each PDF is matched to its page by filename, so keep markdown filenames and PDF filenames consistent (they are generated from the same pages automatically).

## Customization

### Adding New Content

1. **Add a new lecture:**
   - Create `docs/lectures/new-lecture.md`
   - Add to `mkdocs.yml` navigation:
     ```yaml
     - Lecture 4 - New Topic:
       - lectures/04-new-topic.md
     ```
   - Run `mkdocstocanvas upload-modules --add-pdf`

2. **Add a new lab:**
   - Create `docs/labs/new-lab.md`
   - Add to `mkdocs.yml` navigation
   - Run `mkdocstocanvas upload-labs`

### Markdown Features

#### Mathematical Formulas

Inline math: `$E = mc^2$`

Display math:
```markdown
$$
\frac{\partial \mathcal{L}}{\partial \theta} = \frac{1}{n} \sum_{i=1}^{n} (h_\theta(x_i) - y_i)x_i
$$
```

#### Code Blocks

```python
def hello_world():
    """A simple example."""
    print("Hello, World!")
```

#### Admonitions

```markdown
!!! note "Important Information"
    This is a note admonition.

!!! warning "Be Careful"
    This is a warning.

!!! important "Critical"
    This is very important!

!!! tip "Pro Tip"
    Here's a helpful tip.

!!! success "Well Done"
    Great job!
```

#### Images

```markdown
![Alt text](../assets/images/diagram.png)
```

Images are automatically uploaded to Canvas and links are rewritten.

#### Excel Spreadsheets

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

#### Tables

```markdown
| Method | Time Complexity | Space Complexity |
|--------|----------------|------------------|
| Method A | O(n) | O(1) |
| Method B | O(n²) | O(n) |
```

### Customizing Themes

Edit `mkdocs.yml` to change colors, fonts, and features:

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

## Troubleshooting

### Common Issues

**No PDFs generated**:
- PDF generation needs a Playwright browser: run `mkdocstocanvas pdf --install-browser` once
- Make sure the `page-to-pdf` plugin is enabled in `mkdocs.yml`

**Duplicate pages being created**:
- The tool automatically updates existing pages (matched by slug and title)
- If you see duplicates, delete `.canvas_upload_state.json` and re-upload
- Use `mkdocstocanvas delete-pages` to clean up Canvas, then `mkdocstocanvas upload-pages --force`

**Pages not updating**:
- Check if `.canvas_upload_state.json` exists and has correct page URLs
- Force upload to bypass cache: `mkdocstocanvas upload-pages --force`
- Clear metadata: `rm .canvas_upload_state.json && mkdocstocanvas upload-pages`

**Canvas API errors**:
- Verify your Canvas API token in `.env`
- Check course ID is correct
- Ensure token has proper permissions (manage course content)

**Math formulas not rendering**:
- Enable MathJax in Canvas: Course Settings → Feature Options → "New Math Equation Editor"
- Check LaTeX syntax is correct
- Test locally with `mkdocstocanvas serve` first

**Images not displaying**:
- Verify image paths are relative to markdown file location
- Check images exist in `docs/` directory
- Images are automatically uploaded to Canvas

**Excel sheets not rendering**:
- Ensure `openpyxl` is installed (included in project dependencies)
- Place Excel files in `docs/` directory
- Use correct sheet name in render command

### Getting Help

If you encounter issues:

1. Check the error messages in console output
2. Verify all prerequisites are installed
3. Test with `mkdocstocanvas serve` locally first
4. Check Canvas permissions and API token
5. Review the example course structure

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite with `uv run pytest`
5. Submit a pull request

## License

This project is licensed under the Mozilla Public License 2.0 (MPL-2.0). See the [LICENSE](LICENSE) file for details.
