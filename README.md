# Canvas Course Publisher

A complete system for publishing MkDocs-based course content to Canvas LMS, with automatic PDF generation and module organization.

## Features

- 📚 **Automatic Module Creation**: Converts MkDocs navigation structure into Canvas modules
- 📄 **PDF Generation**: Automatically generates PDFs from markdown content
- 🔗 **Smart Linking**: Maintains internal links between course pages
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
- [Workflow](#workflow)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Prerequisites

- Python 3.8 or later
- MkDocs and Material theme
- Canvas API token with course management permissions
- Required Python packages (see [Installation](#installation))

## Installation

### 1. Clone or Download

```bash
pip install --upgrade Pygments && \
pip install pymdown-extensions && \
pip install mkdocs-include-markdown-plugin && \
pip install mkdocs-material && \
pip install mkdocs-excel-plugin && \
pip install mkdocs-page-pdf && \
pip install openpyxl && \
pip install mkdocs
```

```bash 
apt-get update \
   && apt-get install -y wget gnupg \
   && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
   && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list' \
   && apt-get update \
   && apt-get install -y google-chrome-stable fonts-ipafont-gothic fonts-wqy-zenhei fonts-thai-tlwg fonts-kacst fonts-freefont-ttf libxss1 \
      --no-install-recommends \
   && rm -rf /var/lib/apt/lists/* \
   && npm init -y &&  \
      npm i puppeteer
```

### 2. Configure Canvas API Access

Create a `tokens.sh` file with your Canvas credentials:

```bash
cp tokens.sh.example tokens.sh
# Edit tokens.sh with your actual credentials
```

Edit `tokens.sh`:

```bash
#!/bin/bash
export CANVAS_API_TOKEN="your_canvas_api_token_here"
export CANVAS_BASE_URL="https://canvas.instructure.com"  # or your institution's URL
export CANVAS_COURSE_ID="your_course_id_here"
```

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

### 4. Make Scripts Executable

```bash
chmod +x tokens.sh
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
canvas-course-publisher/
├── docs/                    # Course content (markdown files)
│   ├── index.md
│   ├── lectures/
│   ├── labs/
│   └── assets/
├── scripts/                 # Python scripts for Canvas integration
│   ├── upload_modules_to_canvas.py
│   ├── upload_all_pages_to_canvas.py
│   ├── upload_labs_to_canvas.py
│   └── canvas_processing.py
├── Makefile                 # Build and deployment commands
├── mkdocs.yml              # MkDocs configuration
├── tokens.sh               # Canvas API credentials (gitignored)
├── pdf/                    # Generated PDFs (auto-created)
└── site/                   # Built site (auto-created)
```

## Usage

### Local Development

Preview your content locally:

```bash
make serve
```

This starts a local server at `http://localhost:8000` with live reload enabled.

**Note**: After editing markdown files, mkdocs will automatically rebuild. You may need to manually refresh your browser (F5) if auto-refresh doesn't work.

For full plugin support (slower):

```bash
make serve-full
```

### Building PDFs

Generate PDFs from your markdown content:

```bash
make pdf
```

PDFs are saved to the `pdf/` directory.

### Publishing to Canvas

#### Complete Rebuild (Recommended for Initial Setup)

Completely rebuild your Canvas course from scratch:

```bash
./rebuild_canvas.sh
```

This script performs a complete rebuild:
1. Cleans generated files
2. Deletes all modules from Canvas
3. Deletes all pages from Canvas
4. Deletes all lab assignments from Canvas
5. Generates PDFs
6. Uploads all pages to Canvas
7. Creates modules with pages and PDFs
8. Uploads lab assignments

**⚠️ Warning**: This deletes all existing content! Use only for fresh setup or complete updates.

#### Individual Workflows

Upload only pages:

```bash
make upload-pages
```

Upload only lab assignments:

```bash
make upload-labs
```

Upload modules (includes PDF generation):

```bash
make upload-modules
```

This command:
1. Builds the site
2. Generates PDFs
3. Uploads all pages to Canvas
4. Creates modules with pages and PDFs
5. Publishes all the modules 

### Deleting Content

**⚠️ Warning**: These commands permanently delete content from Canvas!

Delete all modules (keeps pages):

```bash
make delete-modules
```

Delete all pages:

```bash
make delete-pages
```

Delete all lab assignments:

```bash
make delete-labs
```

### Cleaning Local Files

Remove generated files:

```bash
make clean
```

### Updating Existing Content

1. Edit your markdown files
2. Run `make upload-pages && make upload-modules` to update Canvas

The script will:
- Update existing pages
- Recreate modules with the new content
- Regenerate PDFs

## Customization

### Adding New Content

1. **Add a new lecture:**
   - Create `docs/lectures/new-lecture.md`
   - Add to `mkdocs.yml` navigation:
     ```yaml
     - Lecture 4 - New Topic:
       - lectures/04-new-topic.md
     ```
   - Run `make upload-modules`

2. **Add a new lab:**
   - Create `docs/labs/new-lab.md`
   - Add to `mkdocs.yml` navigation
   - Run `make upload-labs` or `make upload-modules`

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
- Install `mkdocs-excel-plugin` and `openpyxl`
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


### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is provided as-is for educational purposes. Modify and use as needed for your courses.

