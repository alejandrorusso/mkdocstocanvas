# Script Features Documentation

This document provides a detailed overview of all the scripts used to develop and upload MkDocs course content to Canvas LMS.

## Overview

The scripts in this project automate the process of:
- Converting MkDocs markdown content to Canvas-compatible HTML
- Uploading course pages, labs, and modules to Canvas
- Handling images, PDFs, Excel sheets, and file attachments
- Processing LaTeX math equations and syntax-highlighted code
- Managing internal links between pages

## Scripts

### 1. canvas_processing.py

**Purpose**: Common processing utilities for Canvas uploads.

**Core Features**:

#### LaTeX Math Processing
- Converts LaTeX `align` environments to Canvas-compatible `array` format
- Protects math content from markdown processing interference
- Handles both inline math (`$...$` → `\(...\)`) and display math (`$$...$$`)
- Preserves line breaks (`\\`) in math environments
- Protects special characters (underscores, asterisks, tildes) within math expressions (?)
- Supports standalone align environments by wrapping them in `$$` delimiters

#### Markdown to HTML Conversion
- Comprehensive markdown processing pipeline with multiple extensions:
  - Tables, fenced code blocks (`extra`)
  - Syntax highlighting with Pygments (`codehilite`)
  - Table of contents (`toc`)
  - Note/warning blocks (`admonition`)
  - Definition lists, footnotes, metadata
- Converts markdown to Canvas-compatible HTML while preserving math notation

#### Image Handling
- Uploads images to Canvas course files
- Automatically processes image references in markdown
- Resolves relative image paths
- Returns Canvas URLs for uploaded images
- Creates `/course_images` folder structure in Canvas

#### Excel Sheet Rendering
- Renders Excel sheets as HTML tables with preserved cell colors
- Supports theme colors with tint application
- Processes both direct RGB and indexed colors
- Handles custom macro syntax: `{{ render_excel_sheet('./path/to/file.xlsx', 'SheetName') }}`
- Preserves cell background colors, text colors, and font weights
- Applies appropriate header styling

#### Internal Link Resolution (Issue 1: Pushed or work on this fix)
- Resolves internal markdown links to Canvas page URLs
- Maps local markdown filenames to Canvas pages
- Preserves anchor links (#sections) when converting to Canvas URLs
- Handles relative and absolute paths

#### Syntax Highlighting
- Applies GitHub-light theme colors using inline styles
- Supports wide range of programming languages
- Styles code blocks with proper formatting
- Inline code styling with monospace fonts

#### Admonition Styling
- Styles note, warning, important, tip, danger, and info blocks
- Color-coded border and background for each type
- Proper padding and typography for readability

### 2. upload_all_pages_to_canvas.py

**Purpose**: Bulk upload all markdown pages from MkDocs navigation to Canvas.

**Features**:

#### Page Management (Issue 2: pushed change) (Issue 5: repeating the title for each page in a module)
- Reads `mkdocs.yml` navigation structure
- Extracts page titles from first `#` header in markdown files
- Creates or updates Canvas pages based on existing content
- Prefixes pages with letter ordering (A., B., C., ...) for organization
- Skips lab files (handles them separately)

#### Two-Pass Upload Process (Issue 3: pushed linking)
- **Pass 1**: Uploads all pages with initial content
  - Converts markdown to HTML
  - Uploads images and processes content
  - Creates/updates Canvas pages
- **Pass 2**: Resolves inter-page links
  - Re-uploads pages with all internal links properly resolved
  - Updates links to point to actual Canvas page URLs

#### File Handling  (Issue 4: pushed referring to PDFs files in markdown)
- Uploads PDF files corresponding to markdown pages
- Places PDFs in `/page_files` folder
- Handles images in `/page_images` folder
- Supports relative and absolute file paths


#### Page Deletion
- `--delete-pages` flag to bulk delete all Canvas pages
- Requires user confirmation before deletion
- Provides summary of deleted/failed operations

#### API Management
- Tests API connectivity before operations
- Rate limiting with delays between requests
- Error handling for API failures
- Progress tracking with detailed console output

### 3. upload_labs_to_canvas.py

**Purpose**: Upload lab markdown files as Canvas assignments.

**Features**:

#### Lab Discovery
- Automatically finds all `lab*.md` files in docs directory
- Sorts labs by number (lab1, lab2, etc.)
- Extracts lab titles from markdown headers

#### Assignment Creation (Issue 6: (Ale) do not overwrite configuration of the lab, just the statement)
- Creates or updates Canvas assignments
- Configures assignment settings:
  - 100 points possible (configurable)
  - Online text entry and file upload submission types
  - Points-based grading
  - Allowed extensions: py, ipynb, txt, pdf, zip
  - Published by default

#### Lab Styling
- Applies custom CSS styling for lab content
- Material Design inspired typography
- Responsive layout with max-width container
- Styled code blocks, tables, and admonitions
- Professional color scheme with blue accents

#### Link Resolution (Issue 7: make link resolution only in one file, check that is not re-implementing it)
- Fetches all Canvas pages for internal link resolution
- Converts markdown links to Canvas page URLs
- Preserves anchors in links 

#### Assignment Management
- `--delete` flag to remove all lab assignments
- Filters assignments to only affect lab assignments (matching "Lab \d+" pattern)
- Requires user confirmation before deletion
- Provides detailed summary of operations

### 4. upload_modules_to_canvas.py

**Purpose**: Create Canvas modules from MkDocs navigation structure.

**Features**:

#### Module Creation (Issue 8: Syllabus should be linked to the Syllabus page with deleted from modules. It should also appear in mkdocs.yml)

- Parses `mkdocs.yml` navigation hierarchy
- Creates Canvas modules matching MkDocs structure
- Skips lab modules (handled separately as assignments)
- Deletes existing modules before creating new ones

#### Module Population (Issue 9: push your fix on respecting the mkdocs.yml)
- Adds pages to modules in order
- Matches markdown files to Canvas pages by title
- Handles letter-prefixed titles (A., B., C., ...)
- Positions items sequentially within modules

#### PDF Integration
- Finds PDF files corresponding to each markdown page
- Uploads PDFs to Canvas files in `/module_pdfs` folder
- Adds PDF as module item immediately after corresponding page
- Uses same naming convention as pages

#### Publishing
- Creates modules as unpublished initially
- Publishes all module items first
- Then publishes the module itself
- Ensures proper visibility of content

#### Module Management
- `--delete-only` flag to remove all modules without recreating
- Preserves pages when deleting modules
- Requires user confirmation
- Provides detailed progress output

## Common Functionality Across Scripts

### Environment Variables Required
- `CANVAS_API_TOKEN`: Canvas API access token
- `CANVAS_BASE_URL`: Canvas instance URL (default: https://canvas.instructure.com)
- `CANVAS_COURSE_ID`: Target course ID

### API Request Handling
- Consistent error handling across all scripts
- JSON request/response processing
- Authorization header management
- Rate limiting to avoid API throttling

### File System Operations
- Resolves relative and absolute paths
- Handles missing files gracefully
- Works with standard MkDocs directory structure (docs/, pdf/)

### Progress Reporting
- Detailed console output with status symbols (✓, ✗, ⚠)
- Progress counters showing current/total items
- Summary statistics at completion
- Error messages with context

### Dependencies
- `requests`: HTTP API calls
- `markdown`: Markdown to HTML conversion
- `pyyaml`: MkDocs configuration parsing
- `openpyxl`: Excel file processing (optional)

## Typical Workflow

1. **Develop course content** in MkDocs markdown format
2. **Generate PDFs** (if needed) for each page
3. **Run upload_all_pages_to_canvas.py** to create all course pages
4. **Run upload_labs_to_canvas.py** to create lab assignments
5. **Run upload_modules_to_canvas.py** to organize content into modules

## Notes

- All scripts support dry-run mode via API testing before operations
- Images and files are automatically uploaded and linked
- LaTeX math notation is fully supported via MathJax
- Code syntax highlighting works with Canvas rendering
- Internal links between pages are automatically resolved
- Module structure mirrors MkDocs navigation hierarchy
