# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-10-28

### Added

#### Excel Sheet Rendering
- **Full Excel support**: Embed Excel spreadsheets directly in markdown with `{{ render_excel_sheet('./file.xlsx', 'SheetName') }}`
- **Color preservation**: Extracts and preserves all cell background and text colors
- **Theme color support**: Reads custom Excel theme colors from workbook XML
- **Tint support**: Applies color tints (lightening/darkening) to theme colors
- **Text visibility**: Automatically ensures readable text by defaulting to black when no font color is specified
- Functions added to `canvas_processing.py`:
  - `apply_tint()`: Apply tint values to RGB colors
  - `get_theme_colors()`: Extract theme colors from Excel workbook XML
  - `color_to_hex()`: Convert openpyxl Color objects to hex (RGB, theme, indexed)
  - `render_excel_sheet_to_html()`: Render Excel sheets as styled HTML tables
  - `process_excel_macros()`: Process Excel rendering macros in markdown

#### Build & Deployment Scripts
- **`rebuild_canvas.sh`**: Complete rebuild script for Canvas courses
  - Deletes all modules, pages, and lab assignments
  - Regenerates PDFs
  - Uploads all content fresh
  - Includes progress indicators and summary

#### Improved Development Workflow
- **Better live reload**: Added `--watch-theme` and `--livereload` flags to `make serve`
- **Dual serve modes**:
  - `make serve`: Fast mode with live reload
  - `make serve-full`: Full plugin support (slower)

### Changed

- **canvas_processing.py**: Updated from 628 to 943 lines with Excel support
- **Makefile**: Enhanced serve targets with better flags
- **README.md**: Added Excel documentation and rebuild script instructions

### Dependencies Added

- `openpyxl`: Required for Excel file processing

### Technical Details

#### Color Extraction
The Excel rendering system handles three types of colors:

1. **Direct RGB colors**: `AARRGGBB` format, extracts `RRGGBB`
2. **Theme colors**: Reads from workbook's embedded theme XML
3. **Indexed colors**: Legacy color palette support

#### Theme Color Mapping
Standard Office theme indices:
- 0: Background 1 (light)
- 1: Text 1 (dark)
- 2: Background 2
- 3: Text 2
- 4-9: Accent colors 1-6

Tint values are applied using:
- Positive tint: Mix with white (lighten)
- Negative tint: Mix with black (darken)

### Migration Guide

If updating from version 1.x:

1. Install new dependency:
   ```bash
   pip install openpyxl
   ```

2. Copy updated files:
   ```bash
   # Backup your current scripts
   cp scripts/canvas_processing.py scripts/canvas_processing.py.backup

   # Copy new version
   cp /path/to/new/canvas_processing.py scripts/
   ```

3. Update Makefile for better serve:
   ```bash
   cp /path/to/new/Makefile ./
   ```

4. Add rebuild script:
   ```bash
   cp /path/to/rebuild_canvas.sh ./
   chmod +x rebuild_canvas.sh
   ```

### Compatibility

- Fully backward compatible with existing markdown content
- Excel macros are optional - existing courses work without changes
- All previous Canvas upload features remain unchanged

## [1.0.0] - 2024-10-24

### Initial Release

- Basic MkDocs to Canvas conversion
- PDF generation per page
- Module creation and management
- Lab assignment uploads
- Math rendering (LaTeX/MathJax)
- Code syntax highlighting
- Image uploads
- Internal link resolution
