#!/usr/bin/env python3
"""
Rename PDFs with Letter Prefixes Based on mkdocs.yml Navigation Order

This script reads the mkdocs.yml navigation structure and renames PDFs
in the pdf/ directory with letter prefixes (A, B, C, etc.) following
the order they appear in the navigation.

Usage:
    python3 scripts/rename_pdfs_with_order.py
"""

import os
import yaml
import shutil
from pathlib import Path
import string

def parse_mkdocs_nav(mkdocs_path):
    """Parse mkdocs.yml and extract ordered list of markdown files"""
    try:
        with open(mkdocs_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        nav = config.get('nav', [])
        ordered_files = []

        def extract_files(items):
            """Recursively extract markdown files from nav structure"""
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str) and item.endswith('.md'):
                        ordered_files.append(item)
                    elif isinstance(item, dict):
                        for key, value in item.items():
                            extract_files(value)
            elif isinstance(items, str) and items.endswith('.md'):
                ordered_files.append(items)

        extract_files(nav)
        return ordered_files

    except Exception as e:
        print(f"ERROR: Failed to parse mkdocs.yml: {e}")
        return None

def get_pdf_filename_from_md(md_path):
    """Convert markdown path to expected PDF filename"""
    # Remove extension and get base name
    base_name = Path(md_path).stem
    return f"{base_name}.pdf"

def rename_pdfs_with_prefixes(pdf_dir, ordered_md_files):
    """Rename PDFs in pdf_dir with letter prefixes based on order"""
    pdf_path = Path(pdf_dir)

    if not pdf_path.exists():
        print(f"ERROR: PDF directory {pdf_dir} does not exist")
        return False

    # Create mapping of original PDF names to new names with prefixes
    rename_map = {}
    letters = string.ascii_uppercase

    for idx, md_file in enumerate(ordered_md_files):
        # Skip lab files - they should not be renamed
        if md_file.startswith('labs/'):
            continue

        pdf_name = get_pdf_filename_from_md(md_file)
        pdf_file = pdf_path / pdf_name

        if pdf_file.exists():
            # Create new name with letter prefix
            if idx < len(letters):
                letter = letters[idx]
            else:
                # If we run out of letters, use AA, AB, AC, etc.
                letter = letters[idx // 26] + letters[idx % 26]

            new_name = f"{letter}. {pdf_name}"
            rename_map[pdf_name] = new_name

    # Perform renames
    print(f"Renaming {len(rename_map)} PDFs with letter prefixes...")
    print("=" * 70)

    successful = 0
    failed = 0

    for old_name, new_name in rename_map.items():
        old_path = pdf_path / old_name
        new_path = pdf_path / new_name

        try:
            # If target exists, remove it first
            if new_path.exists():
                new_path.unlink()

            shutil.move(str(old_path), str(new_path))
            print(f"✓ {old_name} → {new_name}")
            successful += 1
        except Exception as e:
            print(f"✗ Failed to rename {old_name}: {e}")
            failed += 1

    print("=" * 70)
    print(f"Summary: {successful} renamed, {failed} failed")

    return failed == 0

def main():
    """Main function"""
    print("PDF Renaming Script")
    print("=" * 70)

    # Parse mkdocs.yml
    mkdocs_path = 'mkdocs.yml'
    print(f"Reading navigation order from {mkdocs_path}...")

    ordered_files = parse_mkdocs_nav(mkdocs_path)
    if not ordered_files:
        print("ERROR: Could not extract navigation order")
        return 1

    print(f"Found {len(ordered_files)} pages in navigation order")
    print()

    # Rename PDFs
    pdf_dir = 'pdf'
    success = rename_pdfs_with_prefixes(pdf_dir, ordered_files)

    if success:
        print()
        print("✓ All PDFs renamed successfully!")
        return 0
    else:
        print()
        print("⚠ Some PDFs failed to rename")
        return 1

if __name__ == "__main__":
    exit(main())
