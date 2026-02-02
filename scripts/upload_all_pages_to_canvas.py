#!/usr/bin/env python3
"""
Upload All Lectures as Canvas Pages

This script reads mkdocs.yml navigation structure and uploads all markdown files
as Canvas pages with proper styling, syntax highlighting, images, and link resolution.
Page titles are extracted from the first # header in each markdown file.

Requirements:
- pip install requests markdown pyyaml
- Canvas API token set as environment variable: CANVAS_API_TOKEN
- Canvas base URL set as environment variable: CANVAS_BASE_URL
- Course ID set as environment variable: CANVAS_COURSE_ID
"""

import os
import sys
import requests
import time
from pathlib import Path
import re
import hashlib
import json
from datetime import datetime

try:
    import markdown
except ImportError:
    print("ERROR: markdown library not installed")
    print("Please run: pip install markdown pyyaml")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("ERROR: yaml library not installed")
    print("Please run: pip install pyyaml")
    sys.exit(1)

# Import shared processing functions
from canvas_processing import process_markdown_to_html

# Configuration
API_TOKEN = os.environ.get('CANVAS_API_TOKEN')
BASE_URL = os.environ.get('CANVAS_BASE_URL', 'https://canvas.instructure.com')
COURSE_ID = os.environ.get('CANVAS_COURSE_ID')

if not API_TOKEN:
    print("ERROR: CANVAS_API_TOKEN environment variable not set")
    sys.exit(1)

if not COURSE_ID:
    print("ERROR: CANVAS_COURSE_ID environment variable not set")
    sys.exit(1)

# API Headers
HEADERS = {
    'Authorization': f'Bearer {API_TOKEN}',
    'Content-Type': 'application/json'
}

# Track uploaded files and pages for link resolution
uploaded_files = {}  # filename -> Canvas URL
uploaded_pages = {}  # markdown_file -> Canvas page URL

# Metadata file for tracking upload state
METADATA_FILE = Path('.canvas_upload_state.json')

def load_upload_metadata():
    """Load upload metadata from file"""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠ Warning: Could not load upload metadata: {e}")
            return {}
    return {}

def save_upload_metadata(metadata):
    """Save upload metadata to file"""
    try:
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"  ⚠ Warning: Could not save upload metadata: {e}")

def compute_file_hash(file_path):
    """Compute MD5 hash of a file's contents"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"  ⚠ Warning: Could not compute hash for {file_path}: {e}")
        return None

def needs_upload(markdown_path, metadata, docs_root):
    """Check if a file needs to be uploaded based on its hash"""
    # Get relative path for metadata key
    try:
        rel_path = str(markdown_path.relative_to(docs_root))
    except:
        return True  # If can't compute relative path, upload it

    # Compute current hash
    current_hash = compute_file_hash(markdown_path)
    if current_hash is None:
        return True  # If can't compute hash, upload it

    # Check if we have metadata for this file
    if rel_path not in metadata:
        return True  # New file, needs upload

    stored_info = metadata[rel_path]
    stored_hash = stored_info.get('hash')

    # Compare hashes
    if stored_hash != current_hash:
        return True  # File changed, needs upload

    # Check if Canvas page still exists
    page_url_slug = stored_info.get('page_url_slug')
    if page_url_slug:
        existing_page = get_existing_page(page_url_slug)
        if not existing_page:
            return True  # Page doesn't exist on Canvas, needs upload

    return False  # File unchanged and page exists

def make_api_request(method, endpoint, **kwargs):
    """Make API request with error handling"""
    url = f"{BASE_URL}/api/v1{endpoint}"
    response = requests.request(method, url, headers=HEADERS, **kwargs)

    if response.status_code >= 400:
        print(f"API Error {response.status_code}: {response.text}")
        return None

    return response.json() if response.text else {}

def upload_image_to_canvas(image_path, filename):
    """Upload an image to Canvas and return its URL"""
    # Check if already uploaded
    if filename in uploaded_files:
        return uploaded_files[filename]

    print(f"    Uploading image: {filename}...")

    upload_data = {
        'name': filename,
        'content_type': 'image/png' if filename.endswith('.png') else 'image/jpeg',
        'parent_folder_path': '/page_images'
    }

    upload_info = make_api_request('POST', f'/courses/{COURSE_ID}/files', json=upload_data)
    if not upload_info:
        return None

    if 'upload_url' in upload_info and 'upload_params' in upload_info:
        with open(image_path, 'rb') as f:
            upload_params = upload_info['upload_params']
            files = {upload_info.get('file_param', 'file'): f}

            upload_response = requests.post(upload_info['upload_url'], data=upload_params, files=files)

            if upload_response.status_code in [200, 201]:
                try:
                    file_info = upload_response.json()
                    url = file_info.get('url')
                    uploaded_files[filename] = url
                    return url
                except:
                    if 'Location' in upload_response.headers:
                        confirm_response = requests.get(upload_response.headers['Location'],
                                                       headers={'Authorization': f'Bearer {API_TOKEN}'})
                        if confirm_response.status_code == 200:
                            file_info = confirm_response.json()
                            url = file_info.get('url')
                            uploaded_files[filename] = url
                            return url

    return None

def upload_file_to_canvas(file_path, filename):
    """Upload a general file (PDF, etc.) to Canvas and return its URL"""
    # Check if already uploaded
    if filename in uploaded_files:
        return uploaded_files[filename]

    print(f"    Uploading file: {filename}...")

    upload_data = {
        'name': filename,
        'content_type': 'application/pdf' if filename.endswith('.pdf') else 'application/octet-stream',
        'parent_folder_path': '/page_files'
    }

    upload_info = make_api_request('POST', f'/courses/{COURSE_ID}/files', json=upload_data)
    if not upload_info:
        return None

    if 'upload_url' in upload_info and 'upload_params' in upload_info:
        with open(file_path, 'rb') as f:
            upload_params = upload_info['upload_params']
            files = {upload_info.get('file_param', 'file'): f}

            upload_response = requests.post(upload_info['upload_url'], data=upload_params, files=files)

            if upload_response.status_code in [200, 201]:
                try:
                    file_info = upload_response.json()
                    url = file_info.get('url')
                    uploaded_files[filename] = url
                    return url
                except:
                    if 'Location' in upload_response.headers:
                        confirm_response = requests.get(upload_response.headers['Location'],
                                                       headers={'Authorization': f'Bearer {API_TOKEN}'})
                        if confirm_response.status_code == 200:
                            file_info = confirm_response.json()
                            url = file_info.get('url')
                            uploaded_files[filename] = url
                            return url

    return None

def resolve_markdown_links(md_content, markdown_path, docs_root):
    """Resolve links to other markdown files and replace with Canvas page URLs"""
    # Pattern: [text](path/to/file.md#anchor) - capture file path and optional anchor separately
    link_pattern = r'\[([^\]]+)\]\(([^\)#]+\.md)(#[^\)]+)?\)'

    def replace_link(match):
        link_text = match.group(1)
        link_path = match.group(2)  # The .md file path without anchor
        anchor = match.group(3) or ''  # The #anchor part (if any)

        # Ensure docs_root is a Path object and resolve it to absolute
        docs_root_abs = Path(docs_root).resolve()

        # Resolve relative path
        markdown_dir = Path(markdown_path).parent
        if link_path.startswith('./') or link_path.startswith('../'):
            full_path = (markdown_dir / link_path).resolve()
        else:
            full_path = (docs_root_abs / link_path).resolve()

        # Convert to relative to docs root for lookup
        try:
            rel_path = full_path.relative_to(docs_root_abs)
            md_key = str(rel_path)

            # Check if we have a Canvas URL for this page
            if md_key in uploaded_pages:
                canvas_url = uploaded_pages[md_key]
                # Append the anchor if it exists
                return f'[{link_text}]({canvas_url}{anchor})'
        except Exception as e:
            pass

        # If not found, keep original
        return match.group(0)

    return re.sub(link_pattern, replace_link, md_content)

def resolve_file_links(md_content, markdown_path):
    """Resolve links to files (PDFs, etc.) and upload/replace with Canvas URLs"""
    # Pattern: [text](path/to/file.pdf) or similar
    file_pattern = r'\[([^\]]+)\]\(([^\)]+\.(pdf|docx?|xlsx?|pptx?))\)'

    def replace_file_link(match):
        link_text = match.group(1)
        file_path = match.group(2)

        # Resolve relative path
        markdown_dir = Path(markdown_path).parent
        if file_path.startswith('./') or file_path.startswith('../'):
            full_path = (markdown_dir / file_path).resolve()
        else:
            full_path = Path(file_path)

        if full_path.exists():
            filename = full_path.name
            canvas_url = upload_file_to_canvas(str(full_path), filename)
            if canvas_url:
                return f'[{link_text}]({canvas_url})'

        # If file doesn't exist or upload failed, keep original
        return match.group(0)

    return re.sub(file_pattern, replace_file_link, md_content, flags=re.IGNORECASE)

def markdown_to_html(markdown_path, docs_root, all_pages_info=None):
    """Convert markdown file to HTML using shared processing pipeline"""
    with open(markdown_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Use shared processing pipeline with all pages info for internal link resolution
    html_content = process_markdown_to_html(
        md_content,
        str(markdown_path),
        BASE_URL,
        API_TOKEN,
        COURSE_ID,
        all_pages_info
    )

    return html_content

def get_all_pages():
    """Get all pages from Canvas"""
    pages = []
    page = 1
    while True:
        result = make_api_request('GET', f'/courses/{COURSE_ID}/pages', params={'page': page, 'per_page': 100})
        if not result or len(result) == 0:
            break
        pages.extend(result)
        page += 1
    return pages

def get_existing_page(page_url):
    """Check if a page already exists with the given URL"""
    result = make_api_request('GET', f'/courses/{COURSE_ID}/pages/{page_url}')
    return result

def create_or_update_page(page_title, html_content, published=True, page_url_slug=None):
    """Create a new page or update existing one"""
    # If page_url_slug is provided, use it directly; otherwise generate from title
    if not page_url_slug:
        page_url_slug = page_title.lower().replace(' ', '-').replace('/', '-').replace('(', '').replace(')', '')

    page_data = {
        'wiki_page': {
            'title': page_title,
            'body': html_content,
            'published': published,
            'editing_roles': 'teachers'
        }
    }

    existing_page = get_existing_page(page_url_slug)

    if existing_page:
        print(f"  Updating existing page: {page_title}")
        result = make_api_request('PUT', f'/courses/{COURSE_ID}/pages/{page_url_slug}', json=page_data)
        if result:
            canvas_url = f"{BASE_URL}/courses/{COURSE_ID}/pages/{result.get('url', page_url_slug)}"
            return canvas_url
    else:
        print(f"  Creating new page: {page_title}")
        result = make_api_request('POST', f'/courses/{COURSE_ID}/pages', json=page_data)
        if result:
            canvas_url = f"{BASE_URL}/courses/{COURSE_ID}/pages/{result.get('url', page_url_slug)}"
            return canvas_url

    return None

def extract_title_from_markdown(markdown_path):
    """Extract the first # header from a markdown file as the title"""
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Match lines starting with exactly one # (not ##, ###, etc.)
                if re.match(r'^#\s+', line):
                    # Remove the # and any extra whitespace
                    title = re.sub(r'^#\s+', '', line).strip()
                    return title
    except Exception as e:
        print(f"  ⚠ Error reading title from {markdown_path}: {e}")
    return None

def parse_mkdocs_nav(mkdocs_path):
    """Parse mkdocs.yml and extract navigation structure with markdown files"""
    try:
        with open(mkdocs_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        nav = config.get('nav', [])
        markdown_files = []

        def extract_files(items):
            """Recursively extract markdown file paths from nav structure"""
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, str) and value.endswith('.md'):
                                markdown_files.append(value)
                            elif isinstance(value, list):
                                extract_files(value)
                    elif isinstance(item, str) and item.endswith('.md'):
                        markdown_files.append(item)

        extract_files(nav)
        return markdown_files

    except Exception as e:
        print(f"ERROR: Failed to parse mkdocs.yml: {e}")
        return None

def find_pdf_file(markdown_path, page_title, pdf_dir):
    """Find the PDF file corresponding to a markdown file"""
    # Extract letter prefix from page title (e.g., "A." from "A. Course Syllabus")
    letter_prefix = page_title.split('.')[0] + '.'

    # Get the basename without extension from markdown path
    md_basename = Path(markdown_path).stem

    # Construct PDF filename: "{Letter}. {basename}.pdf"
    pdf_filename = f"{letter_prefix} {md_basename}.pdf"
    pdf_path = pdf_dir / pdf_filename

    if pdf_path.exists():
        return pdf_path
    return None

def process_all_pages():
    """Main function to process all pages from mkdocs.yml navigation"""
    mkdocs_path = Path('mkdocs.yml')
    docs_root = Path('docs')

    if not mkdocs_path.exists():
        print("ERROR: mkdocs.yml not found")
        return False

    if not docs_root.exists():
        print("ERROR: docs/ directory not found")
        return False

    print(f"Processing pages for Canvas course {COURSE_ID}")
    print(f"Canvas URL: {BASE_URL}")
    print("=" * 70)

    # Test API connectivity
    print("Testing API connectivity...")
    course_info = make_api_request('GET', f'/courses/{COURSE_ID}')
    if not course_info:
        print("Failed to connect to Canvas API or access course")
        return False

    print(f"Connected to course: {course_info.get('name', 'Unknown')}")
    print("=" * 70)

    # Load upload metadata
    print("\nLoading upload metadata...")
    metadata = load_upload_metadata()
    print(f"Found metadata for {len(metadata)} files")

    # Parse mkdocs.yml navigation
    print("\nParsing mkdocs.yml navigation...")
    markdown_files = parse_mkdocs_nav(mkdocs_path)

    if not markdown_files:
        print("ERROR: No markdown files found in mkdocs.yml navigation")
        return False

    # Collect pages to upload with their titles
    pages_to_upload = []
    pages_to_skip = []
    page_index = 0  # Track page index for letter prefix (only for non-lab pages)

    for md_file in markdown_files:
        markdown_path = docs_root / md_file

        # Skip lab files
        if md_file.startswith('labs/'):
            print(f"  ⚠ Skipping {md_file}: lab file (excluded)")
            continue

        if not markdown_path.exists():
            print(f"  ⚠ Skipping {md_file}: file not found")
            continue

        # Extract title from markdown file
        page_title = extract_title_from_markdown(markdown_path)

        if not page_title:
            print(f"  ⚠ Skipping {md_file}: no title found")
            continue

        # Prefix title with letter for ordering (A., B., C., ...)
        letter_prefix = chr(65 + page_index)  # 65 is ASCII for 'A'
        prefixed_title = f"{letter_prefix}. {page_title}"

        # Check if file needs upload
        if needs_upload(markdown_path, metadata, docs_root):
            pages_to_upload.append((markdown_path, prefixed_title))
        else:
            pages_to_skip.append((markdown_path, prefixed_title))
            # Still track this page for link resolution
            rel_path = markdown_path.relative_to(docs_root)
            stored_info = metadata[str(rel_path)]
            canvas_url = stored_info.get('canvas_url')
            if canvas_url:
                uploaded_pages[str(rel_path)] = canvas_url

        page_index += 1

    print(f"Found {len(pages_to_upload)} pages to upload")
    print(f"Skipping {len(pages_to_skip)} unchanged pages\n")

    # Upload pages (two passes for link resolution)
    successful = 0
    failed = 0
    pdf_dir = Path('pdf')
    page_url_by_title = {}  # Track actual Canvas URL slugs by page title

    # Pass 1: Upload all pages (links may not be resolved yet)
    print("PASS 1: Uploading all pages...")
    print("-" * 70)

    for markdown_file, page_title in pages_to_upload:
        print(f"\n[{successful + failed + 1}/{len(pages_to_upload)}] Processing: {page_title}")
        print(f"  Markdown file: {markdown_file}")

        # Check for corresponding PDF and upload it
        pdf_file = find_pdf_file(markdown_file, page_title, pdf_dir)
        if pdf_file:
            print(f"    Uploading PDF: {pdf_file.name}...")
            try:
                pdf_url = upload_file_to_canvas(str(pdf_file), pdf_file.name)
                if pdf_url:
                    print(f"    ✓ PDF uploaded: {pdf_url}")
                else:
                    print(f"    ⚠ PDF upload failed")
            except Exception as e:
                print(f"    ⚠ Error uploading PDF: {e}")

        try:
            html_content = markdown_to_html(str(markdown_file), docs_root)

            # Get stored page_url_slug from metadata if it exists
            rel_path = markdown_file.relative_to(docs_root)
            stored_slug = None
            if str(rel_path) in metadata:
                stored_slug = metadata[str(rel_path)].get('page_url_slug')

            canvas_url = create_or_update_page(page_title, html_content, published=True, page_url_slug=stored_slug)

            if canvas_url:
                # Store for link resolution
                rel_path = markdown_file.relative_to(docs_root)
                uploaded_pages[str(rel_path)] = canvas_url

                # Extract and store the actual page URL slug for Pass 2
                page_url_slug = canvas_url.split('/pages/')[-1]
                page_url_by_title[page_title] = page_url_slug

                # Update metadata
                file_hash = compute_file_hash(str(markdown_file))
                metadata[str(rel_path)] = {
                    'hash': file_hash,
                    'canvas_url': canvas_url,
                    'page_url_slug': page_url_slug,
                    'page_title': page_title,
                    'last_upload': datetime.now().isoformat()
                }

                print(f"  ✓ Success: {canvas_url}")
                successful += 1
            else:
                print(f"  ✗ Failed to create/update page")
                failed += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1

        time.sleep(0.5)  # Rate limiting

    # Save metadata after Pass 1
    if successful > 0:
        print("\nSaving upload metadata...")
        save_upload_metadata(metadata)

    # Pass 2: Re-upload pages with resolved inter-page links
    # Only run if pages were actually updated in Pass 1
    if successful > 0:
        print("\n" + "=" * 70)
        print("PASS 2: Resolving inter-page links...")
        print("-" * 70)

        # Get all pages info for internal link resolution
        print("  Getting all Canvas pages for link resolution...")
        all_pages_info = get_all_pages()
        print(f"  Found {len(all_pages_info)} pages for link resolution")

        updated = 0
        for markdown_file, page_title in pages_to_upload:
            rel_path = markdown_file.relative_to(docs_root)
            if str(rel_path) in uploaded_pages and page_title in page_url_by_title:
                try:
                    # Re-convert with all page URLs now available
                    html_content = markdown_to_html(str(markdown_file), docs_root, all_pages_info)
                    # Use the actual page URL slug from Pass 1
                    page_url_slug = page_url_by_title[page_title]
                    create_or_update_page(page_title, html_content, published=True, page_url_slug=page_url_slug)
                    updated += 1
                    print(f"  ✓ Updated links in: {page_title}")
                except Exception as e:
                    print(f"  ✗ Error updating {page_title}: {e}")

                time.sleep(0.5)  # Rate limiting

        print(f"\n  Updated {updated}/{successful} pages with resolved links")

        # Save metadata after Pass 2
        if updated > 0:
            print("  Saving upload metadata...")
            save_upload_metadata(metadata)

    print("\n" + "=" * 70)
    print(f"Upload Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (unchanged): {len(pages_to_skip)}")
    print(f"  Total: {len(pages_to_upload)}")
    print("=" * 70)

    if successful > 0:
        print("\n✓ Upload completed successfully!")
        print(f"\nPages are available at:")
        print(f"  {BASE_URL}/courses/{COURSE_ID}/pages")
        return True
    elif len(pages_to_skip) > 0 and failed == 0:
        print("\n✓ All pages are up to date - nothing to upload")
        return True
    else:
        print("\n✗ No pages were uploaded successfully")
        return False

def delete_all_pages():
    """Delete all pages from the Canvas course"""
    print("=" * 70)
    print("⚠️  WARNING: This will delete ALL pages from the course!")
    print("=" * 70)

    # Test API connectivity
    print("Testing API connectivity...")
    course_info = make_api_request('GET', f'/courses/{COURSE_ID}')
    if not course_info:
        print("Failed to connect to Canvas API or access course")
        return False

    print(f"Connected to course: {course_info.get('name', 'Unknown')}")
    print("=" * 70)

    # Get all pages
    print("\nFetching all pages...")
    pages = make_api_request('GET', f'/courses/{COURSE_ID}/pages', params={'per_page': 100})

    if pages is None:
        print("Failed to fetch pages from Canvas API")
        return False

    if len(pages) == 0:
        print("No pages found to delete")
        print("✓ Nothing to do - course already has no pages")
        return True

    print(f"Found {len(pages)} pages")

    # Confirm deletion
    print("\n⚠️  Pages to be deleted:")
    for page in pages:
        print(f"  - {page.get('title', 'Untitled')}")

    print("\n" + "=" * 70)
    response = input("Are you sure you want to delete ALL these pages? (type 'yes' to confirm): ")

    if response.lower() != 'yes':
        print("Deletion cancelled")
        return False

    # Delete pages
    print("\n" + "=" * 70)
    print("Deleting pages...")
    print("-" * 70)

    deleted = 0
    failed = 0

    for page in pages:
        page_url = page.get('url')
        page_title = page.get('title', 'Untitled')

        try:
            result = make_api_request('DELETE', f'/courses/{COURSE_ID}/pages/{page_url}')
            if result is not None:
                print(f"  ✓ Deleted: {page_title}")
                deleted += 1
            else:
                print(f"  ✗ Failed to delete: {page_title}")
                failed += 1
        except Exception as e:
            print(f"  ✗ Error deleting {page_title}: {e}")
            failed += 1

        time.sleep(0.3)  # Rate limiting

    print("\n" + "=" * 70)
    print(f"Deletion Summary:")
    print(f"  Deleted: {deleted}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(pages)}")
    print("=" * 70)

    if deleted > 0:
        print("\n✓ Page deletion completed!")
        return True
    else:
        print("\n✗ No pages were deleted")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Canvas Bulk Page Upload/Management Script')
    parser.add_argument('--delete-pages', action='store_true',
                        help='Delete all pages from the course (requires confirmation)')
    parser.add_argument('--force', action='store_true',
                        help='Force upload all pages, ignoring cached metadata')

    args = parser.parse_args()

    print("Canvas Bulk Page Upload Script")
    print("=" * 70)

    if args.delete_pages:
        success = delete_all_pages()
    else:
        # If force flag is set, clear metadata to force re-upload
        if args.force:
            print("⚠️  Force mode enabled - will upload all pages")
            if METADATA_FILE.exists():
                METADATA_FILE.unlink()
                print("  Cleared metadata cache")
            print("=" * 70)

        success = process_all_pages()

    sys.exit(0 if success else 1)
