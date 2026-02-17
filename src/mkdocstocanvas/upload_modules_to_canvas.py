#!/usr/bin/env python3
"""
Canvas Module Upload Script

This script reads mkdocs.yml navigation structure and creates Canvas modules
with links to the uploaded pages.

Requirements:
- pip install requests pyyaml
- Canvas API token set as environment variable: CANVAS_API_TOKEN
- Canvas base URL set as environment variable: CANVAS_BASE_URL
- Course ID set as environment variable: CANVAS_COURSE_ID
"""

import os
import requests
import sys
from pathlib import Path
import time
import re

try:
    import yaml
except ImportError:
    print("ERROR: yaml library not installed")
    print("Please run: pip install pyyaml")
    sys.exit(1)

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

def make_api_request(method, endpoint, **kwargs):
    """Make API request with error handling"""
    url = f"{BASE_URL}/api/v1{endpoint}"
    response = requests.request(method, url, headers=HEADERS, **kwargs)

    if response.status_code >= 400:
        print(f"API Error {response.status_code}: {response.text}")
        return None

    return response.json() if response.text else {}

def extract_title_from_markdown(markdown_path):
    """Extract the first # header from a markdown file as the title"""
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if re.match(r'^#\s+', line):
                    title = re.sub(r'^#\s+', '', line).strip()
                    return title
    except Exception as e:
        print(f"  ⚠ Error reading title from {markdown_path}: {e}")
    return None

def parse_mkdocs_nav(mkdocs_path):
    """Parse mkdocs.yml and extract navigation structure with modules and pages"""
    try:
        with open(mkdocs_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        nav = config.get('nav', [])
        modules = []

        for item in nav:
            if isinstance(item, dict):
                for module_name, pages in item.items():
                    module_data = {
                        'name': module_name,
                        'pages': []
                    }

                    if isinstance(pages, list):
                        for page in pages:
                            if isinstance(page, str) and page.endswith('.md'):
                                module_data['pages'].append(page)

                    if module_data['pages']:  # Only add modules with pages
                        modules.append(module_data)

        return modules

    except Exception as e:
        print(f"ERROR: Failed to parse mkdocs.yml: {e}")
        return None

def get_existing_modules():
    """Get all existing modules in the course"""
    modules = make_api_request('GET', f'/courses/{COURSE_ID}/modules?per_page=100')
    return modules if modules else []

def delete_all_modules():
    """Delete all existing modules in the course"""
    modules = get_existing_modules()
    if not modules:
        print("No existing modules to delete")
        return True

    print(f"Deleting {len(modules)} existing modules...")
    for module in modules:
        result = make_api_request('DELETE', f'/courses/{COURSE_ID}/modules/{module["id"]}')
        print(f"  ✓ Deleted module: '{module['name']}'")
        time.sleep(0.3)  # Rate limiting

    return True

def get_all_pages():
    """Get all Canvas pages in the course"""
    pages = make_api_request('GET', f'/courses/{COURSE_ID}/pages?per_page=100')
    return pages if pages else []

def find_page_by_title(pages, title):
    """Find a page by matching the title (with or without letter prefix)"""
    # Clean the search title (remove letter prefix if present)
    clean_search_title = re.sub(r'^[A-Z]\.\s+', '', title).strip()

    for page in pages:
        page_title = page.get('title', '')
        # Remove letter prefix from page title for comparison
        clean_page_title = re.sub(r'^[A-Z]\.\s+', '', page_title).strip()

        # Match if cleaned titles are the same
        if clean_page_title.lower() == clean_search_title.lower():
            return page

    return None

def create_module(name, position):
    """Create a new module"""
    module_data = {
        'module': {
            'name': name,
            'position': position,
            'published': False  # Create unpublished, will publish all at once later
        }
    }

    result = make_api_request('POST', f'/courses/{COURSE_ID}/modules', json=module_data)
    if result:
        print(f"  ✓ Created module: '{name}'")
        return result['id']
    return None

def publish_module_and_items(module_id, module_name):
    """Publish module and all its items at once"""
    # First, get all items in the module
    items = make_api_request('GET', f'/courses/{COURSE_ID}/modules/{module_id}/items?per_page=100')
    if not items:
        items = []

    # Publish each item
    for item in items:
        item_data = {
            'module_item': {
                'published': True
            }
        }
        make_api_request('PUT', f'/courses/{COURSE_ID}/modules/{module_id}/items/{item["id"]}', json=item_data)
        time.sleep(0.1)  # Rate limiting

    # Then publish the module itself
    module_data = {
        'module': {
            'published': True
        }
    }

    result = make_api_request('PUT', f'/courses/{COURSE_ID}/modules/{module_id}', json=module_data)
    if result:
        print(f"  ✓ Published module and all items: '{module_name}'")
        return True
    else:
        print(f"  ✗ Failed to publish module: '{module_name}'")
        return False

def add_page_to_module(module_id, page_url, page_title, position):
    """Add a page to a module"""
    item_data = {
        'module_item': {
            'title': page_title,  # Use the actual Canvas page title
            'type': 'Page',
            'page_url': page_url,
            'position': position,
            'indent': 0
        }
    }

    result = make_api_request('POST', f'/courses/{COURSE_ID}/modules/{module_id}/items', json=item_data)
    return result

def find_pdf_file(markdown_path, page_title, pdf_dir):
    """Find the corresponding PDF file for a markdown file"""
    # Extract letter prefix from page title (e.g., "A." from "A. Course Syllabus")
    letter_prefix = page_title.split('.')[0] + '.'

    # Get the basename without extension from markdown path
    base_name = markdown_path.stem

    # Construct PDF filename: "{Letter}. {basename}.pdf"
    pdf_filename = f"{letter_prefix} {base_name}.pdf"
    pdf_path = pdf_dir / pdf_filename

    if pdf_path.exists():
        return pdf_path
    return None

def upload_pdf_to_canvas(pdf_path, filename):
    """Upload PDF file to Canvas course files"""
    # Step 1: Get upload URL
    upload_data = {
        'name': filename,
        'content_type': 'application/pdf',
        'parent_folder_path': '/module_pdfs'
    }

    upload_info = make_api_request('POST', f'/courses/{COURSE_ID}/files', json=upload_data)
    if not upload_info:
        return None

    # Step 2: Upload file to the provided URL
    if 'upload_url' in upload_info and 'upload_params' in upload_info:
        with open(pdf_path, 'rb') as f:
            upload_params = upload_info['upload_params']
            files = {upload_info.get('file_param', 'file'): f}

            upload_response = requests.post(upload_info['upload_url'], data=upload_params, files=files)

            if upload_response.status_code >= 400:
                print(f"    ⚠ PDF upload failed: {upload_response.status_code}")
                return None

        # Step 3: The response should contain the file info or a redirect
        if upload_response.status_code in [200, 201]:
            try:
                file_info = upload_response.json()
                return file_info['id']
            except:
                # Sometimes Canvas returns a redirect
                if 'Location' in upload_response.headers:
                    confirm_response = requests.get(upload_response.headers['Location'], headers=HEADERS)
                    if confirm_response.status_code == 200:
                        file_info = confirm_response.json()
                        return file_info['id']

    return None

def add_pdf_to_module(module_id, file_id, pdf_title, position):
    """Add PDF file as module item"""
    item_data = {
        'module_item': {
            'title': pdf_title,
            'type': 'File',
            'content_id': file_id,
            'position': position,
            'indent': 0
        }
    }

    result = make_api_request('POST', f'/courses/{COURSE_ID}/modules/{module_id}/items', json=item_data)
    return result

def process_modules():
    """Main function to create modules from mkdocs.yml"""
    mkdocs_path = Path('mkdocs.yml')
    docs_root = Path('docs')
    pdf_dir = Path('pdf')

    if not mkdocs_path.exists():
        print("ERROR: mkdocs.yml not found")
        return False

    print(f"Creating modules for Canvas course {COURSE_ID}")
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

    # Parse mkdocs.yml navigation
    print("\nParsing mkdocs.yml navigation...")
    modules = parse_mkdocs_nav(mkdocs_path)

    if not modules:
        print("ERROR: No modules found in mkdocs.yml navigation")
        return False

    # Get all Canvas pages
    print("Fetching Canvas pages...")
    canvas_pages = get_all_pages()
    print(f"Found {len(canvas_pages)} pages in Canvas")

    # Delete existing modules
    print("\nDeleting existing modules...")
    delete_all_modules()

    print(f"\nFound {len(modules)} modules to create\n")
    print("=" * 70)

    successful = 0
    failed = 0
    page_index = 0

    for i, module_data in enumerate(modules, 1):
        module_name = module_data['name']
        pages = module_data['pages']

        # Skip lab modules
        if module_name.lower().startswith('lab'):
            print(f"\n[{i}/{len(modules)}] Skipping module: {module_name} (lab module)")
            continue

        print(f"\n[{i}/{len(modules)}] Creating module: {module_name}")
        print(f"  Pages: {len(pages)}")

        # Create module
        module_id = create_module(module_name, i)
        if not module_id:
            print(f"  ✗ Failed to create module")
            failed += 1
            continue

        # Add pages and PDFs to module
        item_position = 1
        for md_file in pages:
            # Skip lab files
            if md_file.startswith('labs/'):
                print(f"    ⚠ Skipping {md_file}: lab file")
                continue

            markdown_path = docs_root / md_file

            if not markdown_path.exists():
                print(f"    ⚠ Skipping {md_file}: file not found")
                continue

            # Extract title from markdown
            page_title = extract_title_from_markdown(markdown_path)
            if not page_title:
                print(f"    ⚠ Skipping {md_file}: no title found")
                continue

            # Find corresponding Canvas page
            canvas_page = find_page_by_title(canvas_pages, page_title)
            if not canvas_page:
                print(f"    ⚠ Skipping {md_file}: Canvas page not found for '{page_title}'")
                continue

            # Add page to module using the actual Canvas page title (with letter prefix)
            canvas_page_title = canvas_page.get('title', page_title)
            result = add_page_to_module(module_id, canvas_page['url'], canvas_page_title, item_position)
            if result:
                print(f"    ✓ Added page: {canvas_page_title}")
                item_position += 1
            else:
                print(f"    ✗ Failed to add page: {canvas_page_title}")
                continue

            time.sleep(0.3)  # Rate limiting

            # Find and upload corresponding PDF
            pdf_path = find_pdf_file(markdown_path, canvas_page_title, pdf_dir)
            if pdf_path:
                pdf_filename = f"{canvas_page_title}.pdf"

                # Upload PDF to Canvas
                file_id = upload_pdf_to_canvas(pdf_path, pdf_filename)
                if file_id:
                    # Add PDF to module
                    pdf_result = add_pdf_to_module(module_id, file_id, pdf_filename, item_position)
                    if pdf_result:
                        print(f"    ✓ Added PDF: {pdf_filename}")
                        item_position += 1
                    else:
                        print(f"    ✗ Failed to add PDF: {pdf_filename}")
                else:
                    print(f"    ⚠ Failed to upload PDF: {pdf_path.name}")

                time.sleep(0.3)  # Rate limiting
            else:
                print(f"    ⚠ No PDF found for: {md_file}")

        # Publish module and all its items
        print(f"  Publishing module and all items...")
        publish_module_and_items(module_id, module_name)

        successful += 1

    print("\n" + "=" * 70)
    print(f"Module Creation Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(modules)}")
    print("=" * 70)

    if successful > 0:
        print("\n✓ Modules created successfully!")
        print(f"\nModules are available at:")
        print(f"  {BASE_URL}/courses/{COURSE_ID}/modules")
        return True
    else:
        print("\n✗ No modules were created")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Canvas Module Upload/Management Script')
    parser.add_argument('--delete-only', action='store_true',
                        help='Delete all modules from the course (does not delete pages)')

    args = parser.parse_args()

    if args.delete_only:
        print("Canvas Module Deletion Script")
        print("=" * 70)
        print(f"Course ID: {COURSE_ID}")
        print(f"Canvas URL: {BASE_URL}")
        print("=" * 70)

        # Test API connectivity
        print("\nTesting API connectivity...")
        course_info = make_api_request('GET', f'/courses/{COURSE_ID}')
        if not course_info:
            print("Failed to connect to Canvas API or access course")
            sys.exit(1)

        print(f"Connected to course: {course_info.get('name', 'Unknown')}")
        print("=" * 70)

        print("\n⚠️  WARNING: This will delete ALL modules from the course!")
        print("Pages will NOT be deleted, only the module structure.")
        print("=" * 70)

        response = input("\nAre you sure you want to delete ALL modules? (type 'yes' to confirm): ")

        if response.lower() == 'yes':
            success = delete_all_modules()
            if success:
                print("\n✓ All modules deleted successfully!")
                sys.exit(0)
            else:
                print("\n✗ Module deletion failed")
                sys.exit(1)
        else:
            print("\nDeletion cancelled")
            sys.exit(0)
    else:
        success = process_modules()
        sys.exit(0 if success else 1)
