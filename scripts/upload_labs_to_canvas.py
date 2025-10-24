#!/usr/bin/env python3
"""
Canvas Assignment Upload Script for Labs

This script converts lab markdown files to HTML and uploads them as Canvas assignments.

Requirements:
- pip install requests markdown
- Canvas API token set as environment variable: CANVAS_API_TOKEN
- Canvas base URL set as environment variable: CANVAS_BASE_URL
- Course ID set as environment variable: CANVAS_COURSE_ID

Usage:
    python3 upload_labs_to_canvas.py

This script automatically finds all lab*.md files and uploads them as assignments.
"""

import os
import sys
import requests
from pathlib import Path
import re
import argparse

try:
    import markdown
except ImportError:
    print("ERROR: markdown library not installed")
    print("Please run: pip install markdown")
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

def make_api_request(method, endpoint, **kwargs):
    """Make API request with error handling"""
    url = f"{BASE_URL}/api/v1{endpoint}"
    response = requests.request(method, url, headers=HEADERS, **kwargs)

    if response.status_code >= 400:
        print(f"API Error {response.status_code}: {response.text}")
        return None

    return response.json() if response.text else {}

def get_all_pages():
    """Get all pages from Canvas for link resolution"""
    pages = []
    page = 1
    while True:
        result = make_api_request('GET', f'/courses/{COURSE_ID}/pages', params={'page': page, 'per_page': 100})
        if not result or len(result) == 0:
            break
        pages.extend(result)
        page += 1
    return pages

def markdown_to_html(markdown_path):
    """Convert markdown file to HTML using shared processing pipeline"""
    with open(markdown_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Get all pages for link resolution
    print("  Fetching Canvas pages for link resolution...")
    all_pages_info = get_all_pages()
    print(f"  Found {len(all_pages_info)} pages for link resolution")

    # Use shared processing pipeline with link resolution
    html_content = process_markdown_to_html(
        md_content,
        str(markdown_path),
        BASE_URL,
        API_TOKEN,
        COURSE_ID,
        all_pages_info
    )

    # Add lab-specific styling for Canvas
    full_html = f"""
    <div class="lab-content" style="max-width: 1200px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 16px; line-height: 1.6; color: rgba(0, 0, 0, 0.87);">
        <style>
            .lab-content h1 {{ font-size: 2em; font-weight: 400; margin: 1.5em 0 0.8em; color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 0.3em; }}
            .lab-content h2 {{ font-size: 1.6em; font-weight: 400; margin: 1.5em 0 0.8em; color: rgba(0, 0, 0, 0.87); border-bottom: 1px solid rgba(0, 0, 0, 0.12); padding-bottom: 0.3em; }}
            .lab-content h3 {{ font-size: 1.3em; font-weight: 500; margin: 1.2em 0 0.6em; color: rgba(0, 0, 0, 0.87); }}
            .lab-content p {{ margin: 1em 0; }}
            .lab-content a {{ color: #1976d2; text-decoration: none; }}
            .lab-content a:hover {{ color: #1565c0; text-decoration: underline; }}
            .lab-content code {{ background-color: rgba(0, 0, 0, 0.05); padding: 0.2em 0.4em; border-radius: 3px; font-family: 'Roboto Mono', monospace; font-size: 0.9em; }}
            .lab-content pre {{ background-color: #f6f8fa; padding: 16px; border-radius: 4px; overflow-x: auto; margin: 1.5em 0; border: 1px solid #e1e4e8; }}
            .lab-content blockquote {{ border-left: 4px solid #1976d2; padding-left: 1em; margin: 1.5em 0; color: rgba(0, 0, 0, 0.54); }}
            .lab-content ul, .lab-content ol {{ padding-left: 2em; margin: 1em 0; }}
            .lab-content li {{ margin: 0.5em 0; }}
            .lab-content table {{ border-collapse: collapse; width: 100%; margin: 1.5em 0; }}
            .lab-content table th {{ background-color: #1976d2; color: white; padding: 12px 16px; text-align: left; }}
            .lab-content table td {{ padding: 12px 16px; border-bottom: 1px solid rgba(0, 0, 0, 0.12); }}
            .lab-content .admonition {{ margin: 1.5em 0; padding: 15px 20px; border-radius: 4px; border-left: 4px solid #2196F3; background-color: #e7f2fa; }}
            .lab-content .admonition-title {{ font-weight: 600; margin: 0 0 10px 0; color: #014361; }}
            .lab-content .highlight {{ background-color: #f6f8fa; padding: 12px; border-radius: 4px; overflow-x: auto; margin: 1em 0; border: 1px solid #e1e4e8; }}
            .lab-content .highlight pre {{ background: none; border: none; padding: 0; margin: 0; }}
        </style>
        {html_content}
    </div>
    """

    return full_html

def get_existing_assignment(assignment_name):
    """Check if an assignment already exists with the given name"""
    assignments = make_api_request('GET', f'/courses/{COURSE_ID}/assignments')
    if assignments:
        for assignment in assignments:
            if assignment.get('name') == assignment_name:
                return assignment
    return None

def create_or_update_assignment(lab_name, html_content, points_possible=100):
    """Create a new assignment or update existing one"""

    assignment_data = {
        'assignment': {
            'name': lab_name,
            'description': html_content,
            'points_possible': points_possible,
            'submission_types': ['online_text_entry', 'online_upload'],
            'grading_type': 'points',
            'published': True,
            'assignment_group_id': None,  # Will use default assignment group
            'allowed_extensions': ['py', 'ipynb', 'txt', 'pdf', 'zip'],
            'notify_of_update': False
        }
    }

    # Check if assignment exists
    existing_assignment = get_existing_assignment(lab_name)

    if existing_assignment:
        assignment_id = existing_assignment['id']
        print(f"Assignment '{lab_name}' already exists, updating...")
        result = make_api_request('PUT', f'/courses/{COURSE_ID}/assignments/{assignment_id}', json=assignment_data)
        if result:
            print(f"✓ Successfully updated assignment: {lab_name}")
            print(f"  URL: {BASE_URL}/courses/{COURSE_ID}/assignments/{assignment_id}")
            return True
    else:
        print(f"Creating new assignment: {lab_name}")
        result = make_api_request('POST', f'/courses/{COURSE_ID}/assignments', json=assignment_data)
        if result:
            print(f"✓ Successfully created assignment: {lab_name}")
            print(f"  URL: {BASE_URL}/courses/{COURSE_ID}/assignments/{result['id']}")
            return True

    print(f"✗ Failed to create/update assignment: {lab_name}")
    return False

def find_lab_files():
    """Find all lab*.md files in the docs directory"""
    docs_root = Path('docs')
    lab_files = []

    # Search for lab*.md files
    for lab_file in docs_root.rglob('lab*.md'):
        lab_files.append(lab_file)

    # Sort by lab number
    def lab_sort_key(path):
        filename = path.stem
        # Extract number from lab file name (e.g., lab1 -> 1, lab2 -> 2)
        match = re.search(r'lab(\d+)', filename)
        return int(match.group(1)) if match else 0

    lab_files.sort(key=lab_sort_key)
    return lab_files

def get_all_assignments():
    """Get all assignments from Canvas"""
    print("Fetching all assignments from Canvas...")
    assignments = make_api_request('GET', f'/courses/{COURSE_ID}/assignments?per_page=100')

    if assignments is None:
        print("✗ Failed to fetch assignments")
        return []

    print(f"Found {len(assignments)} assignments")
    return assignments

def delete_lab_assignments(lab_assignments):
    """Delete lab assignments from Canvas"""
    if not lab_assignments:
        print("No lab assignments found to delete")
        return True

    print(f"Found {len(lab_assignments)} lab assignment(s) to delete:")
    for assignment in lab_assignments:
        print(f"  - {assignment['name']} (ID: {assignment['id']})")

    # Confirm deletion
    confirm = input(f"\nAre you sure you want to delete {len(lab_assignments)} lab assignment(s)? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Deletion cancelled")
        return False

    print("\nDeleting lab assignments...")
    successful = 0
    failed = 0

    for assignment in lab_assignments:
        print(f"Deleting: {assignment['name']}...")

        # Delete assignment
        response = make_api_request('DELETE', f'/courses/{COURSE_ID}/assignments/{assignment["id"]}')

        if response is not None:
            print(f"✓ Successfully deleted: {assignment['name']}")
            successful += 1
        else:
            print(f"✗ Failed to delete: {assignment['name']}")
            failed += 1

    print(f"\nDeletion Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {successful + failed}")

    return failed == 0

def filter_lab_assignments(assignments):
    """Filter assignments to only include lab assignments"""
    lab_assignments = []
    lab_pattern = re.compile(r'^Lab\s+\d+', re.IGNORECASE)

    for assignment in assignments:
        if lab_pattern.match(assignment.get('name', '')):
            lab_assignments.append(assignment)

    return lab_assignments

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Canvas Lab Assignment Management Script')
    parser.add_argument('--delete', action='store_true',
                       help='Delete all lab assignments from Canvas')
    args = parser.parse_args()

    if args.delete:
        print("Canvas Lab Assignment Deletion Script")
    else:
        print("Canvas Lab Assignment Upload Script")

    print("=" * 60)
    print(f"Canvas URL: {BASE_URL}")
    print(f"Course ID: {COURSE_ID}")
    print("=" * 60)

    # Test API connectivity
    print("Testing API connectivity...")
    course_info = make_api_request('GET', f'/courses/{COURSE_ID}')
    if not course_info:
        print("Failed to connect to Canvas API or access course")
        sys.exit(1)

    print(f"Connected to course: {course_info.get('name', 'Unknown')}")
    print("=" * 60)

    if args.delete:
        # Delete lab assignments
        assignments = get_all_assignments()
        if assignments:
            lab_assignments = filter_lab_assignments(assignments)
            if delete_lab_assignments(lab_assignments):
                print("✓ All lab assignments deleted successfully!")
            else:
                print("Some deletions failed. Check the output above for details.")
                sys.exit(1)
        return

    # Find lab files
    lab_files = find_lab_files()

    if not lab_files:
        print("No lab*.md files found in docs directory")
        sys.exit(1)

    print(f"Found {len(lab_files)} lab file(s):")
    for lab_file in lab_files:
        print(f"  - {lab_file}")
    print("=" * 60)

    # Process each lab file
    successful = 0
    failed = 0

    for lab_file in lab_files:
        lab_number = re.search(r'lab(\d+)', lab_file.stem).group(1)
        lab_name = f"Lab {lab_number}"
        print(f"\nProcessing {lab_name} ({lab_file})...")
        print("-" * 40)

        try:
            # Convert markdown to HTML
            print("Converting markdown to HTML...")
            html_content = markdown_to_html(lab_file)
            print(f"✓ Conversion successful ({len(html_content)} characters)")

            # Upload as assignment
            if create_or_update_assignment(lab_name, html_content):
                successful += 1
            else:
                failed += 1

        except Exception as e:
            print(f"✗ Error processing {lab_name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print("Lab Upload Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {successful + failed}")
    print("=" * 60)

    if failed > 0:
        print("Some lab uploads failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("✓ All lab assignments uploaded successfully!")

if __name__ == "__main__":
    main()