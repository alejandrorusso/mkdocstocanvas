#!/bin/bash

# Rebuild Canvas Course Script
# This script rebuilds the entire Canvas course content:
# 1. Delete modules
# 2. Delete pages
# 3. Delete lab assignments
# 4. Generate PDFs
# 5. Upload pages
# 6. Create modules
# 7. Upload lab assignments

set -e  # Exit on error

echo "========================================================================"
echo "Canvas Course Rebuild Script"
echo "========================================================================"
echo ""

make clean

# Source tokens
source ./tokens.sh

echo "Step 1/7: Deleting modules..."
echo "------------------------------------------------------------------------"

make delete-modules

echo "Step 2/7: Deleting pages..."
echo "------------------------------------------------------------------------"

make delete-pages

echo "Step 3/7: Deleting lab assignments..."
echo "------------------------------------------------------------------------"

make delete-labs

echo "Step 4/7: Generating PDFs..."
echo "------------------------------------------------------------------------"

make pdf

echo "Step 5/7: Uploading pages..."
echo "------------------------------------------------------------------------"

make upload-pages

echo "Step 6/7: Creating modules..."
echo "------------------------------------------------------------------------"

make upload-modules

echo "Step 7/7: Uploading lab assignments..."
echo "------------------------------------------------------------------------"

make upload-labs

echo "========================================================================"
echo "✓ Canvas course rebuild completed successfully!"
echo "========================================================================"
echo ""
echo "Course available at:"
echo "  Modules: ${CANVAS_BASE_URL}/courses/${CANVAS_COURSE_ID}/modules"
echo "  Pages: ${CANVAS_BASE_URL}/courses/${CANVAS_COURSE_ID}/pages"
echo "  Assignments: ${CANVAS_BASE_URL}/courses/${CANVAS_COURSE_ID}/assignments"
echo ""
