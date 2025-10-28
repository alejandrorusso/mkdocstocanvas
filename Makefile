.PHONY: serve serve-full pdf delete-pages delete-modules delete-labs upload-pages upload-modules upload-labs clean

# Serve the documentation locally (fast mode - better live reload)
serve:
	mkdocs serve -a 0.0.0.0:8000 --watch-theme --livereload

# Serve with all plugins enabled (slower reload)
serve-full:
	mkdocs serve -a 0.0.0.0:8000

# Build site and generate PDFs (dependency for upload-modules)
pdf:
	mkdocs build --clean
	mkdir -p pdf
	find site -name "*.pdf" -exec cp {} pdf/ \;
	@echo "✓ PDFs generated in /pdf directory"

# Delete all pages from Canvas
delete-pages:
	@echo "⚠️  Deleting all pages from Canvas..."
	bash -c "source ./tokens.sh && echo 'yes' | python3 scripts/upload_all_pages_to_canvas.py --delete-pages"

# Delete all modules from Canvas
delete-modules:
	@echo "⚠️  Deleting all modules from Canvas..."
	bash -c "source ./tokens.sh && echo 'yes' | python3 scripts/upload_modules_to_canvas.py --delete-only"

# Delete all lab assignments from Canvas
delete-labs:
	@echo "⚠️  Deleting all lab assignments from Canvas..."
	bash -c "source ./tokens.sh && echo 'yes' | python3 scripts/upload_labs_to_canvas.py --delete"

# Upload all pages to Canvas (with align environment conversion)
upload-pages:
	@echo "📄 Uploading all pages to Canvas..."
	bash -c "source ./tokens.sh && python3 scripts/upload_all_pages_to_canvas.py"

# Upload modules to Canvas (includes pages + PDFs) - requires PDFs to be built first
upload-modules: pdf
	@echo "📚 Uploading modules (pages + PDFs) to Canvas..."
	bash -c "source ./tokens.sh && python3 scripts/upload_modules_to_canvas.py"

# Upload lab assignments to Canvas
upload-labs:
	@echo "🧪 Uploading lab assignments to Canvas..."
	bash -c "source ./tokens.sh && python3 scripts/upload_labs_to_canvas.py"

# Clean generated files
clean:
	rm -rf site/ pdf/
	@echo "✓ Cleaned site/ and pdf/ directories"
