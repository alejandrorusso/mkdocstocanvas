.PHONY: serve serve-full pdf upload-pages upload-pages-force upload-modules upload-labs upload-all delete-pages delete-modules delete-labs delete-all rebuild test clean

# Host/port for the local dev server (override with: make serve PORT=9000)
HOST ?= 0.0.0.0
PORT ?= 8000

# Runs the CLI. Canvas credentials are read automatically from a .env file
# (copy .env.example to create it) or from the environment.
define canvas
uv run mkdocstocanvas $(1)
endef

# ---------------------------------------------------------------
# Local development
# ---------------------------------------------------------------

# Serve the documentation locally (fast mode - better live reload)
serve:
	@$(call canvas,serve -a $(HOST):$(PORT))

# Serve with all plugins enabled (slower reload)
serve-full:
	@$(call canvas,serve -a $(HOST):$(PORT) --all-plugins)

# Build site and collect generated PDFs into pdf/
pdf:
	@$(call canvas,pdf)

# ---------------------------------------------------------------
# Upload to Canvas
# ---------------------------------------------------------------

# Upload pages (incremental - only changed files)
upload-pages:
	@$(call canvas,upload-pages)

# Force upload pages (ignores the upload cache)
upload-pages-force:
	@$(call canvas,upload-pages --force)

# Upload pages + build modules with PDFs attached
upload-modules:
	@$(call canvas,upload-modules --add-pdf)

# Upload lab*.md files as Canvas assignments
upload-labs:
	@$(call canvas,upload-labs)

# Upload everything: pages, modules and labs
upload-all:
	@$(call canvas,upload-all)

# ---------------------------------------------------------------
# Delete from Canvas
# ---------------------------------------------------------------
# ⚠️ These permanently delete Canvas content. They ask for
# confirmation unless you pass --force (not used here).

delete-pages:
	@$(call canvas,delete-pages)

delete-modules:
	@$(call canvas,delete-modules)

delete-labs:
	@$(call canvas,delete-labs)

# Delete everything: pages, modules and labs
delete-all:
	@$(call canvas,delete-all)

# ⚠️ Complete rebuild: deletes ALL Canvas content (no confirmation),
# then re-uploads everything. Use only for fresh setup or full resets.
rebuild:
	@$(call canvas,delete-all --force)
	@$(call canvas,upload-all --force)

# ---------------------------------------------------------------
# Misc
# ---------------------------------------------------------------

# Run the test suite
test:
	uv run pytest

# Clean generated files (the upload cache .canvas_upload_state.json is kept)
clean:
	rm -rf site/ pdf/
	@echo "✓ Cleaned site/ and pdf/ directories"
