.PHONY: serve serve-full pdf pdf-install-browser upload-pages upload-pages-force upload-modules upload-labs upload-all delete-pages delete-modules delete-labs delete-all rebuild docker-build test lint format typecheck check clean

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

# Install the Playwright browser needed for PDF generation (run once)
pdf-install-browser:
	@$(call canvas,pdf --install-browser)

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

# ⚠️ Complete rebuild: deletes ALL Canvas content (no confirmation), drops the
# upload cache, then re-uploads everything. Use only for fresh setup or full
# resets. (upload-all has no --force option — dropping the cache is what forces
# the full re-upload.)
rebuild:
	@$(call canvas,delete-all --force)
	@rm -f .canvas_upload_state.json
	@$(call canvas,upload-all)

# ---------------------------------------------------------------
# Misc
# ---------------------------------------------------------------

# Run the test suite
test:
	uv run pytest

# Lint with ruff
lint:
	uv run ruff check

# Auto-format and auto-fix lint issues
format:
	uv run ruff check --fix
	uv run ruff format

# Type-check with pyright
typecheck:
	uv run pyright

# Everything CI would run
check: lint typecheck test

# Build the Docker image (CI/CD pipelines pull it from GHCR instead —
# see documentation/CI-CD.md and .github/workflows/docker-publish.yml)
docker-build:
	docker build -t mkdocstocanvas:local .

# Clean generated files (the upload cache .canvas_upload_state.json is kept)
clean:
	rm -rf site/ pdf/
	@echo "✓ Cleaned site/ and pdf/ directories"
