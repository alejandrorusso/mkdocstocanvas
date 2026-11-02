# Canvas course publisher container
# Provides an isolated environment for MkDocs + Canvas upload tooling, designed
# for CI/CD: mount a course directory (mkdocs.yml + docs/) and run the CLI:
#
#   docker run --rm -v "$PWD":/workspace -w /workspace \
#     -e CANVAS_API_TOKEN -e CANVAS_BASE_URL -e CANVAS_COURSE_ID \
#     ghcr.io/<owner>/mkdocstocanvas upload-all
#
# See documentation/CI-CD.md for complete pipeline examples.

FROM python:3.14-slim-bookworm

# Copy uv binary from the official image
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    # Fixed, HOME-independent browser location: the image works no matter which
    # user (--user 1000:1000) or HOME the container runs with
    PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright

WORKDIR /app

# Install dependencies only (not the local package) — this layer is cached
# as long as pyproject.toml and uv.lock don't change. The uv download cache
# lives in a build-cache mount and is not baked into the image.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Make the venv's binaries available without activating it
ENV PATH="/app/.venv/bin:$PATH"

# Preinstall the Playwright Chromium headless shell used by mkdocs-page-pdf —
# cached independently of source code changes
RUN playwright install --only-shell chromium

# System libraries for Chromium plus the fonts it needs to render PDF text,
# and git so CI `container:` jobs can check out the course repository
RUN apt-get update && \
    playwright install-deps chromium && \
    apt-get install -y --no-install-recommends \
      git \
      fonts-noto \
      fonts-noto-cjk \
      fonts-noto-color-emoji \
      fonts-dejavu \
      libxss1 \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    chmod -R a+rX /opt/ms-playwright

# Install the local package — only this layer reruns when src/ changes.
# The smoke test fails the build immediately if the CLI is broken.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev && \
    mkdocstocanvas --help > /dev/null

LABEL org.opencontainers.image.title="mkdocstocanvas" \
      org.opencontainers.image.description="Upload MkDocs course content to Canvas LMS" \
      org.opencontainers.image.licenses="MPL-2.0"

# Mount your course directory at runtime and run the CLI directly, e.g.
#   docker run --rm -v "$PWD":/workspace ghcr.io/<owner>/mkdocstocanvas upload-all
# A shell for interactive use or compound commands:
#   docker run --entrypoint /bin/bash -it <image>
#   docker run --entrypoint /bin/sh <image> -c "mkdocstocanvas pdf && mkdocstocanvas upload-all"
WORKDIR /workspace

ENTRYPOINT ["mkdocstocanvas"]
CMD ["--help"]
