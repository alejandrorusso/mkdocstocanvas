# Canvas course publisher container
# Provides an isolated environment for MkDocs + Canvas upload tooling

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# System deps: fonts for PDF rendering and Playwright runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      fonts-noto \
      fonts-noto-cjk \
      fonts-noto-color-emoji \
      fonts-dejavu \
      libxss1 \
      make \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (leverages uv's cache for fast rebuilds)
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Make the venv's binaries available without activating it
ENV PATH="/app/.venv/bin:$PATH"

# Preinstall Playwright browser and its runtime dependencies for mkdocs-page-pdf
RUN playwright install chromium && \
    playwright install-deps chromium

# Mount your project directory here at runtime:
#   docker run --rm -v $(pwd):/workspace -w /workspace --env-file .env <image> mkdocstocanvas upload-all
WORKDIR /workspace

LABEL org.opencontainers.image.source=https://github.com/Paenda/mkdocstocanvas

CMD ["/bin/bash"]

