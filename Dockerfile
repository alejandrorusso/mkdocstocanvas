# Canvas course publisher container
# Provides an isolated environment for MkDocs + Canvas upload tooling

FROM python:3.14-slim-bookworm

# Copy uv binary from the official image
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# System deps: fonts for PDF rendering and Playwright runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      fonts-noto \
      fonts-noto-cjk \
      fonts-noto-color-emoji \
      fonts-dejavu \
      libxss1 \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies only (not the local package) — this layer is cached
# as long as pyproject.toml and uv.lock don't change
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Make the venv's binaries available without activating it
ENV PATH="/app/.venv/bin:$PATH"

# Preinstall Playwright browser — cached independently of source code changes
RUN playwright install --only-shell chromium && \
    playwright install-deps chromium

# Install the local package — only this layer reruns when src/ changes
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Mount your project directory at runtime:
#   docker run --rm -it -p 8000:8000 -v $(pwd):/workspace -w /workspace --env-file .env <image>
WORKDIR /workspace

CMD ["/bin/bash"]

