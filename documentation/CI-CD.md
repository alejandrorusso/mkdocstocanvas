# Continuous deployment to Canvas

Automatically upload course content to Canvas whenever you push to your content
repository, using the mkdocstocanvas Docker image. No local Python setup is
needed on the CI runner — the image contains the CLI, mkdocs, and a
pre-installed Chromium (for PDF generation).

```
git push  ──▶  CI runner  ──▶  docker run mkdocstocanvas upload-all  ──▶  Canvas
               (checkout)      (mounts the course directory, reads
                                credentials from CI secrets)
```

## Prerequisites

1. **A published image.** This repository ships a workflow that publishes the
   image to GHCR on every push to `main` (see
   `.github/workflows/docker-publish.yml`). The image is
   `ghcr.io/<owner>/mkdocstocanvas` — replace `<owner>` with your GitHub
   username/org in the examples below. If you don't want a registry, see
   [Building the image in CI](#building-the-image-in-ci) instead.

   > **GHCR visibility:** the first published package is **private**, and the
   > `GITHUB_TOKEN` of another repository cannot pull it by default. Either
   > make the package public (*Package → Package settings → Change
   > visibility*) or grant the content repository access (*Package settings →
   > Manage Actions access → add the content repository*).
2. **A content repository** laid out like the examples: `mkdocs.yml` at the
   root plus a `docs/` directory (and `docs/labs/` for labs). Nothing else is
   required — no `.venv`, no `uv`.
3. **Canvas credentials as CI secrets.** Create a Canvas API token
   (Canvas → *Account → Settings → Approved Integrations → New Access Token*),
   then add these secrets to the content repository
   (GitHub → *Settings → Secrets and variables → Actions*):

   | Secret             | Value                                            |
   |--------------------|--------------------------------------------------|
   | `CANVAS_API_TOKEN` | your Canvas API token                            |
   | `CANVAS_BASE_URL`  | your Canvas host, e.g. `https://canvas.example.edu` |
   | `CANVAS_COURSE_ID` | numeric course ID from the course URL            |

   Never commit a `.env` file with real credentials — the CI examples read the
   values from the secret store instead.

## GitHub Actions

Minimal workflow that uploads pages, labs and modules on every push to `main`:

```yaml
# .github/workflows/upload-canvas.yml  (in the CONTENT repository)
name: Upload to Canvas

on:
  push:
    branches: [main]

concurrency:
  group: canvas-upload      # serialize uploads; never run two at once
  cancel-in-progress: false

jobs:
  upload:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Upload to Canvas
        run: |
          docker run --rm \
            -v "$PWD":/workspace -w /workspace \
            -e CANVAS_API_TOKEN -e CANVAS_BASE_URL -e CANVAS_COURSE_ID \
            ghcr.io/<owner>/mkdocstocanvas:latest \
            upload-all
        env:
          CANVAS_API_TOKEN: ${{ secrets.CANVAS_API_TOKEN }}
          CANVAS_BASE_URL: ${{ secrets.CANVAS_BASE_URL }}
          CANVAS_COURSE_ID: ${{ secrets.CANVAS_COURSE_ID }}
```

The image's entrypoint is the CLI itself, so the last line is simply the
subcommand. `upload-all` uploads labs → pages → labs (two passes so
cross-links between pages and labs resolve) → modules.

### With PDFs

To also generate one PDF per page and attach it inside each module, use the
`--add-pdf` flag (Chromium and fonts are pre-installed in the image, so PDF
rendering works out of the box):

```yaml
            ghcr.io/<owner>/mkdocstocanvas:latest \
            upload-all --add-pdf
```

### As a container job

The same thing using GitHub's `container:` syntax. The image contains `git`,
so `actions/checkout` works inside it:

```yaml
jobs:
  upload:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/<owner>/mkdocstocanvas:latest
    env:
      CANVAS_API_TOKEN: ${{ secrets.CANVAS_API_TOKEN }}
      CANVAS_BASE_URL: ${{ secrets.CANVAS_BASE_URL }}
      CANVAS_COURSE_ID: ${{ secrets.CANVAS_COURSE_ID }}
    steps:
      - uses: actions/checkout@v4
      - run: mkdocstocanvas upload-all
```

### Caching the upload state (optional)

Each run starts without `.canvas_upload_state.json`, the file that drives
incremental uploads. That is **safe**: on an empty cache the CLI rebuilds the
state from the ownership markers on the existing Canvas pages before
uploading (see [Recovering a lost
cache](HOW_IT_WORKS.md#recovering-a-lost-cache)), so nothing is re-created or
duplicated — images and files are deduplicated by content hash regardless.
The rebuild just costs some extra API calls.

To skip the rebuild, persist the cache between runs:

```yaml
      - uses: actions/cache@v4
        with:
          path: .canvas_upload_state.json
          key: canvas-${{ github.sha }}
          restore-keys: canvas-
```

### Building the image in CI

If you don't want to publish the image to a registry (e.g. the tool repository
is private and the runner can't pull GHCR packages), build it on the fly from
a second checkout:

```yaml
      - uses: actions/checkout@v4
      - uses: actions/checkout@v4
        with:
          repository: <owner>/mkdocstocanvas
          path: tool
      - name: Build image
        run: docker build -t mkdocstocanvas ./tool
      - name: Upload to Canvas
        run: |
          docker run --rm \
            -v "$PWD":/workspace -w /workspace \
            -e CANVAS_API_TOKEN -e CANVAS_BASE_URL -e CANVAS_COURSE_ID \
            mkdocstocanvas upload-all
        env:
          CANVAS_API_TOKEN: ${{ secrets.CANVAS_API_TOKEN }}
          CANVAS_BASE_URL: ${{ secrets.CANVAS_BASE_URL }}
          CANVAS_COURSE_ID: ${{ secrets.CANVAS_COURSE_ID }}
```

## GitLab CI

Add the token variables in *Settings → CI/CD → Variables* (mask
`CANVAS_API_TOKEN`), then:

```yaml
# .gitlab-ci.yml  (in the CONTENT repository)
upload-canvas:
  stage: deploy
  image:
    name: ghcr.io/<owner>/mkdocstocanvas:latest
    entrypoint: [""]          # let GitLab run its own script
  script:
    - mkdocstocanvas upload-all
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

## Notes and safety

- **Serialize uploads.** Two simultaneous `upload-all` runs against the same
  course can interleave badly. Use a `concurrency` group (GitHub) or
  `resource_group` (GitLab) so runs queue instead of racing.
- **Deletes stay manual.** The `delete-*` commands are destructive and are not
  meant for pipelines; run them locally against the course when needed.
- **First CI run is slower.** With a cold cache the CLI walks the existing
  Canvas pages to rebuild the state (see above) — afterwards uploads are
  incremental.
- **Try it safely first.** Point `CANVAS_COURSE_ID` at a sandbox/test course
  and push to a scratch branch before wiring up the real course.
- **Local one-off with the same image** (files are written as your own user):

  ```bash
  docker run --rm -it --user "$(id -u):$(id -g)" \
    -v "$PWD":/workspace -w /workspace --env-file .env \
    mkdocstocanvas:local upload-all
  ```
