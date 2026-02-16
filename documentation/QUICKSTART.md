# Quick Start Guide

Get your Canvas course published in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install --upgrade Pygments && \
pip install pymdown-extensions && \
pip install mkdocs-include-markdown-plugin && \
pip install mkdocs-material && \
pip install mkdocs-excel-plugin && \
pip install mkdocs-page-pdf && \
pip install mkdocs
```

```bash 
apt-get update \
   && apt-get install -y wget gnupg \
   && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
   && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list' \
   && apt-get update \
   && apt-get install -y google-chrome-stable fonts-ipafont-gothic fonts-wqy-zenhei fonts-thai-tlwg fonts-kacst fonts-freefont-ttf libxss1 \
      --no-install-recommends \
   && rm -rf /var/lib/apt/lists/* \
   && npm init -y &&  \
      npm i puppeteer
```


## Step 2: Configure Canvas Credentials

```bash
# Copy the template
cp tokens.sh.example tokens.sh

# Edit with your actual credentials
nano tokens.sh  # or use your preferred editor
```

Add your Canvas API token, base URL, and course ID:

```bash
export CANVAS_API_TOKEN="1234~abcdefghijklmnop..."
export CANVAS_BASE_URL="https://your-school.instructure.com"
export CANVAS_COURSE_ID="12345"
```

## Step 3: Preview Locally (Optional)

```bash
make serve
```

Visit `http://localhost:8000` to preview your content.

## Step 4: Publish to Canvas

```bash
make pdf  # It generates a PDF per module  
make upload-pages # It uploads pages
make upload-modules # It creates modules and fill them in with pages
```

## Step 5: Check Canvas

Visit your Canvas course and verify:
- ✓ Modules are created
- ✓ Pages are visible
- ✓ PDFs are attached
- ✓ Everything is published

## Next Steps

- Customize content in `docs/` directory
- Modify `mkdocs.yml` to change course structure
- Run `make delete-labs && make delete-modules && make delete-pages` and then 
      `make pdf && make upload-pages && make upload-modules && make upload-labs`


