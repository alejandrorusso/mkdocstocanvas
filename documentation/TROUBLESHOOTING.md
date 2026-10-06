# Troubleshooting

Solutions to common problems, most-specific first. Setup instructions live in the [README](../README.md); tool internals are explained in [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## `mkdocs.yml` or `docs/` missing after cloning

These are gitignored working copies — create them from the tracked templates:

```bash
cp mkdocs.example.yml mkdocs.yml
cp -r docs-example docs
```

## No PDFs generated

- PDF generation needs a Playwright browser: run `mkdocstocanvas pdf --install-browser` once
- Make sure the `page-to-pdf` plugin is enabled in `mkdocs.yml`

## Duplicate pages being created

- The tool automatically updates existing pages (matched by slug and title)
- If you see duplicates, run `mkdocstocanvas rebuild-cache`, then `mkdocstocanvas upload-pages`
- Use `mkdocstocanvas delete-pages` to clean up Canvas, then `mkdocstocanvas upload-pages --force`

## Pages not updating

- Force upload to bypass the cache: `mkdocstocanvas upload-pages --force`
- If the cache is out of sync: `mkdocstocanvas rebuild-cache`
- `--force` re-uploads page bodies (useful to overwrite manual edits made in Canvas) but never duplicates assets

## Canvas API errors

- Verify your Canvas API token in `.env`
- Check the course ID is correct
- Ensure the token has proper permissions (manage course content)

## Math formulas not rendering

- Enable MathJax in Canvas: Course Settings → Feature Options → "New Math Equation Editor"
- Check LaTeX syntax is correct
- Test locally with `mkdocstocanvas serve` first

## Images not displaying

- Verify image paths are relative to the markdown file location
- Check images exist in the `docs/` directory
- Images are automatically uploaded to Canvas

## Excel sheets not rendering

- Ensure `openpyxl` is installed (included in project dependencies)
- Place Excel files in the `docs/` directory
- Use the correct sheet name in the render command

## Still stuck?

1. Check the error messages in the console output
2. Verify all prerequisites are installed
3. Test with `mkdocstocanvas serve` locally first
4. Check Canvas permissions and the API token
5. Review the example course structure in `docs-example/`
