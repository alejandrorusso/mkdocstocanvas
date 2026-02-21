from api import canvas_client
import utils.cache
import utils.config
import canvas_processing

from pathlib import Path
import hashlib
import typer
from rich.console import Console
from datetime import datetime
import re

console = Console()
err_console = Console(stderr=True, style="bold red")


class MarkdownPage:
    def __init__(self, path: Path, root_path: Path = Path("docs")):
        self.path = path
        self.root_path = root_path
        self.rel_path = self.path.relative_to(root_path)
        if not self.path.exists():
            self.title = None
            self.content = ""
            self.md5 = None
            self.linked_files = {}
        else:
            self.title = self.get_title()
            self.content = self.get_content()
            self.md5 = compute_file_hash(self.path)
            self.linked_files = self.get_linked_files()

    def get_title(self) -> str | None:
        """Extract the first # header from a markdown file as the title"""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    # Check if it starts with exactly one # and a space
                    if line.startswith("# "):
                        # Slice off the '#' and strip any remaining whitespace
                        return line[1:].strip()

        except OSError as e:  # More specific than bare Exception
            err_console.print(f"⚠ Error reading file {self.path}: {e}")
        except UnicodeDecodeError as e:
            err_console.print(f"⚠ Encoding error in {self.path}: {e}")

        return None

    def get_linked_files(self):
        """
        Return a mapping of local file links referenced by the markdown file to their hashes.
        Considers ALL local files except markdown files.
        """
        # Simplest possible regex: Just grab whatever is inside the URL parentheses
        file_pattern = re.compile(
            r'\[([^\]]+)\]\(\s*([^\s\)]+)(?:\s+(?:"[^"]*"|\'[^\']*\'))?\s*\)',
            re.IGNORECASE,
        )

        try:
            content = Path(self.path).read_text(encoding="utf-8")
        except Exception:
            return {}

        markdown_dir = Path(self.path).parent
        docs_root_abs = self.root_path.resolve()
        linked_files = {}

        for match in file_pattern.finditer(content):
            file_path = match.group(2)

            # 1. Skip external web links, emails, and page anchor links
            if file_path.startswith(("http://", "https://", "mailto:", "#")):
                continue

            # 2. Skip Markdown files (even if they have anchors like page.md#section)
            if ".md" in file_path.lower():
                continue

            if file_path.startswith("./") or file_path.startswith("../"):
                full_path = (markdown_dir / file_path).resolve()
            else:
                candidate = Path(file_path)
                full_path = (
                    candidate
                    if candidate.is_absolute()
                    else (docs_root_abs / candidate).resolve()
                )
            try:
                rel_path = str(full_path.relative_to(docs_root_abs))
            except Exception:
                rel_path = str(full_path)
            if full_path.exists():
                linked_files[rel_path] = compute_file_hash(full_path)
            else:
                linked_files[rel_path] = None
        return linked_files

    def get_content(self) -> str:
        with open(self.path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Strip leading H1 title; Canvas uses the page title separately
        return re.sub(r"^\s*#\s+.*\n", "", md_content, count=1)

    def __repr__(self) -> str:
        # Returns a clean, readable summary
        return (
            f"--- Page Summary ---\n"
            f"Rel Path: {self.rel_path}\n"
            f"Title:    {self.title}\n"
            f"MD5:      {self.md5}\n"
            f"Links:    {len(self.linked_files)} found\n"
            f"--------------------"
        )


class PageUploader:
    def __init__(self, cache=".canvas_upload_state.json"):
        # Load cache
        self.cache_path = Path(cache)
        self.cache = utils.cache.load_cache(self.cache_path)

    def upload_all_pages(self, pages: list[MarkdownPage]) -> None:
        """
        Uploads all pages to canvas.
        """
        # Test connection
        success, message = canvas_client.test_connection()
        if not success:
            err_console.print(message)
            raise typer.Exit(1)
        console.print(message)

        pages_to_upload = self._pages_to_upload(pages)

        # Stage 1: Upload the pages
        failed = 0
        for page in pages_to_upload:
            try:
                console.print(f"Uploading {page.path}")
                canvas_url = self._upload_page(page)
                page_url_slug = canvas_url.split("/pages/")[-1]

                self.cache[str(page.rel_path)] = {
                    "page_title": page.title,
                    "hash": page.md5,
                    "canvas_url": canvas_url,
                    "page_url_slug": page_url_slug,
                    "last_upload": datetime.now().isoformat(),
                }
            except Exception as e:
                failed += 1
                err_console.print(f"Error uploading {page.path}: {e}")
        utils.cache.save_cache(self.cache_path, self.cache)

        # Stage 2: Resolve links

    def _upload_page(self, md_page: MarkdownPage) -> str:
        """
        Uploads page to canvas.
        """
        md_page.content = self._process_markdown_assets(md_page)

        html_page = canvas_processing.process_markdown_to_html(md_page)

        url_slug = (
            info.get("page_url_slug")
            if (info := self.cache.get(md_page.rel_path))
            else None
        )
        return canvas_client.create_or_update_page(
            md_page.title,  # pyright: ignore
            html_page,
            published=True,
            page_slug=url_slug,  # pyright: ignore
        )

    def _process_markdown_assets(self, md_page: MarkdownPage) -> str:
        """
        Unified processor for links and images.
        Uploads assets to Canvas and replaces references with resolved URLs.
        """
        # Pattern: Captures optional '!' prefix, [text], and (path)
        # Group 1: '!' if image, else None
        # Group 2: alt_text or link_text
        # Group 3: the file path
        combined_pattern = (
            r'(!)?\[([^\]]+)\]\(\s*([^\s\)]+)(?:\s+(?:"[^"]*"|\'[^\']*\'))?\s*\)'
        )

        docs_root_abs = md_page.root_path.resolve()
        markdown_dir = Path(md_page.path).parent
        linked_files = md_page.linked_files

        def replace_asset(match):
            is_image = match.group(1) == "!"
            text = match.group(2)
            file_path = match.group(3)

            # 1. Skip external URLs, emails, and page anchors
            if file_path.startswith(("http://", "https://", "mailto:", "#")):
                return match.group(0)

            # 2. Skip Markdown files (unless you want to upload them as attachments?)
            file_lower = file_path.lower()
            if not is_image and (
                file_lower.endswith((".md", ".markdown")) or ".md#" in file_lower
            ):
                return match.group(0)

            # 3. Resolve Path
            candidate = Path(file_path)
            if file_path.startswith(("./", "../")):
                full_path = (markdown_dir / candidate).resolve()
            else:
                full_path = (
                    candidate
                    if candidate.is_absolute()
                    else (docs_root_abs / candidate).resolve()
                )

            # 4. Validation & Upload Logic
            if full_path.exists() and not full_path.is_dir():
                rel_path = full_path.relative_to(docs_root_abs)

                # Images go to a specific folder, others preserve structure
                # parent_folder = (
                #     "/course_images" if is_image else f"/{rel_path.parent}".rstrip("/")
                # )
                parent_folder = f"/{rel_path.parent}".rstrip("/")

                # Cache Check
                info = self.cache.get(str(rel_path))
                if info and info.get("hash") == linked_files.get(str(rel_path)):
                    canvas_url = info["canvas_url"]
                else:
                    # Actual Upload
                    canvas_url = canvas_client.upload_file(full_path, parent_folder)
                    # Note: You should update your cache here or in the caller loop

                # 5. Return formatted string based on type
                if canvas_url:
                    if is_image:
                        return f'<img src="{canvas_url}" alt="{text}" />'
                    return f"[{text}]({canvas_url})"

            return match.group(0)

        return re.sub(
            combined_pattern,
            replace_asset,
            md_page.content,  # pyright: ignore
            flags=re.IGNORECASE,
        )

    def _pages_to_upload(self, pages: list[MarkdownPage]) -> list[MarkdownPage]:
        """
        )
        Gets a list of pages to uploads.

        Returns:
            A list of (path, title) of the md-files that needs to be uploaded.
        """
        pages_to_upload = []
        for md_page in pages:  # pyright: ignore
            if md_page.path.parent.name == "labs":
                continue

            if not md_page.path.exists():
                console.print(
                    f"⚠ Skipping {md_page.path}: file not found", style="bold yellow"
                )
                continue

            if md_page.title is None:
                console.print(
                    f"⚠ Skipping {md_page.path}: no title found", style="bold yellow"
                )
                continue

            if self._needs_upload(md_page):
                pages_to_upload.append(md_page)

        return pages_to_upload

    def _needs_upload(self, md_page: MarkdownPage) -> bool:
        """
        Check if a file needs to be uploaded based on its hash and Canvas state.
        """
        # Compute current hash
        if md_page.md5 is None:
            return True  # Failed to compute hash, force upload

        rel_path = str(md_page.path.relative_to(md_page.root_path))
        # Fetch metadata in one step
        cached_info = self.cache.get(rel_path)
        if cached_info is None:
            return True  # File not in metadata (new file)

        # Compare hashes
        if cached_info.get("hash") != md_page.md5:
            return True  # File content changed

        page_url_slug = cached_info.get("page_url_slug")
        if not page_url_slug:
            return True  # It has metadata, but no Canvas slug! Force upload.

        # Check Canvas page existence
        if not canvas_client.get_existing_page(page_url_slug):
            return True  # Page was deleted from Canvas, needs re-upload

        return False  # File unchanged and verified on Canvas


def parse_upload_all_pages(docs_root="docs", mkdocs_path="mkdocs.yml"):
    # docs/ directory
    docs_root = Path(docs_root)
    if not docs_root.exists():
        err_console.print("ERROR: docs/ directory not found")
        raise typer.Exit(1)

    # Parse mkdocs.yml
    mkdocs_path = Path(mkdocs_path)
    if not mkdocs_path.exists():
        err_console.print("ERROR: mkdocs.yml not found")
        raise typer.Exit(1)
    markdown_files = utils.config.parse_mkdocs_nav(mkdocs_path)
    if markdown_files is None:
        err_console.print(f"No markdown files found in {mkdocs_path}")
        raise typer.Exit(1)

    markdown_pages = [MarkdownPage(docs_root / p) for p in markdown_files]

    uploader = PageUploader()

    uploader.upload_all_pages(markdown_pages)


def compute_file_hash(path: Path) -> str | None:
    """Compute MD5 hash of a file's contents"""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError as e:
        err_console.print(f"⚠ Warning: Could not compute hash for {path}: {e}")
        return None
