"""
Shared base class for page, module, and lab uploaders.

Handles file/image asset uploading and internal `.md` → Canvas URL link
resolution so that every uploader gets consistent behaviour without
duplicating logic.
"""

import re
from datetime import datetime
from pathlib import Path

from ..utils import cache as utils_cache
from ..api.canvas import CanvasUploader
from ..models.page import MarkdownPage, compute_file_hash
from ..processing.markdown import process_markdown_to_html

# Matches [text](path/to/page.md) and [text](path/to/page.md#anchor)
_MD_LINK_PATTERN = re.compile(
    r"\[([^\]]*)\]\(\s*([^\s\)#]+\.(?:md|markdown))(#[^\s\)\"']*)?(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)",
    re.IGNORECASE,
)

# Matches optional leading '!', [alt/text], and (path) — for assets
_ASSET_PATTERN = re.compile(
    r'(!)?\[([^\]]+)\]\(\s*([^\s\)]+)(?:\s+(?:"[^"]*"|\'[^\']*\'))?\s*\)',
)


class ContentUploader:
    """
    Base class that provides:
    - Shared cache (pages + files) loading/saving.
    - `_process_markdown_assets`: uploads local images/files to Canvas and
      rewrites references with the returned Canvas URL.
    - `_resolve_page_links`: rewrites internal `[text](page.md)` links to the
      corresponding Canvas page URL stored in `pages_cache`.
    - `_prepare_content`: runs both transformations then calls
      `process_markdown_to_html`, returning finished HTML.
    """

    def __init__(
        self,
        client: CanvasUploader,
        cache: str = ".canvas_upload_state.json",
        force: bool = False,
    ) -> None:
        self.client = client
        self.force = force
        self.cache_path = Path(cache)
        raw = utils_cache.load_cache(self.cache_path)
        self.pages_cache: dict[str, dict] = raw.get("pages", {})
        self.files_cache: dict[str, dict] = raw.get("files", {})

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _save_cache(self) -> None:
        """Persist pages_cache and files_cache to disk."""
        try:
            utils_cache.save_cache(
                self.cache_path,
                {"pages": self.pages_cache, "files": self.files_cache},
            )
        except OSError as e:
            raise RuntimeError(
                "Cannot write upload cache at "
                f"'{self.cache_path}'. Cache must be writable to keep links in sync. "
                "Fix ownership/permissions (e.g. chown/chmod) and rerun. "
                f"Original error: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Content pipeline
    # ------------------------------------------------------------------

    def _prepare_content(self, md_page: MarkdownPage) -> str:
        """
        Full content pipeline:
        1. Upload local assets (images, files) and rewrite links.
        2. Resolve internal `.md` page links to Canvas URLs.
        3. Convert markdown to Canvas-compatible HTML.

        Returns the finished HTML string.
        """
        # Important: always start from source markdown on each upload attempt.
        # `md_page.content` is mutated by this pipeline; without resetting,
        # re-uploads can keep stale already-rewritten Canvas links.
        md_page.content = md_page.get_content()
        md_page.content = self._process_markdown_assets(md_page)
        md_page.content = self._resolve_page_links(md_page)
        return process_markdown_to_html(md_page)

    def _process_markdown_assets(self, md_page: MarkdownPage) -> str:
        """
        Upload local images and files referenced in *md_page* to Canvas and
        replace each reference with the Canvas-hosted URL.

        Supports both image syntax (``![alt](path)``) and plain link syntax
        (``[text](path)``).  External URLs, ``mailto:`` links, anchor-only
        links, and ``.md`` file links are left untouched.
        """
        docs_root_abs = md_page.root_path.resolve()
        markdown_dir = Path(md_page.path).parent

        def replace_asset(match: re.Match) -> str:
            is_image = match.group(1) == "!"
            text = match.group(2)
            file_path = match.group(3)

            # Skip external URLs, emails, and page anchors
            if file_path.startswith(("http://", "https://", "mailto:", "#")):
                return match.group(0)

            # Skip .md links — those are handled by _resolve_page_links
            file_lower = file_path.lower()
            if not is_image and (
                file_lower.endswith((".md", ".markdown")) or ".md#" in file_lower
            ):
                return match.group(0)

            # Resolve the local path
            candidate = Path(file_path)
            if file_path.startswith(("./", "../")):
                full_path = (markdown_dir / candidate).resolve()
            else:
                full_path = (
                    candidate
                    if candidate.is_absolute()
                    else (docs_root_abs / candidate).resolve()
                )

            if not full_path.exists() or full_path.is_dir():
                return match.group(0)

            try:
                rel_path = full_path.relative_to(docs_root_abs)
            except ValueError:
                return match.group(0)

            rel_path_str = str(rel_path)
            parent_folder = f"/{rel_path.parent}".rstrip("/")

            if parent_folder in ("", "/."):
                parent_folder = "/course_files"

            current_hash = compute_file_hash(full_path)
            cached = self.files_cache.get(rel_path_str)

            if not self.force and cached and cached.get("hash") == current_hash:
                canvas_url = cached["canvas_url"]
            else:
                canvas_url = self.client.upload_file(full_path, parent_folder)
                if canvas_url:
                    self.files_cache[rel_path_str] = {
                        "hash": current_hash,
                        "canvas_url": canvas_url,
                        "last_upload": datetime.now().isoformat(),
                    }

            if canvas_url:
                if is_image:
                    return f'<img src="{canvas_url}" alt="{text}" />'
                return f"[{text}]({canvas_url})"

            return match.group(0)

        return _ASSET_PATTERN.sub(replace_asset, md_page.content)

    def _resolve_page_links(self, md_page: MarkdownPage) -> str:
        """
        Rewrite ``[text](other/page.md[#anchor])`` links to the Canvas page
        URL stored in *pages_cache* for the target page.

        Links whose target is not in the cache (not yet uploaded) are left
        unchanged.
        """
        docs_root_abs = md_page.root_path.resolve()

        def replace_md_link(match: re.Match) -> str:
            text = match.group(1)
            link_path = match.group(2)
            anchor = match.group(3) or ""

            resolved = (md_page.path.parent / link_path).resolve()
            try:
                rel = str(resolved.relative_to(docs_root_abs))
            except ValueError:
                return match.group(0)

            info = self.pages_cache.get(rel)
            if info and info.get("canvas_url"):
                return f"[{text}]({info['canvas_url']}{anchor})"

            return match.group(0)

        return _MD_LINK_PATTERN.sub(replace_md_link, md_page.content)
