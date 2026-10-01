"""
Shared base class for page, module, and lab uploaders.

Handles file/image asset uploading and internal `.md` → Canvas URL link
resolution so that every uploader gets consistent behaviour without
duplicating logic.
"""

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ..api.canvas import CanvasUploader
from ..models.page import MarkdownPage, compute_file_hash
from ..processing.markdown import process_markdown_to_html
from ..utils import cache as utils_cache
from ..utils.marker import make_marker, parse_marker

# Shared consoles — all uploaders print through these so output is consistent.
console = Console()
err_console = Console(stderr=True, style="bold red")


def make_progress() -> Progress:
    """Standard transient progress bar used by all upload/delete flows."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def require_connection(client: CanvasUploader) -> None:
    """Test the Canvas connection; print the result and exit on failure."""
    success, message = client.test_connection()
    if not success:
        err_console.print(message)
        raise typer.Exit(1)
    console.print(message)


def print_results_summary(
    title: str,
    results: list[dict],
    *,
    name_column: str,
    detail_column: str,
    detail_of: Callable[[dict], str],
    ok_label: str,
) -> None:
    """Print a rich summary table for an upload run."""
    ok = sum(1 for r in results if r["status"] == "ok")
    errors = sum(1 for r in results if r["status"] == "error")

    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Status", min_width=12, no_wrap=True)
    table.add_column(name_column)
    table.add_column(detail_column)
    for r in results:
        status = (
            f"[green]✓ {ok_label}[/green]"
            if r["status"] == "ok"
            else "[red]✗ Failed[/red]"
        )
        table.add_row(status, r["name"], detail_of(r))
    console.print(table)
    console.print(
        f"[bold]Total:[/bold] {len(results)} | "
        f"[green]{ok_label}: {ok}[/green] | "
        f"[red]Failed: {errors}[/red]"
    )


def delete_all_items(
    items: list[dict],
    delete: Callable[[dict], bool],
    *,
    noun: str,
    name_column: str,
    columns: list[tuple[str, Literal["left", "center", "right"] | None]],
    row_values: Callable[[dict], tuple[str, str]],
    assume_yes: bool = False,
) -> None:
    """
    Shared deletion flow: preview table, confirmation, progress bar,
    and a summary table.

    Args:
        items: The Canvas objects to delete, as returned by the list API.
        delete: Called once per item; must return True on success.
        noun: Entity name for messages, e.g. "page" or "module".
        name_column: Header of the name column in the summary table.
        columns: (header, justify) pairs for the preview table.
        row_values: Returns the (name, extra) values shown for each item.
        assume_yes: Skip the interactive confirmation (the --force flag).
    """
    if not items:
        console.print(f"No {noun}s found. Nothing to delete.")
        return

    preview = Table(show_header=True, header_style="bold yellow")
    for header, justify in columns:
        if justify:
            preview.add_column(header, justify=justify)
        else:
            preview.add_column(header)
    for item in items:
        preview.add_row(*row_values(item))
    console.print(preview)
    console.print(
        f"[bold yellow]⚠ {len(items)} {noun}(s) will be permanently deleted.[/bold yellow]"
    )

    if not assume_yes:
        typer.confirm(f"Are you sure you want to delete ALL these {noun}s?", abort=True)

    deleted = 0
    failed = 0
    results: list[dict] = []

    with make_progress() as progress:
        task = progress.add_task(f"[red]Deleting {noun}s...", total=len(items))
        for item in items:
            name = row_values(item)[0]
            progress.update(task, description=f"[red]Deleting [bold]{name}[/bold]...")
            if delete(item):
                deleted += 1
                results.append({"name": name, "status": "ok"})
            else:
                failed += 1
                results.append({"name": name, "status": "error"})
            progress.advance(task)

    summary = Table(
        title="Deletion Summary", show_header=True, header_style="bold cyan"
    )
    summary.add_column("Status", min_width=12, no_wrap=True)
    summary.add_column(name_column)
    for r in results:
        if r["status"] == "ok":
            summary.add_row("[green]✓ Deleted[/green]", r["name"])
        else:
            summary.add_row("[red]✗ Failed[/red]", r["name"])
    console.print(summary)
    console.print(
        f"[bold]Total:[/bold] {len(items)} | "
        f"[green]Deleted: {deleted}[/green] | "
        f"[red]Failed: {failed}[/red]"
    )

    if failed and not deleted:
        raise typer.Exit(code=1)


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
        self.modules_cache: dict[str, dict] = raw.get("modules", {})

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _save_cache(self) -> None:
        """Persist pages_cache, files_cache and modules_cache to disk."""
        try:
            utils_cache.save_cache(
                self.cache_path,
                {
                    "pages": self.pages_cache,
                    "files": self.files_cache,
                    "modules": self.modules_cache,
                },
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
        4. Prepend the ownership marker (used for scoped deletes and
           cache rebuilding).

        Returns the finished HTML string.
        """
        # Important: always start from source markdown on each upload attempt.
        # `md_page.content` is mutated by this pipeline; without resetting,
        # re-uploads can keep stale already-rewritten Canvas links.
        md_page.content = md_page.get_content()
        md_page.content = self._process_markdown_assets(md_page)
        md_page.content = self._resolve_page_links(md_page)
        html = process_markdown_to_html(md_page)
        marker = make_marker(str(md_page.rel_path), md_page.md5)
        return f"{marker}\n{html}"

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

            # Assets are content-addressed, so they are never force-uploaded:
            # the local cache is consulted, then Canvas is checked for an
            # identical file by hash. Only a true miss triggers an upload.
            # (--force only skips the local cache entry, not the dedupe.)
            if cached and cached.get("hash") == current_hash and not self.force:
                canvas_url = cached["canvas_url"]
            else:
                canvas_url = None
                # Skip re-uploading if an identical file already exists in
                # Canvas (e.g. after the local cache file was lost).
                if current_hash:
                    existing = self.client.find_existing_file(
                        md5=current_hash,
                        filename=full_path.name,
                        size=full_path.stat().st_size,
                    )
                    if existing:
                        file_id = existing.get("id")
                        if file_id is not None:
                            canvas_url = (
                                f"{self.client.base_url}/courses/"
                                f"{self.client.course_id}/files/{file_id}?wrap=1"
                            )
                if canvas_url is None:
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

    def _rebuild_cache_if_empty(self) -> None:
        """Rebuild the cache from Canvas when it is missing or empty."""
        if self.pages_cache:
            return
        console.print(
            "[yellow]Upload cache is empty - rebuilding from Canvas...[/yellow]"
        )
        rebuilt = rebuild_cache_from_canvas(self.client, self.cache_path)
        self.pages_cache = rebuilt["pages"]
        self.files_cache = rebuilt["files"]
        self.modules_cache = rebuilt["modules"]


def _canvas_file_url(client: CanvasUploader, file_id: int) -> str:
    return f"{client.base_url}/courses/{client.course_id}/files/{file_id}?wrap=1"


def rebuild_cache_from_canvas(
    client: CanvasUploader,
    cache_path: Path,
    docs_root: Path = Path("docs"),
) -> dict:
    """
    Reconstruct the upload cache from Canvas state.

    Pages, labs and the syllabus carry an ownership marker (rel path + md5)
    in their body, so they can be mapped back to local files without
    re-uploading anything. Local asset files under docs_root are matched
    against Canvas files by content hash.

    Writes the rebuilt cache to cache_path and returns it.
    """
    console.print("Fetching pages, assignments and files from Canvas...")
    canvas_pages = client.list_pages(include_body=True)
    assignments = client.list_assignments()
    syllabus_body = client.get_syllabus()
    client.list_files()  # warm the asset index once

    pages_cache: dict[str, dict] = {}
    files_cache: dict[str, dict] = {}

    def _page_url(slug: str) -> str:
        return f"{client.base_url}/courses/{client.course_id}/pages/{slug}"

    for p in canvas_pages:
        marker = parse_marker(p.get("body") or "")
        if not marker:
            continue  # not uploaded by this tool
        pages_cache[marker["rel"]] = {
            "page_title": p.get("title"),
            "hash": marker.get("md5"),
            "canvas_url": _page_url(p.get("url", "")),
            "page_url_slug": p.get("url", ""),
            "mentioned_by": [],
            "resolved_md_links": {},
        }

    for a in assignments:
        marker = parse_marker(a.get("description") or "")
        if not marker:
            continue
        assignment_id = a.get("id")
        info: dict = {
            "hash": marker.get("md5"),
            "canvas_assignment_id": assignment_id,
            "canvas_name": a.get("name"),
        }
        if assignment_id is not None:
            info["canvas_url"] = (
                f"{client.base_url}/courses/{client.course_id}"
                f"/assignments/{assignment_id}"
            )
        pages_cache[marker["rel"]] = info

    if syllabus_body:
        marker = parse_marker(syllabus_body)
        if marker:
            pages_cache[marker["rel"]] = {
                "hash": marker.get("md5"),
                "canvas_url": (
                    f"{client.base_url}/courses/{client.course_id}/assignments/syllabus"
                ),
                "page_url_slug": "syllabus",
                "mentioned_by": [],
                "resolved_md_links": {},
            }

    # Local asset files (images, PDFs, ...) are matched to Canvas files by
    # content hash, so a lost cache does not cause duplicate uploads.
    docs_root = Path(docs_root)
    if docs_root.is_dir():
        for path in docs_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() in (".md", ".markdown"):
                continue
            file_hash = compute_file_hash(path)
            match = (
                client.find_existing_file(
                    md5=file_hash, filename=path.name, size=path.stat().st_size
                )
                if file_hash
                else None
            )
            if not match:
                continue
            rel = str(path.relative_to(docs_root))
            files_cache[rel] = {
                "hash": file_hash,
                "canvas_url": _canvas_file_url(client, match["id"]),
                "last_upload": datetime.now().isoformat(),
            }

    # Snapshot each page's resolved links so the incremental-upload check
    # does not force a full re-upload right after rebuilding.
    docs_root_abs = docs_root.resolve()
    for rel, info in pages_cache.items():
        md_path = docs_root / rel
        if not md_path.is_file():
            continue
        try:
            raw = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
        links: dict[str, str] = {}
        for match in _MD_LINK_PATTERN.finditer(raw):
            resolved = (md_path.parent / match.group(2)).resolve()
            try:
                link_rel = str(resolved.relative_to(docs_root_abs))
            except ValueError:
                continue
            target = pages_cache.get(link_rel)
            if target and target.get("canvas_url"):
                links[link_rel] = target["canvas_url"]
        info["resolved_md_links"] = links

    pages_found = sum(1 for i in pages_cache.values() if "canvas_assignment_id" not in i)
    labs_found = len(pages_cache) - pages_found
    console.print(
        f"[green]✓ Rebuilt cache:[/green] {pages_found} page(s)/syllabus, "
        f"{labs_found} lab(s), {len(files_cache)} asset file(s)"
    )
    if docs_root.is_dir() and not files_cache:
        console.print(
            "[yellow]Note: no local assets matched Canvas files by hash. "
            "Assets will re-upload once (and are deduplicated afterwards).[/yellow]"
        )

    cache = {"pages": pages_cache, "files": files_cache, "modules": {}}
    utils_cache.save_cache(cache_path, cache)
    return cache
