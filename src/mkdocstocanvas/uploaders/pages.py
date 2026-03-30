from pathlib import Path
from datetime import datetime
import typer
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ..utils import config as utils_config
from ..api.canvas import CanvasUploader
from ..models.page import MarkdownPage
from .base import ContentUploader, _MD_LINK_PATTERN

console = Console()
err_console = Console(stderr=True, style="bold red")


class PageUploader(ContentUploader):
    def __init__(
        self,
        client: CanvasUploader,
        cache: str = ".canvas_upload_state.json",
        force: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(client=client, cache=cache, force=force)
        self.verbose = verbose

    def upload_all_pages(self, pages: list[MarkdownPage]) -> None:
        """
        Uploads all pages to canvas.
        """
        # Test connection
        success, message = self.client.test_connection()
        if not success:
            err_console.print(message)
            raise typer.Exit(1)
        console.print(message)

        # Build/refresh the mentioned_by index before uploading anything
        self._collect_mentions(pages)

        # Get pages that need to be uploaded
        pages_to_upload = self._pages_to_upload(pages)
        pages_by_rel = {str(p.rel_path): p for p in pages}
        skipped = len(pages) - len(pages_to_upload)

        if self.verbose:
            console.print(f"[dim]{pages}")
            console.print(
                f"[dim]Found {len(pages)} total page(s). "
                f"{len(pages_to_upload)} to upload, {skipped} skipped (cached/unchanged).[/dim]"
            )

        # Capture old slugs so we can detect changes after upload
        old_slugs = {
            str(p.rel_path): self.pages_cache.get(str(p.rel_path), {}).get(
                "page_url_slug"
            )
            for p in pages_to_upload
        }

        # Stage 1: Upload dirty pages
        failed = 0
        uploaded_rels: set[str] = set()
        upload_results: list[dict] = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "[cyan]Uploading pages...", total=len(pages_to_upload)
            )
            for page in pages_to_upload:
                rel = str(page.rel_path)
                progress.update(
                    task,
                    description=f"[cyan]Uploading [bold]{page.title or rel}[/bold]...",
                )
                try:
                    canvas_url = self._upload_page(page)
                    self._update_page_cache(rel, page, canvas_url)
                    uploaded_rels.add(rel)
                    upload_results.append(
                        {
                            "title": page.title or rel,
                            "rel": rel,
                            "url": canvas_url,
                            "status": "ok",
                        }
                    )
                    if self.verbose:
                        console.print(
                            f"  [green]✓[/green] {page.title} [dim]→ {canvas_url}[/dim]"
                        )
                except Exception as e:
                    failed += 1
                    upload_results.append(
                        {
                            "title": page.title or rel,
                            "rel": rel,
                            "url": "",
                            "status": "error",
                            "error": str(e),
                        }
                    )
                    err_console.print(f"Error uploading {page.path}: {e}")
                finally:
                    progress.advance(task)

        self._save_cache()

        if failed == len(pages_to_upload) and pages_to_upload:
            err_console.print("All pages failed to upload. Exiting...")
            raise typer.Exit(1)

        # Stage 2: Re-upload pages that link to any page whose slug changed.
        # This ensures their HTML contains the correct Canvas URL.
        slug_changed = {
            rel
            for rel, old_slug in old_slugs.items()
            if rel in uploaded_rels
            and self.pages_cache.get(rel, {}).get("page_url_slug") != old_slug
        }
        mentioners_to_fix: set[str] = set()
        for changed_rel in slug_changed:
            for mentioner_rel in self.pages_cache.get(changed_rel, {}).get(
                "mentioned_by", []
            ):
                if mentioner_rel not in uploaded_rels and mentioner_rel in pages_by_rel:
                    mentioners_to_fix.add(mentioner_rel)

        if mentioners_to_fix:
            console.print(
                f"Stage 2: Re-uploading {len(mentioners_to_fix)} page(s) with stale links..."
            )
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                task2 = progress.add_task(
                    "[cyan]Re-uploading pages...", total=len(mentioners_to_fix)
                )
                for rel in mentioners_to_fix:
                    page = pages_by_rel[rel]
                    progress.update(
                        task2,
                        description=f"[cyan]Re-uploading [bold]{page.title or rel}[/bold]...",
                    )
                    try:
                        canvas_url = self._upload_page(page)
                        self._update_page_cache(rel, page, canvas_url)
                        upload_results.append(
                            {
                                "title": page.title or rel,
                                "rel": rel,
                                "url": canvas_url,
                                "status": "re-upload",
                            }
                        )
                        if self.verbose:
                            console.print(
                                f"  [blue]↺[/blue] {page.title} [dim]→ {canvas_url}[/dim]"
                            )
                    except Exception as e:
                        upload_results.append(
                            {
                                "title": page.title or rel,
                                "rel": rel,
                                "url": "",
                                "status": "error",
                                "error": str(e),
                            }
                        )
                        err_console.print(f"Error re-uploading {page.path}: {e}")
                    finally:
                        progress.advance(task2)

        self._save_cache()

        self._print_summary(upload_results, skipped=skipped, total=len(pages))

    def _print_summary(
        self, results: list[dict], skipped: int = 0, total: int = 0
    ) -> None:
        """Prints a rich summary table after the upload run."""
        ok = sum(1 for r in results if r["status"] == "ok")
        re_uploaded = sum(1 for r in results if r["status"] == "re-upload")
        errors = sum(1 for r in results if r["status"] == "error")

        table = Table(
            title="Upload Summary", show_header=True, header_style="bold cyan"
        )
        table.add_column("Status", style="bold", min_width=14, no_wrap=True)
        table.add_column("Page")
        table.add_column("Canvas URL / Error")

        for r in results:
            if r["status"] == "ok":
                status_str = "[green]✓ Uploaded[/green]"
            elif r["status"] == "re-upload":
                status_str = "[blue]↺ Re-uploaded[/blue]"
            else:
                status_str = "[red]✗ Failed[/red]"
            detail = r.get("url") or r.get("error", "")
            table.add_row(status_str, r["title"], detail)

        console.print(table)
        console.print(
            f"[bold]Total:[/bold] {total} page(s) | "
            f"[green]Uploaded: {ok}[/green] | "
            f"[blue]Re-uploaded: {re_uploaded}[/blue] | "
            f"[yellow]Skipped: {skipped}[/yellow] | "
            f"[red]Failed: {errors}[/red]"
        )

    def _update_page_cache(self, rel: str, page: MarkdownPage, canvas_url: str) -> None:
        """Write a page's upload result into pages_cache, preserving mentioned_by."""
        page_url_slug = canvas_url.split("/pages/")[-1]
        existing = self.pages_cache.get(rel, {})
        self.pages_cache[rel] = {
            "page_title": page.title,
            "hash": page.md5,
            "canvas_url": canvas_url,
            "page_url_slug": page_url_slug,
            "last_upload": datetime.now().isoformat(),
            "mentioned_by": existing.get("mentioned_by", []),
        }

    def _collect_mentions(self, pages: list[MarkdownPage]) -> None:
        """
        Scan all pages for links to other .md pages and build a
        mentioned_by index in pages_cache.

        mentioned_by[B] = [A, C, ...]  means pages A and C contain a
        markdown link pointing to page B.
        """
        # Clear existing mentioned_by lists so we rebuild from scratch
        for entry in self.pages_cache.values():
            entry["mentioned_by"] = []

        if not pages:
            return
        docs_root_abs = pages[0].root_path.resolve()

        for page in pages:
            page_rel = str(page.rel_path)
            try:
                raw = page.path.read_text(encoding="utf-8")
            except OSError:
                continue

            for match in _MD_LINK_PATTERN.finditer(raw):
                link_path_str = match.group(2)
                resolved = (page.path.parent / link_path_str).resolve()
                try:
                    link_rel = str(resolved.relative_to(docs_root_abs))
                except ValueError:
                    continue

                if link_rel not in self.pages_cache:
                    self.pages_cache[link_rel] = {"mentioned_by": []}

                entry = self.pages_cache[link_rel]
                mentioned_by: list = entry.setdefault("mentioned_by", [])
                if page_rel not in mentioned_by:
                    mentioned_by.append(page_rel)

    def _upload_page(self, md_page: MarkdownPage) -> str:
        """
        Uploads page to canvas.

        Returns the Canvas URL
        """
        html_page = self._prepare_content(md_page)

        info = self.pages_cache.get(str(md_page.rel_path))
        url_slug = info.get("page_url_slug") if info else None

        if md_page.title.lower() == "syllabus":
            return self.client.upload_syllabus(html_page)  # pyright: ignore

        return self.client.create_or_update_page(
            md_page.title,  # pyright: ignore
            html_page,
            published=True,
            page_slug=url_slug,  # pyright: ignore
        )

    def _pages_to_upload(self, pages: list[MarkdownPage]) -> list[MarkdownPage]:
        """
        Gets a list of pages that need uploading.

        Returns:
            A list of MarkdownPage objects that need to be uploaded.
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
        if self.force:
            return True

        # Compute current hash
        if md_page.md5 is None:
            return True  # Failed to compute hash, force upload

        rel_path = str(md_page.path.relative_to(md_page.root_path))
        # Fetch metadata in one step
        cached_info = self.pages_cache.get(rel_path)
        if cached_info is None:
            return True  # File not in metadata (new file)

        # Compare hashes
        if cached_info.get("hash") != md_page.md5:
            return True  # File content changed

        page_url_slug = cached_info.get("page_url_slug")
        if not page_url_slug:
            return True  # It has metadata, but no Canvas slug! Force upload.

        # Check Canvas page existence
        if (
            not self.client.get_existing_page(page_url_slug)
            # If syllabus, ignore existence check
            and md_page.title.lower() != "syllabus"
        ):
            return True  # Page was deleted from Canvas, needs re-upload

        return False  # File unchanged and verified on Canvas


def parse_upload_all_pages(
    client: CanvasUploader,
    docs_root="docs",
    mkdocs_path="mkdocs.yml",
    force: bool = False,
    verbose: bool = False,
):
    # docs/ directory
    docs_root = Path(docs_root)
    if not docs_root.exists():
        err_console.print(
            "ERROR: docs/ directory not found. Make sure that your directory contains docs/"
        )
        raise typer.Exit(1)

    # Parse mkdocs.yml
    mkdocs_path = Path(mkdocs_path)
    if not mkdocs_path.exists():
        err_console.print(
            "ERROR: mkdocs.yml not found. Make sure that your directory contains mkdocs.yml"
        )
        raise typer.Exit(1)
    markdown_files = utils_config.parse_mkdocs_nav(mkdocs_path)
    if markdown_files is None:
        err_console.print(f"No markdown files found in {mkdocs_path}")
        raise typer.Exit(1)

    markdown_pages = [MarkdownPage(docs_root / p) for p in markdown_files]

    uploader = PageUploader(client=client, force=force, verbose=verbose)

    uploader.upload_all_pages(markdown_pages)


def delete_all_pages(client: CanvasUploader, force: bool = False) -> None:
    """
    Deletes all pages from the Canvas course.

    Lists pages, asks for confirmation (unless force=True), deletes with a
    progress bar, and prints a rich summary table.
    """
    success, message = client.test_connection()
    if not success:
        err_console.print(message)
        raise typer.Exit(1)
    console.print(message)

    console.print("Fetching pages...")
    pages = client.list_pages()

    if not pages:
        console.print("No pages found. Nothing to delete.")
        return

    preview = Table(show_header=True, header_style="bold yellow")
    preview.add_column("Title")
    preview.add_column("Slug")
    for p in pages:
        preview.add_row(p.get("title", "Untitled"), p.get("url", ""))
    console.print(preview)
    console.print(
        f"[bold yellow]⚠ {len(pages)} page(s) will be permanently deleted.[/bold yellow]"
    )

    if not force:
        typer.confirm("Are you sure you want to delete ALL these pages?", abort=True)

    deleted = 0
    failed = 0
    results: list[dict] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[red]Deleting pages...", total=len(pages))
        for page in pages:
            slug = page.get("url", "")
            title = page.get("title", "Untitled")
            progress.update(task, description=f"[red]Deleting [bold]{title}[/bold]...")
            if client.delete_page(slug):
                deleted += 1
                results.append({"title": title, "status": "ok"})
            else:
                failed += 1
                results.append({"title": title, "status": "error"})
            progress.advance(task)

    summary = Table(
        title="Deletion Summary", show_header=True, header_style="bold cyan"
    )
    summary.add_column("Status", min_width=12, no_wrap=True)
    summary.add_column("Page")
    for r in results:
        if r["status"] == "ok":
            summary.add_row("[green]✓ Deleted[/green]", r["title"])
        else:
            summary.add_row("[red]✗ Failed[/red]", r["title"])
    console.print(summary)
    console.print(
        f"[bold]Total:[/bold] {len(pages)} | "
        f"[green]Deleted: {deleted}[/green] | "
        f"[red]Failed: {failed}[/red]"
    )

    if failed and not deleted:
        raise typer.Exit(code=1)
