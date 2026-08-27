from datetime import datetime
from pathlib import Path

import requests
import typer
from rich.table import Table

from ..api.canvas import CanvasUploader
from ..models.page import MarkdownPage
from ..utils import config as utils_config
from .base import (
    _MD_LINK_PATTERN,
    ContentUploader,
    console,
    delete_all_items,
    err_console,
    make_progress,
    require_connection,
)
from .labs import is_lab_rel_path


class PageUploader(ContentUploader):
    def __init__(
        self,
        client: CanvasUploader,
        cache: str = ".canvas_upload_state.json",
        force: bool = False,
        verbose: bool = False,
        syllabus_rel_path: str | None = None,
    ) -> None:
        super().__init__(client=client, cache=cache, force=force)
        self.verbose = verbose
        self.syllabus_rel_path = (
            Path(syllabus_rel_path).as_posix() if syllabus_rel_path else None
        )

    def _is_syllabus_rel(self, rel: str) -> bool:
        return bool(self.syllabus_rel_path and rel == self.syllabus_rel_path)

    def _is_syllabus_page(self, md_page: MarkdownPage) -> bool:
        return self._is_syllabus_rel(str(md_page.rel_path))

    def upload_all_pages(self, pages: list[MarkdownPage]) -> None:
        """
        Uploads all pages to canvas.
        """
        require_connection(self.client)

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
        upload_order: dict[str, int] = {}
        upload_seq = 0
        upload_results: list[dict] = []

        with make_progress() as progress:
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
                    canvas_url, page_url_slug = self._upload_page(page)
                    self._update_page_cache(rel, page, canvas_url, page_url_slug)
                    uploaded_rels.add(rel)
                    upload_order[rel] = upload_seq
                    upload_seq += 1
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
            changed_index = upload_order.get(changed_rel)
            for mentioner_rel in self.pages_cache.get(changed_rel, {}).get(
                "mentioned_by", []
            ):
                if mentioner_rel not in pages_by_rel:
                    continue

                mentioner_index = upload_order.get(mentioner_rel)

                # Re-upload when either:
                # 1) mentioner was not uploaded in Stage 1 (stale cache entry), or
                # 2) mentioner was uploaded before the changed target got its slug.
                if mentioner_index is None or (
                    changed_index is not None and mentioner_index < changed_index
                ):
                    mentioners_to_fix.add(mentioner_rel)

        if mentioners_to_fix:
            console.print(
                f"Stage 2: Re-uploading {len(mentioners_to_fix)} page(s) with stale links..."
            )
            # Keep syllabus last so it gets the final link state.
            stage2_rels = sorted(
                mentioners_to_fix,
                key=self._is_syllabus_rel,
            )
            with make_progress() as progress:
                task2 = progress.add_task(
                    "[cyan]Re-uploading pages...", total=len(stage2_rels)
                )
                for rel in stage2_rels:
                    page = pages_by_rel[rel]
                    progress.update(
                        task2,
                        description=f"[cyan]Re-uploading [bold]{page.title or rel}[/bold]...",
                    )
                    try:
                        canvas_url, page_url_slug = self._upload_page(page)
                        self._update_page_cache(rel, page, canvas_url, page_url_slug)
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

    def _update_page_cache(
        self, rel: str, page: MarkdownPage, canvas_url: str, page_url_slug: str
    ) -> None:
        """Write a page's upload result into pages_cache, preserving mentioned_by."""
        existing = self.pages_cache.get(rel, {})
        resolved_md_links = self._snapshot_resolved_md_links(page)
        self.pages_cache[rel] = {
            "page_title": page.title,
            "hash": page.md5,
            "canvas_url": canvas_url,
            "page_url_slug": page_url_slug,
            "last_upload": datetime.now().isoformat(),
            "mentioned_by": existing.get("mentioned_by", []),
            "resolved_md_links": resolved_md_links,
        }

    def _iter_md_link_targets(self, md_page: MarkdownPage) -> list[str]:
        """
        Return unique, in-order relative docs paths targeted by markdown `.md` links.
        """
        try:
            raw = md_page.path.read_text(encoding="utf-8")
        except OSError:
            return []

        docs_root_abs = md_page.root_path.resolve()
        seen: set[str] = set()
        targets: list[str] = []

        for match in _MD_LINK_PATTERN.finditer(raw):
            link_path_str = match.group(2)
            resolved = (md_page.path.parent / link_path_str).resolve()
            try:
                link_rel = str(resolved.relative_to(docs_root_abs))
            except ValueError:
                continue

            if link_rel not in seen:
                seen.add(link_rel)
                targets.append(link_rel)

        return targets

    def _snapshot_resolved_md_links(self, md_page: MarkdownPage) -> dict[str, str]:
        """
        Build a mapping of markdown-link targets to their currently known Canvas URLs.

        Returns:
            dict[target_rel_path -> canvas_url]
        """
        resolved_links: dict[str, str] = {}

        for link_rel in self._iter_md_link_targets(md_page):
            info = self.pages_cache.get(link_rel)
            canvas_url = info.get("canvas_url") if isinstance(info, dict) else None
            if canvas_url:
                resolved_links[link_rel] = canvas_url

        return resolved_links

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

        for page in pages:
            page_rel = str(page.rel_path)
            for link_rel in self._iter_md_link_targets(page):
                if link_rel not in self.pages_cache:
                    self.pages_cache[link_rel] = {"mentioned_by": []}

                entry = self.pages_cache[link_rel]
                mentioned_by: list = entry.setdefault("mentioned_by", [])
                if page_rel not in mentioned_by:
                    mentioned_by.append(page_rel)

    def _upload_page(self, md_page: MarkdownPage) -> tuple[str, str]:
        """
        Uploads page to canvas.

        Returns (Canvas URL, page slug).
        """
        html_page = self._prepare_content(md_page)

        info = self.pages_cache.get(str(md_page.rel_path))
        url_slug = info.get("page_url_slug") if info else None

        if self._is_syllabus_page(md_page):
            syllabus_url = self.client.upload_syllabus(html_page)
            if syllabus_url is None:
                # upload_syllabus already printed the API error
                raise RuntimeError(f"Failed to upload syllabus {md_page.path}")
            return syllabus_url, "syllabus"

        result = self.client.create_or_update_page(
            md_page.title,
            html_page,
            published=True,
            page_slug=url_slug,
        )
        if result is None:
            # create_or_update_page already printed the API error
            raise RuntimeError(f"Failed to save page {md_page.path}")
        return result

    def _pages_to_upload(self, pages: list[MarkdownPage]) -> list[MarkdownPage]:
        """
        Gets a list of pages that need uploading.

        Returns:
            A list of MarkdownPage objects that need to be uploaded.
        """
        pages_to_upload = []
        for md_page in pages:  # pyright: ignore
            # Labs are uploaded as Canvas assignments, not pages. Other files
            # under labs/ are regular pages (see labs.is_lab_rel_path).
            if is_lab_rel_path(md_page.rel_path):
                continue

            if not md_page.path.exists():
                console.print(
                    f"⚠ Skipping {md_page.path}: file not found", style="bold yellow"
                )
                continue

            if self._needs_upload(md_page):
                pages_to_upload.append(md_page)

        # Keep syllabus last so its internal links are resolved with the final
        # set of uploaded page/lab URLs.
        pages_to_upload.sort(key=self._is_syllabus_page)

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

        # Re-upload if any linked page/lab now resolves to a different Canvas URL
        # than the one this page was last uploaded with.
        cached_links = cached_info.get("resolved_md_links", {})
        if not isinstance(cached_links, dict):
            cached_links = {}
        current_links = self._snapshot_resolved_md_links(md_page)
        if cached_links != current_links:
            return True

        page_url_slug = cached_info.get("page_url_slug")
        if not page_url_slug:
            return True  # It has metadata, but no Canvas slug! Force upload.

        # Check Canvas page existence. A failed lookup (e.g. network hiccup)
        # just forces a re-upload attempt, where any real error is reported.
        try:
            page_exists = bool(self.client.get_existing_page(page_url_slug))
        except requests.exceptions.RequestException:
            page_exists = False

        # If syllabus, ignore existence check
        return not page_exists and not self._is_syllabus_page(md_page)


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

    syllabus_rel_path = utils_config.parse_syllabus_nav_file(mkdocs_path)

    uploader = PageUploader(
        client=client,
        force=force,
        verbose=verbose,
        syllabus_rel_path=syllabus_rel_path,
    )

    uploader.upload_all_pages(markdown_pages)


def delete_all_pages(client: CanvasUploader, force: bool = False) -> None:
    """
    Deletes all pages from the Canvas course.

    Lists pages, asks for confirmation (unless force=True), deletes with a
    progress bar, and prints a rich summary table.
    """
    require_connection(client)

    console.print("Fetching pages...")
    try:
        pages = client.list_pages()
    except requests.exceptions.RequestException as e:
        err_console.print(f"[bold red]Error listing pages:[/bold red] {e}")
        raise typer.Exit(1) from e

    delete_all_items(
        pages,
        lambda p: client.delete_page(p.get("url", "")),
        noun="page",
        name_column="Page",
        columns=[("Title", None), ("Slug", None)],
        row_values=lambda p: (p.get("title", "Untitled"), p.get("url", "")),
        assume_yes=force,
    )
