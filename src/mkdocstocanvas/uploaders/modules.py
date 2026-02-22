import re
import typer
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ..api.canvas import CanvasUploader
from ..utils import config as utils_config

console = Console()
err_console = Console(stderr=True, style="bold red")


class ModuleUploader:
    def __init__(
        self,
        client: CanvasUploader,
        docs_root: Path = Path("docs"),
        verbose: bool = False,
    ):
        self.client = client
        self.docs_root = docs_root
        self.verbose = verbose

    def upload_all(self, sections: list[dict]) -> None:
        """
        Upload all sections as Canvas modules.

        Deletes existing modules first, then creates one module per section,
        linking the matching Canvas pages inside each one.
        """
        success, message = self.client.test_connection()
        if not success:
            err_console.print(message)
            raise typer.Exit(1)
        console.print(message)

        # Fetch all Canvas pages once for link resolution
        console.print("Fetching Canvas pages for link resolution...")
        canvas_pages = self.client.list_pages()
        pages_by_title = {p["title"]: p for p in canvas_pages}
        if self.verbose:
            console.print(f"[dim]Found {len(canvas_pages)} Canvas page(s).[/dim]")

        # Delete existing modules
        existing = self.client.list_modules()
        if existing:
            console.print(f"Deleting {len(existing)} existing module(s)...")
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                del_task = progress.add_task("[red]Deleting...", total=len(existing))
                for m in existing:
                    progress.update(
                        del_task,
                        description=f"[red]Deleting [bold]{m['name']}[/bold]...",
                    )
                    self.client.delete_module(m["id"])
                    progress.advance(del_task)

        # Skip lab sections
        to_upload = [s for s in sections if not s["name"].lower().startswith("lab")]
        skipped_labs = len(sections) - len(to_upload)
        if self.verbose and skipped_labs:
            console.print(f"[dim]Skipping {skipped_labs} lab section(s).[/dim]")

        results: list[dict] = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "[cyan]Uploading modules...", total=len(to_upload)
            )
            for i, section in enumerate(to_upload, 1):
                name = section["name"]
                progress.update(
                    task,
                    description=f"[cyan]Creating [bold]{name}[/bold]...",
                )
                module = self.client.create_module(name, i)
                if not module:
                    results.append({"name": name, "pages": 0, "status": "error"})
                    progress.advance(task)
                    continue

                pages_added = self._add_pages_to_module(
                    module["id"], section["pages"], pages_by_title
                )
                self.client.publish_module(module["id"])

                if self.verbose:
                    console.print(
                        f"  [green]✓[/green] {name} ({pages_added} page(s))"
                    )
                results.append({"name": name, "pages": pages_added, "status": "ok"})
                progress.advance(task)

        self._print_summary(results)

    def _add_pages_to_module(
        self,
        module_id: int,
        pages: list[str],
        pages_by_title: dict[str, dict],
    ) -> int:
        """Add matching Canvas pages to a module. Returns count of pages added."""
        added = 0
        for i, md_path in enumerate(pages, 1):
            if md_path.startswith("labs/"):
                continue

            full_path = self.docs_root / md_path
            if not full_path.exists():
                if self.verbose:
                    console.print(
                        f"    [yellow]⚠[/yellow] Skipping {md_path}: file not found"
                    )
                continue

            title = self._extract_title(full_path)
            if not title:
                if self.verbose:
                    console.print(
                        f"    [yellow]⚠[/yellow] Skipping {md_path}: no title found"
                    )
                continue

            canvas_page = self._find_page(pages_by_title, title)
            if not canvas_page:
                if self.verbose:
                    console.print(
                        f"    [yellow]⚠[/yellow] No Canvas page found for '{title}'"
                    )
                continue

            result = self.client.add_page_to_module(
                module_id, canvas_page["url"], canvas_page["title"], i
            )
            if result:
                added += 1
                if self.verbose:
                    console.print(
                        f"    [green]✓[/green] Added: {canvas_page['title']}"
                    )

        return added

    def _find_page(
        self, pages_by_title: dict[str, dict], title: str
    ) -> dict | None:
        """Find a Canvas page by title, ignoring leading letter prefixes like 'A. '."""
        clean = re.sub(r"^[A-Z]\.\s+", "", title).strip().lower()
        for page_title, page in pages_by_title.items():
            if re.sub(r"^[A-Z]\.\s+", "", page_title).strip().lower() == clean:
                return page
        return None

    def _extract_title(self, path: Path) -> str | None:
        """Extract the first # heading from a markdown file."""
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if re.match(r"^#\s+", line):
                    return re.sub(r"^#\s+", "", line).strip()
        except OSError:
            pass
        return None

    def _print_summary(self, results: list[dict]) -> None:
        ok = sum(1 for r in results if r["status"] == "ok")
        errors = sum(1 for r in results if r["status"] == "error")

        table = Table(
            title="Module Upload Summary", show_header=True, header_style="bold cyan"
        )
        table.add_column("Status", min_width=12, no_wrap=True)
        table.add_column("Module")
        table.add_column("Pages Added", justify="right")
        for r in results:
            status_str = (
                "[green]✓ Created[/green]"
                if r["status"] == "ok"
                else "[red]✗ Failed[/red]"
            )
            table.add_row(status_str, r["name"], str(r["pages"]))
        console.print(table)
        console.print(
            f"[bold]Total:[/bold] {len(results)} | "
            f"[green]Created: {ok}[/green] | "
            f"[red]Failed: {errors}[/red]"
        )


def upload_all_modules(
    client: CanvasUploader,
    docs_root: str = "docs",
    mkdocs_path: str = "mkdocs.yml",
    verbose: bool = False,
) -> None:
    """Parse mkdocs.yml and upload all sections as Canvas modules."""
    mkdocs_path_obj = Path(mkdocs_path)
    if not mkdocs_path_obj.exists():
        err_console.print("ERROR: mkdocs.yml not found")
        raise typer.Exit(1)

    sections = utils_config.parse_mkdocs_nav_sections(mkdocs_path_obj)
    if not sections:
        err_console.print("No sections found in mkdocs.yml nav.")
        raise typer.Exit(1)

    uploader = ModuleUploader(client=client, docs_root=Path(docs_root), verbose=verbose)
    uploader.upload_all(sections)


def delete_all_modules(client: CanvasUploader, force: bool = False) -> None:
    """Delete all modules from the Canvas course."""
    success, message = client.test_connection()
    if not success:
        err_console.print(message)
        raise typer.Exit(1)
    console.print(message)

    modules = client.list_modules()
    if not modules:
        console.print("No modules found. Nothing to delete.")
        return

    preview = Table(show_header=True, header_style="bold yellow")
    preview.add_column("Module")
    preview.add_column("ID", justify="right")
    for m in modules:
        preview.add_row(m.get("name", "Untitled"), str(m.get("id", "")))
    console.print(preview)
    console.print(
        f"[bold yellow]⚠ {len(modules)} module(s) will be permanently deleted.[/bold yellow]"
    )

    if not force:
        typer.confirm("Are you sure you want to delete ALL these modules?", abort=True)

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
        task = progress.add_task("[red]Deleting modules...", total=len(modules))
        for m in modules:
            name = m.get("name", "Untitled")
            progress.update(
                task, description=f"[red]Deleting [bold]{name}[/bold]..."
            )
            if client.delete_module(m["id"]):
                deleted += 1
                results.append({"name": name, "status": "ok"})
            else:
                failed += 1
                results.append({"name": name, "status": "error"})
            progress.advance(task)

    summary = Table(title="Deletion Summary", show_header=True, header_style="bold cyan")
    summary.add_column("Status", min_width=12, no_wrap=True)
    summary.add_column("Module")
    for r in results:
        if r["status"] == "ok":
            summary.add_row("[green]✓ Deleted[/green]", r["name"])
        else:
            summary.add_row("[red]✗ Failed[/red]", r["name"])
    console.print(summary)
    console.print(
        f"[bold]Total:[/bold] {len(modules)} | "
        f"[green]Deleted: {deleted}[/green] | "
        f"[red]Failed: {failed}[/red]"
    )
