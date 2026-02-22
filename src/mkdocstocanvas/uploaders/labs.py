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

from api.canvas import CanvasUploader
from models.page import MarkdownPage
from uploaders.base import ContentUploader

console = Console()
err_console = Console(stderr=True, style="bold red")

_LAB_PATTERN = re.compile(r"^Lab\s+\d+", re.IGNORECASE)


class LabUploader(ContentUploader):
    def __init__(
        self,
        client: CanvasUploader,
        cache: str = ".canvas_upload_state.json",
        force: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(client=client, cache=cache, force=force)
        self.verbose = verbose

    def upload_all(self, lab_files: list[Path]) -> None:
        """Convert each lab markdown file to HTML and upload as a Canvas assignment."""
        results: list[dict] = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Uploading labs...", total=len(lab_files))
            for lab_file in lab_files:
                name = self._lab_name(lab_file)
                progress.update(
                    task,
                    description=f"[cyan]Uploading [bold]{name}[/bold]...",
                )
                try:
                    html = self._render(lab_file)
                    result = self.client.create_or_update_assignment(name, html)
                    if result:
                        url = (
                            f"{self.client.base_url}/courses/{self.client.course_id}"
                            f"/assignments/{result['id']}"
                        )
                        # Store in pages_cache so other pages/labs can resolve
                        # links that point to this lab's .md file.
                        rel = str(MarkdownPage(lab_file, root_path=lab_file.parent.parent).rel_path)
                        self.pages_cache[rel] = {"canvas_url": url}
                        results.append({"name": name, "url": url, "status": "ok"})
                        if self.verbose:
                            console.print(
                                f"  [green]✓[/green] {name} [dim]→ {url}[/dim]"
                            )
                    else:
                        results.append({"name": name, "url": "", "status": "error"})
                except Exception as e:
                    err_console.print(f"Error uploading {lab_file.name}: {e}")
                    results.append(
                        {"name": name, "url": "", "status": "error", "error": str(e)}
                    )
                finally:
                    progress.advance(task)

        self._save_cache()
        self._print_summary(results)

    def _render(self, path: Path) -> str:
        """Convert a lab markdown file to Canvas-ready HTML (with asset and link resolution)."""
        page = MarkdownPage(path, root_path=path.parent.parent)
        return self._prepare_content(page)

    def _lab_name(self, path: Path) -> str:
        """Derive assignment name from filename, e.g. lab1.md -> 'Lab 1'."""
        match = re.search(r"lab\s*(\d+)", path.stem, re.IGNORECASE)
        return f"Lab {match.group(1)}" if match else path.stem

    def _print_summary(self, results: list[dict]) -> None:
        ok = sum(1 for r in results if r["status"] == "ok")
        errors = sum(1 for r in results if r["status"] == "error")

        table = Table(
            title="Lab Upload Summary", show_header=True, header_style="bold cyan"
        )
        table.add_column("Status", min_width=12, no_wrap=True)
        table.add_column("Assignment")
        table.add_column("Canvas URL / Error")
        for r in results:
            if r["status"] == "ok":
                status_str = "[green]✓ Uploaded[/green]"
            else:
                status_str = "[red]✗ Failed[/red]"
            detail = r.get("url") or r.get("error", "")
            table.add_row(status_str, r["name"], detail)
        console.print(table)
        console.print(
            f"[bold]Total:[/bold] {len(results)} | "
            f"[green]Uploaded: {ok}[/green] | "
            f"[red]Failed: {errors}[/red]"
        )


def find_lab_files(docs_root: Path = Path("docs")) -> list[Path]:
    """Find and sort all lab*.md files under docs_root/labs/."""

    def _sort_key(p: Path) -> int:
        m = re.search(r"lab\s*(\d+)", p.stem, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    return sorted(docs_root.rglob("lab*.md"), key=_sort_key)


def upload_all_labs(
    client: CanvasUploader,
    docs_root: str = "docs",
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Find all lab files and upload them as Canvas assignments."""
    success, message = client.test_connection()
    if not success:
        err_console.print(message)
        raise typer.Exit(1)
    console.print(message)

    lab_files = find_lab_files(Path(docs_root))
    if not lab_files:
        err_console.print(f"No lab*.md files found under {docs_root}/")
        raise typer.Exit(1)

    if verbose:
        console.print(f"[dim]Found {len(lab_files)} lab file(s).[/dim]")
        for f in lab_files:
            console.print(f"  [dim]{f}[/dim]")

    LabUploader(client=client, force=force, verbose=verbose).upload_all(lab_files)


def delete_all_labs(client: CanvasUploader, force: bool = False) -> None:
    """Delete all lab assignments from the Canvas course."""
    success, message = client.test_connection()
    if not success:
        err_console.print(message)
        raise typer.Exit(1)
    console.print(message)

    console.print("Fetching assignments...")
    all_assignments = client.list_assignments()
    labs = [a for a in all_assignments if _LAB_PATTERN.match(a.get("name", ""))]

    if not labs:
        console.print("No lab assignments found. Nothing to delete.")
        return

    preview = Table(show_header=True, header_style="bold yellow")
    preview.add_column("Assignment")
    preview.add_column("ID", justify="right")
    for a in labs:
        preview.add_row(a.get("name", "Untitled"), str(a.get("id", "")))
    console.print(preview)
    console.print(
        f"[bold yellow]⚠ {len(labs)} lab assignment(s) will be permanently deleted.[/bold yellow]"
    )

    if not force:
        typer.confirm(
            "Are you sure you want to delete ALL lab assignments?", abort=True
        )

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
        task = progress.add_task("[red]Deleting labs...", total=len(labs))
        for a in labs:
            name = a.get("name", "Untitled")
            progress.update(task, description=f"[red]Deleting [bold]{name}[/bold]...")
            if client.delete_assignment(a["id"]):
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
    summary.add_column("Assignment")
    for r in results:
        if r["status"] == "ok":
            summary.add_row("[green]✓ Deleted[/green]", r["name"])
        else:
            summary.add_row("[red]✗ Failed[/red]", r["name"])
    console.print(summary)
    console.print(
        f"[bold]Total:[/bold] {len(labs)} | "
        f"[green]Deleted: {deleted}[/green] | "
        f"[red]Failed: {failed}[/red]"
    )

    if failed and not deleted:
        raise typer.Exit(code=1)
