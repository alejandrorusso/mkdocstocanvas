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
        pdf_root: Path = Path("pdf"),
        add_pdf: bool = False,
        verbose: bool = False,
    ):
        self.client = client
        self.docs_root = docs_root
        self.pdf_root = pdf_root
        self.add_pdf = add_pdf
        self.verbose = verbose
        self.pdf_by_stem = self._index_pdfs() if add_pdf else {}

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
            if self.add_pdf:
                console.print(
                    f"[dim]Found {len(self.pdf_by_stem)} PDF file(s) in {self.pdf_root}/.[/dim]"
                )

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
        position = 1
        for md_path in pages:
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
                module_id, canvas_page["url"], canvas_page["title"], position
            )
            if result:
                added += 1
                if self.verbose:
                    console.print(
                        f"    [green]✓[/green] Added: {canvas_page['title']}"
                    )
                if self.add_pdf:
                    self._add_matching_pdf_to_module(
                        module_id=module_id,
                        md_path=md_path,
                        position=position + 1,
                    )
                position += 2 if self.add_pdf else 1

        return added

    def _index_pdfs(self) -> dict[str, Path]:
        """Index PDFs in pdf_root by normalized stem for quick lookup."""
        if not self.pdf_root.exists():
            if self.verbose:
                console.print(
                    f"[yellow]⚠[/yellow] PDF directory not found: {self.pdf_root}"
                )
            return {}

        pdf_by_stem: dict[str, Path] = {}
        for pdf_path in self.pdf_root.glob("*.pdf"):
            pdf_by_stem[self._normalize_stem(pdf_path.stem)] = pdf_path
        return pdf_by_stem

    def _normalize_stem(self, stem: str) -> str:
        """Normalize file stems for matching (strip optional letter prefix)."""
        return re.sub(r"^[A-Z]\.\s+", "", stem).strip().lower()

    def _extract_file_id(self, canvas_file_url: str) -> int | None:
        """Extract Canvas file ID from a /files/{id}/ URL."""
        match = re.search(r"/files/(\d+)/", canvas_file_url)
        return int(match.group(1)) if match else None

    def _add_matching_pdf_to_module(self, module_id: int, md_path: str, position: int) -> None:
        """Upload and add the PDF corresponding to md_path into the same module."""
        md_stem = self._normalize_stem(Path(md_path).stem)
        pdf_path = self.pdf_by_stem.get(md_stem)

        if not pdf_path:
            if self.verbose:
                console.print(
                    f"    [yellow]⚠[/yellow] No matching PDF for: {md_path}"
                )
            return

        try:
            canvas_file_url = self.client.upload_file(pdf_path, "/module_pdfs")
        except Exception as exc:
            if self.verbose:
                console.print(
                    f"    [yellow]⚠[/yellow] PDF upload failed for {pdf_path.name}: {exc}"
                )
            return

        file_id = self._extract_file_id(canvas_file_url)
        if file_id is None:
            if self.verbose:
                console.print(
                    f"    [yellow]⚠[/yellow] Could not parse Canvas file ID for {pdf_path.name}"
                )
            return

        result = self.client.add_file_to_module(
            module_id=module_id,
            file_id=file_id,
            title=pdf_path.name,
            position=position,
        )
        if result and self.verbose:
            console.print(f"    [green]✓[/green] Added PDF: {pdf_path.name}")

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
    pdf_root: str = "pdf",
    mkdocs_path: str = "mkdocs.yml",
    add_pdf: bool = False,
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

    uploader = ModuleUploader(
        client=client,
        docs_root=Path(docs_root),
        pdf_root=Path(pdf_root),
        add_pdf=add_pdf,
        verbose=verbose,
    )
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
