import re
from pathlib import Path

import requests
import typer

from ..api.canvas import CanvasUploader
from ..models.page import extract_title_text
from ..utils import config as utils_config
from .base import (
    console,
    delete_all_items,
    err_console,
    make_progress,
    print_results_summary,
    require_connection,
)
from .labs import is_lab_rel_path


class ModuleUploader:
    def __init__(
        self,
        client: CanvasUploader,
        docs_root: Path = Path("docs"),
        pdf_root: Path = Path("pdf"),
        add_pdf: bool = False,
        verbose: bool = False,
        syllabus_rel_path: str | None = None,
    ):
        self.client = client
        self.docs_root = docs_root
        self.pdf_root = pdf_root
        self.add_pdf = add_pdf
        self.verbose = verbose
        self.syllabus_rel_path = (
            Path(syllabus_rel_path).as_posix() if syllabus_rel_path else None
        )
        self.pdf_by_stem = self._index_pdfs() if add_pdf else {}

    def upload_all(self, sections: list[dict]) -> None:
        """
        Upload all sections as Canvas modules.

        Deletes existing modules first, then creates one module per section,
        linking the matching Canvas pages inside each one.
        """
        require_connection(self.client)

        # Fetch all Canvas pages once for link resolution
        console.print("Fetching Canvas pages for link resolution...")
        try:
            canvas_pages = self.client.list_pages()
        except requests.exceptions.RequestException as e:
            err_console.print(f"[bold red]Error listing pages:[/bold red] {e}")
            raise typer.Exit(1) from e
        pages_by_title = {p["title"]: p for p in canvas_pages}
        if self.verbose:
            console.print(f"[dim]Found {len(canvas_pages)} Canvas page(s).[/dim]")
            if self.add_pdf:
                console.print(
                    f"[dim]Found {len(self.pdf_by_stem)} PDF file(s) in {self.pdf_root}/.[/dim]"
                )

        # Delete existing modules
        try:
            existing = self.client.list_modules()
        except requests.exceptions.RequestException as e:
            err_console.print(f"[bold red]Error listing modules:[/bold red] {e}")
            raise typer.Exit(1) from e
        if existing:
            console.print(f"Deleting {len(existing)} existing module(s)...")
            with make_progress() as progress:
                del_task = progress.add_task("[red]Deleting...", total=len(existing))
                for m in existing:
                    progress.update(
                        del_task,
                        description=f"[red]Deleting [bold]{m['name']}[/bold]...",
                    )
                    self.client.delete_module(m["id"])
                    progress.advance(del_task)

        # Skip lab and syllabus sections. Lab sections are matched by NAME
        # (intentionally different from file-level lab detection): labs are
        # Canvas assignments, not module items, so a whole "Lab ..." nav
        # section never becomes a module.
        to_upload: list[dict] = []
        skipped_labs = 0
        skipped_syllabus_sections = 0
        for section in sections:
            section_name = section["name"].strip().lower()
            if section_name.startswith("lab"):
                skipped_labs += 1
                continue
            if section_name == "syllabus":
                skipped_syllabus_sections += 1
                continue
            to_upload.append(section)

        if self.verbose and skipped_labs:
            console.print(f"[dim]Skipping {skipped_labs} lab section(s).[/dim]")
        if self.verbose and skipped_syllabus_sections:
            console.print(
                f"[dim]Skipping {skipped_syllabus_sections} syllabus section(s).[/dim]"
            )

        results: list[dict] = []

        with make_progress() as progress:
            task = progress.add_task("[cyan]Uploading modules...", total=len(to_upload))
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
                    console.print(f"  [green]✓[/green] {name} ({pages_added} page(s))")
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
            # Labs become Canvas assignments, not module items (see
            # labs.is_lab_rel_path). Files under labs/ that are not labs
            # are added like any other page.
            if is_lab_rel_path(md_path):
                continue
            if self.syllabus_rel_path and Path(md_path).as_posix() == self.syllabus_rel_path:
                if self.verbose:
                    console.print(
                        f"    [dim]Skipping syllabus page in module: {md_path}[/dim]"
                    )
                continue

            full_path = self.docs_root / md_path
            if not full_path.exists():
                if self.verbose:
                    console.print(
                        f"    [yellow]⚠[/yellow] Skipping {md_path}: file not found"
                    )
                continue

            try:
                title = extract_title_text(full_path)
            except (OSError, UnicodeDecodeError):
                title = None
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
                    console.print(f"    [green]✓[/green] Added: {canvas_page['title']}")
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

    def _add_matching_pdf_to_module(
        self, module_id: int, md_path: str, position: int
    ) -> None:
        """Upload and add the PDF corresponding to md_path into the same module."""
        md_stem = self._normalize_stem(Path(md_path).stem)
        pdf_path = self.pdf_by_stem.get(md_stem)

        if not pdf_path:
            if self.verbose:
                console.print(f"    [yellow]⚠[/yellow] No matching PDF for: {md_path}")
            return

        try:
            canvas_file_url = self.client.upload_file(pdf_path, "/module_pdfs")
        except Exception as exc:
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

    def _find_page(self, pages_by_title: dict[str, dict], title: str) -> dict | None:
        """Find a Canvas page by title, ignoring leading letter prefixes like 'A. '."""
        clean = re.sub(r"^[A-Z]\.\s+", "", title).strip().lower()
        for page_title, page in pages_by_title.items():
            if re.sub(r"^[A-Z]\.\s+", "", page_title).strip().lower() == clean:
                return page
        return None

    def _print_summary(self, results: list[dict]) -> None:
        print_results_summary(
            "Module Upload Summary",
            results,
            name_column="Module",
            detail_column="Pages Added",
            detail_of=lambda r: str(r["pages"]),
            ok_label="Created",
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

    syllabus_rel_path = utils_config.parse_syllabus_nav_file(mkdocs_path_obj)

    uploader = ModuleUploader(
        client=client,
        docs_root=Path(docs_root),
        pdf_root=Path(pdf_root),
        add_pdf=add_pdf,
        verbose=verbose,
        syllabus_rel_path=syllabus_rel_path,
    )
    uploader.upload_all(sections)


def delete_all_modules(client: CanvasUploader, force: bool = False) -> None:
    """Delete all modules from the Canvas course."""
    require_connection(client)

    try:
        modules = client.list_modules()
    except requests.exceptions.RequestException as e:
        err_console.print(f"[bold red]Error listing modules:[/bold red] {e}")
        raise typer.Exit(1) from e

    delete_all_items(
        modules,
        lambda m: client.delete_module(m["id"]),
        noun="module",
        name_column="Module",
        columns=[("Module", None), ("ID", "right")],
        row_values=lambda m: (m.get("name", "Untitled"), str(m.get("id", ""))),
        assume_yes=force,
    )
