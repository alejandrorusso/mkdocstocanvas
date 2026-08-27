import re
from pathlib import Path

import requests
import typer

from ..api.canvas import CanvasUploader
from ..models.page import MarkdownPage
from .base import (
    ContentUploader,
    console,
    delete_all_items,
    err_console,
    make_progress,
    print_results_summary,
    require_connection,
)

# A lab FILE is a markdown file directly under labs/ whose name starts with
# "lab" (case-insensitive), e.g. labs/lab1.md or labs/Lab2-intro.md. This is
# the single definition used by the page uploader, the module uploader and
# the lab uploader. Other files under labs/ are regular pages.
_LAB_NAME_PATTERN = re.compile(r"^lab", re.IGNORECASE)

# Fallback for identifying lab ASSIGNMENTS on Canvas that are not in the
# upload cache (e.g. uploaded from another machine): name matches "Lab N ...".
_LAB_ASSIGNMENT_PATTERN = re.compile(r"^Lab\s+\d+", re.IGNORECASE)


def is_lab_rel_path(rel_path: str | Path) -> bool:
    """
    True if a docs-relative path is a lab file (labs/lab*.md, case-insensitive).

    Files under labs/ that do not match are regular pages, so they are
    uploaded as pages and added to modules like any other page.
    """
    path = Path(rel_path)
    return (
        path.parent.as_posix() == "labs"
        and path.suffix.lower() == ".md"
        and bool(_LAB_NAME_PATTERN.match(path.name))
    )


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

        with make_progress() as progress:
            task = progress.add_task("[cyan]Uploading labs...", total=len(lab_files))
            for lab_file in lab_files:
                page = MarkdownPage(lab_file, root_path=lab_file.parent.parent)
                name = page.title or lab_file.stem
                progress.update(
                    task,
                    description=f"[cyan]Uploading [bold]{name}[/bold]...",
                )
                try:
                    html = self._prepare_content(page)
                    result = self.client.create_or_update_assignment(name, html)
                    if result:
                        assignment_id = result["id"]
                        url = (
                            f"{self.client.base_url}/courses/{self.client.course_id}"
                            f"/assignments/{assignment_id}"
                        )
                        # Store in pages_cache so other pages/labs can resolve
                        # links to this lab's .md file, and so delete-labs can
                        # find the exact assignment without name guessing.
                        rel = str(page.rel_path)
                        self.pages_cache[rel] = {
                            "canvas_url": url,
                            "canvas_assignment_id": assignment_id,
                            "canvas_name": name,
                        }
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

    def _print_summary(self, results: list[dict]) -> None:
        print_results_summary(
            "Lab Upload Summary",
            results,
            name_column="Assignment",
            detail_column="Canvas URL / Error",
            detail_of=lambda r: r.get("url") or r.get("error", ""),
            ok_label="Uploaded",
        )


def find_lab_files(docs_root: Path = Path("docs")) -> list[Path]:
    """Find all lab files (via is_lab_rel_path) under docs_root/labs/, sorted by lab number."""
    labs_dir = docs_root / "labs"

    # Ensure the directory exists to avoid errors
    if not labs_dir.is_dir():
        return []

    labs = [
        p for p in labs_dir.glob("*.md") if is_lab_rel_path(p.relative_to(docs_root))
    ]

    def _sort_key(p: Path) -> int:
        # Matches 'lab' followed by optional space and digits
        m = re.search(r"lab\s*(\d+)", p.stem, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    return sorted(labs, key=_sort_key)


def upload_all_labs(
    client: CanvasUploader,
    docs_root: str = "docs",
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Find all lab files and upload them as Canvas assignments."""
    require_connection(client)

    lab_files = find_lab_files(Path(docs_root))
    if not lab_files:
        err_console.print(f"No lab files found under {docs_root}/labs/")

    if verbose:
        console.print(f"[dim]Found {len(lab_files)} lab file(s).[/dim]")
        for f in lab_files:
            console.print(f"  [dim]{f}[/dim]")

    LabUploader(client=client, force=force, verbose=verbose).upload_all(lab_files)


def _find_lab_assignments(
    assignments: list[dict], cache: dict[str, dict]
) -> list[dict]:
    """
    Identify lab assignments among all course assignments.

    Cached labs (uploaded by this tool) are matched by their exact assignment
    ID. The name pattern is only a fallback for labs that were uploaded
    without this cache (e.g. from another machine), so renamed or numberless
    labs are still found as long as they were cached.
    """
    cached_ids = {
        info["canvas_assignment_id"]
        for info in cache.values()
        if isinstance(info, dict) and "canvas_assignment_id" in info
    }
    return [
        a
        for a in assignments
        if a.get("id") in cached_ids
        or _LAB_ASSIGNMENT_PATTERN.match(a.get("name", ""))
    ]


def delete_all_labs(client: CanvasUploader, force: bool = False) -> None:
    """Delete all lab assignments from the Canvas course."""
    require_connection(client)

    console.print("Fetching assignments...")
    try:
        all_assignments = client.list_assignments()
    except requests.exceptions.RequestException as e:
        err_console.print(f"[bold red]Error listing assignments:[/bold red] {e}")
        raise typer.Exit(1) from e
    labs = _find_lab_assignments(all_assignments, LabUploader(client).pages_cache)

    delete_all_items(
        labs,
        lambda a: client.delete_assignment(a["id"]),
        noun="lab assignment",
        name_column="Assignment",
        columns=[("Assignment", None), ("ID", "right")],
        row_values=lambda a: (a.get("name", "Untitled"), str(a.get("id", ""))),
        assume_yes=force,
    )
