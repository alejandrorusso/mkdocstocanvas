import re
from pathlib import Path

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
                        url = (
                            f"{self.client.base_url}/courses/{self.client.course_id}"
                            f"/assignments/{result['id']}"
                        )
                        # Store in pages_cache so other pages/labs can resolve
                        # links that point to this lab's .md file.
                        rel = str(page.rel_path)
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
    """Find and sort all lab*.md files specifically under docs_root/labs/."""
    labs_dir = docs_root / "labs"

    # Ensure the directory exists to avoid errors
    if not labs_dir.is_dir():
        return []

    def _sort_key(p: Path) -> int:
        # Matches 'lab' followed by optional space and digits
        m = re.search(r"lab\s*(\d+)", p.stem, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    # Using glob instead of rglob to stay within the /labs/ folder
    return sorted(labs_dir.glob("lab*.md"), key=_sort_key)


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
        err_console.print(f"No lab*.md files found under {docs_root}/")

    if verbose:
        console.print(f"[dim]Found {len(lab_files)} lab file(s).[/dim]")
        for f in lab_files:
            console.print(f"  [dim]{f}[/dim]")

    LabUploader(client=client, force=force, verbose=verbose).upload_all(lab_files)


def delete_all_labs(client: CanvasUploader, force: bool = False) -> None:
    """Delete all lab assignments from the Canvas course."""
    require_connection(client)

    console.print("Fetching assignments...")
    all_assignments = client.list_assignments()
    labs = [a for a in all_assignments if _LAB_PATTERN.match(a.get("name", ""))]

    delete_all_items(
        labs,
        lambda a: client.delete_assignment(a["id"]),
        noun="lab assignment",
        name_column="Assignment",
        columns=[("Assignment", None), ("ID", "right")],
        row_values=lambda a: (a.get("name", "Untitled"), str(a.get("id", ""))),
        assume_yes=force,
    )
