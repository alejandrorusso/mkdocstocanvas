from pathlib import Path

import typer
import yaml
from rich.console import Console

err_console = Console(stderr=True)


def _load_mkdocs_config(mkdocs_path: Path) -> dict:
    """
    Load and parse a mkdocs.yml file, exiting with a clear error on failure.

    All mkdocs parsing functions share this, so YAML/file errors are reported
    identically everywhere.
    """
    try:
        with open(mkdocs_path, encoding="utf-8") as f:
            # `or {}` handles an empty file (safe_load returns None)
            return yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        err_console.print(f"[bold red]Error:[/bold red] Could not find {mkdocs_path}")
        raise typer.Exit(1) from e
    except yaml.YAMLError as e:
        err_console.print(
            f"[bold red]Error parsing YAML in {mkdocs_path}:[/bold red] {e}"
        )
        raise typer.Exit(1) from e
    except Exception as e:
        err_console.print_exception()
        raise typer.Exit(1) from e


def _collect_markdown_files(items) -> list[str]:
    """Recursively collect all .md file paths under a nav node."""
    files: list[str] = []
    if not isinstance(items, list):
        items = [items]
    for item in items:
        if isinstance(item, str) and item.endswith(".md"):
            files.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str) and value.endswith(".md"):
                    files.append(value)
                elif isinstance(value, list):
                    files.extend(_collect_markdown_files(value))
    return files


def parse_mkdocs_nav(mkdocs_path: Path) -> list[str] | None:
    """Parse mkdocs.yml and extract a flattened list of markdown files from the nav."""
    config = _load_mkdocs_config(mkdocs_path)

    # If there is no nav, MkDocs auto-discovers files.
    # Returning an empty list here is safer than failing.
    nav = config.get("nav", [])
    markdown_files = []

    def extract_files(items):
        """Recursively extract markdown file paths from nav structure."""
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    for value in item.values():
                        if isinstance(value, str) and value.endswith(".md"):
                            markdown_files.append(value)
                        elif isinstance(value, list):
                            extract_files(value)
                elif isinstance(item, str) and item.endswith(".md"):
                    markdown_files.append(item)

    extract_files(nav)
    return markdown_files


def parse_mkdocs_nav_sections(mkdocs_path: Path) -> list[dict] | None:
    """
    Parse mkdocs.yml and return top-level sections (modules) with their pages.

    Returns:
        [{"name": "Section Name", "pages": ["path/to/file.md", ...]}, ...]
        or None if no sections are found.
    """
    config = _load_mkdocs_config(mkdocs_path)

    sections: list[dict] = []
    for item in config.get("nav", []):
        if isinstance(item, dict):
            for section_name, section_content in item.items():
                pages = _collect_markdown_files(
                    section_content if isinstance(section_content, list) else [section_content]
                )
                if pages:
                    sections.append({"name": section_name, "pages": pages})

    return sections or None


def parse_syllabus_nav_file(
    mkdocs_path: Path, section_name: str = "Syllabus"
) -> str | None:
    """
    Return the first markdown file under the top-level syllabus nav section.

    Example nav:
      - Syllabus:
        - index.md
        - other.md

    Returns "index.md" in the example above.
    """
    config = _load_mkdocs_config(mkdocs_path)

    for item in config.get("nav", []):
        if not isinstance(item, dict):
            continue
        for name, content in item.items():
            if str(name).strip().lower() != section_name.strip().lower():
                continue
            files = _collect_markdown_files(content)
            return files[0] if files else None

    return None
