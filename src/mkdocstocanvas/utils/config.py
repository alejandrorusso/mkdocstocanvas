import yaml
import typer
from rich.console import Console
from pathlib import Path

err_console = Console(stderr=True)


def parse_mkdocs_nav(mkdocs_path: Path) -> list[str] | None:
    """Parse mkdocs.yml and extract a flattened list of markdown files from the nav."""
    try:
        with open(mkdocs_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # If there is no nav, MkDocs auto-discovers files.
        # Returning an empty list here is safer than failing.
        nav = config.get("nav", [])
        markdown_files = []

        def extract_files(items):
            """Recursively extract markdown file paths from nav structure."""
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, str) and value.endswith(".md"):
                                markdown_files.append(value)
                            elif isinstance(value, list):
                                extract_files(value)
                    elif isinstance(item, str) and item.endswith(".md"):
                        markdown_files.append(item)

        extract_files(nav)
        return markdown_files
    except FileNotFoundError:
        err_console.print(f"[bold red]Error:[/bold red] Could not find {mkdocs_path}")
        raise typer.Exit(1)
    except yaml.YAMLError as e:
        err_console.print(
            f"[bold red]Error parsing YAML in {mkdocs_path}:[/bold red] {e}"
        )
        raise typer.Exit(1)
    except Exception as e:
        err_console.print_exception(show_locals=True)
        raise typer.Exit(1)
