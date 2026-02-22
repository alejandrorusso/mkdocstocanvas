import hashlib
import re
from pathlib import Path

from rich.console import Console

err_console = Console(stderr=True, style="bold red")


class MarkdownPage:
    def __init__(self, path: Path, root_path: Path = Path("docs")):
        self.path = path
        self.root_path = root_path
        self.rel_path = self.path.relative_to(root_path)
        if not self.path.exists():
            self.title = None
            self.content = ""
            self.md5 = None
        else:
            self.title = self.get_title()
            self.content = self.get_content()
            self.md5 = compute_file_hash(self.path)

    def get_title(self) -> str | None:
        """Extract the first # header from a markdown file as the title"""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    # Check if it starts with exactly one # and a space
                    if line.startswith("# "):
                        # Slice off the '#' and strip any remaining whitespace
                        return line[1:].strip()

        except OSError as e:
            err_console.print(f"⚠ Error reading file {self.path}: {e}")
        except UnicodeDecodeError as e:
            err_console.print(f"⚠ Encoding error in {self.path}: {e}")

        return None

    def get_content(self) -> str:
        with open(self.path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Strip leading H1 title; Canvas uses the page title separately
        return re.sub(r"^\s*#\s+.*\n", "", md_content, count=1)

    def __repr__(self) -> str:
        return (
            f"--- Page Summary ---\n"
            f"Rel Path: {self.rel_path}\n"
            f"Title:    {self.title}\n"
            f"MD5:      {self.md5}\n"
            f"--------------------"
        )


def compute_file_hash(path: Path) -> str | None:
    """Compute MD5 hash of a file's contents."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError as e:
        err_console.print(f"⚠ Warning: Could not compute hash for {path}: {e}")
        return None
