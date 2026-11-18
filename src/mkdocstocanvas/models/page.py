import hashlib
import re
from pathlib import Path

from rich.console import Console

from ..processing.includes import expand_includes

err_console = Console(stderr=True, style="bold red")


def _first_h1(text: str) -> str | None:
    """First "# " heading in a markdown string, or None."""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return None


def extract_title_text(path: Path, docs_dir: Path | None = None) -> str | None:
    """
    First "# " heading of a markdown file (after expanding include directives),
    or None if there is none.

    Raises OSError / UnicodeDecodeError if the file cannot be read.
    Shared by MarkdownPage and the module uploader so both use the same
    title rule. ``docs_dir`` is the docs root used to resolve includes; it
    defaults to the file's own directory.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    return _first_h1(expand_includes(raw, path, docs_dir or path.parent))


class MarkdownPage:
    def __init__(self, path: Path, root_path: Path = Path("docs")):
        # Path that also contains the root path.
        # Ex: docs/lectures/lecture1.md
        self.path = path
        # Ex: docs/
        self.root_path = root_path
        # Ex: lectures/lecture1.md
        self.rel_path = self.path.relative_to(root_path)
        # Read once and expand include-markdown directives, so title, body and
        # the change-detection hash all reflect the real (included) content.
        self._raw, self._expanded = self._read()
        self.title = self.get_title()
        self.content = self.get_content()
        # Preserve the previous byte-for-byte file hash for ordinary pages
        # (no includes); only wrappers whose content is assembled from other
        # files hash the expanded text, so upstream edits are detected.
        self.md5 = (
            compute_file_hash(self.path)
            if self._expanded == self._raw
            else hashlib.md5(self._expanded.encode("utf-8")).hexdigest()
        )

    def _read(self) -> tuple[str, str]:
        """Return (raw, expanded) markdown; ("", "") if the file is missing."""
        try:
            with open(self.path, encoding="utf-8") as f:
                raw = f.read()
        except (OSError, UnicodeDecodeError) as e:
            err_console.print(f"⚠ Error reading file {self.path}: {e}")
            return "", ""
        return raw, expand_includes(raw, self.path, self.root_path)

    def get_title(self) -> str:
        """First `# ` heading of the (expanded) markdown, used as the title."""
        title = _first_h1(self._expanded)
        if title is None:
            err_console.print(
                f"Couldn't find title in {self.path}. Make sure that the file contains a level 1 heading."
            )
            return self.path.stem
        return title

    def get_content(self) -> str:
        # Strip the leading H1 (Canvas uses the title separately). In an
        # include wrapper the H1 may be preceded by a single HTML comment, so
        # optionally consume that too. For ordinary pages this is identical to
        # stripping the first "# " line.
        return re.sub(
            r"^\s*(?:<!--.*?-->\s*)?#\s+.*\n",
            "",
            self._expanded,
            count=1,
        )

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
