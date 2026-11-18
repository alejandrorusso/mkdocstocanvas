"""Expand mkdocs-include-markdown directives in page markdown.

mkdocstocanvas reads raw markdown files; it does not run the mkdocs build
pipeline, so ``{% include-markdown %}`` / ``{% include %}`` directives would
reach Canvas unexpanded — and a wrapper file that is *only* an include has no
level-1 heading to use as a title. We reuse the include-markdown plugin's own
text-in/text-out function so uploaded content matches what ``mkdocs build``
produces.

Pages without a directive are returned unchanged (the plugin is never called),
so non-wrapper pages behave exactly as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from mkdocs_include_markdown_plugin.directive import create_include_tag
from mkdocs_include_markdown_plugin.event import Settings, get_file_content

# Mirror the include-markdown plugin's own config defaults so expansion here
# matches a normal `mkdocs build`.
_DEFAULTS = {
    "encoding": "utf-8",
    "preserve-includer-indent": True,
    "dedent": False,
    "trailing-newlines": True,
    "comments": False,
    "rewrite-relative-urls": True,
    "heading-offset": 0,
    "recursive": True,
    "start": None,
    "end": None,
    "order": "default",
}


def expand_includes(markdown: str, page_path: Path, docs_dir: Path) -> str:
    """Return *markdown* with include-markdown directives resolved.

    ``page_path`` is the file the markdown came from (used to resolve relative
    includes); ``docs_dir`` is the docs root. Markdown with no directive is
    returned unchanged (the plugin is never invoked).
    """
    if "{%" not in markdown:
        return markdown

    tags = {
        "include": create_include_tag("{%", "%}", "include"),
        "include-markdown": create_include_tag("{%", "%}", "include-markdown"),
    }
    return get_file_content(
        markdown,
        str(page_path),
        str(docs_dir),
        cast(Any, tags),
        cast(Any, _DEFAULTS),
        Settings(exclude=[]),
    )
