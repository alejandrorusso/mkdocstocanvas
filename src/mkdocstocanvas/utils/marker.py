"""Ownership markers embedded in Canvas content uploaded by this tool.

Every page/assignment/syllabus body gets an invisible marker element of the
form:

    <span class="mkdocstocanvas-marker" data-rel="lectures/01 intro.md"
     data-md5="ab12cd…"></span>

Note: an HTML comment is NOT usable here — Canvas strips comments from page
bodies, but keeps data-* attributes on span elements.

The marker identifies the tool as the owner of the content (so deletes can
skip manually created items) and lets the upload cache be rebuilt from
Canvas after the local cache file is lost.
"""

from html import escape
from html.parser import HTMLParser

MARKER_CLASS = "mkdocstocanvas-marker"


def make_marker(rel_path: str, md5: str | None) -> str:
    """Build the invisible ownership marker element for a docs-relative path."""
    rel = escape(rel_path, quote=True)
    fields = f'<span class="{MARKER_CLASS}" data-rel="{rel}"'
    if md5:
        fields += f' data-md5="{escape(md5, quote=True)}"'
    return fields + "></span>"


class _MarkerParser(HTMLParser):
    """Find the first marker element in an HTML body."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.fields: dict[str, str] | None = None

    def handle_starttag(self, tag, attrs):
        if self.fields is not None:
            return
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if MARKER_CLASS in classes:
            fields = {}
            if attributes.get("data-rel"):
                fields["rel"] = attributes["data-rel"]
            if attributes.get("data-md5"):
                fields["md5"] = attributes["data-md5"]
            if fields:
                self.fields = fields


def parse_marker(html: str) -> dict[str, str] | None:
    """
    Extract the first ownership marker from an HTML body.

    Returns {"rel": ..., "md5": ...} or None if no marker is present
    (i.e. the content was not uploaded by this tool).
    """
    parser = _MarkerParser()
    try:
        parser.feed(html)
    except Exception:
        return None
    return parser.fields
