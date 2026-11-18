"""Tests for include-markdown expansion in the upload path (offline).

Ensures wrapper pages (only an `{% include-markdown %}` directive) upload with
the included title/body, and that ordinary pages are untouched.
"""

from pathlib import Path

from mkdocstocanvas.models.page import (
    MarkdownPage,
    compute_file_hash,
    extract_title_text,
)
from mkdocstocanvas.processing.includes import expand_includes

SOURCE = "# Real Title\n\nThe body paragraph.\n\n## Readings\n\n- a reference\n"


def _tree(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "src.md").write_text(SOURCE)
    # A wrapper: leading comment + an include that drops the Readings section.
    (docs / "wrap.md").write_text(
        '<!-- wrapper -->\n{% include-markdown "src.md" end="## Readings" %}\n'
    )
    (docs / "plain.md").write_text("# Plain\n\njust text\n")
    return docs


class TestExpandIncludes:
    def test_resolves_directive_with_end_anchor(self, tmp_path):
        docs = _tree(tmp_path)
        out = expand_includes((docs / "wrap.md").read_text(), docs / "wrap.md", docs)
        assert "# Real Title" in out
        assert "The body paragraph." in out
        assert "Readings" not in out
        assert "{%" not in out

    def test_passthrough_without_directive(self, tmp_path):
        docs = _tree(tmp_path)
        raw = (docs / "plain.md").read_text()
        assert expand_includes(raw, docs / "plain.md", docs) == raw


class TestMarkdownPageWithIncludes:
    def test_wrapper_title_body_and_hash(self, tmp_path):
        docs = _tree(tmp_path)
        page = MarkdownPage(docs / "wrap.md", docs)

        assert page.title == "Real Title"
        assert "The body paragraph." in page.content
        assert "Readings" not in page.content
        assert "{%" not in page.content
        # hash reflects the expanded content, not the (tiny) wrapper file
        assert page.md5 != compute_file_hash(docs / "wrap.md")

    def test_wrapper_hash_tracks_source_changes(self, tmp_path):
        docs = _tree(tmp_path)
        before = MarkdownPage(docs / "wrap.md", docs).md5
        (docs / "src.md").write_text(SOURCE.replace("body paragraph", "edited body"))
        after = MarkdownPage(docs / "wrap.md", docs).md5
        assert before != after

    def test_plain_page_unchanged(self, tmp_path):
        docs = _tree(tmp_path)
        page = MarkdownPage(docs / "plain.md", docs)
        assert page.title == "Plain"
        assert page.content == "\njust text\n"
        # ordinary pages keep the byte-for-byte file hash
        assert page.md5 == compute_file_hash(docs / "plain.md")

    def test_extract_title_text_expands(self, tmp_path):
        docs = _tree(tmp_path)
        assert extract_title_text(docs / "wrap.md", docs) == "Real Title"
