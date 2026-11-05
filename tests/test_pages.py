"""Tests for uploading pages linked from the nav but not listed in it."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mkdocstocanvas.models.page import MarkdownPage
from mkdocstocanvas.uploaders.pages import PageUploader


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    (docs / "extra").mkdir(parents=True)
    (docs / "labs").mkdir()
    (docs / "a.md").write_text(
        "# A\n\n[b](extra/b.md) [lab](labs/lab1.md) [gone](missing.md) "
        "[out](../outside.md) [again](extra/b.md#x)\n"
    )
    (docs / "extra" / "b.md").write_text("# B\n\n[c](c.md) [a](../a.md)\n")
    (docs / "extra" / "c.md").write_text("# C\n")
    (docs / "labs" / "lab1.md").write_text("# Lab 1\n\n[d](../d.md)\n")
    (docs / "d.md").write_text("# D\n")
    (tmp_path / "outside.md").write_text("# Outside\n")
    return docs


def _client() -> MagicMock:
    client = MagicMock()
    client.base_url = "https://canvas.example.com"
    client.course_id = "123"
    client.test_connection.return_value = (True, "Connected")
    client.create_or_update_page.side_effect = lambda title, html, **_: (
        f"https://canvas.example.com/courses/123/pages/{title.lower()}",
        title.lower(),
    )
    return client


def _uploader(tmp_path: Path, client: MagicMock) -> PageUploader:
    uploader = PageUploader(client, cache=str(tmp_path / "state.json"))
    uploader._rebuild_cache_if_empty = lambda: None  # type: ignore[method-assign]
    return uploader


class TestWithLinkedPages:
    def test_discovers_transitively_and_skips_labs_missing_outside(
        self, tmp_path: Path, docs: Path
    ):
        uploader = _uploader(tmp_path, _client())
        pages = [MarkdownPage(docs / "a.md", docs)]

        result = uploader._with_linked_pages(pages, [])

        # a -> b -> c; the a<->b cycle terminates; no duplicates.
        assert [str(p.rel_path) for p in result] == [
            "a.md",
            "extra/b.md",
            "extra/c.md",
        ]

    def test_link_seeds_scanned_but_not_returned(self, tmp_path: Path, docs: Path):
        uploader = _uploader(tmp_path, _client())
        lab = MarkdownPage(docs / "labs" / "lab1.md", docs)

        result = uploader._with_linked_pages([], [lab])

        assert [str(p.rel_path) for p in result] == ["d.md"]


class TestUploadLinkedPages:
    def test_linked_pages_uploaded_and_links_resolved(self, tmp_path: Path, docs: Path):
        client = _client()
        uploader = _uploader(tmp_path, client)

        uploader.upload_all_pages([MarkdownPage(docs / "a.md", docs)])

        cache = uploader.pages_cache
        b_url = cache["extra/b.md"]["canvas_url"]
        assert cache["extra/c.md"]["canvas_url"]

        # The final upload of A must contain B's Canvas URL (Stage 2 fixes it
        # if A was uploaded before B had a slug).
        a_bodies = [
            c.args[1]
            for c in client.create_or_update_page.call_args_list
            if c.args[0] == "A"
        ]
        assert b_url in a_bodies[-1]


class TestCachedUrlHostChange:
    """Regression: after CANVAS_BASE_URL moves to another domain for the same
    Canvas, cached URLs still carry the old host. Links to pages uploaded later
    in the run (e.g. linked-only pages) used to keep that stale host."""

    OLD = "https://old.example.com/courses/123/pages/b"

    def _setup(self, tmp_path: Path, docs: Path):
        (docs / "a.md").write_text("# A\n\n[b](extra/b.md#x)\n")
        uploader = _uploader(tmp_path, _client())
        page = MarkdownPage(docs / "a.md", docs)
        uploader.pages_cache["extra/b.md"] = {
            "canvas_url": self.OLD,
            "page_url_slug": "b",
        }
        return uploader, page

    def test_link_uses_configured_host(self, tmp_path: Path, docs: Path):
        uploader, page = self._setup(tmp_path, docs)
        html = uploader._prepare_content(page)
        assert 'href="https://canvas.example.com/courses/123/pages/b#x"' in html
        assert "old.example.com" not in html

    def test_host_change_marks_mentioner_dirty(self, tmp_path: Path, docs: Path):
        # The page itself is unchanged, but its stored link snapshot has the
        # old host, so it is re-uploaded without --force.
        uploader, page = self._setup(tmp_path, docs)
        uploader.pages_cache["a.md"] = {
            "hash": page.md5,
            "page_url_slug": "a",
            "resolved_md_links": {"extra/b.md": self.OLD},
        }
        assert uploader._needs_upload(page)
