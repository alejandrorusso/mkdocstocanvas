"""Tests for cache rebuilding and scoped deletes."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mkdocstocanvas.models.page import compute_file_hash
from mkdocstocanvas.uploaders.base import rebuild_cache_from_canvas
from mkdocstocanvas.uploaders.pages import delete_all_pages
from mkdocstocanvas.utils.cache import load_cache
from mkdocstocanvas.utils.marker import make_marker

# ---------------------------------------------------------------------------
# rebuild_cache_from_canvas
# ---------------------------------------------------------------------------


def _make_client(pages, assignments, syllabus_body, files):
    client = MagicMock()
    client.base_url = "https://canvas.example.com"
    client.course_id = "123"
    client.list_pages.return_value = pages
    client.list_assignments.return_value = assignments
    client.get_syllabus.return_value = syllabus_body
    client.list_files.return_value = files
    return client


class TestRebuildCacheFromCanvas:
    def test_pages_and_labs_restored(self, tmp_path: Path):
        pages = [
            {
                "url": "intro",
                "title": "Intro",
                "body": make_marker("intro.md", "h1") + "<p>hi</p>",
            },
            {"url": "foreign", "title": "Not ours", "body": "<p>no marker</p>"},
        ]
        assignments = [
            {
                "id": 55,
                "name": "Lab 1",
                "description": make_marker("labs/lab1.md", "h2")
                + "<p>lab</p>",
            }
        ]
        client = _make_client(pages, assignments, None, [])
        cache = rebuild_cache_from_canvas(client, tmp_path / "state.json", tmp_path)

        assert set(cache["pages"].keys()) == {"intro.md", "labs/lab1.md"}
        intro = cache["pages"]["intro.md"]
        assert intro["page_url_slug"] == "intro"
        assert intro["canvas_url"].endswith("/pages/intro")
        assert intro["hash"] == "h1"
        lab = cache["pages"]["labs/lab1.md"]
        assert lab["canvas_assignment_id"] == 55
        assert lab["canvas_url"].endswith("/assignments/55")
        # foreign page must not enter the cache
        assert all("foreign" not in str(v) for v in cache["pages"].values())
        # cache was persisted
        assert load_cache(tmp_path / "state.json") == cache

    def test_syllabus_restored(self, tmp_path: Path):
        client = _make_client(
            [], [], make_marker("syllabus.md", "h9"), []
        )
        cache = rebuild_cache_from_canvas(client, tmp_path / "state.json", tmp_path)
        syllabus = cache["pages"]["syllabus.md"]
        assert syllabus["page_url_slug"] == "syllabus"
        assert syllabus["canvas_url"].endswith("/assignments/syllabus")

    def test_assets_matched_by_md5(self, tmp_path: Path):
        docs = tmp_path / "docs"
        docs.mkdir()
        image = docs / "banana.jpg"
        image.write_bytes(b"fake-image-bytes")
        file_hash = compute_file_hash(image)

        client = _make_client(
            [],
            [],
            None,
            [{"id": 7, "md5": file_hash, "filename": "banana.jpg", "size": 16}],
        )
        # find_existing_file hits the (mocked) Canvas file list
        client.find_existing_file.side_effect = (
            lambda *, md5=None, filename=None, size=None: (
                {"id": 7, "md5": file_hash, "filename": "banana.jpg", "size": 16}
                if md5 == file_hash or (filename == "banana.jpg" and size == 16)
                else None
            )
        )
        cache = rebuild_cache_from_canvas(client, tmp_path / "state.json", docs)

        assert cache["files"]["banana.jpg"]["canvas_url"].endswith("/files/7?wrap=1")
        assert cache["files"]["banana.jpg"]["hash"] == file_hash

    def test_resolved_links_snapshot(self, tmp_path: Path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# A\n\n[b](b.md)\n")
        (docs / "b.md").write_text("# B\n")

        pages = [
            {
                "url": "a",
                "title": "A",
                "body": make_marker("a.md", "ha") + "<p>a</p>",
            },
            {
                "url": "b",
                "title": "B",
                "body": make_marker("b.md", "hb") + "<p>b</p>",
            },
        ]
        client = _make_client(pages, [], None, [])
        cache = rebuild_cache_from_canvas(client, tmp_path / "state.json", docs)

        b_url = cache["pages"]["b.md"]["canvas_url"]
        assert cache["pages"]["a.md"]["resolved_md_links"] == {"b.md": b_url}


# ---------------------------------------------------------------------------
# Scoped page deletion
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_pages():
    client = MagicMock()
    client.test_connection.return_value = (True, "Connected")
    client.delete_page.return_value = True
    return client


class TestDeleteAllPagesScope:
    def test_only_marker_pages_deleted(self, client_with_pages):
        client_with_pages.list_pages.return_value = [
            {"url": "ours", "title": "Ours", "body": make_marker("ours.md", "h1")},
            {"url": "theirs", "title": "Theirs", "body": "<p>manual page</p>"},
        ]
        delete_all_pages(client_with_pages, force=True)
        deleted = [c.args[0] for c in client_with_pages.delete_page.call_args_list]
        assert deleted == ["ours"]

    def test_all_flag_deletes_everything(self, client_with_pages):
        client_with_pages.list_pages.return_value = [
            {"url": "ours", "title": "Ours", "body": make_marker("ours.md", "h1")},
            {"url": "theirs", "title": "Theirs", "body": "<p>manual page</p>"},
        ]
        delete_all_pages(client_with_pages, force=True, all_items=True)
        deleted = [c.args[0] for c in client_with_pages.delete_page.call_args_list]
        assert sorted(deleted) == ["ours", "theirs"]

    def test_body_not_fetched_with_all_flag(self, client_with_pages):
        client_with_pages.list_pages.return_value = []
        delete_all_pages(client_with_pages, force=True, all_items=True)
        client_with_pages.list_pages.assert_called_once_with(include_body=False)

    def test_body_fetched_for_scope_check(self, client_with_pages):
        client_with_pages.list_pages.return_value = []
        delete_all_pages(client_with_pages, force=True)
        client_with_pages.list_pages.assert_called_once_with(include_body=True)


# ---------------------------------------------------------------------------
# Module scoping
# ---------------------------------------------------------------------------


class TestModuleScoping:
    def _uploader(self, tmp_path: Path):
        from mkdocstocanvas.uploaders.modules import ModuleUploader

        return ModuleUploader(
            client=MagicMock(),
            cache=str(tmp_path / "state.json"),
        )

    def test_cached_module_ids_are_managed(self, tmp_path: Path):
        uploader = self._uploader(tmp_path)
        uploader.modules_cache["Week 1"] = {"canvas_module_id": 11}
        modules = [{"id": 11, "name": "Week 1"}, {"id": 22, "name": "Quizzes"}]
        managed = uploader._managed_modules(modules, set())
        assert [m["id"] for m in managed] == [11]

    def test_section_name_match_is_managed(self, tmp_path: Path):
        uploader = self._uploader(tmp_path)
        modules = [{"id": 33, "name": "Week 2"}, {"id": 44, "name": "Quizzes"}]
        managed = uploader._managed_modules(modules, {"Week 2"})
        assert [m["id"] for m in managed] == [33]

    def test_foreign_modules_never_managed(self, tmp_path: Path):
        uploader = self._uploader(tmp_path)
        modules = [{"id": 44, "name": "Quizzes"}, {"id": 55, "name": "Imported"}]
        assert uploader._managed_modules(modules, set()) == []
