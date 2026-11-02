"""Tests for the content pipeline ownership marker and asset dedupe."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mkdocstocanvas.models.page import MarkdownPage, compute_file_hash
from mkdocstocanvas.uploaders.base import ContentUploader
from mkdocstocanvas.utils.marker import parse_marker


@pytest.fixture
def uploader(tmp_path: Path):
    client = MagicMock()
    client.base_url = "https://canvas.example.com"
    client.course_id = "123"
    return PageUploaderFixture(
        ContentUploader(client=client, cache=str(tmp_path / "state.json")),
        tmp_path,
    )


class PageUploaderFixture:
    def __init__(self, uploader: ContentUploader, docs_root: Path):
        self.uploader = uploader
        self.docs_root = docs_root

    def page(self, rel: str, text: str) -> MarkdownPage:
        path = self.docs_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return MarkdownPage(path, root_path=self.docs_root)


class TestPrepareContentMarker:
    def test_marker_prepended(self, uploader):
        page = uploader.page("intro.md", "# Intro\n\nHello.\n")
        html = uploader.uploader._prepare_content(page)

        fields = parse_marker(html)
        assert fields is not None
        assert fields["rel"] == "intro.md"
        assert fields["md5"] == compute_file_hash(page.path)
        assert "<p>Hello.</p>" in html

    def test_marker_not_duplicated_on_reprocess(self, uploader):
        page = uploader.page("intro.md", "# Intro\n\nHello.\n")
        html1 = uploader.uploader._prepare_content(page)
        html2 = uploader.uploader._prepare_content(page)
        assert parse_marker(html1) == parse_marker(html2)
        assert html1.count('class="mkdocstocanvas-marker"') == 1
        # source markdown is reset from disk each time, so no marker leaks
        # into the markdown body of the second run
        assert "mkdocstocanvas-marker" not in page.content


class TestAssetDedupe:
    def test_existing_canvas_file_reused(self, uploader, tmp_path: Path):
        docs = tmp_path
        image = docs / "img.png"
        image.write_bytes(b"png-bytes")
        content_hash = compute_file_hash(image)
        page = uploader.page("page.md", "![pic](img.png)\n")

        client = uploader.uploader.client
        client.find_existing_file.return_value = {
            "id": 42,
            "md5": content_hash,
            "filename": "img.png",
            "size": image.stat().st_size,
        }

        html = uploader.uploader._prepare_content(page)
        client.upload_file.assert_not_called()
        assert 'src="https://canvas.example.com/courses/123/files/42?wrap=1"' in html
        assert uploader.uploader.files_cache["img.png"]["canvas_url"].endswith(
            "/files/42?wrap=1"
        )

    def test_missing_file_is_uploaded(self, uploader, tmp_path: Path):
        image = tmp_path / "img.png"
        image.write_bytes(b"png-bytes")
        page = uploader.page("page.md", "![pic](img.png)\n")

        client = uploader.uploader.client
        client.find_existing_file.return_value = None
        client.upload_file.return_value = (
            "https://canvas.example.com/courses/123/files/99?wrap=1"
        )

        html = uploader.uploader._prepare_content(page)
        client.upload_file.assert_called_once()
        assert "/files/99?wrap=1" in html

    def test_force_still_dedupes(self, uploader, tmp_path: Path):
        # Assets are content-addressed: even with --force, an identical file
        # in Canvas is reused instead of creating a duplicate upload.
        image = tmp_path / "img.png"
        image.write_bytes(b"png-bytes")
        page = uploader.page("page.md", "![pic](img.png)\n")

        client = uploader.uploader.client
        uploader.uploader.force = True
        client.find_existing_file.return_value = {"id": 42}

        uploader.uploader._prepare_content(page)
        client.find_existing_file.assert_called_once()
        client.upload_file.assert_not_called()

    def test_force_skips_local_asset_cache(self, uploader, tmp_path: Path):
        # --force ignores the local asset cache entry (but not the dedupe)
        image = tmp_path / "img.png"
        image.write_bytes(b"png-bytes")
        content_hash = compute_file_hash(image)
        page = uploader.page("page.md", "![pic](img.png)\n")

        uploader.uploader.force = True
        uploader.uploader.files_cache["img.png"] = {
            "hash": content_hash,
            "canvas_url": "https://stale.example/files/1?wrap=1",
        }
        client = uploader.uploader.client
        client.find_existing_file.return_value = None
        client.upload_file.return_value = (
            "https://canvas.example.com/courses/123/files/99?wrap=1"
        )

        uploader.uploader._prepare_content(page)
        client.find_existing_file.assert_called_once()
        client.upload_file.assert_called_once()

    def test_local_cache_hit_skips_canvas_lookup(self, uploader, tmp_path: Path):
        # Without force, a matching cache entry is used without API calls
        image = tmp_path / "img.png"
        image.write_bytes(b"png-bytes")
        content_hash = compute_file_hash(image)
        page = uploader.page("page.md", "![pic](img.png)\n")

        uploader.uploader.files_cache["img.png"] = {
            "hash": content_hash,
            "canvas_url": "https://canvas.example.com/courses/123/files/7?wrap=1",
        }
        client = uploader.uploader.client

        html = uploader.uploader._prepare_content(page)
        client.find_existing_file.assert_not_called()
        client.upload_file.assert_not_called()
        assert "/files/7?wrap=1" in html
