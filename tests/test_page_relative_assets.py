"""Bare relative asset paths resolve relative to the page (like mkdocs), with a
fallback to the docs root for backward compatibility. Offline — the Canvas
client is mocked.
"""

from pathlib import Path
from unittest.mock import MagicMock

from mkdocstocanvas.models.page import MarkdownPage
from mkdocstocanvas.uploaders.base import ContentUploader


def _uploader(docs: Path) -> ContentUploader:
    client = MagicMock()
    client.base_url = "https://canvas.example.com"
    client.course_id = "123"
    client.find_existing_file.return_value = None
    client.upload_file.return_value = (
        "https://canvas.example.com/courses/123/files/7?wrap=1"
    )
    return ContentUploader(client=client, cache=str(docs / "state.json"))


def _page(docs: Path, rel: str, text: str) -> MarkdownPage:
    path = docs / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return MarkdownPage(path, root_path=docs)


def test_bare_path_resolves_relative_to_page(tmp_path: Path):
    # Image sits next to a page in a subdirectory; a bare reference must find
    # it page-relative (mkdocs semantics), not at the docs root.
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "img.png").write_bytes(b"png-bytes")
    up = _uploader(tmp_path)
    page = _page(tmp_path, "sub/page.md", "# P\n\n![pic](img.png)\n")

    html = up._prepare_content(page)

    up.client.upload_file.assert_called_once()
    assert "/files/7/preview" in html


def test_bare_path_falls_back_to_docs_root(tmp_path: Path):
    # Backward compatibility: a bare path that only exists at the docs root
    # still resolves even when the page is in a subdirectory.
    (tmp_path / "img.png").write_bytes(b"png-bytes")
    (tmp_path / "sub").mkdir()
    up = _uploader(tmp_path)
    page = _page(tmp_path, "sub/page.md", "# P\n\n![pic](img.png)\n")

    html = up._prepare_content(page)

    up.client.upload_file.assert_called_once()
    assert "/files/7/preview" in html
