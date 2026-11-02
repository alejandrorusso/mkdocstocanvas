"""Tests for module uploads: PDF attachment (--add-pdf) and file-ID parsing."""

from pathlib import Path
from unittest.mock import MagicMock

from mkdocstocanvas.uploaders.modules import ModuleUploader


def _uploader(
    tmp_path: Path,
    client: MagicMock | None = None,
    add_pdf: bool = False,
) -> ModuleUploader:
    return ModuleUploader(
        client=client or MagicMock(),
        docs_root=tmp_path / "docs",
        pdf_root=tmp_path / "pdf",
        cache=str(tmp_path / "state.json"),
        add_pdf=add_pdf,
    )


class TestExtractFileId:
    def test_parses_upload_file_url_without_trailing_slash(self, tmp_path):
        """Regression: upload_file() returns '/files/{id}?wrap=1' — no trailing
        slash — so the old r'/files/(\\d+)/' pattern never matched and no PDF
        was ever attached to a module."""
        uploader = _uploader(tmp_path)
        url = "https://canvas.example.com/courses/1/files/67890?wrap=1"
        assert uploader._extract_file_id(url) == 67890

    def test_parses_url_with_trailing_slash(self, tmp_path):
        uploader = _uploader(tmp_path)
        assert uploader._extract_file_id("https://x.example.com/courses/1/files/42/") == 42

    def test_returns_none_for_non_file_url(self, tmp_path):
        uploader = _uploader(tmp_path)
        assert uploader._extract_file_id("https://x.example.com/courses/1/pages/foo") is None


class TestAddPdf:
    def _client(self) -> MagicMock:
        client = MagicMock()
        client.test_connection.return_value = (True, "Connected")
        client.list_pages.return_value = [{"title": "Lecture 1", "url": "lecture-1"}]
        client.list_modules.return_value = []
        client.create_module.return_value = {"id": 7, "name": "Module 1"}
        client.add_page_to_module.return_value = {"id": 1}
        client.upload_file.return_value = (
            "https://canvas.example.com/courses/1/files/99?wrap=1"
        )
        client.add_file_to_module.return_value = {"id": 2}
        return client

    def _seed_project(self, tmp_path: Path) -> None:
        (tmp_path / "docs" / "lectures").mkdir(parents=True)
        (tmp_path / "pdf").mkdir()
        (tmp_path / "docs" / "lectures" / "01-intro.md").write_text(
            "# Lecture 1\n\nBody\n"
        )
        (tmp_path / "pdf" / "01-intro.pdf").write_bytes(b"%PDF-1.4 fake")

    def test_matching_pdf_is_added_after_its_page(self, tmp_path):
        self._seed_project(tmp_path)
        client = self._client()
        uploader = _uploader(tmp_path, client=client, add_pdf=True)
        uploader.upload_all([{"name": "Module 1", "pages": ["lectures/01-intro.md"]}])

        client.add_file_to_module.assert_called_once()
        kwargs = client.add_file_to_module.call_args.kwargs
        assert kwargs["module_id"] == 7
        assert kwargs["file_id"] == 99
        page_position = client.add_page_to_module.call_args.args[3]
        assert kwargs["position"] == page_position + 1

    def test_no_pdf_attached_without_flag(self, tmp_path):
        self._seed_project(tmp_path)
        client = self._client()
        uploader = _uploader(tmp_path, client=client, add_pdf=False)
        uploader.upload_all([{"name": "Module 1", "pages": ["lectures/01-intro.md"]}])

        client.upload_file.assert_not_called()
        client.add_file_to_module.assert_not_called()

    def test_unmatched_pdf_is_skipped(self, tmp_path):
        self._seed_project(tmp_path)
        (tmp_path / "pdf" / "01-intro.pdf").unlink()
        client = self._client()
        uploader = _uploader(tmp_path, client=client, add_pdf=True)
        uploader.upload_all([{"name": "Module 1", "pages": ["lectures/01-intro.md"]}])

        client.add_file_to_module.assert_not_called()
