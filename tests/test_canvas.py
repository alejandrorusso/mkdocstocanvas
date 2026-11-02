"""Tests for the Canvas API client (api/canvas.py), using a mocked session."""

from unittest.mock import MagicMock

import pytest
import requests

from mkdocstocanvas.api.canvas import DEFAULT_TIMEOUT, CanvasUploader, _TimeoutSession


def _http_error(status: int) -> requests.exceptions.HTTPError:
    response = MagicMock()
    response.status_code = status
    response.text = f"HTTP {status}"
    return requests.exceptions.HTTPError(response=response)


@pytest.fixture
def uploader():
    client = CanvasUploader("https://canvas.example.com", "token", 123)
    client.session = MagicMock()
    return client


def _ok_response(json_data=None):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = json_data if json_data is not None else {}
    return response


class TestTimeoutSession:
    def test_default_timeout_applied_to_all_verbs(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(requests.Session, "request", spy)
        session = _TimeoutSession()
        session.get("http://x")
        session.post("http://x")
        session.put("http://x")
        session.delete("http://x")
        assert spy.call_count == 4
        for call in spy.call_args_list:
            assert call.kwargs["timeout"] == DEFAULT_TIMEOUT == 30

    def test_explicit_timeout_wins(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(requests.Session, "request", spy)
        session = _TimeoutSession()
        session.get("http://x", timeout=5)
        assert spy.call_args.kwargs["timeout"] == 5


class TestGetExistingPage:
    def test_returns_json_on_200(self, uploader):
        uploader.session.get.return_value = _ok_response({"title": "Hi"})
        assert uploader.get_existing_page("hi") == {"title": "Hi"}

    def test_404_returns_none(self, uploader):
        response = MagicMock()
        response.raise_for_status.side_effect = _http_error(404)
        uploader.session.get.return_value = response
        assert uploader.get_existing_page("missing") is None

    def test_401_raises_not_none(self, uploader):
        # An auth failure must never be mistaken for "page doesn't exist",
        # otherwise create_or_update_page would POST duplicate pages.
        response = MagicMock()
        response.raise_for_status.side_effect = _http_error(401)
        uploader.session.get.return_value = response
        with pytest.raises(requests.exceptions.HTTPError):
            uploader.get_existing_page("hi")


class TestFindExistingFile:
    def _files(self):
        return [
            {"id": 1, "md5": "aaa", "filename": "a.png", "size": 10},
            {"id": 2, "md5": None, "filename": "b.png", "size": 20},
            {"id": 3, "md5": None, "filename": "no-size.png", "size": None},
        ]

    def test_lazy_index_built_once(self, uploader):
        uploader.list_files = MagicMock(return_value=self._files())
        assert uploader.find_existing_file(md5="aaa")["id"] == 1
        assert uploader.find_existing_file(md5="missing") is None
        assert uploader.list_files.call_count == 1

    def test_fallback_matches_filename_and_size(self, uploader):
        uploader.list_files = MagicMock(return_value=self._files())
        assert uploader.find_existing_file(filename="b.png", size=20)["id"] == 2
        # same name, wrong size -> no match
        assert uploader.find_existing_file(filename="b.png", size=21) is None
        # md5 wins over name+size
        assert uploader.find_existing_file(
            md5="aaa", filename="b.png", size=20
        )["id"] == 1

    def test_files_without_usable_identity_ignored(self, uploader):
        uploader.list_files = MagicMock(return_value=self._files())
        assert uploader.find_existing_file(filename="no-size.png", size=5) is None
        assert uploader.find_existing_file(filename="b.png", size=None) is None


class TestCreateOrUpdatePage:
    def test_slug_generated_from_title(self, uploader):
        uploader.session.get.return_value = _ok_response({"url": "my-cool-page"})
        uploader.session.put.return_value = _ok_response({"url": "my-cool-page"})
        url, slug = uploader.create_or_update_page("My Cool Page!", "body")
        assert url == "https://canvas.example.com/courses/123/pages/my-cool-page"
        assert slug == "my-cool-page"
        put_url = uploader.session.put.call_args.args[0]
        assert put_url.endswith("/pages/my-cool-page")

    def test_existing_page_is_updated_not_created(self, uploader):
        uploader.session.get.return_value = _ok_response({"url": "hi"})
        uploader.session.put.return_value = _ok_response({"url": "hi"})
        uploader.create_or_update_page("Hi", "body")
        uploader.session.put.assert_called_once()
        uploader.session.post.assert_not_called()

    def test_missing_page_is_created(self, uploader):
        missing = MagicMock()
        missing.raise_for_status.side_effect = _http_error(404)
        uploader.session.get.return_value = missing
        uploader.list_pages = MagicMock(return_value=[])
        uploader.session.post.return_value = _ok_response({"url": "brand-new"})

        url, slug = uploader.create_or_update_page("Brand New", "body")

        uploader.session.post.assert_called_once()
        uploader.session.put.assert_not_called()
        assert url == "https://canvas.example.com/courses/123/pages/brand-new"
        assert slug == "brand-new"

    def test_post_failure_returns_none(self, uploader):
        missing = MagicMock()
        missing.raise_for_status.side_effect = _http_error(404)
        uploader.session.get.return_value = missing
        uploader.list_pages = MagicMock(return_value=[])
        failed = MagicMock()
        failed.raise_for_status.side_effect = _http_error(500)
        uploader.session.post.return_value = failed

        assert uploader.create_or_update_page("Brand New", "body") is None

    def test_slug_taken_from_response_not_generated(self, uploader):
        # Canvas can normalize slugs differently than we generate them
        uploader.session.get.return_value = _ok_response({"url": "canvas-slug"})
        uploader.session.put.return_value = _ok_response({"url": "canvas-slug"})
        _, slug = uploader.create_or_update_page("My Title", "body")
        assert slug == "canvas-slug"
