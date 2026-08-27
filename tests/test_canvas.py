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


class TestCreateOrUpdatePage:
    def test_slug_generated_from_title(self, uploader):
        uploader.session.get.return_value = _ok_response({"url": "my-cool-page"})
        uploader.session.put.return_value = _ok_response({"url": "my-cool-page"})
        url = uploader.create_or_update_page("My Cool Page!", "body")
        assert url == "https://canvas.example.com/courses/123/pages/my-cool-page"
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

        url = uploader.create_or_update_page("Brand New", "body")

        uploader.session.post.assert_called_once()
        uploader.session.put.assert_not_called()
        assert url == "https://canvas.example.com/courses/123/pages/brand-new"

    def test_post_failure_returns_none(self, uploader):
        missing = MagicMock()
        missing.raise_for_status.side_effect = _http_error(404)
        uploader.session.get.return_value = missing
        uploader.list_pages = MagicMock(return_value=[])
        failed = MagicMock()
        failed.raise_for_status.side_effect = _http_error(500)
        uploader.session.post.return_value = failed

        assert uploader.create_or_update_page("Brand New", "body") is None
