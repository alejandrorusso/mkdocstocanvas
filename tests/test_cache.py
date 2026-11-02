"""Tests for cache load/save (utils/cache.py)."""

import json

from mkdocstocanvas.utils.cache import load_cache, save_cache


class TestLoadCache:
    def test_missing_file_created_empty(self, tmp_path):
        cache = tmp_path / "state.json"
        assert load_cache(cache) == {}
        assert cache.read_text() == "{}"

    def test_existing_file_roundtrip(self, tmp_path):
        cache = tmp_path / "state.json"
        save_cache(cache, {"pages": {"a.md": {"hash": "x"}}})
        assert load_cache(cache) == {"pages": {"a.md": {"hash": "x"}}}

    def test_empty_file_returns_empty_dict(self, tmp_path):
        cache = tmp_path / "state.json"
        cache.write_text("")
        assert load_cache(cache) == {}

    def test_whitespace_only_file_returns_empty_dict(self, tmp_path):
        cache = tmp_path / "state.json"
        cache.write_text("   \n  ")
        assert load_cache(cache) == {}


class TestSaveCache:
    def test_writes_valid_json(self, tmp_path):
        cache = tmp_path / "state.json"
        save_cache(cache, {"pages": {}})
        assert json.loads(cache.read_text()) == {"pages": {}}

    def test_creates_parent_directories(self, tmp_path):
        cache = tmp_path / "deep" / "nested" / "state.json"
        save_cache(cache, {"a": 1})
        assert cache.exists()

    def test_no_tmp_file_left_behind(self, tmp_path):
        cache = tmp_path / "state.json"
        save_cache(cache, {"a": 1})
        assert not cache.with_name("state.json.tmp").exists()

    def test_overwrites_previous_content(self, tmp_path):
        cache = tmp_path / "state.json"
        save_cache(cache, {"a": 1})
        save_cache(cache, {"b": 2})
        assert load_cache(cache) == {"b": 2}
