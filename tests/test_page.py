"""Tests for MarkdownPage (models/page.py)."""

import hashlib

import pytest

from mkdocstocanvas.models.page import MarkdownPage, compute_file_hash


@pytest.fixture
def md_file(tmp_path):
    """A markdown file inside a docs/ root, returning (file, root)."""
    root = tmp_path / "docs"
    sub = root / "lectures"
    sub.mkdir(parents=True)
    file = sub / "lecture1.md"
    file.write_text("# My Lecture\n\nSome content.\n")
    return file, root


class TestMarkdownPage:
    def test_title_from_h1(self, md_file):
        page = MarkdownPage(md_file[0], root_path=md_file[1])
        assert page.title == "My Lecture"

    def test_title_falls_back_to_stem(self, tmp_path):
        file = tmp_path / "untitled.md"
        file.write_text("No heading here\n")
        page = MarkdownPage(file, root_path=tmp_path)
        assert page.title == "untitled"

    def test_rel_path(self, md_file):
        page = MarkdownPage(md_file[0], root_path=md_file[1])
        assert str(page.rel_path) == "lectures/lecture1.md"

    def test_content_strips_h1(self, md_file):
        # The H1 line is removed; the blank line after it is kept
        page = MarkdownPage(md_file[0], root_path=md_file[1])
        assert page.content == "\nSome content.\n"

    def test_md5(self, md_file):
        page = MarkdownPage(md_file[0], root_path=md_file[1])
        expected = hashlib.md5(md_file[0].read_bytes()).hexdigest()
        assert page.md5 == expected


class TestComputeFileHash:
    def test_missing_file_returns_none(self, tmp_path):
        assert compute_file_hash(tmp_path / "missing.md") is None

    def test_matches_content(self, tmp_path):
        file = tmp_path / "f.txt"
        file.write_text("hello")
        assert compute_file_hash(file) == hashlib.md5(b"hello").hexdigest()
