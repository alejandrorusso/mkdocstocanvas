"""Tests for mkdocs.yml parsing (utils/config.py)."""

from pathlib import Path

import pytest
import typer

from mkdocstocanvas.utils.config import (
    _collect_markdown_files,
    parse_mkdocs_nav,
    parse_mkdocs_nav_sections,
    parse_syllabus_nav_file,
)


def write_mkdocs(tmp_path: Path, nav: str) -> Path:
    path = tmp_path / "mkdocs.yml"
    path.write_text(f"site_name: Test\ncopyright: X\nnav:\n{nav}")
    return path


class TestParseMkdocsNav:
    def test_flattens_nested_nav(self, tmp_path):
        path = write_mkdocs(
            tmp_path,
            dedent_nav(
                """
                - Syllabus:
                  - index.md
                - Lectures:
                  - lectures/a.md
                  - lectures/b.md
                """
            ),
        )
        assert parse_mkdocs_nav(path) == [
            "index.md",
            "lectures/a.md",
            "lectures/b.md",
        ]

    def test_no_nav_returns_empty_list(self, tmp_path):
        path = tmp_path / "mkdocs.yml"
        path.write_text("site_name: Test\n")
        assert parse_mkdocs_nav(path) == []

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(typer.Exit):
            parse_mkdocs_nav(tmp_path / "missing.yml")

    def test_invalid_yaml_exits(self, tmp_path):
        path = tmp_path / "mkdocs.yml"
        path.write_text("a: [unclosed")
        with pytest.raises(typer.Exit):
            parse_mkdocs_nav(path)


class TestParseMkdocsNavSections:
    def test_sections_with_pages(self, tmp_path):
        path = write_mkdocs(
            tmp_path,
            dedent_nav(
                """
                - Syllabus:
                  - index.md
                - Lectures:
                  - lectures/a.md
                """
            ),
        )
        sections = parse_mkdocs_nav_sections(path)
        assert sections == [
            {"name": "Syllabus", "pages": ["index.md"]},
            {"name": "Lectures", "pages": ["lectures/a.md"]},
        ]

    def test_section_without_markdown_pages_skipped(self, tmp_path):
        # Sections whose entries are not .md files collect no pages and are skipped
        path = tmp_path / "mkdocs.yml"
        path.write_text(
            "site_name: Test\nnav:\n"
            "  - 'Empty':\n"
            "    - 'logo.png'\n"
            "  - 'Real':\n"
            "    - 'real.md'\n"
        )
        sections = parse_mkdocs_nav_sections(path)
        assert sections == [{"name": "Real", "pages": ["real.md"]}]

    def test_no_sections_returns_none(self, tmp_path):
        path = tmp_path / "mkdocs.yml"
        path.write_text("site_name: Test\n")
        assert parse_mkdocs_nav_sections(path) is None


class TestParseSyllabusNavFile:
    def test_finds_syllabus_section_case_insensitive(self, tmp_path):
        path = write_mkdocs(
            tmp_path,
            dedent_nav(
                """
                - Syllabus:
                  - syllabus.md
                - Lectures:
                  - lectures/a.md
                """
            ),
        )
        assert parse_syllabus_nav_file(path) == "syllabus.md"

    def test_no_syllabus_section(self, tmp_path):
        path = write_mkdocs(tmp_path, dedent_nav("\n- Lectures:\n  - a.md\n"))
        assert parse_syllabus_nav_file(path) is None


class TestCollectMarkdownFiles:
    def test_nested_structures(self):
        nav = ["a.md", {"Section": ["b.md", {"Sub": ["c.md"]}]}]
        assert _collect_markdown_files(nav) == ["a.md", "b.md", "c.md"]

    def test_ignores_non_markdown(self):
        assert _collect_markdown_files(["a.md", "image.png", {"S": ["b.md"]}]) == [
            "a.md",
            "b.md",
        ]


def dedent_nav(text: str) -> str:
    """Dedent a triple-quoted YAML fragment while keeping relative indentation."""
    lines = text.strip("\n").splitlines()
    base = len(lines[0]) - len(lines[0].lstrip(" "))
    return "\n".join(line[base:] if line.strip() else line for line in lines)
