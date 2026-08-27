"""Tests for lab detection and delete matching (uploaders/labs.py)."""

from pathlib import Path

from mkdocstocanvas.uploaders.labs import (
    _LAB_ASSIGNMENT_PATTERN,
    _find_lab_assignments,
    find_lab_files,
    is_lab_rel_path,
)


class TestIsLabRelPath:
    def test_lab_file(self):
        assert is_lab_rel_path("labs/lab1.md")

    def test_case_insensitive(self):
        assert is_lab_rel_path("labs/Lab2-intro.md")

    def test_numberless_lab(self):
        assert is_lab_rel_path("labs/lab-intro.md")

    def test_non_lab_file_in_labs_dir(self):
        assert not is_lab_rel_path("labs/notes.md")

    def test_lab_named_file_outside_labs_dir(self):
        assert not is_lab_rel_path("lectures/lab1.md")

    def test_non_markdown(self):
        assert not is_lab_rel_path("labs/lab1.txt")

    def test_labs_subdirectory_is_not_a_lab(self):
        # must be directly under labs/, not deeper
        assert not is_lab_rel_path("labs/sub/lab1.md")


class TestFindLabFiles:
    def test_finds_sorted_by_number(self, tmp_path: Path):
        labs = tmp_path / "labs"
        labs.mkdir()
        (labs / "lab10.md").write_text("# Lab 10")
        (labs / "lab2.md").write_text("# Lab 2")
        (labs / "Lab1.md").write_text("# Lab 1")

        result = find_lab_files(tmp_path)
        assert [p.name for p in result] == ["Lab1.md", "lab2.md", "lab10.md"]

    def test_ignores_non_lab_files(self, tmp_path: Path):
        labs = tmp_path / "labs"
        labs.mkdir()
        (labs / "notes.md").write_text("# Notes")
        (labs / "lab1.md").write_text("# Lab 1")

        result = find_lab_files(tmp_path)
        assert [p.name for p in result] == ["lab1.md"]

    def test_ignores_subdirectories(self, tmp_path: Path):
        labs = tmp_path / "labs"
        (labs / "old").mkdir(parents=True)
        (labs / "lab1.md").write_text("# Lab 1")
        (labs / "old" / "lab2.md").write_text("# Lab 2")

        result = find_lab_files(tmp_path)
        assert [p.name for p in result] == ["lab1.md"]

    def test_missing_labs_dir(self, tmp_path: Path):
        assert find_lab_files(tmp_path) == []


class TestFindLabAssignments:
    def test_cached_id_matched_even_if_renamed(self):
        cache = {
            "labs/lab1.md": {"canvas_assignment_id": 101, "canvas_name": "Lab 1"},
        }
        assignments = [
            {"id": 101, "name": "Renamed Lab"},  # no longer matches the pattern
            {"id": 202, "name": "Other course work"},
        ]
        result = _find_lab_assignments(assignments, cache)
        assert [a["id"] for a in result] == [101]

    def test_name_pattern_fallback_for_uncached(self):
        assignments = [
            {"id": 301, "name": "Lab 2 - Debugging"},  # not in cache
            {"id": 302, "name": "Reading quiz"},
        ]
        result = _find_lab_assignments(assignments, {})
        assert [a["id"] for a in result] == [301]

    def test_cached_and_fallback_combined(self):
        cache = {
            "labs/lab1.md": {"canvas_assignment_id": 101},
        }
        assignments = [
            {"id": 101, "name": "Renamed Lab"},
            {"id": 102, "name": "Lab 9 - Extras"},
            {"id": 103, "name": "Project"},
        ]
        result = _find_lab_assignments(assignments, cache)
        assert [a["id"] for a in result] == [101, 102]


class TestLabAssignmentPattern:
    def test_matches_numbered_labs(self):
        assert _LAB_ASSIGNMENT_PATTERN.match("Lab 1 - Intro")
        assert _LAB_ASSIGNMENT_PATTERN.match("lab 12: something")

    def test_requires_number(self):
        assert not _LAB_ASSIGNMENT_PATTERN.match("Lab intro")
        assert not _LAB_ASSIGNMENT_PATTERN.match("Laboratory safety")
