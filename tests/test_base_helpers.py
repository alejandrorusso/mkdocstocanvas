"""Tests for the shared uploader helpers (uploaders/base.py)."""

from typing import ClassVar
from unittest.mock import MagicMock

import pytest
import typer

from mkdocstocanvas.uploaders.base import (
    delete_all_items,
    make_progress,
    print_results_summary,
    require_connection,
)


@pytest.fixture
def no_confirm(monkeypatch):
    """Replace the interactive confirmation with one that always says yes."""
    calls = []
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: calls.append(a) or True)
    return calls


class TestRequireConnection:
    def test_success_prints_message(self, capsys):
        client = MagicMock()
        client.test_connection.return_value = (True, "Connected!")
        require_connection(client)
        assert "Connected!" in capsys.readouterr().out

    def test_failure_exits(self, capsys):
        client = MagicMock()
        client.test_connection.return_value = (False, "Bad token")
        with pytest.raises(typer.Exit) as exc:
            require_connection(client)
        assert exc.value.exit_code == 1
        assert "Bad token" in capsys.readouterr().err


class TestDeleteAllItems:
    items: ClassVar[list[dict]] = [
        {"id": 1, "name": "A"},
        {"id": 2, "name": "B"},
        {"id": 3, "name": "C"},
    ]

    def _kwargs(self, **overrides):
        kwargs = {
            "noun": "widget",
            "name_column": "Widget",
            "columns": [("Widget", None), ("ID", "right")],
            "row_values": lambda w: (w["name"], str(w["id"])),
            "assume_yes": True,
        }
        kwargs.update(overrides)
        return kwargs

    def test_deletes_every_item(self, capsys):
        deleted = []
        delete_all_items(
            self.items,
            lambda i: deleted.append(i["id"]) or True,
            **self._kwargs(),
        )
        assert deleted == [1, 2, 3]
        out = capsys.readouterr().out
        assert "Deleted: 3" in out
        assert "Failed: 0" in out

    def test_partial_failure_counted(self, capsys):
        delete_all_items(
            self.items,
            lambda i: i["id"] != 3,
            **self._kwargs(),
        )
        out = capsys.readouterr().out
        assert "Deleted: 2" in out
        assert "Failed: 1" in out
        assert "✗ Failed" in out

    def test_empty_list_no_delete(self, capsys):
        deleted = []

        def record(item):
            deleted.append(item)
            return True

        delete_all_items([], record, **self._kwargs())
        assert deleted == []
        assert "Nothing to delete" in capsys.readouterr().out

    def test_confirmation_skipped_with_assume_yes(self, no_confirm, capsys):
        delete_all_items(self.items, lambda i: True, **self._kwargs())
        assert no_confirm == []

    def test_confirmation_asked_without_assume_yes(self, no_confirm, capsys):
        delete_all_items(self.items, lambda i: True, **self._kwargs(assume_yes=False))
        assert len(no_confirm) == 1
        assert "widget" in no_confirm[0][0]

    def test_all_failed_exits_1(self, capsys):
        with pytest.raises(typer.Exit) as exc:
            delete_all_items(self.items[:1], lambda i: False, **self._kwargs())
        assert exc.value.exit_code == 1


class TestPrintResultsSummary:
    def test_ok_and_failed_rows(self, capsys):
        print_results_summary(
            "My Summary",
            [
                {"name": "Good", "status": "ok", "url": "https://x"},
                {"name": "Bad", "status": "error", "error": "boom"},
            ],
            name_column="Thing",
            detail_column="Detail",
            detail_of=lambda r: r.get("url") or r.get("error", ""),
            ok_label="Uploaded",
        )
        out = capsys.readouterr().out
        assert "My Summary" in out
        assert "Uploaded: 1" in out
        assert "Failed: 1" in out
        assert "boom" in out


class TestMakeProgress:
    def test_returns_transient_progress(self):
        progress = make_progress()
        assert progress.live.transient is True
