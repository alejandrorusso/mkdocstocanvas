"""Tests for table styling in the markdown → HTML pipeline."""

from pathlib import Path

from mkdocstocanvas.models.page import MarkdownPage
from mkdocstocanvas.processing.markdown import (
    process_markdown_to_html,
    style_tables,
)

_TABLE = """# Schedule

| activity | when | where | who | topic |
|---|---|---|---|---|
| L1 | Tue 2026-11-03, 08:00 | *(not given)* | Elena | |
| L2 | Fri 2026-11-06, 10:00 | HB1 | Elena | |
"""

_TD_LEFT = (
    '<td style="border: 1px solid #ddd; padding: 8px 12px; vertical-align: top; '
    'text-align: left;">'
)
_TH_LEFT = (
    '<th style="border: 1px solid #ddd; padding: 8px 12px; vertical-align: top; '
    'text-align: left; background-color: #f5f5f5; font-weight: bold;">'
)
_TABLE_TAG = (
    '<table style="border-collapse: collapse; width: 100%; margin: 1.5em 0;">'
)


def _render(tmp_path: Path, content: str) -> str:
    docs = tmp_path / "docs"
    docs.mkdir()
    page_file = docs / "page.md"
    page_file.write_text(content)
    page = MarkdownPage(page_file, root_path=docs)
    return process_markdown_to_html(page)


class TestStyleTables:
    def test_table_tag_is_styled(self):
        html = style_tables("<table>\n<tr><td>x</td></tr>\n</table>")
        assert _TABLE_TAG in html

    def test_bare_cells_get_borders_and_padding(self):
        html = style_tables("<table><tr><td>a</td><th>h</th></tr></table>")
        assert _TD_LEFT in html
        assert _TH_LEFT in html

    def test_header_gets_background_and_bold(self):
        html = style_tables("<th>h</th>")
        assert "background-color: #f5f5f5" in html
        assert "font-weight: bold" in html

    def test_alignment_attribute_becomes_inline_style(self):
        html = style_tables('<td align="center">a</td><th align="right">h</th>')
        assert 'text-align: center;">a</td>' in html
        assert (
            'text-align: right; background-color: #f5f5f5; font-weight: bold;">h</th>'
            in html
        )

    def test_styled_cells_are_untouched(self):
        """Excel-rendered cells always carry a style attribute — leave them."""
        cell = '<td style="padding: 12px 16px; border: 1px solid #ddd;">a</td>'
        assert style_tables(cell) == cell

    def test_styled_table_is_untouched(self):
        """Excel-rendered tables carry their own style — leave them."""
        table = '<table style="border-collapse: collapse; width: 100%;">'
        assert style_tables(table) == table


class TestPipelineTable:
    def test_markdown_table_is_fully_styled(self, tmp_path):
        html = _render(tmp_path, _TABLE)
        assert _TABLE_TAG in html
        assert html.count(_TD_LEFT) == 10  # 2 rows x 5 columns
        assert html.count(_TH_LEFT) == 5

    def test_empty_cells_keep_their_borders(self, tmp_path):
        html = _render(tmp_path, _TABLE)
        assert _TD_LEFT + "</td>" in html

    def test_center_alignment_from_separator(self, tmp_path):
        content = _TABLE.replace("|---|---|", "|:---:|---|", 1)
        html = _render(tmp_path, content)
        assert 'text-align: center;">activity</th>' in html

    def test_italic_cell_content_preserved(self, tmp_path):
        html = _render(tmp_path, _TABLE)
        assert "<em>(not given)</em>" in html
