"""Tests for ownership markers (utils/marker.py)."""

from mkdocstocanvas.utils.marker import make_marker, parse_marker


class TestMakeAndParseMarker:
    def test_roundtrip(self):
        marker = make_marker("lectures/01-intro.md", "ab12cd")
        fields = parse_marker(marker)
        assert fields == {"rel": "lectures/01-intro.md", "md5": "ab12cd"}

    def test_rel_with_spaces(self):
        marker = make_marker("lectures/01 intro.md", "ab12cd")
        fields = parse_marker(marker)
        assert fields is not None
        assert fields["rel"] == "lectures/01 intro.md"

    def test_without_md5(self):
        marker = make_marker("index.md", None)
        assert parse_marker(marker) == {"rel": "index.md"}

    def test_is_an_invisible_empty_span(self):
        marker = make_marker("index.md", None)
        assert marker.startswith('<span class="mkdocstocanvas-marker"')
        assert marker.endswith("></span>")
        assert "<!--" not in marker

    def test_parses_regardless_of_attribute_order(self):
        html = '<p>x</p><span data-md5="aa" class="mkdocstocanvas-marker" data-rel="b.md"></span>'
        assert parse_marker(html) == {"rel": "b.md", "md5": "aa"}

    def test_marker_inside_html_body_found(self):
        marker = make_marker("index.md", "aa")
        html = f"<h1>Hi</h1>\n<p>Stuff</p>\n{marker}\n<p>more</p>"
        fields = parse_marker(html)
        assert fields is not None
        assert fields["rel"] == "index.md"

    def test_other_comments_ignored(self):
        assert parse_marker("<!-- a normal comment -->") is None
        assert parse_marker("<p>no comments</p>") is None
