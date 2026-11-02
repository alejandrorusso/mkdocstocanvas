"""Tests for LaTeX protection and restoration (processing/math.py)."""

from mkdocstocanvas.processing.math import (
    convert_align_to_array,
    restore_protected_math_content,
    robust_math_protection,
)


def roundtrip(md: str) -> str:
    return restore_protected_math_content(robust_math_protection(md))


class TestConvertAlignToArray:
    def test_align_star(self):
        assert (
            convert_align_to_array(r"\begin{align*} x \end{align*}")
            == r"\begin{array}{rl} x \end{array}"
        )

    def test_align(self):
        assert (
            convert_align_to_array(r"\begin{align} x \end{align}")
            == r"\begin{array}{rl} x \end{array}"
        )

    def test_no_align_unchanged(self):
        assert convert_align_to_array("plain text") == "plain text"


class TestInlineMath:
    def test_roundtrip(self):
        assert roundtrip("$a_1 + b$") == r"\(a_1 + b\)"

    def test_underscores_survive_markdown(self):
        # _ would otherwise be interpreted as markdown emphasis
        assert roundtrip("$x_i y_j$") == r"\(x_i y_j\)"

    def test_asterisks_survive_markdown(self):
        assert roundtrip("$a * b$") == r"\(a * b\)"

    def test_no_closing_dollar_left_alone(self):
        # A single unmatched $ is left as-is
        assert roundtrip("price is $5") == "price is $5"

    def test_two_dollars_on_one_line_treated_as_math(self):
        # Known limitation: "$5 and $10" looks like inline math delimiters
        # to the scanner and is protected as math.
        assert roundtrip("cost is $5 and $10") == r"cost is \(5 and \)10"


class TestDisplayMath:
    def test_roundtrip(self):
        assert roundtrip("$$x + y$$") == "$$x + y$$"

    def test_linebreak_preserved(self):
        # A LaTeX \\ line break survives as a line break (the raw newline is
        # folded into the " \\\\ " replacement, which is valid LaTeX).
        md = "$$a \\\\\nb$$"
        restored = roundtrip(md)
        assert restored.startswith("$$") and restored.endswith("$$")
        assert "\\\\" in restored.replace("§", "")

    def test_align_converted_to_array(self):
        restored = roundtrip("$$\\begin{align*}\nx &= 1\n\\end{align*}$$")
        assert r"\begin{array}{rl}" in restored
        assert r"\begin{align*}" not in restored

    def test_unclosed_display_left_alone(self):
        assert roundtrip("$$ never closed") == "$$ never closed"


class TestMixedContent:
    def test_inline_and_display(self):
        restored = roundtrip("Text $x_2$ and $$y_2$$ here")
        assert restored == r"Text \(x_2\) and $$y_2$$ here"

    def test_text_outside_math_untouched(self):
        md = "a_b *c* ~d~ $e_f$"
        restored = roundtrip(md)
        assert restored.startswith("a_b *c* ~d~ ")
        assert restored.endswith(r"\(e_f\)")
