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


class TestBackslashDelimiters:
    """\\( \\) and \\[ \\] are valid arithmatex delimiters and must survive."""

    def test_inline_paren_roundtrip(self):
        assert roundtrip(r"where \( \pi_{Low} \) is") == r"where \( \pi_{Low} \) is"

    def test_display_bracket_becomes_dollars(self):
        md = "\\[\n\\forall h_1, h_2 \\in H\n\\]"
        restored = roundtrip(md)
        assert restored.startswith("$$") and restored.endswith("$$")
        assert r"\forall h_1, h_2 \in H" in restored

    def test_escaped_backslash_is_not_math(self):
        # "\\(" in markdown is a literal backslash followed by "("
        assert roundtrip(r"a \\(b\) c") == r"a \\(b\) c"

    def test_unclosed_left_alone(self):
        assert roundtrip(r"see \[1 for details") == r"see \[1 for details"

    def test_full_pipeline_in_admonition(self):
        from pathlib import Path

        from mkdocstocanvas.processing.markdown import process_markdown_to_html

        class _Page:
            path = Path("page.md")
            content = (
                "!!! Note\n"
                "    **Non-interference**:\n\n"
                "    \\[\n"
                "    \\forall h_1, h_2 \\in H, \\; \\forall l \\in L, \\quad\n"
                "    \\pi_{Low}(P(h_1, l)) = \\pi_{Low}(P(h_2, l))\n"
                "    \\]\n\n"
                "    where \\( \\pi_{Low} \\) is the projection.\n"
            )

        html = process_markdown_to_html(_Page())  # type: ignore[arg-type]
        assert r"\pi_{Low}(P(h_1, l)) = \pi_{Low}(P(h_2, l))" in html
        assert "$$" in html
        assert r"\( \pi_{Low} \)" in html
        assert "<em>" not in html


class TestCodeIsNotMath:
    """$ and \\( in code are code, not math delimiters."""

    def test_inline_code_dollars_untouched(self):
        md = "The operator `$` is application: `f $ x = f x`."
        assert roundtrip(md) == md

    def test_text_between_code_spans_untouched(self):
        md = "Use `$` here and `$` there, with *emphasis* between."
        assert roundtrip(md) == md

    def test_double_backtick_span(self):
        md = "Write ``a ` $b$ `` literally."
        assert roundtrip(md) == md

    def test_unmatched_backtick_is_text(self):
        assert roundtrip("a ` b $x$") == r"a ` b \(x\)"

    def test_fenced_code_dollars_untouched(self):
        md = "```haskell\nmain = print $ sum $ map f xs\n```\n\nAfter $y$."
        assert (
            roundtrip(md)
            == "```haskell\nmain = print $ sum $ map f xs\n```\n\nAfter \\(y\\)."
        )

    def test_tilde_fence_and_unclosed_fence(self):
        assert roundtrip("~~~\n$a$\n~~~\n") == "~~~\n$a$\n~~~\n"
        assert roundtrip("```\n$a$ and $b$") == "```\n$a$ and $b$"

    def test_backslash_paren_in_code_untouched(self):
        md = 'Write `printf("\\(%d\\)")` or `\\[x\\]`.'
        assert roundtrip(md) == md

    def test_math_still_works_around_code(self):
        assert roundtrip("$x$ and `$` and $y$") == r"\(x\) and `$` and \(y\)"
        assert roundtrip(r"\(x\) and `$` and \(y\)") == r"\(x\) and `$` and \(y\)"

    def test_escaped_dollar_is_literal(self):
        assert roundtrip(r"## The \$ operator") == "## The $ operator"
        assert roundtrip(r"costs \$5 and \$10") == "costs $5 and $10"

    def test_full_pipeline_haskell(self):
        from pathlib import Path

        from mkdocstocanvas.processing.markdown import process_markdown_to_html

        class _Page:
            path = Path("page.md")
            content = (
                "Use `<$>` for `fmap` and `$` for application.\n\n"
                "```haskell\n"
                "main = print $ length <$> Just [1, 2, 3]\n"
                "```\n\n"
                "Here *emphasis* is still emphasis, and \\( x_1 \\) is math.\n"
            )

        html = process_markdown_to_html(_Page())  # type: ignore[arg-type]
        assert "\\(" in html and "x_1" in html
        assert "<em>emphasis</em>" in html
        assert "MATH" not in html
        assert html.count("$") == 4
