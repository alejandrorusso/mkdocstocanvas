import re


def convert_align_to_array(content: str) -> str:
    """Convert LaTeX align environments to Canvas-compatible array format."""
    content = re.sub(r"\\begin\{align\*?\}", r"\\begin{array}{rl}", content)
    content = re.sub(r"\\end\{align\*?\}", r"\\end{array}", content)
    return content


def _protect_display(math_content: str) -> str:
    """Shield display-math content from markdown and wrap it in sentinels."""
    math_content = convert_align_to_array(math_content)
    math_content = math_content.replace("\\\\", "§LATEXLINEBREAK§")
    math_content = math_content.replace("\\", "\\\\")
    math_content = math_content.replace("_", "§UNDERSCORE§")
    math_content = math_content.replace("*", "§ASTERISK§")
    math_content = math_content.replace("~", "§TILDE§")
    return f"§MATH_DISPLAY_START§{math_content}§MATH_DISPLAY_END§"


def _protect_inline(math_content: str) -> str:
    """Shield inline-math content from markdown and wrap it in sentinels."""
    math_content = math_content.replace("\\", "\\\\")
    math_content = math_content.replace("_", "§UNDERSCORE§")
    math_content = math_content.replace("*", "§ASTERISK§")
    math_content = math_content.replace("~", "§TILDE§")
    return f"§MATH_INLINE_START§{math_content}§MATH_INLINE_END§"


def robust_math_protection(content: str) -> str:
    """
    Robust math protection using character-by-character parsing.
    This avoids complex regex issues with multiline math content.

    Recognised delimiters (the same set pymdownx.arithmatex accepts):
    ``$$...$$`` and ``\\[...\\]`` for display math, ``$...$`` and
    ``\\(...\\)`` for inline math. Without protection, Python-Markdown
    treats ``\\(`` / ``\\[`` as escaped characters and drops the backslash,
    so the math would reach Canvas as plain text.
    """
    result = []
    i = 0
    length = len(content)

    while i < length:
        # Look for $$ (display math)
        if content.startswith("$$", i):
            j = content.find("$$", i + 2)
            if j == -1:
                result.append(content[i])
                i += 1
            else:
                result.append(_protect_display(content[i + 2 : j]))
                i = j + 2

        # Look for \[ ... \] or \( ... \) (not when the backslash is itself
        # escaped, e.g. a literal "\\(" in markdown).
        elif (
            content[i] == "\\"
            and i + 1 < length
            and content[i + 1] in "[("
            and (i == 0 or content[i - 1] != "\\")
        ):
            closing = "\\]" if content[i + 1] == "[" else "\\)"
            j = content.find(closing, i + 2)
            if j == -1:
                result.append(content[i])
                i += 1
            elif closing == "\\]":
                result.append(_protect_display(content[i + 2 : j]))
                i = j + 2
            else:
                result.append(_protect_inline(content[i + 2 : j]))
                i = j + 2

        # Look for single $ (inline math) - but not if it's part of $$
        elif (
            content[i] == "$"
            and (i == 0 or content[i - 1] != "$")
            and (i == length - 1 or content[i + 1] != "$")
        ):
            # Find the closing $
            j = i + 1
            while j < length and content[j] != "$":
                # Skip escaped dollars
                if content[j] == "\\" and j + 1 < length:
                    j += 2
                else:
                    j += 1

            if j < length:
                result.append(_protect_inline(content[i + 1 : j]))
                i = j + 1
            else:
                # No closing $ found, treat as regular character
                result.append(content[i])
                i += 1
        else:
            result.append(content[i])
            i += 1

    return "".join(result)


def restore_protected_math_content(content: str) -> str:
    """Restore protected math content with Canvas-compatible delimiters."""
    content = content.replace("§MATH_INLINE_START§", r"\(")
    content = content.replace("§MATH_INLINE_END§", r"\)")
    content = content.replace("§MATH_DISPLAY_START§", "$$")
    content = content.replace("§MATH_DISPLAY_END§", "$$")

    def fix_math_backslashes(match):
        math_content = match.group(0)
        math_content = math_content.replace("\\\\", "\\")
        math_content = re.sub(r"\s*§LATEXLINEBREAK§\s*", r" \\\\ ", math_content)
        math_content = math_content.replace("§ASTERISK§", "*")
        math_content = math_content.replace("§UNDERSCORE§", "_")
        math_content = math_content.replace("§TILDE§", "~")
        return math_content

    content = re.sub(r"\$\$.*?\$\$", fix_math_backslashes, content, flags=re.DOTALL)
    content = re.sub(r"\\\(.*?\\\)", fix_math_backslashes, content, flags=re.DOTALL)

    return content
