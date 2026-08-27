import re


def convert_align_to_array(content: str) -> str:
    """Convert LaTeX align environments to Canvas-compatible array format."""
    content = re.sub(r"\\begin\{align\*?\}", r"\\begin{array}{rl}", content)
    content = re.sub(r"\\end\{align\*?\}", r"\\end{array}", content)
    return content


def robust_math_protection(content: str) -> str:
    """
    Robust math protection using character-by-character parsing.
    This avoids complex regex issues with multiline math content.
    """
    result = []
    i = 0
    length = len(content)

    while i < length:
        # Look for $$ (display math)
        if i < length - 1 and content[i : i + 2] == "$$":
            # Find the closing $$
            j = i + 2
            while j < length - 1:
                if content[j : j + 2] == "$$":
                    # Found closing $$
                    math_content = content[i + 2 : j]

                    # Convert align to array FIRST
                    math_content = convert_align_to_array(math_content)
                    # Preserve line breaks
                    math_content = math_content.replace("\\\\", "§LATEXLINEBREAK§")
                    # Double all remaining backslashes to protect from markdown
                    math_content = math_content.replace("\\", "\\\\")
                    # Protect underscores, asterisks, and tildes from markdown processing
                    math_content = math_content.replace("_", "§UNDERSCORE§")
                    math_content = math_content.replace("*", "§ASTERISK§")
                    math_content = math_content.replace("~", "§TILDE§")

                    result.append(
                        f"§MATH_DISPLAY_START§{math_content}§MATH_DISPLAY_END§"
                    )
                    i = j + 2
                    break
                j += 1
            else:
                # No closing $$ found, treat as regular character
                result.append(content[i])
                i += 1

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
                # Found closing $
                math_content = content[i + 1 : j]

                math_content = math_content.replace("\\", "\\\\")
                math_content = math_content.replace("_", "§UNDERSCORE§")
                math_content = math_content.replace("*", "§ASTERISK§")
                math_content = math_content.replace("~", "§TILDE§")

                result.append(f"§MATH_INLINE_START§{math_content}§MATH_INLINE_END§")
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
