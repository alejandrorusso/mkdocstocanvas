import re
import markdown
from pathlib import Path
from typing import Dict, List, Optional

from ..models.page import MarkdownPage
from .math import robust_math_protection, restore_protected_math_content
from .excel import process_excel_macros


def process_markdown_to_html(md_page: MarkdownPage) -> str:
    """
    Complete processing pipeline to convert markdown to Canvas-compatible HTML.

    Args:
        md_page: MarkdownPage instance (used for path context)
        all_pages_info: List of all Canvas pages for internal link resolution
        content: Pre-processed markdown content. Falls back to md_page.content if omitted.

    Returns:
        Processed HTML content ready for Canvas
    """
    # Step 0: Process Excel macros (before any other processing)
    content = process_excel_macros(md_page.content, md_page.path)

    # Step 1a: Handle standalone align environments that need $$ wrapping
    def wrap_standalone_align(content):
        """Wrap standalone align environments with $$ delimiters."""
        result = []
        i = 0
        length = len(content)

        while i < length:
            if content[i : i + 13] == "\\begin{align}":
                j = i - 1
                found_dollar_signs = False
                while j >= 1:
                    if content[j - 1 : j + 1] == "$$":
                        found_dollar_signs = True
                        break
                    if content[j] not in " \t\n":
                        break
                    j -= 1

                if not found_dollar_signs:
                    align_start = i
                    k = i + 13
                    align_depth = 1

                    while k < length and align_depth > 0:
                        if content[k : k + 13] == "\\begin{align}":
                            align_depth += 1
                            k += 13
                        elif content[k : k + 11] == "\\end{align}":
                            align_depth -= 1
                            if align_depth == 0:
                                k += 11
                                break
                            else:
                                k += 11
                        else:
                            k += 1

                    if align_depth == 0:
                        align_content = content[align_start:k]
                        result.append(f"$${align_content}$$")
                        i = k
                    else:
                        result.append(content[i])
                        i += 1
                else:
                    result.append(content[i])
                    i += 1
            else:
                result.append(content[i])
                i += 1

        return "".join(result)

    content = wrap_standalone_align(content)

    # Step 1b: Use robust character-by-character math protection
    content = robust_math_protection(content)

    # Step 2: Convert markdown to HTML (math is already protected)
    md = markdown.Markdown(
        extensions=[
            "extra",
            "codehilite",
            "toc",
            "sane_lists",
            "admonition",
            "pymdownx.details",
            "pymdownx.superfences",
            "attr_list",
            "def_list",
            "footnotes",
            "meta",
        ],
        extension_configs={
            "codehilite": {
                "css_class": "codehilite",
                "guess_lang": True,
                "linenums": False,
                "use_pygments": True,
            }
        },
    )

    html_content = md.convert(content)

    # Step 3: Restore protected math content
    html_content = restore_protected_math_content(html_content)

    # Step 4: Add Pygments inline styles for Canvas compatibility
    html_content = add_pygments_inline_styles(html_content)

    # Step 5: Style code blocks and admonitions
    html_content = style_code_blocks(html_content)
    html_content = style_admonitions(html_content)

    return html_content


def add_pygments_inline_styles(html_content: str) -> str:
    """Add inline styles for Pygments syntax highlighting (GitHub light theme)"""
    pygments_styles = {
        'class="c"': 'class="c" style="color: #6a737d; font-style: italic;"',
        'class="k"': 'class="k" style="color: #d73a49; font-weight: 600;"',
        'class="n"': 'class="n" style="color: #24292e;"',
        'class="o"': 'class="o" style="color: #d73a49;"',
        'class="p"': 'class="p" style="color: #24292e;"',
        'class="cm"': 'class="cm" style="color: #6a737d; font-style: italic;"',
        'class="c1"': 'class="c1" style="color: #6a737d; font-style: italic;"',
        'class="kc"': 'class="kc" style="color: #005cc5;"',
        'class="kd"': 'class="kd" style="color: #d73a49; font-weight: 600;"',
        'class="kn"': 'class="kn" style="color: #d73a49; font-weight: 600;"',
        'class="kp"': 'class="kp" style="color: #d73a49; font-weight: 600;"',
        'class="kr"': 'class="kr" style="color: #d73a49; font-weight: 600;"',
        'class="kt"': 'class="kt" style="color: #005cc5; font-weight: 600;"',
        'class="m"': 'class="m" style="color: #005cc5;"',
        'class="s"': 'class="s" style="color: #032f62;"',
        'class="na"': 'class="na" style="color: #22863a;"',
        'class="nb"': 'class="nb" style="color: #005cc5;"',
        'class="nc"': 'class="nc" style="color: #6f42c1; font-weight: 600;"',
        'class="no"': 'class="no" style="color: #005cc5;"',
        'class="nd"': 'class="nd" style="color: #6f42c1;"',
        'class="nf"': 'class="nf" style="color: #6f42c1; font-weight: 600;"',
        'class="nn"': 'class="nn" style="color: #24292e;"',
        'class="nt"': 'class="nt" style="color: #22863a;"',
        'class="nv"': 'class="nv" style="color: #e36209;"',
        'class="ow"': 'class="ow" style="color: #d73a49; font-weight: 600;"',
        'class="w"': 'class="w" style="color: #24292e;"',
        'class="mf"': 'class="mf" style="color: #005cc5;"',
        'class="mh"': 'class="mh" style="color: #005cc5;"',
        'class="mi"': 'class="mi" style="color: #005cc5;"',
        'class="mo"': 'class="mo" style="color: #005cc5;"',
        'class="sb"': 'class="sb" style="color: #032f62;"',
        'class="sc"': 'class="sc" style="color: #032f62;"',
        'class="sd"': 'class="sd" style="color: #032f62;"',
        'class="s2"': 'class="s2" style="color: #032f62;"',
        'class="se"': 'class="se" style="color: #005cc5;"',
        'class="sh"': 'class="sh" style="color: #032f62;"',
        'class="si"': 'class="si" style="color: #005cc5;"',
        'class="sx"': 'class="sx" style="color: #032f62;"',
        'class="sr"': 'class="sr" style="color: #032f62;"',
        'class="s1"': 'class="s1" style="color: #032f62;"',
        'class="ss"': 'class="ss" style="color: #032f62;"',
    }

    for old, new in pygments_styles.items():
        html_content = html_content.replace(old, new)

    return html_content


def style_code_blocks(html_content: str) -> str:
    """Add inline styles to code blocks for Canvas compatibility"""
    html_content = html_content.replace(
        '<div class="codehilite">',
        '<div class="codehilite" style="background-color: #f6f8fa; padding: 16px; border-radius: 6px; margin: 1.5em 0; border: 1px solid #e1e4e8;">',
    )

    html_content = html_content.replace(
        "<pre>",
        '<pre style="background-color: #f6f8fa; margin: 0; overflow-x: auto; color: #24292e; line-height: 1.5;">',
    )

    html_content = html_content.replace(
        "<code>",
        "<code style=\"background-color: transparent; font-family: 'SFMono-Regular', 'Consolas', 'Liberation Mono', 'Menlo', monospace; font-size: 0.9em; color: #24292e;\">",
    )

    return html_content


def style_admonitions(html_content: str) -> str:
    """Add inline styles for admonitions (Note, Warning, etc.)"""
    admonition_styles = {
        "note": "background-color: #e7f2fa; border-left: 4px solid #2196F3; color: #014361;",
        "warning": "background-color: #fff4e5; border-left: 4px solid #ff9800; color: #663c00;",
        "important": "background-color: #ffe5e5; border-left: 4px solid #f44336; color: #5f2120;",
        "tip": "background-color: #e8f5e9; border-left: 4px solid #4caf50; color: #1b5e20;",
        "danger": "background-color: #ffebee; border-left: 4px solid #f44336; color: #5f2120;",
        "info": "background-color: #e1f5fe; border-left: 4px solid #03a9f4; color: #01579b;",
    }

    for admonition_type, style in admonition_styles.items():
        html_content = html_content.replace(
            f'<div class="admonition {admonition_type}">',
            f'<div class="admonition {admonition_type}" style="padding: 15px 20px; margin: 1.5em 0; border-radius: 4px; {style}">',
        )
        html_content = html_content.replace(
            f'<p class="admonition-title">',
            f'<p class="admonition-title" style="font-weight: 600; margin: 0 0 10px 0; font-size: 1.1em;">',
        )

    return html_content
