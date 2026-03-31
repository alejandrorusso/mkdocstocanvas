# Canvas Upload Test

This document tests the conversion pipeline from Markdown to Canvas HTML, ensuring all custom processors (Math, Excel, Images, Code) are functioning correctly.

---

## 1. Mathematical Formulas (LaTeX)
*Tests: `protect_math_content`, `restore_math_content`, `convert_align_to_array`*

### Inline Math
The mass-energy equivalence is denoted by $E=mc^2$.
We also test protection of special markdown characters inside math: $x_{i} * y^{2} \sim \mathcal{N}(0,1)$.
*(The `*` and `_` inside the dollars should not italicize the text)*

### Standard Display Math
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

### Align Environment (The Tricky Part)
*Tests: `convert_align_to_array` and `wrap_standalone_align`*

Your script converts `align` blocks into `array` blocks for Canvas compatibility.

$$
\begin{align}
    f(x) &= (x+a)(x+b) \\
    &= x^2 + (a+b)x + ab
\end{align}
$$

### Robustness Test
*Tests: `robust_math_protection`*

This block contains double backslashes and complex nesting to ensure the regex doesn't break.
$$
A = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix} \\
\text{where } \lambda_{1,2} = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

---

## 2. Code Syntax Highlighting
*Tests: `style_code_blocks` and `add_pygments_inline_styles`*

This should render with a gray background, rounded corners, and **inline colors** (no CSS classes) so it looks colorful in Canvas.

```python
import os

def hello_canvas(name: str = "Student") -> None:
    """
    A simple greeting function.
    """
    # This is a comment
    greeting = f"Hello, {name}!"
    print(greeting)
    
    if name == "Canvas":
        return True
    return False

```

```javascript
// Testing another language
const api = {
    upload: (file) => {
        console.log(`Uploading ${file}...`);
    }
};
```

---

## 3. Admonitions & Collapsible Details

*Tests: `style_admonitions`, `pymdownx.details`, `pymdownx.superfences`*

!!! note "Standard Note"
This is a standard note. It should be blue with a bold title.

!!! tip "Tip"
This is a tip. It should be green.

!!! warning "Crucial Warning"
This is a warning. It should be orange/yellow.

!!! danger "Do Not Delete"
This is a danger block. It should be red.

??? tip "Click to Reveal (Collapsible)"
    This content is hidden by default.
    It supports **markdown** inside.

    And even code blocks:
    ```python
    secret_code = "hidden"
    ```

---

## 4. Excel Embedding

*Tests: `process_excel_macros` and `render_excel_sheet_to_html`*

If `data.xlsx` exists, this will be replaced by an HTML table with preserved cell colors.

{{ render_excel_sheet('./data.xlsx', 'Sheet1') }}

---

## 5. Media & Assets

*Tests: `process_images` and `upload_image_to_canvas`*
![Banananana](./banana.jpg)

The script should detect this image, upload it to the Canvas "Files" tab, and replace the path with the new Canvas URL.

---

## 6. Internal Linking

*Tests: `resolve_internal_links`*

* [Link to the other test file](./test2.md) (Should resolve if page exists)
* [Link to some anchor](#robustness-test) (Should handle anchors)
* [Link to some anchor](./test2.md#some-anchor-here) (Should handle anchors in a different page)
* [Link to syllabus](../index.md) (Should correctly link to syllabus)
* [Link to lab](../labs/lab1.md) (Should correctly link to lab)
* [Link to a website](https://nohello.net/en/) (Should link to some external website)
