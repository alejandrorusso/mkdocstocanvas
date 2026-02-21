from typing import Annotated
import typer
from rich.console import Console

import upload_all_pages_to_canvas
import upload_modules_to_canvas
import upload_labs_to_canvas

from uploaders.pages import parse_upload_all_pages

app = typer.Typer(
    help="Uploads mkdocs to canvas",
    pretty_exceptions_short=False,
    pretty_exceptions_show_locals=True,
)
console = Console()
err_console = Console(stderr=True, style="bold red")


@app.command()
def upload_all(
    force: Annotated[
        bool, typer.Option(help="Force upload (ignores cache) for pages.")
    ] = False,
    add_pdf: Annotated[
        bool, typer.Option(help="Add corresponding page pdf:s to the modules.")
    ] = False,
):
    """
    Uploads EVERYTHING: Pages, Modules, and Labs.
    """
    console.print("Starting full upload sequence...")

    # 2. Call the other commands directly as normal functions
    # We pass the arguments we received manually
    upload_pages(force=force)
    upload_modules(add_pdf=add_pdf)
    upload_labs()

    console.print("All uploads finished!")


@app.command()
def upload_pages(
    force: Annotated[bool, typer.Option(help="Force upload (ignores cache).")] = False,
):
    """
    Uploads pages.
    """
    if force:
        typer.echo("Forcing upload. Ignoring cache.")
    # upload_all_pages_to_canvas.process_all_pages()
    parse_upload_all_pages()


@app.command()
def upload_modules(
    add_pdf: Annotated[
        bool, typer.Option(help="Add corresponding page pdf:s to the modules.")
    ] = False,
):
    """
    Uploads modules.
    """
    upload_modules_to_canvas.process_modules()


@app.command()
def upload_labs():
    """
    Uploads labs.
    """
    upload_labs_to_canvas.upload_lab_assignments()


@app.command()
def delete_pages(
    force: Annotated[
        bool,
        typer.Option(
            prompt="Are you sure you want to delete ALL pages?",
            help="Force deletion without confirmation.",
        ),
    ],
):
    """
    Deletes all pages.

    Asks for confirmation unless --force is used
    """
    if force:
        success = upload_all_pages_to_canvas.delete_all_pages()
        if success:
            console.print("✓ All pages deleted successfully!", style="bold green")
            raise typer.Exit()
        else:
            err_console.print("✗ Page deletion failed")
            raise typer.Exit(code=1)
    else:
        raise typer.Abort()


@app.command()
def delete_modules(
    force: Annotated[
        bool,
        typer.Option(
            prompt="Are you sure you want to delete ALL modules?",
            help="Force deletion without confirmation.",
        ),
    ],
):
    """
    Deletes all modules.

    Asks for confirmation unless --force is used
    """
    if force:
        success = upload_modules_to_canvas.delete_all_modules()
        if success:
            console.print("✓ All modules deleted successfully!", style="bold green")
            raise typer.Exit()
        else:
            err_console.print("✗ Module deletion failed")
            raise typer.Exit(code=1)
    else:
        raise typer.Abort()


@app.command()
def delete_labs(
    force: Annotated[
        bool, typer.Option(help="Force deletion without confirmation.")
    ] = False,
):
    """
    Deletes all labs.

    Asks for confirmation unless --force is used
    """
    assignments = upload_labs_to_canvas.get_all_assignments()
    if assignments:
        lab_assignments = upload_labs_to_canvas.filter_lab_assignments(assignments)

    if not lab_assignments:
        console.print("No lab assignments found to delete")
        return True

    console.print(f"Found {len(lab_assignments)} lab assignment(s) to delete:")
    for assignment in lab_assignments:
        console.print(f"  - {assignment['name']} (ID: {assignment['id']})")

    if not force:
        typer.confirm(
            f"Are you sure you want to delete {len(lab_assignments)} lab assignment(s)?",
            abort=True,
        )

    upload_labs_to_canvas.delete_lab_assignments()


if __name__ == "__main__":
    app()
