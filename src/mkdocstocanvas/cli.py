from typing import Annotated
import typer
from rich.console import Console

from .api import create_client
from .api.canvas import CanvasUploader
from .uploaders.pages import parse_upload_all_pages, delete_all_pages
from .uploaders.modules import upload_all_modules, delete_all_modules
from .uploaders.labs import upload_all_labs, delete_all_labs

app = typer.Typer(
    pretty_exceptions_short=False,
    pretty_exceptions_show_locals=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()


@app.callback()
def main(ctx: typer.Context):
    """Uploads a mkdocs project to canvas."""
    ctx.ensure_object(dict)
    ctx.obj["client"] = create_client()


@app.command()
def upload_all(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option(help="Force upload (ignores cache) for pages.")
    ] = False,
    add_pdf: Annotated[
        bool, typer.Option(help="Add corresponding page pdf:s to the modules.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose output.")
    ] = False,
):
    """
    Uploads EVERYTHING: Pages, Modules, and Labs.
    """
    if verbose:
        console.print("[bold]Verbose mode enabled.[/bold]")
    console.print("Starting full upload sequence...")

    upload_pages(ctx, force=force, verbose=verbose)
    upload_modules(ctx, add_pdf=add_pdf, verbose=verbose)
    upload_labs(ctx, verbose=verbose)

    console.print("All uploads finished!")


@app.command()
def upload_pages(
    ctx: typer.Context,
    force: Annotated[bool, typer.Option(help="Force upload (ignores cache).")] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose output.")
    ] = False,
):
    """
    Uploads pages.
    """
    client: CanvasUploader = ctx.obj["client"]
    if force:
        typer.echo("Forcing upload. Ignoring cache.")
    parse_upload_all_pages(client, force=force, verbose=verbose)


@app.command()
def upload_modules(
    ctx: typer.Context,
    add_pdf: Annotated[
        bool, typer.Option(help="Add corresponding page pdf:s to the modules.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose output.")
    ] = False,
):
    """
    Uploads modules.
    """
    upload_all_modules(ctx.obj["client"], verbose=verbose)


@app.command()
def upload_labs(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Re-upload even if assets are cached.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose output.")
    ] = False,
):
    """
    Uploads labs as Canvas assignments.
    """
    upload_all_labs(ctx.obj["client"], force=force, verbose=verbose)


@app.command()
def delete_all(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip all confirmation prompts.")
    ] = False,
):
    """
    Deletes EVERYTHING: Pages, Modules, and Labs.
    """
    console.print("Deleting all content...")
    delete_pages(ctx, force=force)
    delete_modules(ctx, force=force)
    delete_labs(ctx, force=force)
    console.print("All content deleted!")


@app.command()
def delete_pages(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation prompt.")
    ] = False,
):
    """Deletes ALL pages from the Canvas course."""
    delete_all_pages(ctx.obj["client"], force=force)


@app.command()
def delete_modules(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation prompt.")
    ] = False,
):
    """Deletes all modules."""
    delete_all_modules(ctx.obj["client"], force=force)


@app.command()
def delete_labs(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation prompt.")
    ] = False,
):
    """Deletes all lab assignments."""
    delete_all_labs(ctx.obj["client"], force=force)


if __name__ == "__main__":
    app()
