import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .api import create_client
from .api.canvas import CanvasUploader
from .uploaders.base import rebuild_cache_from_canvas
from .uploaders.labs import delete_all_labs, upload_all_labs
from .uploaders.modules import delete_all_modules, upload_all_modules
from .uploaders.pages import delete_all_pages, parse_upload_all_pages

app = typer.Typer(
    pretty_exceptions_short=False,
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()


@app.callback()
def main(ctx: typer.Context):
    """Uploads a mkdocs project to canvas."""
    ctx.ensure_object(dict)


def get_client(ctx: typer.Context) -> CanvasUploader:
    """Create the Canvas client lazily, when a command actually needs it.

    Creating it in the callback would require credentials even for
    `--help`, `--version` and other non-uploading invocations — which
    breaks e.g. inspecting the CLI in CI before secrets are configured.
    """
    client = ctx.obj.get("client")
    if client is None:
        client = create_client()
        ctx.obj["client"] = client
    return client


@app.command()
def serve(
    host: Annotated[
        str, typer.Option("--host", help="Host address to bind to.")
    ] = "0.0.0.0",
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port to listen on.")
    ] = 8000,
    all_plugins: Annotated[
        bool,
        typer.Option(
            "--all-plugins",
            help="Run with all plugins enabled (slower reloads, production-accurate).",
        ),
    ] = False,
):
    """
    Serve the MkDocs site locally.

    Default: fast mode — uses --watch-theme and --livereload, skips heavy plugins.
    Use --all-plugins for a complete build on every reload (e.g. to verify PDF output).
    """
    addr = f"{host}:{port}"
    cmd = [sys.executable, "-m", "mkdocs", "serve", "-a", addr]
    if not all_plugins:
        cmd += ["--watch-theme", "--livereload"]
    mode = "all plugins" if all_plugins else "fast"
    console.print(
        f"[bold]Serving docs ({mode} mode) at http://{addr}[/bold]  (Ctrl+C to stop)"
    )
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\nServer stopped.")
    except subprocess.CalledProcessError as e:
        raise typer.Exit(e.returncode) from e


@app.command()
def pdf(
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o", help="Directory to copy generated PDFs into."
        ),
    ] = Path("pdf"),
    install_browser: Annotated[
        bool,
        typer.Option(
            "--install-browser",
            help="Run 'playwright install chromium' before building (needed on first run or in fresh environments).",
        ),
    ] = False,
):
    """
    Build the MkDocs site and collect generated PDFs into a directory.

    PDF generation requires a Playwright browser. If you see 'Browser not available'
    warnings and no PDFs are produced, run once with --install-browser.
    """
    if install_browser:
        console.print("[bold]Installing Playwright browser...[/bold]")
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"]
        )
        if result.returncode != 0:
            console.print("[red]Playwright browser installation failed.[/red]")
            raise typer.Exit(result.returncode)
        console.print("[green]✓ Browser installed.[/green]")

    console.print("[bold]Building MkDocs site...[/bold]")
    result = subprocess.run([sys.executable, "-m", "mkdocs", "build", "--clean"])
    if result.returncode != 0:
        console.print("[red]mkdocs build failed.[/red]")
        raise typer.Exit(result.returncode)

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(Path("site").rglob("*.pdf"))
    if not pdf_files:
        console.print(
            "[yellow]No PDF files found in site/. "
            "If you see 'Browser not available' above, re-run with --install-browser.[/yellow]"
        )
    else:
        for pdf_file in pdf_files:
            dest = output_dir / pdf_file.name
            shutil.copy2(pdf_file, dest)
        console.print(
            f"[green]✓ {len(pdf_files)} PDF(s) copied to {output_dir}/[/green]"
        )

    console.print(
        f"[bold green]✓ PDFs generated in /{output_dir} directory[/bold green]"
    )


@app.command()
def upload_all(
    ctx: typer.Context,
    add_pdf: Annotated[
        bool, typer.Option(help="Add corresponding page pdf:s to the modules.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose output.")
    ] = False,
):
    """
    Uploads EVERYTHING: Pages, Modules, and Labs.

    Uses a two-pass strategy to resolve cross-links between pages and labs:
    1. Labs uploaded first  → lab URLs enter the cache.
    2. Pages uploaded       → page links to labs resolve; page URLs enter the cache.
    3. Labs re-uploaded     → page links inside labs now resolve.
    """
    if verbose:
        console.print("[bold]Verbose mode enabled.[/bold]")
    console.print("Starting full upload sequence...")

    console.print("[dim]Pass 1: uploading labs to seed cache...[/dim]")
    upload_labs(ctx, verbose=verbose)

    console.print("[dim]Pass 2: uploading pages...[/dim]")
    upload_pages(ctx, verbose=verbose)

    console.print("[dim]Pass 3: re-uploading labs to resolve page links...[/dim]")
    upload_labs(ctx, verbose=verbose)

    upload_modules(ctx, add_pdf=add_pdf, verbose=verbose)

    console.print("All uploads finished!")


@app.command()
def upload_pages(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force upload (ignores cache).")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose output.")
    ] = False,
):
    """
    Uploads pages.
    """
    client: CanvasUploader = get_client(ctx)
    if force:
        typer.echo("Forcing upload. Ignoring cache.")
    parse_upload_all_pages(client, force=force, verbose=verbose)


@app.command()
def upload_modules(
    ctx: typer.Context,
    add_pdf: Annotated[
        bool, typer.Option(help="Make and add corresponding page pdf:s to the modules.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose output.")
    ] = False,
):
    """
    Uploads modules.
    """
    if add_pdf:
        pdf()

    upload_all_modules(get_client(ctx), add_pdf=add_pdf, verbose=verbose)


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
    upload_all_labs(get_client(ctx), force=force, verbose=verbose)


@app.command()
def rebuild_cache(
    ctx: typer.Context,
    docs_root: Annotated[
        Path, typer.Option("--docs-root", help="Path to the docs/ directory.")
    ] = Path("docs"),
):
    """
    Rebuild the local upload cache from Canvas state.

    Use this when .canvas_upload_state.json is missing or deleted (new
    machine, cleaned repo). Pages, labs and the syllabus are matched back
    via their ownership markers and assets via content hashes, so nothing
    is re-uploaded and no duplicates are created.
    """
    client: CanvasUploader = get_client(ctx)
    rebuild_cache_from_canvas(
        client, Path(".canvas_upload_state.json"), docs_root=docs_root
    )
    console.print("[bold green]Cache rebuilt.[/bold green]")


@app.command()
def delete_all(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip all confirmation prompts.")
    ] = False,
    all_items: Annotated[
        bool,
        typer.Option(
            "--all", help="Delete ALL content, including anything not uploaded by mkdocstocanvas."
        ),
    ] = False,
):
    """
    Deletes EVERYTHING the tool has uploaded: Pages, Modules, and Labs.

    Content created manually in Canvas (quizzes, pages, modules) is kept
    unless --all is given.
    """
    console.print("Deleting all managed content...")
    delete_pages(ctx, force=force, all_items=all_items)
    delete_modules(ctx, force=force, all_items=all_items)
    delete_labs(ctx, force=force)
    console.print("All content deleted!")


@app.command()
def delete_pages(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation prompt.")
    ] = False,
    all_items: Annotated[
        bool,
        typer.Option(
            "--all", help="Delete ALL pages, including ones not uploaded by mkdocstocanvas."
        ),
    ] = False,
):
    """Deletes the pages uploaded by mkdocstocanvas (or ALL pages with --all)."""
    delete_all_pages(get_client(ctx), force=force, all_items=all_items)


@app.command()
def delete_modules(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation prompt.")
    ] = False,
    all_items: Annotated[
        bool,
        typer.Option(
            "--all", help="Delete ALL modules, including ones not uploaded by mkdocstocanvas."
        ),
    ] = False,
):
    """Deletes the modules uploaded by mkdocstocanvas (or ALL modules with --all)."""
    delete_all_modules(get_client(ctx), force=force, all_items=all_items)


@app.command()
def delete_labs(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation prompt.")
    ] = False,
):
    """Deletes all lab assignments."""
    delete_all_labs(get_client(ctx), force=force)


if __name__ == "__main__":
    app()
