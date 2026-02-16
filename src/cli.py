from typing import Annotated
import typer

import upload_all_pages_to_canvas

# from upload_all_pages_to_canvas import process_all_pages
# from upload_modules_to_canvas import process_modules

app = typer.Typer(help="Uploads mkdocs to canvas")


@app.command()
def upload_pages(
    force: Annotated[bool, typer.Option(help="Force upload (ignores cache).")] = False,
):
    if force:
        typer.echo("Forcing upload...")
    upload_all_pages_to_canvas.process_all_pages()


if __name__ == "__main__":
    app()
