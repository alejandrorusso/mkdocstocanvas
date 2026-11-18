import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .canvas import CanvasUploader


def create_client(env_file: str | Path | None = None) -> CanvasUploader:
    """Create a CanvasUploader from environment variables.

    When ``env_file`` is given (e.g. from ``--env-file``), that file is loaded
    and its values take precedence over variables already in the environment,
    so an explicitly chosen credentials file always wins. When it is ``None``
    the previous behaviour is kept unchanged.
    """
    if env_file is not None:
        env_path = Path(env_file)
        if not env_path.is_file():
            raise ValueError(f"❌ --env-file not found: {env_path}")
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        # When installed as a tool (e.g. via `uv tool install`), the package lives in an
        # isolated environment. `load_dotenv()` without a path starts searching from the
        # caller file location (site-packages), which can miss the user's project `.env`.
        # Prefer the current working directory and its parents.
        cwd_env = Path.cwd() / ".env"
        if cwd_env.is_file():
            load_dotenv(dotenv_path=cwd_env)
        else:
            env_path = find_dotenv(usecwd=True)
            if env_path:
                load_dotenv(dotenv_path=env_path)

    api_token = os.environ.get("CANVAS_API_TOKEN")
    base_url = os.environ.get("CANVAS_BASE_URL", "https://canvas.instructure.com")
    course_id = os.environ.get("CANVAS_COURSE_ID")

    if not api_token or not course_id:
        raise ValueError(
            "❌ Missing required Canvas configuration! "
            "Please ensure CANVAS_API_TOKEN and CANVAS_COURSE_ID are set in your .env file."
        )

    return CanvasUploader(base_url=base_url, api_token=api_token, course_id=course_id)
