import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from .canvas import CanvasUploader


def create_client() -> CanvasUploader:
    """Create a CanvasUploader from environment variables."""
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
