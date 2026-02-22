import os
from dotenv import load_dotenv
from .canvas import CanvasUploader


def create_client() -> CanvasUploader:
    """Create a CanvasUploader from environment variables."""
    load_dotenv()
    api_token = os.environ.get("CANVAS_API_TOKEN")
    base_url = os.environ.get("CANVAS_BASE_URL", "https://canvas.instructure.com")
    course_id = os.environ.get("CANVAS_COURSE_ID")

    if not api_token or not course_id:
        raise ValueError(
            "❌ Missing required Canvas configuration! "
            "Please ensure CANVAS_API_TOKEN and CANVAS_COURSE_ID are set in your .env file."
        )

    return CanvasUploader(base_url=base_url, api_token=api_token, course_id=course_id)
