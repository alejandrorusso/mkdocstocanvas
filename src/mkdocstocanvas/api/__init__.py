import os
from dotenv import load_dotenv
from .canvas import CanvasUploader

# 1. Load the environment variables from .env immediately
load_dotenv()

# 2. Fetch the variables (using os.environ ensures the script crashes
#    loudly and early if you forgot to set one in your .env file)
API_TOKEN = os.environ.get("CANVAS_API_TOKEN")
BASE_URL = os.environ.get("CANVAS_BASE_URL", "https://canvas.instructure.com")
COURSE_ID = os.environ.get("CANVAS_COURSE_ID")

# if not API_TOKEN:
#     print("Error: No CANVAS_API_TOKEN")
#     sys.exit(1)

if not API_TOKEN or not COURSE_ID:
    raise ValueError(
        "❌ Missing required Canvas configuration! "
        "Please ensure CANVAS_API_TOKEN and CANVAS_COURSE_ID are set in your .env file."
    )

# 3. Create the single, global instance of your uploader
canvas_client = CanvasUploader(
    base_url=BASE_URL, api_token=API_TOKEN, course_id=COURSE_ID
)
