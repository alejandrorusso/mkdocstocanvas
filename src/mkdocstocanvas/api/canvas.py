import os
import mimetypes
from pathlib import Path
import requests
import re  # Make sure this is at the top of your file

from rich.console import Console

# Initialize the error console at the module level
err_console = Console(stderr=True)


class CanvasUploader:
    def __init__(self, base_url: str, api_token: str, course_id: str | int):
        """Initializes the Canvas API client with a persistent session."""
        self.base_url = base_url.rstrip("/")
        self.course_id = str(course_id)

        # 1. Setup global session for connection pooling
        self.session = requests.Session()

        # 2. Set global headers (Auth and Accept ONLY)
        self.session.headers.update(
            {"Authorization": f"Bearer {api_token}", "Accept": "application/json"}
        )

    def test_connection(self) -> tuple[bool, str]:
        """
        Tests the Canvas connection by attempting to fetch the course details.
        Returns a tuple: (Success Boolean, Message String)
        """
        url = f"{self.base_url}/api/v1/courses/{self.course_id}"

        try:
            response = self.session.get(url)

            # If the token is bad, this throws a 401.
            # If the course ID is wrong, this throws a 404.
            response.raise_for_status()

            course_name = response.json().get("name", "Unknown Course")
            return (
                True,
                f"Successfully connected to Canvas course: [bold green]{course_name}[/bold green]",
            )

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code == 401:
                return (
                    False,
                    "Authentication failed. Is your CANVAS_API_TOKEN correct and unexpired?",
                )
            elif status_code == 404:
                return (
                    False,
                    f"Course not found. Are you sure CANVAS_COURSE_ID '{self.course_id}' is correct?",
                )
            else:
                return False, f"HTTP Error {status_code}: {e.response.text}"

        except requests.exceptions.RequestException as e:
            return False, f"Network error connecting to Canvas: {str(e)}"

    def get_existing_page(self, page_slug: str) -> dict | None:
        """
        Fetches an existing page from Canvas using its URL slug.

        Args:
            page_slug: The URL-friendly identifier of the page (e.g., 'my-cool-page')

        Returns:
            A dictionary containing the page data if found, or None if it does not exist.
        """
        # Note: Canvas API calls the identifier 'url', but it means the slug!
        url = f"{self.base_url}/api/v1/courses/{self.course_id}/pages/{page_slug}"

        try:
            response = self.session.get(url)

            # If the page exists, Canvas returns a 200 OK
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code == 404:
                # 404 just means the page hasn't been created yet.
                # This is normal, so we fail silently and return None.
                return None
            else:
                # 401 Unauthorized or other unexpected errors
                err_console.print(
                    f"[bold red]HTTP Error fetching page '{page_slug}':[/bold red] {e.response.text}"
                )
                return None

        except requests.exceptions.RequestException as e:
            err_console.print(
                f"[bold red]Network Error fetching page '{page_slug}':[/bold red] {str(e)}"
            )
            return None

    def create_or_update_page(
        self,
        title: str,
        html_content: str,
        published: bool = True,
        page_slug: str | None = None,
    ) -> str | None:
        """
        Creates a new Wiki Page in Canvas or updates it if it already exists.
        Returns the Canvas URL of the created/updated page.

        Returns:
            URL of the new canvas page if successful. Otherwise None.
        """
        # 1. Generate a Canvas-safe URL slug if one isn't provided
        if not page_slug:
            # Lowercase, replace spaces/special characters with hyphens, and strip trailing hyphens
            page_slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")

        # 2. Build the exact payload Canvas expects
        payload = {
            "wiki_page": {
                "title": title,
                "body": html_content,
                "published": published,
                "editing_roles": "teachers",
            }
        }

        # 3. Check existence to decide between PUT (update) and POST (create)
        existing_page = self.get_existing_page(page_slug)

        try:
            if existing_page:
                print(f"  [~] Updating existing page: {title}")
                url = (
                    f"{self.base_url}/api/v1/courses/{self.course_id}/pages/{page_slug}"
                )
                response = self.session.put(url, json=payload)
            else:
                print(f"  [+] Creating new page: {title}")
                url = f"{self.base_url}/api/v1/courses/{self.course_id}/pages"
                response = self.session.post(url, json=payload)

            response.raise_for_status()
            result = response.json()

            # 4. Construct and return the final user-facing Canvas URL
            final_slug = result.get("url", page_slug)
            return f"{self.base_url}/courses/{self.course_id}/pages/{final_slug}"

        except requests.exceptions.HTTPError as e:
            err_console.print(
                f"[bold red]API Error saving page '{title}':[/bold red] {e.response.text}"
            )
            return None
        except requests.exceptions.RequestException as e:
            err_console.print(
                f"[bold red]Network Error saving page '{title}':[/bold red] {str(e)}"
            )
            return None

    def upload_file(self, file: Path, folder_path: str = "/course_files") -> str:
        """
        Safely handles the Canvas 3-Step file upload process.
        Returns the Canvas URL of the uploaded file.
        """
        if not file.exists():
            raise FileNotFoundError(f"Cannot upload missing file: {file}")

        # --- STEP 0: Prepare Metadata ---
        content_type, _ = mimetypes.guess_type(file)
        content_type = content_type or "application/octet-stream"
        file_size = file.stat().st_size  # Crucial for Canvas!

        # --- STEP 1: Tell Canvas about the file ---
        # Note: We use data=payload (Form Data), NOT json=payload
        payload = {
            "name": file.name,
            "parent_folder_path": folder_path,
            "content_type": content_type,
            "size": file_size,
        }

        init_url = f"{self.base_url}/api/v1/courses/{self.course_id}/files"
        response = self.session.post(init_url, data=payload)

        # If this fails, crash early with a helpful error
        response.raise_for_status()
        canvas_data = response.json()

        # --- STEP 2: Upload the actual bytes to the provided URL ---
        upload_url = canvas_data["upload_url"]
        upload_params = canvas_data["upload_params"]

        with open(file, "rb") as f:
            # We pass files={"file": f}. The requests library automatically
            # sets the Content-Type to multipart/form-data for us here.
            upload_response = self.session.post(
                upload_url,
                data=upload_params,
                files={"file": f},
                allow_redirects=False,  # We want to handle the confirmation manually
            )

        # --- STEP 3: Confirm the upload ---
        # Depending on where Canvas routed the file (e.g., AWS S3), it either
        # returns the file object directly (2xx), or gives us a Location header (3xx)
        # to go fetch the final confirmation.
        if "Location" in upload_response.headers:
            confirm = self.session.get(upload_response.headers["Location"])
            confirm.raise_for_status()
            final_data = confirm.json()
        else:
            upload_response.raise_for_status()
            final_data = upload_response.json()

        # Canvas gives us an internal ID and a preview URL.
        # For embedding in HTML, the preview/download URL is usually what you want.
        file_id = final_data.get("id")
        return f"{self.base_url}/courses/{self.course_id}/files/{file_id}/preview"
