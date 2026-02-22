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
                url = (
                    f"{self.base_url}/api/v1/courses/{self.course_id}/pages/{page_slug}"
                )
                response = self.session.put(url, json=payload)
            else:
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

    def list_pages(self) -> list[dict]:
        """
        Returns all wiki pages in the course (title + url slug).
        Handles Canvas pagination automatically.
        """
        pages: list[dict] = []
        url = f"{self.base_url}/api/v1/courses/{self.course_id}/pages"
        params: dict = {"per_page": 100}

        while url:
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                pages.extend(response.json())
                # Follow Canvas Link header pagination
                next_url = None
                link_header = response.headers.get("Link", "")
                for part in link_header.split(","):
                    if 'rel="next"' in part:
                        next_url = part.split(";")[0].strip().strip("<>")
                        break
                url = next_url
                params = {}  # params are already baked into the next URL
            except requests.exceptions.RequestException as e:
                err_console.print(f"[bold red]Error listing pages:[/bold red] {e}")
                break

        return pages

    def delete_page(self, page_slug: str) -> bool:
        """
        Deletes a single Canvas wiki page by its URL slug.
        Returns True on success, False otherwise.
        """
        url = f"{self.base_url}/api/v1/courses/{self.course_id}/pages/{page_slug}"
        try:
            response = self.session.delete(url)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            err_console.print(
                f"[bold red]Error deleting page '{page_slug}':[/bold red] {e}"
            )
            return False

    # ------------------------------------------------------------------
    # Module API
    # ------------------------------------------------------------------

    def list_modules(self) -> list[dict]:
        """Returns all modules in the course (handles pagination)."""
        modules: list[dict] = []
        url = f"{self.base_url}/api/v1/courses/{self.course_id}/modules"
        params: dict = {"per_page": 100}
        while url:
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                modules.extend(response.json())
                url = None
                for part in response.headers.get("Link", "").split(","):
                    if 'rel="next"' in part:
                        url = part.split(";")[0].strip().strip("<>")
                        break
                params = {}
            except requests.exceptions.RequestException as e:
                err_console.print(f"[bold red]Error listing modules:[/bold red] {e}")
                break
        return modules

    def create_module(self, name: str, position: int) -> dict | None:
        """Creates a new unpublished module. Returns the module dict or None."""
        url = f"{self.base_url}/api/v1/courses/{self.course_id}/modules"
        payload = {"module": {"name": name, "position": position, "published": False}}
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            err_console.print(f"[bold red]Error creating module '{name}':[/bold red] {e}")
            return None

    def delete_module(self, module_id: int) -> bool:
        """Deletes a module by ID."""
        url = f"{self.base_url}/api/v1/courses/{self.course_id}/modules/{module_id}"
        try:
            response = self.session.delete(url)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            err_console.print(f"[bold red]Error deleting module {module_id}:[/bold red] {e}")
            return False

    def add_page_to_module(
        self, module_id: int, page_url: str, title: str, position: int
    ) -> dict | None:
        """Adds a wiki page item to a module."""
        url = f"{self.base_url}/api/v1/courses/{self.course_id}/modules/{module_id}/items"
        payload = {
            "module_item": {
                "title": title,
                "type": "Page",
                "page_url": page_url,
                "position": position,
                "indent": 0,
            }
        }
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            err_console.print(
                f"[bold red]Error adding page '{title}' to module:[/bold red] {e}"
            )
            return None

    def add_file_to_module(
        self, module_id: int, file_id: int, title: str, position: int
    ) -> dict | None:
        """Adds an uploaded file item to a module."""
        url = f"{self.base_url}/api/v1/courses/{self.course_id}/modules/{module_id}/items"
        payload = {
            "module_item": {
                "title": title,
                "type": "File",
                "content_id": file_id,
                "position": position,
                "indent": 0,
            }
        }
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            err_console.print(
                f"[bold red]Error adding file '{title}' to module:[/bold red] {e}"
            )
            return None

    def publish_module(self, module_id: int) -> bool:
        """Publishes a module and all its items."""
        base = f"{self.base_url}/api/v1/courses/{self.course_id}/modules/{module_id}"
        try:
            items = self.session.get(f"{base}/items", params={"per_page": 100})
            items.raise_for_status()
            for item in items.json():
                self.session.put(
                    f"{base}/items/{item['id']}",
                    json={"module_item": {"published": True}},
                )
            resp = self.session.put(base, json={"module": {"published": True}})
            resp.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            err_console.print(f"[bold red]Error publishing module {module_id}:[/bold red] {e}")
            return False

    # ------------------------------------------------------------------
    # Assignment API
    # ------------------------------------------------------------------

    def list_assignments(self) -> list[dict]:
        """Returns all assignments in the course (handles pagination)."""
        assignments: list[dict] = []
        url = f"{self.base_url}/api/v1/courses/{self.course_id}/assignments"
        params: dict = {"per_page": 100}
        while url:
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                assignments.extend(response.json())
                url = None
                for part in response.headers.get("Link", "").split(","):
                    if 'rel="next"' in part:
                        url = part.split(";")[0].strip().strip("<>")
                        break
                params = {}
            except requests.exceptions.RequestException as e:
                err_console.print(f"[bold red]Error listing assignments:[/bold red] {e}")
                break
        return assignments

    def create_or_update_assignment(
        self,
        name: str,
        html_content: str,
        points_possible: int = 100,
    ) -> dict | None:
        """
        Creates a new assignment or updates the description of an existing one.
        Preserves deadlines, points, and other settings on update.
        Returns the assignment dict or None on failure.
        """
        # Look for an existing assignment with this name
        existing = next(
            (a for a in self.list_assignments() if a.get("name") == name), None
        )

        try:
            if existing:
                url = (
                    f"{self.base_url}/api/v1/courses/{self.course_id}"
                    f"/assignments/{existing['id']}"
                )
                payload = {
                    "assignment": {
                        "description": html_content,
                        "notify_of_update": False,
                    }
                }
                response = self.session.put(url, json=payload)
            else:
                url = f"{self.base_url}/api/v1/courses/{self.course_id}/assignments"
                payload = {
                    "assignment": {
                        "name": name,
                        "description": html_content,
                        "points_possible": points_possible,
                        "submission_types": ["online_text_entry", "online_upload"],
                        "grading_type": "points",
                        "published": True,
                        "allowed_extensions": ["py", "ipynb", "txt", "pdf", "zip"],
                        "notify_of_update": False,
                    }
                }
                response = self.session.post(url, json=payload)

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            err_console.print(
                f"[bold red]Error creating/updating assignment '{name}':[/bold red] {e}"
            )
            return None

    def delete_assignment(self, assignment_id: int) -> bool:
        """Deletes an assignment by ID."""
        url = (
            f"{self.base_url}/api/v1/courses/{self.course_id}"
            f"/assignments/{assignment_id}"
        )
        try:
            response = self.session.delete(url)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            err_console.print(
                f"[bold red]Error deleting assignment {assignment_id}:[/bold red] {e}"
            )
            return False
