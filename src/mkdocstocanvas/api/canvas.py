import mimetypes
import re
from pathlib import Path

import requests
from rich.console import Console

# Initialize the error console at the module level
err_console = Console(stderr=True)

# Default timeout (seconds) for every HTTP request, so a hung connection
# can never block the CLI forever.
DEFAULT_TIMEOUT = 30


class _TimeoutSession(requests.Session):
    """A requests Session that applies a default timeout to all requests."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        super().__init__()
        self.timeout = timeout

    def request(self, *args, **kwargs):
        # All HTTP verbs (get/post/put/delete) funnel through this method.
        kwargs.setdefault("timeout", self.timeout)
        return super().request(*args, **kwargs)


class CanvasUploader:
    def __init__(self, base_url: str, api_token: str, course_id: str | int):
        """Initializes the Canvas API client with a persistent session."""
        self.base_url = base_url.rstrip("/")
        self.course_id = str(course_id)

        # 1. Setup global session for connection pooling
        self.session = _TimeoutSession()

        # 2. Set global headers (Auth and Accept ONLY)
        self.session.headers.update(
            {"Authorization": f"Bearer {api_token}", "Accept": "application/json"}
        )

        # Lazily built indexes for asset deduplication
        self._file_index: tuple[dict[str, dict], dict[tuple[str, int], dict]] | None = (
            None
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
            return False, f"Network error connecting to Canvas: {e!s}"

    def get_existing_page(self, page_slug: str) -> dict | None:
        """
        Fetches an existing page from Canvas using its URL slug.

        Args:
            page_slug: The URL-friendly identifier of the page (e.g., 'my-cool-page')

        Returns:
            A dictionary containing the page data if found, or None if it does not exist.

        Raises:
            requests.exceptions.RequestException: For any failure other than
                "page not found" (e.g. an expired token). Must not be confused
                with None, or callers would create duplicate pages.
        """
        # Note: Canvas API calls the identifier 'url', but it means the slug!
        url = f"{self.base_url}/api/v1/courses/{self.course_id}/pages/{page_slug}"

        try:
            response = self.session.get(url)

            # If the page exists, Canvas returns a 200 OK
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # 404 just means the page hasn't been created yet.
                # This is normal, so we return None.
                return None
            # 401 Unauthorized or other unexpected errors must be raised,
            # never treated as "page missing".
            raise

    def create_or_update_page(
        self,
        title: str,
        html_content: str,
        published: bool = True,
        page_slug: str | None = None,
    ) -> tuple[str, str] | None:
        """
        Creates a new Wiki Page in Canvas or updates it if it already exists.

        Returns:
            (Canvas URL, page slug) of the saved page if successful, else None.
            The slug is taken from the API response, so callers never have to
            parse it out of the URL.
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

        # Fallback: if slug lookup fails (often due to stale/missing cache),
        # try to find a unique page by title and update that page instead of
        # creating a duplicate.
        if not existing_page:
            title_slug = self._find_unique_page_slug_by_title(title)
            if title_slug:
                page_slug = title_slug
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
            canvas_url = f"{self.base_url}/courses/{self.course_id}/pages/{final_slug}"
            return canvas_url, final_slug

        except requests.exceptions.HTTPError as e:
            err_console.print(
                f"[bold red]API Error saving page '{title}':[/bold red] {e.response.text}"
            )
            return None
        except requests.exceptions.RequestException as e:
            err_console.print(
                f"[bold red]Network Error saving page '{title}':[/bold red] {e!s}"
            )
            return None

    def _find_unique_page_slug_by_title(self, title: str) -> str | None:
        """
        Return a page slug when there is exactly one Canvas page with this title.

        This avoids creating duplicate pages when local cache is missing/stale.
        If multiple pages share the same title, returns None to avoid ambiguity.
        """
        normalized = title.strip().lower()
        matches = [
            p
            for p in self.list_pages()
            if p.get("title", "").strip().lower() == normalized
        ]
        if len(matches) == 1:
            return matches[0].get("url")
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
        upload_url = canvas_data.get("upload_url")
        upload_params = canvas_data.get("upload_params")
        if not upload_url:
            # Don't KeyError on an unexpected Canvas response shape
            raise ValueError(
                f"Canvas did not return an upload URL for {file.name}. "
                f"Response was: {canvas_data}"
            )

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
        return f"{self.base_url}/courses/{self.course_id}/files/{file_id}?wrap=1"

    def _paginated_get(self, first_url: str, params: dict) -> list[dict]:
        """
        GET all pages of a paginated Canvas endpoint, following the Link header.

        Raises requests.exceptions.RequestException on any failure - callers
        must not receive a silently truncated list.
        """
        items: list[dict] = []
        url: str | None = first_url
        params = dict(params)

        while url:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            items.extend(response.json())
            # Follow Canvas Link header pagination
            url = None
            for part in response.headers.get("Link", "").split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().strip("<>")
                    break
            params = {}  # params are already baked into the next URL

        return items

    def list_pages(self, include_body: bool = False) -> list[dict]:
        """
        Returns all wiki pages in the course (title + url slug).
        Handles Canvas pagination automatically.

        Args:
            include_body: Also fetch each page's HTML body (slower, but
                needed for ownership markers and cache rebuilding).
        """
        params: dict[str, str | int] = {"per_page": 100}
        if include_body:
            params["include[]"] = "body"
        return self._paginated_get(
            f"{self.base_url}/api/v1/courses/{self.course_id}/pages",
            params,
        )

    def list_files(self) -> list[dict]:
        """
        Returns all files in the course (id, filename, md5, size, ...).
        Handles Canvas pagination automatically.
        """
        return self._paginated_get(
            f"{self.base_url}/api/v1/courses/{self.course_id}/files",
            {"per_page": 100},
        )

    def find_existing_file(
        self,
        *,
        md5: str | None = None,
        filename: str | None = None,
        size: int | None = None,
    ) -> dict | None:
        """
        Find an existing Canvas file matching the given identity, if any.

        Matching is by content hash when Canvas reports one, falling back to
        filename + byte size (Canvas does not always return a md5 attribute).
        Used to skip re-uploading assets that already exist in Canvas. The
        file list is fetched once and cached for the lifetime of the client.
        """
        if self._file_index is None:
            by_md5: dict[str, dict] = {}
            by_name_size: dict[tuple[str, int], dict] = {}
            for f in self.list_files():
                file_id = f.get("id")
                if file_id is None:
                    continue
                file_md5 = f.get("md5")
                if file_md5:
                    by_md5.setdefault(file_md5, f)
                name, file_size = f.get("filename"), f.get("size")
                if name and isinstance(file_size, int):
                    by_name_size.setdefault((name, file_size), f)
            self._file_index = (by_md5, by_name_size)
        by_md5, by_name_size = self._file_index
        if md5 and md5 in by_md5:
            return by_md5[md5]
        if filename is not None and size is not None:
            return by_name_size.get((filename, size))
        return None

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
        return self._paginated_get(
            f"{self.base_url}/api/v1/courses/{self.course_id}/modules",
            {"per_page": 100},
        )

    def create_module(self, name: str, position: int) -> dict | None:
        """Creates a new unpublished module. Returns the module dict or None."""
        url = f"{self.base_url}/api/v1/courses/{self.course_id}/modules"
        payload = {"module": {"name": name, "position": position, "published": False}}
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            err_console.print(
                f"[bold red]Error creating module '{name}':[/bold red] {e}"
            )
            return None

    def delete_module(self, module_id: int) -> bool:
        """Deletes a module by ID."""
        url = f"{self.base_url}/api/v1/courses/{self.course_id}/modules/{module_id}"
        try:
            response = self.session.delete(url)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            err_console.print(
                f"[bold red]Error deleting module {module_id}:[/bold red] {e}"
            )
            return False

    def add_page_to_module(
        self, module_id: int, page_url: str, title: str, position: int
    ) -> dict | None:
        """Adds a wiki page item to a module."""
        url = (
            f"{self.base_url}/api/v1/courses/{self.course_id}/modules/{module_id}/items"
        )
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
        url = (
            f"{self.base_url}/api/v1/courses/{self.course_id}/modules/{module_id}/items"
        )
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
                put = self.session.put(
                    f"{base}/items/{item['id']}",
                    json={"module_item": {"published": True}},
                )
                # Unchecked item publishes would silently stay unpublished.
                put.raise_for_status()
            resp = self.session.put(base, json={"module": {"published": True}})
            resp.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            err_console.print(
                f"[bold red]Error publishing module {module_id}:[/bold red] {e}"
            )
            return False

    # ------------------------------------------------------------------
    # Syllabus API
    # ------------------------------------------------------------------

    def get_syllabus(self) -> str | None:
        """
        Fetches the current syllabus body for the course.

        Returns:
            The HTML syllabus body string, or None on failure.
        """
        url = f"{self.base_url}/api/v1/courses/{self.course_id}"
        try:
            response = self.session.get(url, params={"include[]": "syllabus_body"})
            response.raise_for_status()
            return response.json().get("syllabus_body")
        except requests.exceptions.RequestException as e:
            err_console.print(f"[bold red]Error fetching syllabus:[/bold red] {e}")
            return None

    def upload_syllabus(self, html_content: str) -> str | None:
        """
        Uploads or replaces the course syllabus body.

        Args:
            html_content: HTML content to set as the syllabus body.

        Returns:
            The Canvas URL of the syllabus page on success, or None on failure.
        """
        url = f"{self.base_url}/api/v1/courses/{self.course_id}"
        payload = {"course": {"syllabus_body": html_content}}
        try:
            response = self.session.put(url, json=payload)
            response.raise_for_status()
            return f"{self.base_url}/courses/{self.course_id}/assignments/syllabus"
        except requests.exceptions.HTTPError as e:
            err_console.print(
                f"[bold red]Error uploading syllabus:[/bold red] {e.response.text}"
            )
            return None
        except requests.exceptions.RequestException as e:
            err_console.print(
                f"[bold red]Network error uploading syllabus:[/bold red] {e!s}"
            )
            return None

    # ------------------------------------------------------------------
    # Assignment API
    # ------------------------------------------------------------------

    def list_assignments(self) -> list[dict]:
        """Returns all assignments in the course (handles pagination)."""
        return self._paginated_get(
            f"{self.base_url}/api/v1/courses/{self.course_id}/assignments",
            {"per_page": 100},
        )

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
