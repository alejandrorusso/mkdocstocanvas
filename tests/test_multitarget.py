"""Tests for multi-target support: configurable labs dir, site_dir parsing,
and --env-file credential selection. All offline.

These lock in that the new options are opt-in — the defaults reproduce the
previous single-target behaviour.
"""

from pathlib import Path

import pytest

from mkdocstocanvas.api import create_client
from mkdocstocanvas.uploaders.labs import find_lab_files, is_lab_rel_path
from mkdocstocanvas.utils.config import parse_site_dir

CANVAS_VARS = ("CANVAS_API_TOKEN", "CANVAS_BASE_URL", "CANVAS_COURSE_ID")


class TestLabsDirConfigurable:
    def test_default_dir_unchanged(self):
        # BC: default labs_dir is still "labs"
        assert is_lab_rel_path("labs/lab1.md")
        assert not is_lab_rel_path("wasp/lab1.md")

    def test_custom_dir(self):
        assert is_lab_rel_path("wasp/lab1.md", "wasp")
        # a file under the custom dir that is not lab-named stays a page
        assert not is_lab_rel_path("wasp/reconstruction.md", "wasp")
        # the default dir is no longer treated as labs when another is chosen
        assert not is_lab_rel_path("labs/lab1.md", "wasp")

    def test_find_lab_files_custom_dir(self, tmp_path: Path):
        wasp = tmp_path / "wasp"
        wasp.mkdir()
        (wasp / "lab2.md").write_text("# Lab 2")
        (wasp / "lab1.md").write_text("# Lab 1")
        (wasp / "reconstruction.md").write_text("# not a lab")

        result = find_lab_files(tmp_path, "wasp")
        assert [p.name for p in result] == ["lab1.md", "lab2.md"]

    def test_find_lab_files_default_dir_bc(self, tmp_path: Path):
        # With only a custom dir present, the default ("labs") finds nothing.
        wasp = tmp_path / "wasp"
        wasp.mkdir()
        (wasp / "lab1.md").write_text("# Lab 1")
        assert find_lab_files(tmp_path) == []


class TestParseSiteDir:
    def test_default_site_dir(self, tmp_path: Path):
        cfg = tmp_path / "mkdocs.yml"
        cfg.write_text("site_name: Test\n")
        assert parse_site_dir(cfg) == "site"

    def test_custom_site_dir(self, tmp_path: Path):
        cfg = tmp_path / "mkdocs.wasp.yml"
        cfg.write_text("site_name: Test\nsite_dir: site-wasp\n")
        assert parse_site_dir(cfg) == "site-wasp"


class TestCreateClientEnvFile:
    @staticmethod
    def _write_env(path: Path, token: str, base_url: str, course_id: str) -> None:
        path.write_text(
            f'CANVAS_API_TOKEN="{token}"\n'
            f'CANVAS_BASE_URL="{base_url}"\n'
            f'CANVAS_COURSE_ID="{course_id}"\n'
        )

    def test_env_file_selected_and_overrides_environment(self, monkeypatch, tmp_path):
        # A value already in the environment must be overridden by the file.
        for k in CANVAS_VARS:
            monkeypatch.setenv(k, "shell-value")
        env = tmp_path / ".env.kth"
        self._write_env(env, "tok", "https://kth.instructure.com", "42")

        client = create_client(env_file=env)

        assert client.base_url == "https://kth.instructure.com"
        assert client.course_id == "42"

    def test_missing_env_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="--env-file not found"):
            create_client(env_file=tmp_path / "does-not-exist.env")

    def test_default_loads_cwd_env(self, monkeypatch, tmp_path):
        # BC: with no env_file, ./.env in the cwd is still used.
        for k in CANVAS_VARS:
            monkeypatch.delenv(k, raising=False)
        self._write_env(
            tmp_path / ".env", "tok", "https://chalmers.instructure.com", "7"
        )
        monkeypatch.chdir(tmp_path)

        client = create_client()

        assert client.base_url == "https://chalmers.instructure.com"
        assert client.course_id == "7"
