"""CLI wiring tests: the global --config / --env-file / --labs-dir options reach
the uploader functions, and defaults reproduce the previous behaviour. Offline —
the client and uploaders are stubbed so nothing touches Canvas.
"""

from typer.testing import CliRunner

import mkdocstocanvas.cli as cli

runner = CliRunner()


def _stub_client(monkeypatch):
    monkeypatch.setattr(cli, "get_client", lambda ctx: object())


def test_defaults_unchanged(monkeypatch):
    captured = {}
    _stub_client(monkeypatch)

    def fake_parse(client, mkdocs_path="mkdocs.yml", labs_dir="labs", **kw):
        captured["mkdocs_path"] = mkdocs_path
        captured["labs_dir"] = labs_dir

    monkeypatch.setattr(cli, "parse_upload_all_pages", fake_parse)

    result = runner.invoke(cli.app, ["upload-pages"])
    assert result.exit_code == 0, result.output
    assert captured == {"mkdocs_path": "mkdocs.yml", "labs_dir": "labs"}


def test_config_and_labs_dir_threaded_to_pages(monkeypatch):
    captured = {}
    _stub_client(monkeypatch)

    def fake_parse(client, mkdocs_path="mkdocs.yml", labs_dir="labs", **kw):
        captured["mkdocs_path"] = mkdocs_path
        captured["labs_dir"] = labs_dir

    monkeypatch.setattr(cli, "parse_upload_all_pages", fake_parse)

    result = runner.invoke(
        cli.app,
        ["-c", "mkdocs.wasp.yml", "--labs-dir", "wasp", "upload-pages"],
    )
    assert result.exit_code == 0, result.output
    assert captured == {"mkdocs_path": "mkdocs.wasp.yml", "labs_dir": "wasp"}


def test_labs_dir_threaded_to_labs(monkeypatch):
    captured = {}
    _stub_client(monkeypatch)

    def fake_upload(client, labs_dir="labs", **kw):
        captured["labs_dir"] = labs_dir

    monkeypatch.setattr(cli, "upload_all_labs", fake_upload)

    result = runner.invoke(cli.app, ["--labs-dir", "wasp", "upload-labs"])
    assert result.exit_code == 0, result.output
    assert captured["labs_dir"] == "wasp"


def test_env_file_threaded_to_create_client(monkeypatch):
    captured = {}

    def fake_create_client(env_file=None):
        captured["env_file"] = env_file
        return object()

    monkeypatch.setattr(cli, "create_client", fake_create_client)
    monkeypatch.setattr(cli, "parse_upload_all_pages", lambda *a, **k: None)

    result = runner.invoke(cli.app, ["-e", ".env.kth", "upload-pages"])
    assert result.exit_code == 0, result.output
    assert captured["env_file"] == ".env.kth"


def test_config_threaded_to_modules(monkeypatch):
    captured = {}
    _stub_client(monkeypatch)

    def fake_upload(client, mkdocs_path="mkdocs.yml", labs_dir="labs", **kw):
        captured["mkdocs_path"] = mkdocs_path
        captured["labs_dir"] = labs_dir

    monkeypatch.setattr(cli, "upload_all_modules", fake_upload)

    result = runner.invoke(
        cli.app, ["-c", "mkdocs.wasp.yml", "--labs-dir", "wasp", "upload-modules"]
    )
    assert result.exit_code == 0, result.output
    assert captured == {"mkdocs_path": "mkdocs.wasp.yml", "labs_dir": "wasp"}


def test_cache_threaded_to_pages(monkeypatch):
    captured = {}
    _stub_client(monkeypatch)

    def fake_parse(client, cache=".canvas_upload_state.json", **kw):
        captured["cache"] = cache

    monkeypatch.setattr(cli, "parse_upload_all_pages", fake_parse)

    result = runner.invoke(
        cli.app, ["--cache", ".canvas_upload_state.kth.json", "upload-pages"]
    )
    assert result.exit_code == 0, result.output
    assert captured["cache"] == ".canvas_upload_state.kth.json"


def test_default_cache_unchanged(monkeypatch):
    captured = {}
    _stub_client(monkeypatch)

    def fake_parse(client, cache=".canvas_upload_state.json", **kw):
        captured["cache"] = cache

    monkeypatch.setattr(cli, "parse_upload_all_pages", fake_parse)

    result = runner.invoke(cli.app, ["upload-pages"])
    assert result.exit_code == 0, result.output
    assert captured["cache"] == ".canvas_upload_state.json"
