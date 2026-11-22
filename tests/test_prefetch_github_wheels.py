"""Unit and integration tests for `scripts/prefetch_github_wheels.py`."""

from __future__ import annotations

import http.server
import io
import json
import os
import runpy
import stat
import threading
import urllib.error
import urllib.request
import zipfile
from http.client import HTTPMessage
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from scripts.prefetch_github_wheels import (
    CrossOriginAuthStripRedirectHandler,
    GitHubReleaseWheelSpec,
    PrefetchedWheel,
    collect_wheel_specs,
    default_fetch_bytes,
    download_wheel_spec,
    ensure_github_netrc,
    extract_asset_api_url,
    fetch_bytes_with_retry,
    main,
    parse_requirement_line,
    prefetch_all_wheels,
    render_netrc_block,
    render_uv_overrides,
    resolve_github_token,
)

if TYPE_CHECKING:
    from scripts.prefetch_github_wheels import JsonValue

_NL = chr(10)


def _make_valid_wheel_bytes(entry_name: str = "pkg/mod.py", content: bytes = b"X = 1") -> bytes:
    """Create an in-memory valid `.whl` ZIP archive byte payload for test assertions.

    Args:
        entry_name: Internal ZIP member filename.
        content: Raw bytes written inside `entry_name`.

    Returns:
        Serialized ZIP archive bytes.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(entry_name, content)
    return buf.getvalue()


def test_parse_requirement_line_and_collect_wheel_specs(tmp_path: Path) -> None:
    """Verify PEP 508 GitHub Release wheel parsing, marker/comment stripping, and deduplication.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert parse_requirement_line("") is None
    assert parse_requirement_line("# comment only") is None
    assert parse_requirement_line("pydantic>=2.0.0") is None
    assert parse_requirement_line("griffe @ git+https://github.com/SamuelMarks/griffe.git@rest") is None

    cdd_line = (
        "cdd @ https://github.com/offscale/cdd-python/releases/download/"
        "v0.0.99rc46/cdd-0.0.99rc46-py3-none-any.whl ; python_version >= '3.10' # pinned"
    )
    parsed = parse_requirement_line(cdd_line)
    assert parsed == GitHubReleaseWheelSpec(
        package_name="cdd",
        browser_url=(
            "https://github.com/offscale/cdd-python/releases/download/v0.0.99rc46/cdd-0.0.99rc46-py3-none-any.whl"
        ),
        owner="offscale",
        repo="cdd-python",
        tag="v0.0.99rc46",
        asset_name="cdd-0.0.99rc46-py3-none-any.whl",
    )

    req_a = tmp_path / "req_a.txt"
    req_b = tmp_path / "req_b.txt"
    ir_line = (
        "ml-switcheroo-ir @ https://github.com/SamuelMarks/ml-switcheroo-ir/"
        "releases/download/v0.0.8/ml_switcheroo_ir-0.0.8-py3-none-any.whl"
    )
    req_a.write_text(_NL.join(["pydantic>=2.0.0", ir_line, cdd_line, ""]), encoding="utf-8")
    req_b.write_text(_NL.join([ir_line, "rich", ""]), encoding="utf-8")

    assert collect_wheel_specs(()) == ()
    collected = collect_wheel_specs([req_a, req_b])
    assert len(collected) == 2
    assert [s.package_name for s in collected] == ["ml-switcheroo-ir", "cdd"]


def test_repo_requirements_contain_expected_github_release_wheels() -> None:
    """Verify repository `requirements.txt` and `requirements-gen.txt` parse cleanly."""
    root = Path(__file__).resolve().parent.parent
    specs = collect_wheel_specs([root / "requirements.txt", root / "requirements-gen.txt"])
    names = {spec.package_name for spec in specs}
    assert "ml-switcheroo-ir" in names
    assert "cdd" in names


def test_extract_asset_api_url_validation_branches() -> None:
    """Verify `extract_asset_api_url` validates payload shape and locates matching assets."""
    with pytest.raises(ValueError, match="must be a JSON object"):
        extract_asset_api_url(["invalid"], "pkg.whl")

    with pytest.raises(ValueError, match="missing an 'assets' list"):
        extract_asset_api_url({"assets": "not-a-list"}, "pkg.whl")

    payload: JsonValue = {
        "assets": [
            "malformed-entry",
            {"name": "other.whl", "url": "https://api.github.com/assets/1"},
            {"name": "target.whl", "url": ""},
            {"name": "target.whl", "url": "https://api.github.com/repos/org/repo/releases/assets/42"},
        ]
    }
    resolved = extract_asset_api_url(payload, "target.whl")
    assert resolved == "https://api.github.com/repos/org/repo/releases/assets/42"

    with pytest.raises(ValueError, match="not found"):
        extract_asset_api_url(payload, "missing.whl")


def test_netrc_token_and_override_rendering_helpers(tmp_path: Path) -> None:
    """Verify `.netrc` writing, permission hardening, token resolution, and override formatting.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    block = render_netrc_block("tok_123")
    assert _NL.join(["machine github.com", "login x-access-token", "password tok_123", ""]) in block
    assert f"machine api.github.com{_NL}" in block

    netrc_a = tmp_path / "sub" / ".netrc"
    assert ensure_github_netrc("tok_123", netrc_a) is True
    assert stat.S_IMODE(os.stat(netrc_a).st_mode) == 0o600
    assert ensure_github_netrc("tok_456", netrc_a) is False
    assert "tok_456" not in netrc_a.read_text(encoding="utf-8")

    netrc_b = tmp_path / ".netrc_existing"
    netrc_b.write_text(_NL.join(["machine pypi.org", "login __token__", "password pypi_tok", ""]), encoding="utf-8")
    assert ensure_github_netrc("gh_tok", netrc_b) is True
    combined = netrc_b.read_text(encoding="utf-8")
    assert "machine pypi.org" in combined
    assert "machine github.com" in combined

    assert resolve_github_token({}) is None
    assert resolve_github_token({"GITHUB_TOKEN": "  ", "GH_TOKEN": " fallback_tok "}) == "fallback_tok"
    assert resolve_github_token({"GITHUB_TOKEN": "primary_tok", "GH_TOKEN": "fallback_tok"}) == "primary_tok"

    assert render_uv_overrides(()) == ""
    sample_spec = GitHubReleaseWheelSpec(
        package_name="cdd",
        browser_url="https://github.com/offscale/cdd-python/releases/download/v1/cdd-1.whl",
        owner="offscale",
        repo="cdd-python",
        tag="v1",
        asset_name="cdd-1.whl",
    )
    sample_wheel = PrefetchedWheel(
        spec=sample_spec,
        local_path=tmp_path / "cdd-1.whl",
        override_line="cdd @ file:///tmp/cdd-1.whl",
    )
    assert render_uv_overrides([sample_wheel]) == f"cdd @ file:///tmp/cdd-1.whl{_NL}"


def test_cross_origin_redirect_strips_auth_while_same_origin_preserves_auth() -> None:
    """Verify `default_fetch_bytes` and `CrossOriginAuthStripRedirectHandler` over loopback HTTP."""
    recorded_blob_auth: list[str | None] = []
    recorded_same_origin_auth: list[str | None] = []

    class _BlobServerHandler(http.server.BaseHTTPRequestHandler):
        """Loopback handler simulating `objects.githubusercontent.com` signed asset storage."""

        def do_GET(self) -> None:
            """Record incoming `Authorization` header and return wheel payload bytes."""
            recorded_blob_auth.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(b"blob-ok")

    blob_server = http.server.HTTPServer(("127.0.0.1", 0), _BlobServerHandler)
    blob_port = blob_server.server_port

    class _ApiServerHandler(http.server.BaseHTTPRequestHandler):
        """Loopback handler simulating `api.github.com` cross-origin and same-origin redirects."""

        def do_GET(self) -> None:
            """Redirect `/cross` to `blob_server` and `/same` to `/final` on the same host:port."""
            if self.path == "/cross":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{blob_port}/asset.whl")
                self.end_headers()
            elif self.path == "/same":
                host_hdr = self.headers.get("Host", "127.0.0.1")
                self.send_response(302)
                self.send_header("Location", f"http://{host_hdr}/final")
                self.end_headers()
            else:
                recorded_same_origin_auth.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"same-ok")

    api_server = http.server.HTTPServer(("127.0.0.1", 0), _ApiServerHandler)
    api_port = api_server.server_port

    blob_thread = threading.Thread(target=blob_server.serve_forever, daemon=True)
    api_thread = threading.Thread(target=api_server.serve_forever, daemon=True)
    blob_thread.start()
    api_thread.start()

    try:
        cross_req = urllib.request.Request(
            f"http://127.0.0.1:{api_port}/cross",
            headers={"Authorization": "Bearer secret_gh_token"},
        )
        cross_req.add_unredirected_header("Authorization", "Bearer secret_gh_token")
        assert default_fetch_bytes(cross_req, 5.0) == b"blob-ok"
        assert recorded_blob_auth == [None]

        same_req = urllib.request.Request(
            f"http://127.0.0.1:{api_port}/same",
            headers={"Authorization": "Bearer secret_gh_token"},
        )
        assert default_fetch_bytes(same_req, 5.0) == b"same-ok"
        assert recorded_same_origin_auth == ["Bearer secret_gh_token"]

        handler = CrossOriginAuthStripRedirectHandler()
        post_req = urllib.request.Request(f"http://127.0.0.1:{api_port}/cross", data=b"body", method="POST")
        empty_headers = HTTPMessage()
        with pytest.raises(urllib.error.HTTPError):
            handler.redirect_request(post_req, io.BytesIO(b""), 307, "Temporary Redirect", empty_headers, "/")
    finally:
        api_server.shutdown()
        blob_server.shutdown()
        api_server.server_close()
        blob_server.server_close()


def test_fetch_bytes_with_retry_backoff_and_exhaustion() -> None:
    """Verify exponential-backoff retry succeeds on transient errors and raises on exhaustion."""
    req = urllib.request.Request("https://api.github.com/repos/org/repo/releases/tags/v1")

    with pytest.raises(ValueError, match="at least 1"):
        fetch_bytes_with_retry(req, max_attempts=0)

    delays: list[float] = []
    calls = 0

    def _flaky_fetch(request: urllib.request.Request, timeout: float) -> bytes:
        """Fail twice with `URLError` before returning payload bytes on the third attempt.

        Args:
            request: Outbound HTTP request.
            timeout: Per-attempt timeout in seconds.

        Returns:
            Byte payload once `calls == 3`.

        Raises:
            urllib.error.URLError: On the first two attempts.
        """
        nonlocal calls
        del request, timeout
        calls += 1
        if calls < 3:
            raise urllib.error.URLError("transient network blip")
        return b"recovered"

    result = fetch_bytes_with_retry(
        req,
        fetch_fn=_flaky_fetch,
        max_attempts=4,
        base_delay=0.5,
        sleep_fn=delays.append,
    )
    assert result == b"recovered"
    assert delays == [0.5, 1.0]

    def _always_fail(request: urllib.request.Request, timeout: float) -> bytes:
        """Unconditionally raise `TimeoutError` to exercise retry exhaustion.

        Args:
            request: Outbound HTTP request.
            timeout: Per-attempt timeout in seconds.

        Returns:
            Never returns normally.

        Raises:
            TimeoutError: Always raised.
        """
        del request, timeout
        raise TimeoutError("timed out")

    exhaust_delays: list[float] = []
    with pytest.raises(RuntimeError, match="after 2 attempt"):
        fetch_bytes_with_retry(
            req,
            fetch_fn=_always_fail,
            max_attempts=2,
            base_delay=0.25,
            sleep_fn=exhaust_delays.append,
        )
    assert exhaust_delays == [0.25]


def test_download_wheel_spec_authenticated_unauthenticated_and_corrupt(tmp_path: Path) -> None:
    """Verify `download_wheel_spec` handles REST API auth, direct browser fallback, and corrupt archives.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    spec = GitHubReleaseWheelSpec(
        package_name="ml-switcheroo-ir",
        browser_url=(
            "https://github.com/SamuelMarks/ml-switcheroo-ir/releases/download/v0.0.8/"
            "ml_switcheroo_ir-0.0.8-py3-none-any.whl"
        ),
        owner="SamuelMarks",
        repo="ml-switcheroo-ir",
        tag="v0.0.8",
        asset_name="ml_switcheroo_ir-0.0.8-py3-none-any.whl",
    )
    valid_whl = _make_valid_wheel_bytes()
    seen_urls: list[tuple[str, str | None, str | None]] = []

    def _mock_auth_fetch(request: urllib.request.Request, timeout: float) -> bytes:
        """Serve mock GitHub REST API metadata JSON and binary wheel bytes.

        Args:
            request: Outbound HTTP request.
            timeout: Socket timeout in seconds.

        Returns:
            JSON release metadata or binary wheel archive bytes.
        """
        del timeout
        seen_urls.append(
            (
                request.full_url,
                request.get_header("Accept"),
                request.get_header("Authorization"),
            )
        )
        if request.full_url.endswith("/releases/tags/v0.0.8"):
            body = {
                "assets": [
                    {
                        "name": "ml_switcheroo_ir-0.0.8-py3-none-any.whl",
                        "url": "https://api.github.com/repos/SamuelMarks/ml-switcheroo-ir/releases/assets/99",
                    }
                ]
            }
            return json.dumps(body).encode("utf-8")
        return valid_whl

    auth_wheel = download_wheel_spec(spec, dest_dir=tmp_path / "auth", token="ghs_test", fetch_fn=_mock_auth_fetch)
    assert auth_wheel.local_path.is_file()
    assert auth_wheel.override_line.startswith("ml-switcheroo-ir @ file://")
    assert seen_urls == [
        (
            "https://api.github.com/repos/SamuelMarks/ml-switcheroo-ir/releases/tags/v0.0.8",
            "application/vnd.github+json",
            "Bearer ghs_test",
        ),
        (
            "https://api.github.com/repos/SamuelMarks/ml-switcheroo-ir/releases/assets/99",
            "application/octet-stream",
            "Bearer ghs_test",
        ),
    ]

    unauth_urls: list[tuple[str, str | None]] = []

    def _mock_unauth_fetch(request: urllib.request.Request, timeout: float) -> bytes:
        """Serve direct browser wheel download without authentication headers.

        Args:
            request: Outbound HTTP request.
            timeout: Socket timeout in seconds.

        Returns:
            Valid wheel bytes.
        """
        del timeout
        unauth_urls.append((request.full_url, request.get_header("Authorization")))
        return valid_whl

    unauth_wheel = download_wheel_spec(spec, dest_dir=tmp_path / "unauth", token=None, fetch_fn=_mock_unauth_fetch)
    assert unauth_wheel.local_path.is_file()
    assert unauth_urls == [(spec.browser_url, None)]

    with pytest.raises(ValueError, match="not a valid wheel ZIP archive"):
        download_wheel_spec(
            spec,
            dest_dir=tmp_path / "bad",
            token=None,
            fetch_fn=lambda _req, _timeout: b"<html>403 Forbidden</html>",
        )


def test_prefetch_all_wheels_and_cli_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `prefetch_all_wheels`, `$GITHUB_ENV` export, `main()`, and `__main__` execution.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    req_file = tmp_path / "requirements.txt"
    req_file.write_text(
        f"cdd @ https://github.com/offscale/cdd-python/releases/download/v0.0.99rc46/cdd-0.0.99rc46-py3-none-any.whl{_NL}",
        encoding="utf-8",
    )
    valid_whl = _make_valid_wheel_bytes()

    def _fake_fetch(request: urllib.request.Request, timeout: float) -> bytes:
        """Serve release metadata JSON or wheel bytes depending on URL suffix.

        Args:
            request: Outbound HTTP request.
            timeout: Socket timeout in seconds.

        Returns:
            Mock HTTP response bytes.
        """
        del timeout
        if "/releases/tags/" in request.full_url:
            return json.dumps(
                {
                    "assets": [
                        {
                            "name": "cdd-0.0.99rc46-py3-none-any.whl",
                            "url": "https://api.github.com/repos/offscale/cdd-python/releases/assets/77",
                        }
                    ]
                }
            ).encode("utf-8")
        return valid_whl

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    gh_env_file = tmp_path / "gh_env" / "env.txt"
    out_dir = tmp_path / "wheels_with_token"

    overrides_path = prefetch_all_wheels(
        requirement_files=[req_file],
        dest_dir=out_dir,
        env={"GITHUB_TOKEN": "ghs_ci_tok", "GITHUB_ENV": str(gh_env_file)},
        netrc_path=None,
        fetch_fn=_fake_fetch,
    )
    assert overrides_path.is_file()
    assert "cdd @ file://" in overrides_path.read_text(encoding="utf-8")
    assert (fake_home / ".netrc").is_file()
    gh_env_text = gh_env_file.read_text(encoding="utf-8")
    assert f"UV_OVERRIDE={overrides_path}" in gh_env_text
    assert f"GH_WHEEL_PREFETCH_DIR={out_dir.resolve()}" in gh_env_text

    # Also cover explicit netrc_path, default env=None, no token, and no GITHUB_ENV
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_ENV", raising=False)
    explicit_netrc = tmp_path / "custom.netrc"
    overrides_no_tok = prefetch_all_wheels(
        requirement_files=[req_file],
        dest_dir=tmp_path / "wheels_no_token",
        env=None,
        netrc_path=explicit_netrc,
        fetch_fn=_fake_fetch,
    )
    assert overrides_no_tok.is_file()
    assert not explicit_netrc.exists()

    # Cover CLI main() success (using empty requirements file -> 0 network calls) and failure
    empty_req = tmp_path / "empty.txt"
    empty_req.write_text(f"pydantic>=2.0.0{_NL}", encoding="utf-8")
    cli_dest = tmp_path / "cli_dest"
    assert main(["--requirements", str(empty_req), "--dest-dir", str(cli_dest)]) == 0
    assert (cli_dest / "uv-overrides.txt").is_file()

    assert main(["--requirements", str(tmp_path / "missing_req.txt"), "--dest-dir", str(cli_dest)]) == 1

    # Cover __main__ entry point via runpy
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "prefetch_github_wheels.py"
    monkeypatch.setattr(
        "sys.argv",
        [str(script_path), "--requirements", str(empty_req), "--dest-dir", str(tmp_path / "runpy_dest")],
    )
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(script_path), run_name="__main__")
    assert exc_info.value.code == 0
