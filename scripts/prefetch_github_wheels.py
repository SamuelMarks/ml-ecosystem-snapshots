"""Prefetch GitHub Release wheel dependencies via the authenticated GitHub REST API.

GitHub Actions runners scheduled on shared cloud NAT pools (such as AWS `us-west-2`
Hosted Compute runners) can receive `HTTP 403 Forbidden` from `github.com` when `uv`
or `pip` performs unauthenticated web downloads against
`https://github.com/<owner>/<repo>/releases/download/<tag>/<wheel>.whl`.

This pure-stdlib bootstrap script runs prior to package installation in CI:
1. Parses PEP 508 direct GitHub Release `.whl` dependencies from `requirements*.txt`.
2. Resolves and downloads each wheel asset via `https://api.github.com/repos/...`
   using `Bearer $GITHUB_TOKEN` (stripping `Authorization` headers across cross-origin
   CDN/blob redirects) with exponential-backoff retries and ZIP archive verification.
3. Configures `~/.netrc` (`0600`) for `github.com` and `api.github.com` so fallback
   `uv` (`AuthMiddleware`) and `pip` requests also authenticate cleanly.
4. Emits a `uv-overrides.txt` file mapping each package to its local `file://` wheel URI
   and exports `UV_OVERRIDE` and `GH_WHEEL_PREFETCH_DIR` to `$GITHUB_ENV`.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, TYPE_CHECKING, Final

if TYPE_CHECKING:
    from typing import TypeAlias

    JsonScalar: TypeAlias = str | int | float | bool | None
    JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
    FetchBytesFn: TypeAlias = Callable[[urllib.request.Request, float], bytes]

_GITHUB_RELEASE_WHEEL_RE: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z0-9_.-]+)\s*@\s*"
    r"(https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/releases/download/([^/\s]+)/([^/\s]+\.whl))$"
)

_USER_AGENT: Final[str] = "ml-ecosystem-snapshots-ci-wheel-prefetch/1.0"
_GITHUB_API_VERSION: Final[str] = "2022-11-28"
_NL: Final[str] = chr(10)


@dataclass(frozen=True)
class GitHubReleaseWheelSpec:
    """Immutable specification of a PEP 508 GitHub Release wheel dependency.

    Attributes:
        package_name: Distribution name declared on the left-hand side of `@`.
        browser_url: Original `https://github.com/.../releases/download/...` wheel URL.
        owner: GitHub repository owner or organization name.
        repo: GitHub repository name.
        tag: Git release tag hosting the wheel asset.
        asset_name: Exact `.whl` asset filename attached to the release.
    """

    package_name: str
    browser_url: str
    owner: str
    repo: str
    tag: str
    asset_name: str


@dataclass(frozen=True)
class PrefetchedWheel:
    """Verified local wheel artifact paired with its `uv` override directive.

    Attributes:
        spec: Parsed upstream GitHub Release wheel specification.
        local_path: Resolved filesystem path to the verified `.whl` archive.
        override_line: PEP 508 `pkg @ file:///...` line for `UV_OVERRIDE`.
    """

    spec: GitHubReleaseWheelSpec
    local_path: Path
    override_line: str


class CrossOriginAuthStripRedirectHandler(urllib.request.HTTPRedirectHandler):
    """HTTP redirect handler that removes `Authorization` headers across host boundaries."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Construct a redirected HTTP request while stripping cross-origin credentials.

        GitHub's REST API and release endpoints redirect asset downloads (`302 Found`)
        to pre-signed `githubusercontent.com` blob URLs that reject requests carrying
        an `Authorization` header alongside signed query parameters.

        Args:
            req: Original outbound HTTP request.
            fp: Underlying HTTP response byte stream.
            code: HTTP 3xx redirect status code.
            msg: HTTP status reason phrase.
            headers: HTTP response headers containing the `Location` target.
            newurl: Target URL for the redirected request.

        Returns:
            Redirected `urllib.request.Request` instance, or `None` when not redirected.
        """
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and req.host != redirected.host:
            redirected.headers.pop("Authorization", None)
            redirected.unredirected_hdrs.pop("Authorization", None)
        return redirected


def parse_requirement_line(line: str) -> GitHubReleaseWheelSpec | None:
    """Parse a single `requirements.txt` line into a `GitHubReleaseWheelSpec` if applicable.

    Args:
        line: Raw requirement line (optionally including `#` comments or `;` markers).

    Returns:
        Parsed `GitHubReleaseWheelSpec` when `line` pins a GitHub Release `.whl`, else `None`.
    """
    cleaned = line.split("#", 1)[0].split(";", 1)[0].strip()
    match = _GITHUB_RELEASE_WHEEL_RE.match(cleaned)
    if match is None:
        return None
    return GitHubReleaseWheelSpec(
        package_name=match.group(1),
        browser_url=match.group(2),
        owner=match.group(3),
        repo=match.group(4),
        tag=match.group(5),
        asset_name=match.group(6),
    )


def collect_wheel_specs(requirement_files: Sequence[Path]) -> tuple[GitHubReleaseWheelSpec, ...]:
    """Collect deduplicated GitHub Release wheel specifications across requirement files.

    Args:
        requirement_files: Ordered sequence of requirement file paths to inspect.

    Returns:
        Immutable tuple of unique `GitHubReleaseWheelSpec` entries in discovery order.
    """
    seen: set[tuple[str, str]] = set()
    specs: list[GitHubReleaseWheelSpec] = []
    for req_path in requirement_files:
        for raw_line in req_path.read_text(encoding="utf-8").splitlines():
            spec = parse_requirement_line(raw_line)
            if spec is not None:
                dedup_key = (spec.package_name.lower(), spec.browser_url)
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    specs.append(spec)
    return tuple(specs)


def extract_asset_api_url(release_payload: JsonValue, asset_name: str) -> str:
    """Extract the REST API asset URL for `asset_name` from a GitHub Release JSON payload.

    Args:
        release_payload: Decoded JSON payload from `/repos/{owner}/{repo}/releases/tags/{tag}`.
        asset_name: Exact wheel filename to locate inside the `assets` array.

    Returns:
        API asset endpoint URL (`https://api.github.com/repos/.../releases/assets/<id>`).

    Raises:
        ValueError: If `release_payload` is malformed or does not contain `asset_name`.
    """
    if not isinstance(release_payload, dict):
        raise ValueError("GitHub release metadata response must be a JSON object.")
    assets = release_payload.get("assets")
    if not isinstance(assets, list):
        raise ValueError("GitHub release metadata response is missing an 'assets' list.")
    for item in assets:
        if isinstance(item, dict) and item.get("name") == asset_name:
            api_url = item.get("url")
            if isinstance(api_url, str) and api_url:
                return api_url
    raise ValueError(f"Release asset '{asset_name}' not found in GitHub release metadata.")


def render_netrc_block(token: str) -> str:
    """Format a `.netrc` stanza authenticating `github.com` and `api.github.com`.

    Args:
        token: GitHub personal access or installation token.

    Returns:
        Formatted multi-line `.netrc` text block.
    """
    lines = (
        "machine github.com",
        "login x-access-token",
        f"password {token}",
        "machine api.github.com",
        "login x-access-token",
        f"password {token}",
        "",
    )
    return _NL.join(lines)


def ensure_github_netrc(token: str, netrc_path: Path) -> bool:
    """Ensure `netrc_path` contains `github.com` credentials with `0600` permissions.

    Args:
        token: Non-empty GitHub authentication token.
        netrc_path: Target `.netrc` file path (typically `Path.home() / ".netrc"`).

    Returns:
        `True` if credentials were written, or `False` if `machine github.com` already existed.
    """
    existing = netrc_path.read_text(encoding="utf-8") if netrc_path.exists() else ""
    if "machine github.com" in existing:
        return False
    prefix = f"{existing.rstrip()}{_NL}" if existing.strip() else ""
    netrc_path.parent.mkdir(parents=True, exist_ok=True)
    netrc_path.write_text(f"{prefix}{render_netrc_block(token)}", encoding="utf-8")
    os.chmod(netrc_path, 0o600)
    return True


def render_uv_overrides(wheels: Sequence[PrefetchedWheel]) -> str:
    """Render PEP 508 local `file://` override lines for `uv pip install --overrides`.

    Args:
        wheels: Sequence of prefetched wheel descriptors.

    Returns:
        Newline-terminated requirements override file content.
    """
    if not wheels:
        return ""
    return _NL.join(wheel.override_line for wheel in wheels) + _NL


def resolve_github_token(env: Mapping[str, str]) -> str | None:
    """Resolve a non-empty GitHub token from `GITHUB_TOKEN` or `GH_TOKEN`.

    Args:
        env: Environment variable mapping to inspect.

    Returns:
        Trimmed token string if available, otherwise `None`.
    """
    for key in ("GITHUB_TOKEN", "GH_TOKEN"):
        candidate = env.get(key, "").strip()
        if candidate:
            return candidate
    return None


def build_redirect_safe_opener() -> urllib.request.OpenerDirector:
    """Construct a `urllib` opener with cross-origin `Authorization` stripping enabled.

    Returns:
        Configured `urllib.request.OpenerDirector` instance.
    """
    return urllib.request.build_opener(CrossOriginAuthStripRedirectHandler())


def default_fetch_bytes(request: urllib.request.Request, timeout: float) -> bytes:
    """Execute `request` using `build_redirect_safe_opener` and return raw body bytes.

    Args:
        request: Outbound HTTP request object.
        timeout: Socket timeout in seconds.

    Returns:
        Raw HTTP response payload bytes.
    """
    opener = build_redirect_safe_opener()
    with opener.open(request, timeout=timeout) as response:
        return bytes(response.read())


def fetch_bytes_with_retry(
    request: urllib.request.Request,
    fetch_fn: FetchBytesFn = default_fetch_bytes,
    timeout: float = 60.0,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bytes:
    """Fetch raw HTTP bytes with bounded exponential-backoff retry on transient errors.

    Args:
        request: Outbound `urllib.request.Request` to execute.
        fetch_fn: Callable performing a single HTTP attempt.
        timeout: Per-attempt timeout in seconds.
        max_attempts: Maximum number of attempts (>= 1).
        base_delay: Initial backoff delay in seconds before doubling.
        sleep_fn: Delay function invoked between failed attempts.

    Returns:
        Downloaded response body bytes.

    Raises:
        ValueError: If `max_attempts` is less than 1.
        RuntimeError: If all attempts fail with `URLError`, `TimeoutError`, or `OSError`.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    attempt = 1
    while True:
        try:
            return fetch_fn(request, timeout)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"Failed to download '{request.full_url}' after {max_attempts} attempt(s): {exc}"
                ) from exc
            sleep_fn(base_delay * float(2 ** (attempt - 1)))
            attempt += 1


def download_wheel_spec(
    spec: GitHubReleaseWheelSpec,
    dest_dir: Path,
    token: str | None,
    fetch_fn: FetchBytesFn = default_fetch_bytes,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> PrefetchedWheel:
    """Download and ZIP-verify a single GitHub Release wheel into `dest_dir`.

    When `token` is provided, resolves the release tag via `api.github.com` and downloads
    the binary wheel asset with `Accept: application/octet-stream` so requests use
    authenticated REST API quotas instead of unauthenticated `github.com` web limits.

    Args:
        spec: Parsed GitHub Release wheel specification.
        dest_dir: Local directory in which to store the verified `.whl` file.
        token: Optional GitHub token for authenticated REST API downloads.
        fetch_fn: HTTP byte-fetching callable.
        sleep_fn: Backoff sleep callable passed to `fetch_bytes_with_retry`.

    Returns:
        `PrefetchedWheel` pointing to the verified wheel archive on disk.

    Raises:
        RuntimeError: If HTTP metadata or asset download fails across all retries.
        ValueError: If release JSON is invalid or the downloaded bytes are not a valid ZIP archive.
    """
    if token is not None:
        meta_url = f"https://api.github.com/repos/{spec.owner}/{spec.repo}/releases/tags/{spec.tag}"
        meta_req = urllib.request.Request(
            meta_url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": _GITHUB_API_VERSION,
                "User-Agent": _USER_AGENT,
            },
        )
        raw_meta = fetch_bytes_with_retry(meta_req, fetch_fn=fetch_fn, sleep_fn=sleep_fn)
        release_payload: JsonValue = json.loads(raw_meta.decode("utf-8"))
        asset_api_url = extract_asset_api_url(release_payload, spec.asset_name)
        wheel_req = urllib.request.Request(
            asset_api_url,
            headers={
                "Accept": "application/octet-stream",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": _GITHUB_API_VERSION,
                "User-Agent": _USER_AGENT,
            },
        )
    else:
        wheel_req = urllib.request.Request(
            spec.browser_url,
            headers={
                "Accept": "application/octet-stream",
                "User-Agent": _USER_AGENT,
            },
        )

    wheel_bytes = fetch_bytes_with_retry(wheel_req, fetch_fn=fetch_fn, sleep_fn=sleep_fn)
    if not zipfile.is_zipfile(io.BytesIO(wheel_bytes)):
        raise ValueError(f"Downloaded payload for '{spec.asset_name}' is not a valid wheel ZIP archive.")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = (dest_dir / spec.asset_name).resolve()
    dest_path.write_bytes(wheel_bytes)
    return PrefetchedWheel(
        spec=spec,
        local_path=dest_path,
        override_line=f"{spec.package_name} @ {dest_path.as_uri()}",
    )


def prefetch_all_wheels(
    requirement_files: Sequence[Path],
    dest_dir: Path,
    env: Mapping[str, str] | None = None,
    netrc_path: Path | None = None,
    fetch_fn: FetchBytesFn = default_fetch_bytes,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Path:
    """Prefetch all GitHub Release wheels, write `uv-overrides.txt`, and wire CI environment.

    Args:
        requirement_files: Requirement files to scan for GitHub Release wheel URLs.
        dest_dir: Output directory for downloaded `.whl` archives and `uv-overrides.txt`.
        env: Optional environment mapping (defaults to `os.environ`).
        netrc_path: Optional target `.netrc` path (defaults to `Path.home() / ".netrc"`).
        fetch_fn: HTTP byte-fetching callable.
        sleep_fn: Backoff sleep callable.

    Returns:
        Resolved path to the generated `uv-overrides.txt` file.
    """
    active_env: Mapping[str, str] = env if env is not None else os.environ
    token = resolve_github_token(active_env)
    if token is not None:
        target_netrc = netrc_path if netrc_path is not None else (Path.home() / ".netrc")
        ensure_github_netrc(token, target_netrc)

    specs = collect_wheel_specs(requirement_files)
    prefetched = [
        download_wheel_spec(spec, dest_dir=dest_dir, token=token, fetch_fn=fetch_fn, sleep_fn=sleep_fn)
        for spec in specs
    ]

    dest_dir.mkdir(parents=True, exist_ok=True)
    overrides_path = (dest_dir / "uv-overrides.txt").resolve()
    overrides_path.write_text(render_uv_overrides(prefetched), encoding="utf-8")

    github_env_raw = active_env.get("GITHUB_ENV", "").strip()
    if github_env_raw:
        github_env_path = Path(github_env_raw)
        github_env_path.parent.mkdir(parents=True, exist_ok=True)
        with github_env_path.open("a", encoding="utf-8") as env_handle:
            env_handle.write(f"UV_OVERRIDE={overrides_path}{_NL}")
            env_handle.write(f"GH_WHEEL_PREFETCH_DIR={dest_dir.resolve()}{_NL}")

    return overrides_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for prefetching GitHub Release wheels in CI workflows.

    Args:
        argv: Optional sequence of CLI arguments (defaults to `sys.argv[1:]`).

    Returns:
        Exit status `0` on success, or `1` if wheel download or validation fails.
    """
    default_temp_root = os.environ.get("RUNNER_TEMP", "").strip() or tempfile.gettempdir()
    parser = argparse.ArgumentParser(
        description="Prefetch GitHub Release wheel dependencies via authenticated GitHub REST API."
    )
    parser.add_argument(
        "--requirements",
        nargs="+",
        default=["requirements.txt", "requirements-gen.txt"],
        help="Requirement files to scan for GitHub Release .whl URLs.",
    )
    parser.add_argument(
        "--dest-dir",
        default=os.path.join(default_temp_root, "ml-snapshots-gh-wheels"),
        help="Directory to store downloaded wheels and uv-overrides.txt.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    req_paths = [Path(str(item)) for item in args.requirements]
    dest_dir = Path(str(args.dest_dir))

    try:
        overrides_path = prefetch_all_wheels(requirement_files=req_paths, dest_dir=dest_dir)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"scripts/prefetch_github_wheels.py failed: {exc}", file=sys.stderr)
        return 1

    print(f"Prefetched GitHub Release wheels -> {overrides_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
