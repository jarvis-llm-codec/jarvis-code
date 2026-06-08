from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx


USER_AGENT = "JARVIS-Code/1.01 (+https://github.com/jarvis-llm-codec/jarvis-code)"
DEFAULT_TIMEOUT_SEC = 10.0
DEFAULT_MAX_CHARS = 12_000
MAX_FETCH_BYTES = 2_000_000
PRIVATE_HOST_ERROR = "Private, loopback, and link-local hosts are blocked for this tool."

OFFICIAL_DOC_HINT_DOMAINS = [
    "docs.python.org",
    "developer.mozilla.org",
    "nodejs.org",
    "react.dev",
    "vite.dev",
    "nextjs.org",
    "typescriptlang.org",
    "docs.npmjs.com",
    "docs.github.com",
    "learn.microsoft.com",
    "fastapi.tiangolo.com",
    "docs.pydantic.dev",
    "playwright.dev",
    "docs.anthropic.com",
    "platform.openai.com",
    "developers.openai.com",
]


class _ReadableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._title_capture = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.meta_description = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template", "svg", "canvas"}:
            self._skip_depth += 1
            return
        if tag == "title":
            self._title_capture = True
        if tag == "meta":
            attr_map = {name.lower(): value or "" for name, value in attrs}
            name = attr_map.get("name", "").lower()
            prop = attr_map.get("property", "").lower()
            if name == "description" or prop == "og:description":
                self.meta_description = attr_map.get("content", "").strip()
        if tag in {"p", "div", "section", "article", "header", "footer", "main", "br", "li", "tr"}:
            self.text_parts.append("\n")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.text_parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template", "svg", "canvas"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._title_capture = False
        if tag in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = data.strip()
        if not text:
            return
        if self._title_capture:
            self.title_parts.append(text)
            return
        self.text_parts.append(text)
        self.text_parts.append(" ")

    @property
    def title(self) -> str:
        return _clean_text(" ".join(self.title_parts), max_chars=500)

    @property
    def text(self) -> str:
        return _clean_text(" ".join(self.text_parts), max_chars=DEFAULT_MAX_CHARS)


@dataclass
class FetchResult:
    ok: bool
    data: dict[str, Any]


def web_fetch(url: str, *, max_chars: int = DEFAULT_MAX_CHARS, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> dict[str, Any]:
    url = str(url or "").strip()
    max_chars = max(500, min(int(max_chars or DEFAULT_MAX_CHARS), 50_000))
    timeout_sec = max(1.0, min(float(timeout_sec or DEFAULT_TIMEOUT_SEC), 30.0))
    validation = _validate_public_url(url)
    if validation:
        return validation

    try:
        response = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain,application/json,*/*"},
            follow_redirects=True,
            timeout=httpx.Timeout(timeout_sec),
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"web_fetch failed: {exc}", "url": url}

    content = response.content[:MAX_FETCH_BYTES]
    content_type = response.headers.get("content-type", "")
    encoding = response.encoding or "utf-8"
    raw_text = content.decode(encoding, errors="replace")
    parsed = _extract_readable_text(raw_text, content_type, max_chars=max_chars)

    return {
        "ok": 200 <= response.status_code < 400,
        "url": url,
        "final_url": str(response.url),
        "status_code": response.status_code,
        "content_type": content_type,
        "title": parsed.get("title", ""),
        "description": parsed.get("description", ""),
        "text": _clean_text(str(parsed.get("text", "")), max_chars=max_chars),
        "truncated": len(response.content) > len(content) or len(str(parsed.get("text", ""))) > max_chars,
        "fetched_at": datetime.now(UTC).isoformat(),
    }


def docs_search(
    query: str,
    *,
    search_handler,
    domains: list[str] | None = None,
    top_k: int = 5,
    fetch_top: int = 0,
    max_chars: int = 4_000,
) -> dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        return {"ok": False, "error": "query is required"}
    domains = [_normalize_domain(domain) for domain in domains or [] if _normalize_domain(domain)]
    top_k = max(1, min(int(top_k or 5), 10))
    fetch_top = max(0, min(int(fetch_top or 0), top_k, 3))
    max_chars = max(500, min(int(max_chars or 4_000), 12_000))

    search_query = _docs_search_query(query, domains)
    search = search_handler(search_query, top_k=max(top_k, 8 if domains else top_k))
    if not search.get("ok"):
        return {**search, "query": query, "search_query": search_query, "domains": domains}

    results = search.get("results") or []
    if domains:
        results = [result for result in results if _url_matches_domains(str(result.get("url") or ""), domains)]
    results = results[:top_k]

    enriched: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        item = dict(result)
        if index < fetch_top and item.get("url"):
            fetched = web_fetch(str(item["url"]), max_chars=max_chars)
            item["fetch"] = fetched
        enriched.append(item)

    return {
        "ok": True,
        "provider": search.get("provider", "brave"),
        "query": query,
        "search_query": search_query,
        "domains": domains,
        "results": enriched,
    }


def package_info(ecosystem: str, package: str, *, include_release_notes: bool = False) -> dict[str, Any]:
    ecosystem = str(ecosystem or "").strip().lower()
    package = str(package or "").strip()
    if not package:
        return {"ok": False, "error": "package is required"}
    if ecosystem == "npm":
        return _npm_info(package)
    if ecosystem == "pypi":
        return _pypi_info(package)
    if ecosystem == "github":
        return _github_info(package, include_release_notes=include_release_notes)
    return {"ok": False, "error": "ecosystem must be one of: npm, pypi, github"}


def browser_check(
    url: str,
    *,
    wait_ms: int = 1000,
    screenshot: bool = False,
    timeout_sec: float = 15.0,
    max_chars: int = 4_000,
) -> dict[str, Any]:
    url = str(url or "").strip()
    validation = _validate_browser_url(url)
    if validation:
        return validation
    wait_ms = max(0, min(int(wait_ms or 1000), 10_000))
    timeout_sec = max(2.0, min(float(timeout_sec or 15.0), 45.0))
    max_chars = max(500, min(int(max_chars or 4_000), 12_000))

    http_result = _http_smoke(url, timeout_sec=timeout_sec, max_chars=max_chars)
    browser = _find_browser_executable()
    if not browser:
        return {
            "ok": bool(http_result.get("ok")),
            "url": url,
            "mode": "http_smoke",
            "browser": None,
            "http": http_result,
            "warnings": ["No local Chrome/Edge/Chromium executable found; skipped headless browser render."],
        }

    with tempfile.TemporaryDirectory(prefix="jarvis-browser-check-") as tmp:
        tmp_path = Path(tmp)
        screenshot_path = tmp_path / "screenshot.png"
        args = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--disable-dev-shm-usage",
            f"--virtual-time-budget={wait_ms}",
        ]
        if screenshot:
            args.append(f"--screenshot={screenshot_path}")
        args.extend(["--dump-dom", url])
        try:
            proc = subprocess.run(
                args,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": bool(http_result.get("ok")),
                "url": url,
                "mode": "http_smoke",
                "browser": browser,
                "http": http_result,
                "warnings": [f"Headless browser check failed: {exc}"],
            }
        dom = _clean_text(proc.stdout or "", max_chars=max_chars)
        browser_result: dict[str, Any] = {
            "returncode": proc.returncode,
            "dom_chars": len(proc.stdout or ""),
            "dom_preview": dom,
            "stderr": _clean_text(proc.stderr or "", max_chars=2_000),
        }
        if screenshot and screenshot_path.exists():
            browser_result["screenshot_bytes"] = screenshot_path.stat().st_size

    ok = bool(http_result.get("ok")) and proc.returncode == 0 and len(proc.stdout or "") > 0
    return {
        "ok": ok,
        "url": url,
        "mode": "headless_browser",
        "browser": browser,
        "http": http_result,
        "browser_result": browser_result,
    }


def _http_smoke(url: str, *, timeout_sec: float, max_chars: int) -> dict[str, Any]:
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,*/*"},
            follow_redirects=True,
            timeout=httpx.Timeout(timeout_sec),
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "url": url}
    parsed = _extract_readable_text(response.text, response.headers.get("content-type", ""), max_chars=max_chars)
    return {
        "ok": 200 <= response.status_code < 400,
        "status_code": response.status_code,
        "final_url": str(response.url),
        "content_type": response.headers.get("content-type", ""),
        "title": parsed.get("title", ""),
        "text_preview": parsed.get("text", "")[:max_chars],
    }


def _npm_info(package: str) -> dict[str, Any]:
    encoded = quote(package, safe="@")
    data = _json_get(f"https://registry.npmjs.org/{encoded}")
    if not data.ok:
        return data.data
    body = data.data
    latest = str((body.get("dist-tags") or {}).get("latest") or "")
    latest_meta = (body.get("versions") or {}).get(latest) or {}
    times = body.get("time") or {}
    return {
        "ok": True,
        "ecosystem": "npm",
        "package": package,
        "latest": latest,
        "description": body.get("description") or latest_meta.get("description"),
        "license": latest_meta.get("license") or body.get("license"),
        "homepage": latest_meta.get("homepage") or body.get("homepage"),
        "repository": latest_meta.get("repository") or body.get("repository"),
        "deprecated": latest_meta.get("deprecated"),
        "latest_published_at": times.get(latest),
        "dist_tags": body.get("dist-tags") or {},
    }


def _pypi_info(package: str) -> dict[str, Any]:
    data = _json_get(f"https://pypi.org/pypi/{quote(package, safe='')}/json")
    if not data.ok:
        return data.data
    body = data.data
    info = body.get("info") or {}
    version = str(info.get("version") or "")
    releases = body.get("releases") or {}
    latest_files = releases.get(version) or []
    latest_published_at = None
    if latest_files:
        latest_published_at = latest_files[0].get("upload_time_iso_8601") or latest_files[0].get("upload_time")
    return {
        "ok": True,
        "ecosystem": "pypi",
        "package": package,
        "latest": version,
        "summary": info.get("summary"),
        "home_page": info.get("home_page"),
        "project_urls": info.get("project_urls") or {},
        "license": info.get("license"),
        "requires_python": info.get("requires_python"),
        "latest_published_at": latest_published_at,
    }


def _github_info(repo: str, *, include_release_notes: bool) -> dict[str, Any]:
    owner_repo = _parse_github_repo(repo)
    if not owner_repo:
        return {"ok": False, "error": "github package must be owner/repo or a GitHub repo URL"}
    owner, name = owner_repo
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    repo_data = _json_get(f"https://api.github.com/repos/{owner}/{name}", headers=headers)
    if not repo_data.ok:
        return repo_data.data
    release_data = _json_get(f"https://api.github.com/repos/{owner}/{name}/releases/latest", headers=headers)
    latest_release = None
    if release_data.ok:
        release = release_data.data
        latest_release = {
            "tag_name": release.get("tag_name"),
            "name": release.get("name"),
            "published_at": release.get("published_at"),
            "html_url": release.get("html_url"),
        }
        if include_release_notes:
            latest_release["body"] = _clean_text(str(release.get("body") or ""), max_chars=8_000)
    return {
        "ok": True,
        "ecosystem": "github",
        "package": f"{owner}/{name}",
        "default_branch": repo_data.data.get("default_branch"),
        "description": repo_data.data.get("description"),
        "html_url": repo_data.data.get("html_url"),
        "license": (repo_data.data.get("license") or {}).get("spdx_id"),
        "latest_release": latest_release,
        "release_error": None if release_data.ok else release_data.data.get("error"),
    }


def _json_get(url: str, *, headers: dict[str, str] | None = None) -> FetchResult:
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})},
            follow_redirects=True,
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_SEC),
        )
        if response.status_code >= 400:
            return FetchResult(False, {"ok": False, "error": f"HTTP {response.status_code}", "url": url})
        return FetchResult(True, response.json())
    except Exception as exc:  # noqa: BLE001
        return FetchResult(False, {"ok": False, "error": str(exc), "url": url})


def _extract_readable_text(raw_text: str, content_type: str, *, max_chars: int) -> dict[str, str]:
    content_type = content_type.lower()
    if "json" in content_type:
        try:
            parsed = json.loads(raw_text)
            return {"title": "", "description": "", "text": _clean_text(json.dumps(parsed, ensure_ascii=False, indent=2), max_chars=max_chars)}
        except Exception:
            return {"title": "", "description": "", "text": _clean_text(raw_text, max_chars=max_chars)}
    if "html" not in content_type and "<html" not in raw_text[:1000].lower():
        return {"title": "", "description": "", "text": _clean_text(raw_text, max_chars=max_chars)}
    parser = _ReadableHtmlParser()
    try:
        parser.feed(raw_text)
    except Exception:
        return {"title": "", "description": "", "text": _clean_text(re.sub(r"<[^>]+>", " ", raw_text), max_chars=max_chars)}
    return {
        "title": parser.title,
        "description": _clean_text(parser.meta_description, max_chars=1_000),
        "text": _clean_text(parser.text, max_chars=max_chars),
    }


def _clean_text(text: str, *, max_chars: int) -> str:
    cleaned = re.sub(r"[ \t\f\v]+", " ", str(text or ""))
    cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > max_chars:
        return cleaned[: max(0, max_chars - 16)].rstrip() + "\n...[truncated]"
    return cleaned


def _validate_public_url(url: str) -> dict[str, Any] | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {"ok": False, "error": "url must be an absolute http(s) URL", "url": url}
    if os.environ.get("JARVIS_WEB_FETCH_ALLOW_PRIVATE", "0") != "1" and _host_is_private(parsed.hostname or ""):
        return {"ok": False, "error": PRIVATE_HOST_ERROR, "url": url}
    return None


def _validate_browser_url(url: str) -> dict[str, Any] | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {"ok": False, "error": "url must be an absolute http(s) URL", "url": url}
    host = parsed.hostname or ""
    if _host_is_private(host) and not _is_loopback_host(host) and os.environ.get("JARVIS_BROWSER_CHECK_ALLOW_PRIVATE", "0") != "1":
        return {"ok": False, "error": "browser_check allows public and loopback URLs by default; set JARVIS_BROWSER_CHECK_ALLOW_PRIVATE=1 for LAN/private hosts.", "url": url}
    return None


def _host_is_private(host: str) -> bool:
    host = host.strip().strip("[]").lower()
    if not host:
        return True
    if _is_loopback_host(host) or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return True
    return False


def _is_loopback_host(host: str) -> bool:
    host = host.strip().strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _normalize_domain(domain: str) -> str:
    domain = str(domain or "").strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = domain.split("/")[0].strip()
    return domain[4:] if domain.startswith("www.") else domain


def _docs_search_query(query: str, domains: list[str]) -> str:
    if domains:
        sites = " OR ".join(f"site:{domain}" for domain in domains[:8])
        return f"{query} ({sites})"
    hint_sites = " OR ".join(f"site:{domain}" for domain in OFFICIAL_DOC_HINT_DOMAINS[:10])
    return f"{query} official documentation ({hint_sites})"


def _url_matches_domains(url: str, domains: list[str]) -> bool:
    host = _normalize_domain(urlparse(url).hostname or "")
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _parse_github_repo(value: str) -> tuple[str, str] | None:
    value = value.strip()
    if value.startswith("http"):
        parsed = urlparse(value)
        if parsed.hostname not in {"github.com", "www.github.com"}:
            return None
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            return None
        return parts[0], parts[1].removesuffix(".git")
    match = re.match(r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$", value)
    if not match:
        return None
    return match.group(1), match.group(2).removesuffix(".git")


def _find_browser_executable() -> str | None:
    env_path = os.environ.get("JARVIS_BROWSER_PATH", "").strip()
    if env_path and Path(env_path).exists():
        return env_path

    candidates: list[Path] = []
    if os.name == "nt":
        for root in [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LocalAppData")]:
            if not root:
                continue
            candidates.extend(
                [
                    Path(root) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
                    Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe",
                    Path(root) / "Chromium" / "Application" / "chrome.exe",
                ]
            )
    candidates.extend(
        [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
            Path("/usr/bin/google-chrome"),
            Path("/usr/bin/chromium"),
            Path("/usr/bin/chromium-browser"),
            Path("/usr/bin/microsoft-edge"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    for name in ["msedge", "chrome", "google-chrome", "chromium", "chromium-browser", "microsoft-edge"]:
        resolved = shutil_which(name)
        if resolved:
            return resolved
    return None


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)
