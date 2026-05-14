import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    from .openai_module import generate_email
except ImportError:
    from openai_module import generate_email

REQUEST_TIMEOUT = 15
MIN_ARTICLE_LENGTH = 500

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _validate_url(url: str) -> None:
    """Ensure URL is a non-empty http(s) URL with a host."""
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")
    stripped = url.strip()
    if not stripped:
        raise ValueError("URL must be a non-empty string.")
    parsed = urlparse(stripped)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must use the http or https scheme.")
    if not parsed.netloc:
        raise ValueError("URL must include a domain (host).")


def _fetch_html(url: str) -> str:
    """Download HTML for the given URL."""
    headers = {"User-Agent": _USER_AGENT}
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise RuntimeError(
            f"The server did not respond within {REQUEST_TIMEOUT} seconds."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to load the page: {exc}") from exc
    return response.text


def _extract_article_text(html: str) -> str:
    """Parse HTML and return article body text with basic whitespace cleanup."""
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in (
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
    ):
        for element in soup.find_all(tag_name):
            element.decompose()

    container = soup.find("article") or soup.find("main") or soup.find("body")
    if container is None:
        container = soup

    raw = container.get_text(separator="\n", strip=False)
    lines: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r"\s+", " ", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _clean_text(text: str) -> str:
    """Normalize whitespace, drop short lines, drop duplicate lines (order preserved)."""
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    seen: set[str] = set()
    result: list[str] = []
    for line in normalized.split("\n"):
        line = line.strip()
        line = re.sub(r"\s+", " ", line)
        if len(line) < 3:
            continue
        if line in seen:
            continue
        seen.add(line)
        result.append(line)
    return "\n".join(result)


def run(url: str) -> str:
    """Fetch article by URL, extract text, then return generated email newsletter."""
    _validate_url(url)
    fetch_url = url.strip()
    html = _fetch_html(fetch_url)
    extracted = _extract_article_text(html)
    cleaned = _clean_text(extracted)
    if len(cleaned) < MIN_ARTICLE_LENGTH:
        raise ValueError(
            f"Extracted text is too short ({len(cleaned)} characters). "
            f"At least {MIN_ARTICLE_LENGTH} characters are required."
        )
    return generate_email(cleaned)
