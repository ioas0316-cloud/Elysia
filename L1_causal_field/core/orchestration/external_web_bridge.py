r"""
ELYSIA External Web Sensory Bridges (GitHub · YouTube · Public Web/Art)
======================================================================

Pure stdlib external network observers connecting Elysia to the real open web:
- **GitHub (`github.com/{owner}/{repo}`):** Fetches public repository metadata
  and Base64-decoded README causal prose via GitHub REST API.
- **YouTube (`youtube.com` / `youtu.be` / RSS feeds):** Fetches public oEmbed
  metadata (title, channel/author) + public video thumbnail bytes for genuine
  visual perception.
- **Public Web / Art (including `pixiv.net`, blogs, arXiv, wikis):** Fetches
  public HTML/RSS `<title>`, OpenGraph meta tags, and text paragraphs. If an
  origin enforces login walls, HTTP 401/403/429, or anti-bot/CAPTCHA gates,
  **never bypasses** — honestly marks `blocked=True` with `block_reason` so
  Elysia experiences the true boundary friction.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from html.parser import HTMLParser
import io
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


# Callable type: (url, timeout_s, headers) -> (status_code, body_bytes, content_type)
HttpFetcher = Callable[[str, float, Optional[Dict[str, str]]], Tuple[int, bytes, str]]


@dataclass
class ExternalWebObservation:
    """Raw observation returned from a live public URL probe."""

    url: str
    platform: str  # "github" | "youtube" | "web_art"
    title: str
    text_body: str
    code_snippets: List[str] = field(default_factory=list)
    image_bytes: Optional[bytes] = None
    outbound_urls: List[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""
    status_code: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


_ALLOWED_OUTBOUND_HOSTS = (
    "github.com",
    "en.wikipedia.org",
    "ko.wikipedia.org",
    "arxiv.org",
    "docs.python.org",
    "youtube.com",
)
_EXCLUDED_ASSET_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".css",
    ".js",
)


def _extract_safe_outbound_urls(text_or_html: str, max_urls: int = 6) -> List[str]:
    """Extracts deduplicated public knowledge/media outbound URLs without tracking assets."""
    if not text_or_html:
        return []
    candidates = re.findall(r"https://[^\s\"'<>)\]]+", text_or_html)
    seen = set()
    selected: List[str] = []
    for raw_url in candidates:
        cleaned = raw_url.rstrip(".,;:!?\"'")
        parsed = urlparse(cleaned)
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
        if not host:
            continue
        if not any(host == dom or host.endswith("." + dom) for dom in _ALLOWED_OUTBOUND_HOSTS):
            continue
        if host.startswith("api."):
            continue
        if not path or path == "/":
            continue
        lower_path = path.lower()
        if any(lower_path.endswith(ext) for ext in _EXCLUDED_ASSET_SUFFIXES):
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        selected.append(cleaned)
        if len(selected) >= max_urls:
            break
    return selected


_ANTIBOT_MARKERS = (
    "captcha",
    "recaptcha",
    "hcaptcha",
    "cf-browser-verification",
    "cloudflare",
    "verify you are human",
    "just a moment",
    "access denied",
    "login required",
    "sign in to continue",
)


def default_http_get(
    url: str,
    timeout: float = 8.0,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, bytes, str]:
    """Safe, honest HTTP GET returning `(status_code, body_bytes, content_type)` without crashing."""
    req_headers = {
        "User-Agent": "ElysiaOpenObserver/1.0 (+https://github.com/elysia-causal)",
        "Accept": "application/json, text/html, application/xml, image/*, */*;q=0.8",
    }
    if headers:
        req_headers.update(headers)
    req = Request(url, headers=req_headers, method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            ctype = str(resp.headers.get("Content-Type", ""))
            body = resp.read(2_000_000)
            return status, body, ctype
    except HTTPError as err:
        body = b""
        try:
            body = err.read(64_000)
        except Exception:
            pass
        return int(err.code), body, str(err.headers.get("Content-Type", "")) if err.headers else ""
    except (URLError, TimeoutError, OSError) as exc:
        return 0, str(exc).encode("utf-8", errors="replace"), "error/network"


class _SimpleHTMLInspector(HTMLParser):
    """Lightweight stdlib HTML parser extracting `<title>`, meta og tags, and visible text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_title = False
        self.skip_depth = 0
        self.title_parts: List[str] = []
        self.text_parts: List[str] = []
        self.meta: Dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t == "title":
            self.in_title = True
        elif t in ("script", "style", "noscript"):
            self.skip_depth += 1
        elif t == "meta":
            ad = {k.lower(): (v or "") for k, v in attrs}
            key = (ad.get("property") or ad.get("name") or "").lower()
            val = ad.get("content", "").strip()
            if key and val:
                self.meta[key] = val

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "title":
            self.in_title = False
        elif t in ("script", "style", "noscript") and self.skip_depth > 0:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        s = data.strip()
        if not s:
            return
        if self.in_title:
            self.title_parts.append(s)
        elif self.skip_depth == 0:
            self.text_parts.append(s)


def _extract_code_blocks(text: str) -> List[str]:
    blocks = re.findall(r"```[a-zA-Z0-9_-]*\n(.*?)```", text, flags=re.DOTALL)
    return [b.strip()[:600] for b in blocks if b.strip()][:5]


def fetch_github_target(
    url: str,
    fetcher: Optional[HttpFetcher] = None,
) -> ExternalWebObservation:
    """Fetches public GitHub repository metadata & README via GitHub REST API."""
    get_fn = fetcher or default_http_get
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return ExternalWebObservation(
            url=url,
            platform="github",
            title=url,
            text_body="",
            blocked=True,
            block_reason="Invalid GitHub repository URL (expected /owner/repo)",
            status_code=400,
        )

    owner, repo = parts[0], parts[1].removesuffix(".git")
    api_headers = {"Accept": "application/vnd.github+json"}

    repo_url = f"https://api.github.com/repos/{owner}/{repo}"
    st_code, repo_raw, _ = get_fn(repo_url, 8.0, api_headers)
    if st_code in (401, 403, 429) or st_code == 0 or st_code >= 400:
        reason = f"GitHub API HTTP {st_code}" if st_code else f"Network error: {repo_raw.decode('utf-8', 'replace')[:180]}"
        return ExternalWebObservation(
            url=url,
            platform="github",
            title=f"{owner}/{repo}",
            text_body="",
            blocked=True,
            block_reason=reason,
            status_code=st_code,
        )

    try:
        repo_info = json.loads(repo_raw.decode("utf-8", errors="replace"))
    except Exception as exc:
        return ExternalWebObservation(
            url=url,
            platform="github",
            title=f"{owner}/{repo}",
            text_body="",
            blocked=True,
            block_reason=f"GitHub JSON parse error: {exc}",
            status_code=st_code,
        )

    readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    rm_code, rm_raw, _ = get_fn(readme_url, 8.0, api_headers)
    readme_text = ""
    if rm_code == 200:
        try:
            rm_json = json.loads(rm_raw.decode("utf-8", errors="replace"))
            b64 = str(rm_json.get("content", ""))
            if b64:
                readme_text = base64.b64decode(b64).decode("utf-8", errors="replace")
        except Exception:
            pass

    desc = str(repo_info.get("description") or "")
    lang = str(repo_info.get("language") or "")
    topics = list(repo_info.get("topics") or [])
    full_name = str(repo_info.get("full_name") or f"{owner}/{repo}")
    combined_text = "\n".join(x for x in (desc, f"Primary language: {lang}" if lang else "", readme_text[:3500]) if x).strip()

    return ExternalWebObservation(
        url=url,
        platform="github",
        title=full_name,
        text_body=combined_text,
        code_snippets=_extract_code_blocks(readme_text),
        image_bytes=None,
        outbound_urls=_extract_safe_outbound_urls(readme_text),
        blocked=False,
        block_reason="",
        status_code=st_code,
        metadata={
            "owner": owner,
            "repo": repo,
            "stars": int(repo_info.get("stargazers_count") or 0),
            "language": lang,
            "topics": topics[:10],
        },
    )


def fetch_youtube_target(
    url: str,
    fetcher: Optional[HttpFetcher] = None,
) -> ExternalWebObservation:
    """Fetches public YouTube metadata via oEmbed (or Atom RSS) + thumbnail image bytes."""
    get_fn = fetcher or default_http_get
    parsed = urlparse(url)

    # Support public YouTube Atom feed URLs directly
    if "feeds/videos.xml" in parsed.path:
        st_code, xml_raw, _ = get_fn(url, 8.0, None)
        if st_code != 200:
            return ExternalWebObservation(
                url=url,
                platform="youtube",
                title=url,
                text_body="",
                blocked=True,
                block_reason=f"YouTube feed HTTP {st_code}",
                status_code=st_code,
            )
        root = ET.fromstring(xml_raw.decode("utf-8", errors="replace"))
        titles = [el.text.strip() for el in root.iter() if el.tag.endswith("title") and el.text]
        feed_title = titles[0] if titles else url
        entries_text = "\n".join(titles[1:8])
        return ExternalWebObservation(
            url=url,
            platform="youtube",
            title=feed_title,
            text_body=entries_text,
            blocked=False,
            status_code=st_code,
            metadata={"feed_entries": len(titles[1:])},
        )

    oembed_url = f"https://www.youtube.com/oembed?url={quote(url, safe='')}&format=json"
    st_code, raw, _ = get_fn(oembed_url, 8.0, None)
    if st_code != 200:
        reason = (
            f"YouTube oEmbed HTTP {st_code}"
            if st_code
            else f"Network error: {raw.decode('utf-8', 'replace')[:180]}"
        )
        return ExternalWebObservation(
            url=url,
            platform="youtube",
            title=url,
            text_body="",
            blocked=True,
            block_reason=reason,
            status_code=st_code,
        )

    info = json.loads(raw.decode("utf-8", errors="replace"))
    title = str(info.get("title") or url)
    author = str(info.get("author_name") or "")
    thumb_url = str(info.get("thumbnail_url") or "")

    thumb_bytes: Optional[bytes] = None
    if thumb_url:
        t_code, t_bytes, t_type = get_fn(thumb_url, 8.0, None)
        if t_code == 200 and t_bytes and ("image" in t_type or thumb_url.endswith((".jpg", ".jpeg", ".png", ".webp"))):
            thumb_bytes = t_bytes

    body = "\n".join(x for x in (title, f"Creator: {author}" if author else "", "YouTube audiovisual stream (metadata + thumbnail accessible without full video download)") if x)
    return ExternalWebObservation(
        url=url,
        platform="youtube",
        title=title,
        text_body=body,
        image_bytes=thumb_bytes,
        blocked=False,
        status_code=st_code,
        metadata={
            "author_name": author,
            "provider_name": info.get("provider_name", "YouTube"),
            "thumbnail_url": thumb_url,
            "width": info.get("width"),
            "height": info.get("height"),
            "has_thumbnail_bytes": thumb_bytes is not None,
        },
    )


def fetch_web_or_art_target(
    url: str,
    fetcher: Optional[HttpFetcher] = None,
) -> ExternalWebObservation:
    """
    Fetches public HTML/OpenGraph content (including Pixiv/art/blogs/docs).
    Honors HTTP 401/403/429 and login/CAPTCHA gates immediately with `blocked=True`.
    """
    get_fn = fetcher or default_http_get
    st_code, body, ctype = get_fn(url, 8.0, None)
    text_raw = body.decode("utf-8", errors="replace")
    lower = text_raw.lower()

    if st_code in (401, 403, 429) or st_code == 0 or st_code >= 400:
        reason = (
            f"Access gated (HTTP {st_code})"
            if st_code
            else f"Network unreachable: {text_raw[:180]}"
        )
        return ExternalWebObservation(
            url=url,
            platform="web_art",
            title=url,
            text_body="",
            blocked=True,
            block_reason=reason,
            status_code=st_code,
        )

    if any(marker in lower for marker in _ANTIBOT_MARKERS) and len(text_raw) < 25000:
        return ExternalWebObservation(
            url=url,
            platform="web_art",
            title=url,
            text_body="",
            blocked=True,
            block_reason="Access gated (login/CAPTCHA/anti-bot check detected)",
            status_code=st_code,
        )

    parser = _SimpleHTMLInspector()
    try:
        parser.feed(text_raw)
    except Exception:
        pass

    title = (
        parser.meta.get("og:title")
        or (" ".join(parser.title_parts).strip())
        or url
    )
    desc = parser.meta.get("og:description") or parser.meta.get("description") or ""
    body_lines = [p for p in parser.text_parts if len(p) >= 8][:30]
    combined = "\n".join(x for x in (title, desc, *body_lines) if x)[:3500]

    img_bytes: Optional[bytes] = None
    og_img = parser.meta.get("og:image", "")
    if og_img.startswith(("http://", "https://")):
        ic, ib, itype = get_fn(og_img, 6.0, None)
        if ic == 200 and ib and "image" in itype:
            img_bytes = ib

    return ExternalWebObservation(
        url=url,
        platform="web_art",
        title=title,
        text_body=combined,
        image_bytes=img_bytes,
        outbound_urls=_extract_safe_outbound_urls(text_raw),
        blocked=False,
        status_code=st_code,
        metadata={"content_type": ctype, "og_image": og_img},
    )


def search_github_repositories(
    query: str,
    max_results: int = 3,
    fetcher: Optional[HttpFetcher] = None,
) -> List[str]:
    """Queries public GitHub Search API for high-signal repositories matching a concept."""
    clean_q = str(query or "").strip()
    if not clean_q:
        return []
    get_fn = fetcher or default_http_get
    search_url = (
        f"https://api.github.com/search/repositories?q={quote(clean_q)}"
        f"&sort=stars&per_page={max(1, int(max_results))}"
    )
    st_code, raw_bytes, _ = get_fn(
        search_url,
        8.0,
        {"Accept": "application/vnd.github+json"},
    )
    if st_code != 200 or not raw_bytes:
        return []
    try:
        payload = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    except Exception:
        return []
    urls: List[str] = []
    for item in payload.get("items") or []:
        html_url = str(item.get("html_url") or "").strip()
        if html_url.startswith("https://") and html_url not in urls:
            urls.append(html_url)
        if len(urls) >= max_results:
            break
    return urls


def fetch_wikipedia_concept(
    topic: str,
    lang: str = "en",
    fetcher: Optional[HttpFetcher] = None,
) -> ExternalWebObservation:
    """Fetches structured concept summary & optional thumbnail via Wikipedia REST API."""
    clean_topic = str(topic or "").strip().replace(" ", "_")
    safe_lang = (lang or "en").strip().lower() or "en"
    fallback_url = f"https://{safe_lang}.wikipedia.org/wiki/{quote(clean_topic)}"
    if not clean_topic:
        return ExternalWebObservation(
            url=fallback_url,
            platform="wikipedia",
            title=topic or fallback_url,
            text_body="",
            blocked=True,
            block_reason="Empty Wikipedia concept topic",
            status_code=400,
        )

    get_fn = fetcher or default_http_get
    api_url = f"https://{safe_lang}.wikipedia.org/api/rest_v1/page/summary/{quote(clean_topic)}"
    st_code, raw_bytes, _ = get_fn(
        api_url,
        8.0,
        {"Accept": "application/json"},
    )
    if st_code in (401, 403, 429) or st_code == 0 or st_code >= 400:
        reason = (
            f"Wikipedia REST HTTP {st_code}"
            if st_code
            else f"Network error: {raw_bytes.decode('utf-8', 'replace')[:180]}"
        )
        return ExternalWebObservation(
            url=fallback_url,
            platform="wikipedia",
            title=topic,
            text_body="",
            blocked=True,
            block_reason=reason,
            status_code=st_code,
        )

    try:
        data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        return ExternalWebObservation(
            url=fallback_url,
            platform="wikipedia",
            title=topic,
            text_body="",
            blocked=True,
            block_reason=f"Wikipedia JSON parse error: {exc}",
            status_code=st_code,
        )

    title = str(data.get("title") or topic)
    desc = str(data.get("description") or "")
    extract = str(data.get("extract") or "")
    page_url = str(
        ((data.get("content_urls") or {}).get("desktop") or {}).get("page")
        or fallback_url
    )
    combined = "\n".join(x for x in (title, desc, extract) if x).strip()[:3500]

    img_bytes: Optional[bytes] = None
    thumb_url = str((data.get("thumbnail") or {}).get("source") or "")
    if thumb_url.startswith(("http://", "https://")):
        ic, ib, itype = get_fn(thumb_url, 6.0, None)
        if ic == 200 and ib and "image" in itype:
            img_bytes = ib

    return ExternalWebObservation(
        url=page_url,
        platform="wikipedia",
        title=title,
        text_body=combined,
        image_bytes=img_bytes,
        outbound_urls=_extract_safe_outbound_urls(combined),
        blocked=False,
        status_code=st_code,
        metadata={"lang": safe_lang, "description": desc, "thumbnail": thumb_url},
    )


def _extract_concise_dictionary_summary(text_body: str, max_chars: int = 260) -> str:
    """Extracts a clean 1-2 line definition summary from Wiktionary/Wikipedia plain text."""
    lines: List[str] = []
    for raw_line in (text_body or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if line.startswith("==") or line.endswith("[편집]"):
            continue
        if re.match(r"^(발음|어원|한자|로마자|국어의\s*로마자|IPA)\s*[:：]?", line):
            continue
        line = re.sub(r"^(?:\([^)]+\)\s*)?IPA\([^)]*\)\s*[:：]\s*\[[^\]]+\](?:\s*~\s*\[[^\]]+\])?", "", line).strip()
        line = re.sub(r"발음\s*[:：]\s*\[[^\]]+\]", "", line).strip()
        if not line:
            continue
        # Strip leading numbered sense markers like "1." or "1)"
        cleaned = re.sub(r"^\d+[.)]\s*", "", line).strip()
        if len(cleaned) >= 6:
            lines.append(cleaned)
        if sum(len(x) for x in lines) >= max_chars or len(lines) >= 2:
            break
    joined = " ".join(lines).strip()
    if not joined:
        joined = re.sub(r"\s+", " ", (text_body or "")).strip()
    return joined[:max_chars].strip()


def fetch_dictionary_meaning(
    word: str,
    lang: str = "ko",
    fetcher: Optional[HttpFetcher] = None,
) -> ExternalWebObservation:
    """
    Consults live Wiktionary (`{lang}.wiktionary.org`) for a word's lexical meaning,
    falling back to Wikipedia summary only when Wiktionary lacks a clean entry and
    network connectivity is available.
    """
    clean_word = str(word or "").strip()
    safe_lang = (lang or "ko").strip().lower() or "ko"
    wiktionary_url = f"https://{safe_lang}.wiktionary.org/wiki/{quote(clean_word)}"
    if not clean_word:
        return ExternalWebObservation(
            url=wiktionary_url,
            platform="wiktionary",
            title=word or wiktionary_url,
            text_body="",
            blocked=True,
            block_reason="Empty dictionary headword",
            status_code=400,
        )

    get_fn = fetcher or default_http_get
    api_url = (
        f"https://{safe_lang}.wiktionary.org/w/api.php?action=query&prop=extracts"
        f"&explaintext=1&titles={quote(clean_word)}&format=json"
    )
    st_code, raw_bytes, _ = get_fn(api_url, 8.0, {"Accept": "application/json"})
    if st_code == 0:
        return ExternalWebObservation(
            url=wiktionary_url,
            platform="wiktionary",
            title=clean_word,
            text_body="",
            blocked=True,
            block_reason=f"Network error: {raw_bytes.decode('utf-8', 'replace')[:180]}",
            status_code=0,
        )

    if 200 <= st_code < 400:
        try:
            payload = json.loads(raw_bytes.decode("utf-8", errors="replace"))
            pages = (payload.get("query") or {}).get("pages") or {}
            for page in pages.values():
                if not isinstance(page, dict) or "missing" in page:
                    continue
                extract_txt = str(page.get("extract") or "").strip()
                if len(extract_txt) >= 15:
                    page_title = str(page.get("title") or clean_word)
                    summary = _extract_concise_dictionary_summary(extract_txt)
                    return ExternalWebObservation(
                        url=f"https://{safe_lang}.wiktionary.org/wiki/{quote(page_title)}",
                        platform="wiktionary",
                        title=page_title,
                        text_body=extract_txt[:2500],
                        outbound_urls=_extract_safe_outbound_urls(extract_txt),
                        blocked=False,
                        status_code=st_code,
                        metadata={
                            "lang": safe_lang,
                            "dictionary_source": "wiktionary",
                            "definition_summary": summary,
                        },
                    )
        except Exception:
            pass

    wiki_obs = fetch_wikipedia_concept(clean_word, lang=safe_lang, fetcher=fetcher)
    if not wiki_obs.blocked and wiki_obs.text_body:
        body_lines = [ln.strip() for ln in wiki_obs.text_body.splitlines() if ln.strip()]
        if body_lines and body_lines[0].lower() == wiki_obs.title.lower():
            body_lines = body_lines[1:]
        wiki_summary = _extract_concise_dictionary_summary("\n".join(body_lines) or wiki_obs.text_body)
        meta = dict(wiki_obs.metadata or {})
        meta["dictionary_source"] = "wikipedia_fallback"
        meta["definition_summary"] = wiki_summary
        wiki_obs.metadata = meta
        return wiki_obs

    return ExternalWebObservation(
        url=wiktionary_url,
        platform="wiktionary",
        title=clean_word,
        text_body="",
        blocked=True,
        block_reason=wiki_obs.block_reason or f"Dictionary entry not found (HTTP {st_code})",
        status_code=wiki_obs.status_code or st_code,
    )


def observe_external_url(
    url: str,
    fetcher: Optional[HttpFetcher] = None,
) -> ExternalWebObservation:
    """Platform router dispatching GitHub, YouTube, Wikipedia, or general public web/art targets."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if "github.com" in host:
        return fetch_github_target(url, fetcher=fetcher)
    if "youtube.com" in host or "youtu.be" in host:
        return fetch_youtube_target(url, fetcher=fetcher)
    if "wikipedia.org" in host:
        host_parts = [p for p in host.split(".") if p]
        lang = host_parts[0] if len(host_parts) >= 3 else "en"
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) >= 2 and path_parts[0] == "wiki":
            topic = unquote(path_parts[1])
            return fetch_wikipedia_concept(topic, lang=lang, fetcher=fetcher)
    return fetch_web_or_art_target(url, fetcher=fetcher)


def inspect_image_bytes_summary(image_bytes: bytes) -> Dict[str, Any]:
    """Extracts chromatic warmth, contrast, and luminance palette swatches from image bytes."""
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as im:
        rgb_im = im.convert("RGB")
        w, h = rgb_im.size
        arr = np.asarray(rgb_im.resize((64, 64)), dtype=np.float64) / 255.0

    mean_rgb = [round(float(arr[..., c].mean()), 3) for c in range(3)]
    contrast_std = round(float(arr.std()), 3)
    warm_cool_ratio = round(
        float((arr[..., 0].mean() + 0.1) / (arr[..., 2].mean() + 0.1)),
        3,
    )
    flat_pixels = arr.reshape(-1, 3)
    lum = 0.2126 * flat_pixels[:, 0] + 0.7152 * flat_pixels[:, 1] + 0.0722 * flat_pixels[:, 2]
    order = np.argsort(lum)
    n_pix = len(order)
    quantiles = (0.2, 0.5, 0.8)
    dominant_palette_rgb = [
        [round(float(v), 3) for v in flat_pixels[order[int(q * (n_pix - 1))]]]
        for q in quantiles
    ]

    return {
        "width": w,
        "height": h,
        "mean_rgb_01": mean_rgb,
        "contrast_std": contrast_std,
        "warm_cool_ratio": warm_cool_ratio,
        "dominant_palette_rgb": dominant_palette_rgb,
    }


def fetch_live_github_firehose_events(
    limit: int = 8,
    fetcher: Optional[HttpFetcher] = None,
) -> Dict[str, Any]:
    """
    Fetches genuinely uncurated, real-time global public GitHub events (`api.github.com/events`)
    happening right now — completely unknown ahead of time to the developer or LLM.
    """
    get_fn = fetcher or default_http_get
    headers = {"Accept": "application/vnd.github+json"}
    st_code, raw_bytes, _ = get_fn(
        f"https://api.github.com/events?per_page={max(1, min(30, int(limit) * 2))}",
        8.0,
        headers,
    )
    if st_code != 200:
        return {
            "ok": False,
            "status_code": st_code,
            "error": (
                f"GitHub Events HTTP {st_code}"
                if st_code
                else f"Network error: {raw_bytes.decode('utf-8', 'replace')[:180]}"
            ),
            "events": [],
        }
    try:
        arr = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        return {"ok": False, "status_code": st_code, "error": f"JSON parse error: {exc}", "events": []}

    parsed: List[Dict[str, Any]] = []
    seen_repos: set[str] = set()
    for ev in arr if isinstance(arr, list) else []:
        repo_name = str(((ev or {}).get("repo") or {}).get("name") or "").strip()
        if not repo_name or "/" not in repo_name or repo_name in seen_repos:
            continue
        seen_repos.add(repo_name)
        parsed.append(
            {
                "event_id": str(ev.get("id", "")),
                "event_type": str(ev.get("type", "")),
                "repo": repo_name,
                "repo_url": f"https://github.com/{repo_name}",
                "actor": str(((ev or {}).get("actor") or {}).get("login") or ""),
                "created_at": str(ev.get("created_at", "")),
            }
        )
        if len(parsed) >= limit:
            break
    return {"ok": True, "status_code": st_code, "events": parsed}


def fetch_github_repo_tree(
    owner_repo: str,
    fetcher: Optional[HttpFetcher] = None,
) -> Dict[str, Any]:
    """Inspects top-level files and directories of an unknown public GitHub repository."""
    get_fn = fetcher or default_http_get
    headers = {"Accept": "application/vnd.github+json"}
    st_code, raw_bytes, _ = get_fn(
        f"https://api.github.com/repos/{owner_repo}/contents",
        8.0,
        headers,
    )
    if st_code != 200:
        return {
            "ok": False,
            "status_code": st_code,
            "error": f"GitHub contents HTTP {st_code}",
            "entries": [],
        }
    try:
        items = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        return {"ok": False, "status_code": st_code, "error": str(exc), "entries": []}

    entries = [
        {
            "name": str(it.get("name", "")),
            "path": str(it.get("path", "")),
            "type": str(it.get("type", "")),
            "size": int(it.get("size") or 0),
        }
        for it in (items if isinstance(items, list) else [])
        if isinstance(it, dict)
    ]
    return {"ok": True, "status_code": st_code, "entries": entries}


def fetch_github_raw_file(
    owner_repo: str,
    file_path: str,
    fetcher: Optional[HttpFetcher] = None,
) -> Dict[str, Any]:
    """Fetches actual Base64-decoded source content of a specific file inside an unknown GitHub repo."""
    get_fn = fetcher or default_http_get
    headers = {"Accept": "application/vnd.github+json"}
    st_code, raw_bytes, _ = get_fn(
        f"https://api.github.com/repos/{owner_repo}/contents/{quote(file_path)}",
        8.0,
        headers,
    )
    if st_code != 200:
        return {"ok": False, "status_code": st_code, "error": f"GitHub file HTTP {st_code}", "content": ""}
    try:
        payload = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        b64 = str(payload.get("content", ""))
        text = base64.b64decode(b64).decode("utf-8", errors="replace") if b64 else ""
        return {
            "ok": True,
            "status_code": st_code,
            "path": file_path,
            "size": int(payload.get("size") or len(text)),
            "content": text[:8000],
        }
    except Exception as exc:
        return {"ok": False, "status_code": st_code, "error": str(exc), "content": ""}


def fetch_github_issues(
    owner_repo: str,
    limit: int = 3,
    fetcher: Optional[HttpFetcher] = None,
) -> Dict[str, Any]:
    """Fetches real open human discussions/issues inside an unknown GitHub repository."""
    get_fn = fetcher or default_http_get
    headers = {"Accept": "application/vnd.github+json"}
    st_code, raw_bytes, _ = get_fn(
        f"https://api.github.com/repos/{owner_repo}/issues?state=all&per_page={max(1, min(10, int(limit)))}",
        8.0,
        headers,
    )
    if st_code != 200:
        return {"ok": False, "status_code": st_code, "issues": []}
    try:
        arr = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    except Exception:
        return {"ok": False, "status_code": st_code, "issues": []}
    issues = [
        {
            "number": int(it.get("number") or 0),
            "title": str(it.get("title") or "")[:140],
            "state": str(it.get("state") or ""),
            "body_excerpt": str(it.get("body") or "")[:240],
        }
        for it in (arr if isinstance(arr, list) else [])
        if isinstance(it, dict)
    ]
    return {"ok": True, "status_code": st_code, "issues": issues}


def interact_with_unknown_github_repo(
    owner_repo: Optional[str] = None,
    fetcher: Optional[HttpFetcher] = None,
) -> Dict[str, Any]:
    """
    True multi-turn two-way exploration of a genuinely unknown live GitHub repository:
      Turn 1: If `owner_repo` is None, pull live uncurated global GitHub events right now.
      Turn 2: Inspect repository metadata & root file tree.
      Turn 3: Select a real source/config file from the tree and fetch its actual source text.
      Turn 4: Inspect real human issues/discussions to ground why this codebase exists.
    """
    chosen_repo = str(owner_repo or "").strip()
    source_event: Dict[str, Any] = {}
    if not chosen_repo:
        firehose = fetch_live_github_firehose_events(limit=6, fetcher=fetcher)
        if not firehose.get("ok") or not firehose.get("events"):
            return {
                "ok": False,
                "stage": "firehose",
                "error": firehose.get("error", "no_live_github_events"),
            }
        source_event = firehose["events"][0]
        chosen_repo = source_event["repo"]

    repo_obs = fetch_github_target(f"https://github.com/{chosen_repo}", fetcher=fetcher)
    if repo_obs.blocked:
        return {
            "ok": False,
            "stage": "repo_meta",
            "repo": chosen_repo,
            "source_event": source_event,
            "error": repo_obs.block_reason,
            "status_code": repo_obs.status_code,
        }

    tree = fetch_github_repo_tree(chosen_repo, fetcher=fetcher)
    entries = tree.get("entries") or []
    # Pick a real non-README source/config file from the actual unknown repo
    preferred_exts = (".py", ".rs", ".go", ".ts", ".js", ".c", ".cpp", ".java", ".toml", ".json", ".yaml", ".yml", ".md")
    picked_entry = None
    for ext in preferred_exts:
        for ent in entries:
            if ent.get("type") == "file" and ent.get("name", "").lower().endswith(ext) and ent.get("size", 0) > 0:
                if ent.get("size", 0) <= 60_000:
                    picked_entry = ent
                    break
        if picked_entry is not None:
            break
    if picked_entry is None:
        for ent in entries:
            if ent.get("type") == "file" and 0 < ent.get("size", 0) <= 60_000:
                picked_entry = ent
                break

    fetched_file: Dict[str, Any] = {}
    if picked_entry is not None:
        fetched_file = fetch_github_raw_file(chosen_repo, picked_entry["path"], fetcher=fetcher)

    issues = fetch_github_issues(chosen_repo, limit=3, fetcher=fetcher)
    return {
        "ok": True,
        "repo": chosen_repo,
        "url": f"https://github.com/{chosen_repo}",
        "source_event": source_event,
        "metadata": dict(repo_obs.metadata),
        "readme_excerpt": repo_obs.text_body[:1200],
        "outbound_urls": list(repo_obs.outbound_urls[:5]),
        "tree_count": len(entries),
        "tree_names": [e["name"] for e in entries[:15]],
        "inspected_file": {
            "path": fetched_file.get("path", ""),
            "size": fetched_file.get("size", 0),
            "content": fetched_file.get("content", ""),
            "ok": bool(fetched_file.get("ok", False)),
        },
        "issues": issues.get("issues", []),
    }
