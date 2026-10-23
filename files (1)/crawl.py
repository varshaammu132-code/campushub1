"""
crawl.py - polite, domain-locked, privacy-filtered website crawler.

* Crawls ONLY the configured SITE_URL host, and only under ALLOWED_PATHS.
* Production crawls require CRAWL_AUTHORIZED=true (written permission).
* Obeys robots.txt, delays between requests, and a maximum page count.
* Breadth-first. HTML -> Trafilatura, PDF -> pypdf.
* Pages are passed through privacy.py BEFORE being saved.
* Output: JSONL (one page per line) + a failed-URL log. The output file is
  written to a temporary file and moved into place only when the crawl ends,
  so an interrupted crawl never leaves a half-written pages file.

CLI:
    python crawl.py                       # production, uses config
    python crawl.py --max-pages 20
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import ai_config as cfg
import privacy

log = logging.getLogger("campusai.crawl")

SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico", ".tif", ".tiff",
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".exe", ".msi", ".dmg", ".apk",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flv", ".wmv",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".csv", ".css", ".js",
    ".woff", ".woff2", ".ttf", ".eot", ".xml", ".json", ".rss",
}
HTML_TYPES = ("text/html", "application/xhtml+xml")
PDF_TYPES = ("application/pdf",)


class CrawlNotAuthorized(RuntimeError):
    """Raised when a production crawl is attempted without authorization."""


class CrawlConfigError(RuntimeError):
    """Raised when SITE_URL / ALLOWED_PATHS are missing or invalid."""


@dataclass
class CrawlReport:
    started_at: float = 0.0
    finished_at: float = 0.0
    site: str = ""
    pages_saved: int = 0
    pages_fetched: int = 0
    skipped_short: int = 0
    skipped_privacy: int = 0
    skipped_robots: int = 0
    skipped_other: int = 0
    failed: list[dict] = field(default_factory=list)       # {"url","reason"}
    privacy_blocked: list[dict] = field(default_factory=list)
    output_file: str = ""
    stopped_reason: str = "completed"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_s"] = round(self.finished_at - self.started_at, 2)
        return d


# --------------------------------------------------------------------------
# URL policy
# --------------------------------------------------------------------------
def normalise_url(url: str) -> str:
    """Strip fragment and query (calendars/session ids), drop trailing index."""
    url, _ = urldefrag(url.strip())
    p = urlparse(url)
    path = re.sub(r"/{2,}", "/", p.path or "/")
    return p._replace(path=path, query="", params="").geturl()


def _host(url: str) -> str:
    h = (urlparse(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


class UrlPolicy:
    """Decides whether a URL may be fetched. No arbitrary URLs ever pass."""

    def __init__(self, site_url: str, allowed_paths: list[str]):
        p = urlparse(site_url)
        if p.scheme not in ("http", "https") or not p.hostname:
            raise CrawlConfigError(f"SITE_URL is not a valid http(s) URL: {site_url!r}")
        self.site_url = site_url
        self.scheme = p.scheme
        self.host = _host(site_url)
        self.paths = [("/" + a.strip("/")).rstrip("/") or "/" for a in allowed_paths] or ["/"]

    def allowed(self, url: str) -> tuple[bool, str]:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False, "unsupported scheme"
        if _host(url) != self.host:
            return False, "external domain"
        path = p.path or "/"
        if not any(prefix == "/" or path == prefix or path.startswith(prefix + "/")
                   for prefix in self.paths):
            return False, "path not approved"
        ext = os.path.splitext(path.lower())[1]
        if ext in SKIP_EXTENSIONS:
            return False, f"unsupported extension {ext}"
        d = privacy.is_blocked_url(url)
        if d.blocked:
            return False, f"privacy: {d.reason}"
        return True, ""


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or "", flags=re.UNICODE))


def extract_html(html: str, url: str) -> tuple[str, str, list[str]]:
    """Return (title, main_text, links). Trafilatura first, bs4 fallback."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)
             if not a["href"].lower().startswith(("mailto:", "tel:", "javascript:"))]
    text = ""
    try:
        import trafilatura

        text = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=True,
            include_links=False, favor_recall=True) or ""
        if not title:
            meta = trafilatura.extract_metadata(html)
            title = (getattr(meta, "title", "") or "").strip()
    except ImportError:
        log.warning("trafilatura not installed; using BeautifulSoup fallback")
    except Exception as exc:  # malformed html etc.
        log.debug("trafilatura failed on %s: %s", url, exc)
    if not text:
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))
    return title, text.strip(), links


def extract_pdf(data: bytes) -> tuple[str, str]:
    """Return (title, text). Raises ValueError for malformed/encrypted PDFs."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("encrypted pdf")
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        meta_title = ""
        try:
            meta_title = (reader.metadata.title or "").strip() if reader.metadata else ""
        except Exception:
            pass
        return meta_title, "\n".join(parts).strip()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"malformed pdf: {exc}") from exc


# --------------------------------------------------------------------------
# Crawler
# --------------------------------------------------------------------------
def _make_session():
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": str(cfg.get("CRAWL_USER_AGENT")),
                      "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.1"})
    return s


def _load_robots(session, policy: UrlPolicy, timeout: int) -> Optional[RobotFileParser]:
    robots_url = f"{policy.scheme}://{urlparse(policy.site_url).netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        r = session.get(robots_url, timeout=timeout)
        if r.status_code == 200:
            rp.parse(r.text.splitlines())
            return rp
        if r.status_code in (401, 403):
            rp.disallow_all = True      # be conservative
            return rp
        rp.parse([])                    # 404 etc. => no restrictions
        return rp
    except Exception as exc:
        log.warning("could not read robots.txt (%s); proceeding cautiously", exc)
        rp.parse([])
        return rp


def crawl(
    out_path: Optional[Path | str] = None,
    *,
    site_url: Optional[str] = None,
    allowed_paths: Optional[list[str]] = None,
    max_pages: Optional[int] = None,
    delay: Optional[float] = None,
    require_authorization: bool = True,
    session=None,
    progress: Optional[Callable[[CrawlReport], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> CrawlReport:
    """
    Crawl the configured college site. Returns a CrawlReport.

    Safety:
      * ``site_url`` defaults to the shared SITE_URL. Callers that expose this to
        users (admin endpoints) must NOT pass a user-supplied URL.
      * Set ``require_authorization=False`` only for the offline demo/tests.
    """
    site_url = site_url or str(cfg.get("SITE_URL"))
    if not site_url:
        raise CrawlConfigError("SITE_URL is not configured")
    if require_authorization and not cfg.get_bool("CRAWL_AUTHORIZED"):
        raise CrawlNotAuthorized(
            "Production crawling needs written permission from the college. "
            "Set CRAWL_AUTHORIZED=true once you have it.")
    allowed_paths = allowed_paths or cfg.get_list("ALLOWED_PATHS") or ["/"]
    max_pages = int(max_pages if max_pages is not None else cfg.get("MAX_PAGES"))
    delay = float(delay if delay is not None else cfg.get("CRAWL_DELAY"))
    timeout = int(cfg.get("CRAWL_TIMEOUT"))
    max_bytes = int(cfg.get("CRAWL_MAX_BYTES"))
    min_words = int(cfg.get("MIN_PAGE_WORDS"))
    out_path = Path(out_path or cfg.profile_paths("production")["pages_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    failed_path = out_path.with_name("failed_urls.jsonl")

    policy = UrlPolicy(site_url, allowed_paths)
    session = session or _make_session()
    robots = _load_robots(session, policy, timeout)
    ua = str(cfg.get("CRAWL_USER_AGENT"))
    robots_delay = None
    try:
        robots_delay = robots.crawl_delay(ua) if robots else None
    except Exception:
        pass
    if robots_delay:
        delay = max(delay, float(robots_delay))

    report = CrawlReport(started_at=time.time(), site=site_url, output_file=str(out_path))
    queue: deque[str] = deque()
    seen: set[str] = set()
    start_urls = []
    for prefix in policy.paths:
        start_urls.append(normalise_url(urljoin(site_url, prefix + "/" if prefix != "/" else "/")))
    for u in start_urls:
        if u not in seen:
            seen.add(u)
            queue.append(u)

    def fail(url: str, reason: str) -> None:
        report.failed.append({"url": url, "reason": reason})
        log.info("FAILED %s: %s", url, reason)

    first = True
    try:
        with tmp_path.open("w", encoding="utf-8") as out:
            while queue and report.pages_saved < max_pages:
                if cancel and cancel():
                    report.stopped_reason = "cancelled"
                    break
                url = queue.popleft()
                ok, why = policy.allowed(url)
                if not ok:
                    if why.startswith("privacy"):
                        report.skipped_privacy += 1
                        report.privacy_blocked.append({"url": url, "reason": why})
                    else:
                        report.skipped_other += 1
                    continue
                if robots and not robots.can_fetch(ua, url):
                    report.skipped_robots += 1
                    continue

                if not first:
                    sleep(delay)
                first = False

                try:
                    resp = session.get(url, timeout=timeout, stream=True,
                                       allow_redirects=True)
                except Exception as exc:
                    fail(url, f"network error: {type(exc).__name__}")
                    continue
                try:
                    final_url = normalise_url(resp.url or url)
                    ok, why = policy.allowed(final_url)
                    if not ok:               # redirected off-site / to a login page
                        report.skipped_other += 1
                        fail(url, f"redirect rejected: {why}")
                        continue
                    if resp.status_code != 200:
                        fail(url, f"http {resp.status_code}")
                        continue
                    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                    if ctype not in HTML_TYPES + PDF_TYPES:
                        report.skipped_other += 1
                        continue
                    declared = resp.headers.get("Content-Length")
                    if declared and declared.isdigit() and int(declared) > max_bytes:
                        fail(url, "too large")
                        continue
                    body = b""
                    for chunk in resp.iter_content(65536):
                        body += chunk
                        if len(body) > max_bytes:
                            break
                    if len(body) > max_bytes:
                        fail(url, "too large")
                        continue
                finally:
                    try:
                        resp.close()
                    except Exception:
                        pass

                report.pages_fetched += 1
                links: list[str] = []
                try:
                    if ctype in PDF_TYPES:
                        title, text = extract_pdf(body)
                        title = title or os.path.basename(urlparse(final_url).path)
                        kind = "pdf"
                    else:
                        html = body.decode(resp.encoding or "utf-8", errors="replace")
                        title, text, links = extract_html(html, final_url)
                        kind = "html"
                except ValueError as exc:
                    fail(url, str(exc))
                    continue
                except Exception as exc:
                    fail(url, f"extraction error: {type(exc).__name__}")
                    continue

                for link in links:
                    n = normalise_url(link)
                    if n not in seen:
                        seen.add(n)
                        queue.append(n)       # BFS; policy checked when popped

                if _word_count(text) < min_words:
                    report.skipped_short += 1
                    continue
                decision = privacy.check_page(final_url, title, text)
                if decision.blocked:
                    report.skipped_privacy += 1
                    report.privacy_blocked.append({"url": final_url, "reason": decision.reason})
                    continue

                out.write(json.dumps({
                    "url": final_url, "title": title or final_url, "text": text,
                    "type": kind, "crawled_at": int(time.time()),
                }, ensure_ascii=False) + "\n")
                report.pages_saved += 1
                if progress:
                    progress(report)
            else:
                if queue and report.pages_saved >= max_pages:
                    report.stopped_reason = "max_pages_reached"
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise
    report.finished_at = time.time()
    with failed_path.open("w", encoding="utf-8") as f:
        for item in report.failed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Crawl the configured college website")
    ap.add_argument("--out", help="output JSONL (default data/pages.jsonl)")
    ap.add_argument("--max-pages", type=int)
    ap.add_argument("--delay", type=float)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        rep = crawl(args.out, max_pages=args.max_pages, delay=args.delay,
                    progress=lambda r: print(f"\rsaved={r.pages_saved} failed={len(r.failed)}", end=""))
    except (CrawlNotAuthorized, CrawlConfigError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print()
    print(json.dumps(rep.to_dict(), indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
