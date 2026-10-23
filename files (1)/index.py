"""
index.py - chunk, embed (locally via Ollama bge-m3) and store pages in ChromaDB.

Safe rebuilds
-------------
Every rebuild writes a NEW versioned collection ``<name>__v<timestamp>``.
The live index is whichever version the *registry file* points at
(``<chroma_dir>/index_registry.json``). Promotion is one atomic file replace,
so:

  * a failed rebuild never touches the live index,
  * readers never see a half-built index,
  * the previous version is kept (one generation) for manual rollback.

Demo and production never share a directory OR a collection name; the guard
lives in ``ai_config.assert_profiles_separate``.

CLI:
    python index.py                  # production: data/pages.jsonl -> index
    python index.py --demo           # demo_data/sample_pages.jsonl -> demo index
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import ai_config as cfg
import privacy

log = logging.getLogger("campusai.index")

REGISTRY_NAME = "index_registry.json"
KEEP_OLD_VERSIONS = 1


class IndexError_(RuntimeError):
    """Base class for indexing failures."""


class OllamaDown(IndexError_):
    pass


class EmbedModelMissing(IndexError_):
    pass


class ValidationFailed(IndexError_):
    pass


@dataclass
class IndexReport:
    profile: str = "production"
    version: str = ""
    collection: str = ""
    pages_read: int = 0
    pages_indexed: int = 0
    pages_skipped_privacy: int = 0
    pages_skipped_empty: int = 0
    pages_skipped_invalid: int = 0
    chunks: int = 0
    embedding_dim: int = 0
    errors: list[str] = field(default_factory=list)
    promoted: bool = False
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_text(text: str, words: Optional[int] = None, overlap: Optional[int] = None) -> list[str]:
    """Split into word windows of ``words`` with ``overlap`` words repeated."""
    words = int(words if words is not None else cfg.get("CHUNK_WORDS"))
    overlap = int(overlap if overlap is not None else cfg.get("CHUNK_OVERLAP"))
    if words <= 0:
        raise ValueError("CHUNK_WORDS must be positive")
    if not 0 <= overlap < words:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and < CHUNK_WORDS")
    tokens = (text or "").split()
    if not tokens:
        return []
    step = words - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        window = tokens[start:start + words]
        chunks.append(" ".join(window))
        if start + words >= len(tokens):
            break
    return chunks


# --------------------------------------------------------------------------
# Ollama embeddings
# --------------------------------------------------------------------------
def _post(path: str, payload: dict, timeout: int):
    import requests

    url = str(cfg.get("OLLAMA_URL")).rstrip("/") + path
    try:
        return requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaDown(
            f"Cannot reach Ollama at {cfg.get('OLLAMA_URL')}. Is it running? (ollama serve)") from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaDown("Ollama timed out while embedding") from exc


def embed_texts(texts: list[str], model: Optional[str] = None) -> list[list[float]]:
    """Embed a batch locally. Raises OllamaDown / EmbedModelMissing / IndexError_."""
    model = model or str(cfg.get("EMBED_MODEL"))
    timeout = int(cfg.get("OLLAMA_EMBED_TIMEOUT"))
    if not texts:
        return []
    r = _post("/api/embed", {"model": model, "input": texts}, timeout)
    if r.status_code == 404:
        if "not found" in r.text.lower():
            raise EmbedModelMissing(f"Embedding model '{model}' is not installed. Run: ollama pull {model}")
        # very old Ollama without /api/embed -> per-text fallback
        vectors = []
        for t in texts:
            rr = _post("/api/embeddings", {"model": model, "prompt": t}, timeout)
            if rr.status_code != 200:
                raise IndexError_(f"embedding failed: HTTP {rr.status_code}: {rr.text[:200]}")
            vectors.append(rr.json().get("embedding") or [])
    elif r.status_code != 200:
        raise IndexError_(f"embedding failed: HTTP {r.status_code}: {r.text[:200]}")
    else:
        vectors = r.json().get("embeddings") or []
    if len(vectors) != len(texts):
        raise IndexError_(f"expected {len(texts)} embeddings, got {len(vectors)}")
    dims = {len(v) for v in vectors}
    if len(dims) != 1 or 0 in dims:
        raise IndexError_(f"inconsistent or empty embedding dimensions: {sorted(dims)}")
    return vectors


# --------------------------------------------------------------------------
# Registry (which collection version is live)
# --------------------------------------------------------------------------
def _registry_path(chroma_dir: Path) -> Path:
    return Path(chroma_dir) / REGISTRY_NAME


def read_registry(chroma_dir: Path) -> dict:
    p = _registry_path(chroma_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_registry(chroma_dir: Path, data: dict) -> None:
    p = _registry_path(chroma_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, p)                   # atomic on Windows and POSIX


def active_collection_name(chroma_dir: Path, base_name: str) -> Optional[str]:
    return (read_registry(chroma_dir).get(base_name) or {}).get("active")


def get_client(chroma_dir: Path):
    try:
        import chromadb
    except ImportError as exc:
        raise IndexError_("chromadb is not installed (pip install chromadb)") from exc
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    try:
        return chromadb.PersistentClient(path=str(chroma_dir))
    except Exception as exc:
        raise IndexError_(f"cannot open ChromaDB at {chroma_dir}: {exc}") from exc


# --------------------------------------------------------------------------
# Reading pages
# --------------------------------------------------------------------------
def read_pages(path: Path) -> Iterator[tuple[int, Optional[dict], str]]:
    """Yield (line_no, page_or_None, error)."""
    with Path(path).open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                page = json.loads(line)
            except json.JSONDecodeError:
                yield n, None, "invalid json"
                continue
            if not isinstance(page, dict) or not all(
                    isinstance(page.get(k), str) and page.get(k) for k in ("url", "text")):
                yield n, None, "missing url/text"
                continue
            yield n, page, ""


def _chunk_id(url: str, i: int) -> str:
    return hashlib.sha1(f"{url}#{i}".encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Build / validate / promote
# --------------------------------------------------------------------------
def build_index(
    pages_path: Optional[Path | str] = None,
    *,
    profile: str = "production",
    promote: bool = True,
    embed: Callable[[list[str]], list[list[float]]] = embed_texts,
    progress: Optional[Callable[[str], None]] = None,
    min_chunks: int = 1,
) -> IndexReport:
    """
    Build a fresh index version from ``pages_path`` and (optionally) promote it.

    The live index is untouched unless validation passes. On any failure the
    half-built version is deleted and the previous index stays live.
    """
    cfg.assert_profiles_separate()
    paths = cfg.profile_paths(profile)
    pages_path = Path(pages_path or paths["pages_file"])
    if not pages_path.exists():
        raise IndexError_(f"pages file not found: {pages_path}")
    say = progress or (lambda m: log.info(m))
    base = str(paths["collection"])
    chroma_dir = Path(paths["chroma_dir"])
    version = time.strftime("%Y%m%d%H%M%S")
    staging_name = f"{base}__v{version}"
    report = IndexReport(profile=profile, version=version, collection=staging_name)
    t0 = time.time()

    client = get_client(chroma_dir)
    collection = client.create_collection(staging_name, metadata={"hnsw:space": "cosine"})
    batch_size = max(1, int(cfg.get("EMBED_BATCH_SIZE")))
    pend_ids: list[str] = []
    pend_docs: list[str] = []
    pend_meta: list[dict] = []
    seen_urls: set[str] = set()

    def flush() -> None:
        if not pend_docs:
            return
        vectors = embed(pend_docs)
        dim = len(vectors[0])
        if report.embedding_dim and dim != report.embedding_dim:
            raise IndexError_(f"embedding dimension changed {report.embedding_dim} -> {dim}")
        report.embedding_dim = dim
        collection.add(ids=list(pend_ids), documents=list(pend_docs),
                       embeddings=vectors, metadatas=list(pend_meta))
        report.chunks += len(pend_docs)
        pend_ids.clear(); pend_docs.clear(); pend_meta.clear()
        say(f"  indexed {report.chunks} chunks from {report.pages_indexed} pages")

    try:
        for n, page, err in read_pages(pages_path):
            report.pages_read += 1
            if page is None:
                report.pages_skipped_invalid += 1
                report.errors.append(f"line {n}: {err}")
                continue
            url, title, text = page["url"], str(page.get("title") or page["url"]), page["text"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            decision = privacy.check_page(url, title, text)
            if decision.blocked:
                report.pages_skipped_privacy += 1
                report.errors.append(f"privacy skip {url}: {decision.reason}")
                continue
            chunks = chunk_text(text)
            if not chunks:
                report.pages_skipped_empty += 1
                continue
            report.pages_indexed += 1
            for i, ch in enumerate(chunks):
                if privacy.check_text(ch).blocked:
                    report.errors.append(f"privacy skip chunk {url}#{i}")
                    continue
                pend_ids.append(_chunk_id(url, i))
                pend_docs.append(ch)
                pend_meta.append({"url": url, "title": title, "chunk": i})
                if len(pend_docs) >= batch_size:
                    flush()
        flush()
        validate_collection(client, staging_name, min_chunks=min_chunks, embed=embed)
        report.seconds = round(time.time() - t0, 2)
        if promote:
            promote_version(chroma_dir, base, staging_name, client=client)
            report.promoted = True
        say(f"done: {report.chunks} chunks, promoted={report.promoted}")
        return report
    except BaseException:
        # Roll back the half-built version; the live index was never touched.
        try:
            client.delete_collection(staging_name)
        except Exception:
            pass
        raise


def validate_collection(client, name: str, *, min_chunks: int = 1,
                        embed: Callable[[list[str]], list[list[float]]] = embed_texts) -> dict:
    """
    Validate an index version BEFORE promotion. Raises ValidationFailed.

    Checks: exists, has >= min_chunks, every chunk has url metadata, no chunk
    violates the privacy rules, and a probe query works with the live embedder
    (same dimension as the stored vectors).
    """
    try:
        col = client.get_collection(name)
    except Exception as exc:
        raise ValidationFailed(f"collection {name} not found: {exc}") from exc
    count = col.count()
    if count < min_chunks:
        raise ValidationFailed(f"index has {count} chunks (< {min_chunks})")
    offset, page = 0, 200
    while offset < count:
        got = col.get(limit=page, offset=offset, include=["documents", "metadatas"])
        for doc, meta in zip(got["documents"], got["metadatas"]):
            if not meta or not meta.get("url"):
                raise ValidationFailed("chunk without url metadata")
            if privacy.check_text(doc).blocked:
                raise ValidationFailed(f"privacy violation in chunk of {meta.get('url')}")
        offset += page
    try:
        probe = embed(["college admission"])[0]
        res = col.query(query_embeddings=[probe], n_results=1)
        if not res["ids"] or not res["ids"][0]:
            raise ValidationFailed("probe query returned nothing")
    except IndexError_ as exc:
        raise ValidationFailed(f"probe query failed: {exc}") from exc
    except ValidationFailed:
        raise
    except Exception as exc:
        raise ValidationFailed(f"probe query failed (dimension mismatch?): {exc}") from exc
    return {"collection": name, "chunks": count}


def promote_version(chroma_dir: Path, base_name: str, version_name: str, *, client=None) -> None:
    """Atomically make ``version_name`` live; prune old versions."""
    reg = read_registry(chroma_dir)
    entry = reg.get(base_name) or {}
    previous = entry.get("active")
    history = [v for v in entry.get("history", []) if v != version_name]
    if previous and previous != version_name:
        history.insert(0, previous)
    reg[base_name] = {"active": version_name, "history": history[:KEEP_OLD_VERSIONS],
                      "promoted_at": int(time.time())}
    _write_registry(chroma_dir, reg)
    client = client or get_client(chroma_dir)
    keep = {version_name, *reg[base_name]["history"]}
    for col in client.list_collections():
        name = getattr(col, "name", col)
        if str(name).startswith(f"{base_name}__v") and name not in keep:
            try:
                client.delete_collection(name)
            except Exception as exc:     # pruning is best-effort
                log.warning("could not prune %s: %s", name, exc)


def index_status(profile: str = "production") -> dict:
    """Cheap status for dashboards. Never raises."""
    paths = cfg.profile_paths(profile)
    base, chroma_dir = str(paths["collection"]), Path(paths["chroma_dir"])
    active = active_collection_name(chroma_dir, base)
    info = {"profile": profile, "collection": base, "active_version": active,
            "chunks": 0, "ready": False}
    if not active:
        return info
    try:
        info["chunks"] = get_client(chroma_dir).get_collection(active).count()
        info["ready"] = info["chunks"] > 0
    except Exception as exc:
        info["error"] = str(exc)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the ChromaDB index")
    ap.add_argument("--demo", action="store_true", help="index demo_data/sample_pages.jsonl into the DEMO index")
    ap.add_argument("--pages", help="pages JSONL (default: data/pages.jsonl, or demo sample with --demo)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    profile = "demo" if args.demo else "production"
    pages = args.pages or (str(Path(__file__).parent / "demo_data" / "sample_pages.jsonl")
                           if args.demo else None)
    try:
        rep = build_index(pages, profile=profile)
    except IndexError_ as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(rep.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
