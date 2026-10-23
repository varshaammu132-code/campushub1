"""
general.py - Study Helper backend (coding / project tutor).

Default provider : local Ollama, model qwen2.5-coder:3b
Optional provider: Anthropic API (Study Helper ONLY)

!!! PRIVACY NOTICE ---------------------------------------------------------
If STUDY_PROVIDER=anthropic AND STUDY_CLOUD_ENABLED=true, every Study Helper
question and the conversation history are SENT TO AN EXTERNAL PROVIDER
(Anthropic). Tell your users. College Info (rag.py) never uses this module and
never touches a cloud API.
-----------------------------------------------------------------------------

Callable interface for Member B's routes (/ask, /ask/stream, /study):

    ask(question, history=None)         -> str
    ask_stream(question, history=None)  -> iterator of str (call .close() if you stop early)
    provider_info()                     -> dict   (for UI notice / status)

Errors (map to HTTP):
    StudyInputError -> 400/422     StudyBusy -> 503 (+ Retry-After)
    StudyUnavailable -> 503        StudyConfigError -> 500
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Iterator, Optional

import ai_config as cfg

log = logging.getLogger("campusai.general")

ALLOWED_ROLES = {"user", "assistant"}

TUTOR_SYSTEM_PROMPT = """You are CampusAI Study Helper, a patient tutor for engineering students.

- Put all code in fenced code blocks with the language name (for example ```python).
- When asked about a project, explain the plan step by step.
- Do not help with cheating in exams, tests or graded assessments (for example live exam answers, impersonation or hidden devices). Offer to teach the concept instead.
- Do not write malware, ransomware, keyloggers, credential stealers, exploits or tools for attacking systems you do not own.
- If you are not sure about something, say so clearly instead of guessing. Do not invent library functions, APIs, facts or citations.
- You do not know SJCIT-specific facts (fees, dates, rules, staff, timetables, results). For those, tell the student to use the College Info assistant or contact the college office.
- Keep answers focused and as short as the question allows."""


class StudyError(RuntimeError):
    pass


class StudyInputError(StudyError, ValueError):
    pass


class StudyBusy(StudyError):
    pass


class StudyUnavailable(StudyError):
    pass


class StudyConfigError(StudyError):
    pass


# --------------------------------------------------------------------------
# Concurrency: non-blocking semaphore
# --------------------------------------------------------------------------
_slots_lock = threading.Lock()
_slots: Optional[threading.BoundedSemaphore] = None
_slots_size = 0


def _get_slots() -> threading.BoundedSemaphore:
    global _slots, _slots_size
    size = max(1, int(cfg.get("STUDY_MAX_PARALLEL")))
    with _slots_lock:
        if _slots is None or size != _slots_size:
            _slots, _slots_size = threading.BoundedSemaphore(size), size
        return _slots


def _acquire_or_busy() -> threading.BoundedSemaphore:
    sem = _get_slots()
    if not sem.acquire(blocking=False):
        raise StudyBusy("The Study Helper is busy. Please try again in a moment.")
    return sem


# --------------------------------------------------------------------------
# Provider selection / validation
# --------------------------------------------------------------------------
def provider_info() -> dict:
    provider = str(cfg.get("STUDY_PROVIDER")).strip().lower()
    cloud = provider == "anthropic"
    return {
        "provider": provider,
        "model": str(cfg.get("STUDY_ANTHROPIC_MODEL") if cloud else cfg.get("STUDY_MODEL")),
        "external": cloud,
        "notice": ("Study Helper requests are sent to an external provider (Anthropic)."
                   if cloud else "Study Helper runs locally; nothing leaves this machine."),
    }


def _resolve_provider() -> str:
    provider = str(cfg.get("STUDY_PROVIDER")).strip().lower()
    if provider == "ollama":
        return "ollama"
    if provider == "anthropic":
        if not cfg.get_bool("STUDY_CLOUD_ENABLED"):
            raise StudyConfigError(
                "STUDY_PROVIDER=anthropic requires STUDY_CLOUD_ENABLED=true (explicit opt-in).")
        if not str(cfg.get("ANTHROPIC_API_KEY")).strip():
            raise StudyConfigError("ANTHROPIC_API_KEY is not set in the environment.")
        return "anthropic"
    raise StudyConfigError(f"unknown STUDY_PROVIDER {provider!r} (use 'ollama' or 'anthropic')")


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
def validate_question(question) -> str:
    if not isinstance(question, str) or not question.strip():
        raise StudyInputError("question must be a non-empty string")
    limit = int(cfg.get("STUDY_MAX_QUESTION_CHARS"))
    if len(question) > limit:
        raise StudyInputError(f"question is longer than {limit} characters")
    return question.strip()


def validate_history(history) -> list[dict]:
    """
    Accepts [{"role": "user"|"assistant", "content": str}, ...].
    Keeps only the last STUDY_MAX_TURNS turns (a turn = user + assistant message).
    Rejects other roles (e.g. "system") and any message over the char limit.
    """
    if history is None:
        return []
    if not isinstance(history, (list, tuple)):
        raise StudyInputError("history must be a list")
    limit = int(cfg.get("STUDY_MAX_HISTORY_CHARS"))
    clean = []
    for i, m in enumerate(history):
        if not isinstance(m, dict):
            raise StudyInputError(f"history[{i}] must be an object")
        role, content = m.get("role"), m.get("content")
        if role not in ALLOWED_ROLES:
            raise StudyInputError(f"history[{i}].role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise StudyInputError(f"history[{i}].content must be a non-empty string")
        if len(content) > limit:
            raise StudyInputError(f"history[{i}] is longer than {limit} characters")
        clean.append({"role": role, "content": content})
    keep = max(0, int(cfg.get("STUDY_MAX_TURNS"))) * 2
    clean = clean[-keep:] if keep else []
    while clean and clean[0]["role"] != "user":      # must start with a user message
        clean.pop(0)
    return clean


def _messages(question: str, history: list[dict]) -> list[dict]:
    return history + [{"role": "user", "content": question}]


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
def _ollama_stream(messages: list[dict]) -> Iterator[str]:
    import requests

    model = str(cfg.get("STUDY_MODEL"))
    url = str(cfg.get("OLLAMA_URL")).rstrip("/") + "/api/chat"
    payload = {"model": model, "stream": True,
               "messages": [{"role": "system", "content": TUTOR_SYSTEM_PROMPT}] + messages,
               "options": {"temperature": 0.2, "num_ctx": 4096,
                           "num_predict": int(cfg.get("STUDY_MAX_TOKENS"))}}
    try:
        with requests.post(url, json=payload, stream=True,
                           timeout=int(cfg.get("OLLAMA_TIMEOUT"))) as r:
            if r.status_code == 404:
                raise StudyUnavailable(f"Model '{model}' is not installed. Run: ollama pull {model}")
            if r.status_code != 200:
                raise StudyUnavailable(f"Ollama error HTTP {r.status_code}")
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("error"):
                    raise StudyUnavailable(str(data["error"])[:200])
                piece = (data.get("message") or {}).get("content")
                if piece:
                    yield piece
                if data.get("done"):
                    return
    except requests.exceptions.ConnectionError as exc:
        raise StudyUnavailable("The local AI service (Ollama) is not reachable.") from exc
    except requests.exceptions.Timeout as exc:
        raise StudyUnavailable("The local AI service timed out.") from exc


def _anthropic_stream(messages: list[dict]) -> Iterator[str]:
    """Anthropic Messages API over plain HTTPS (no SDK needed). Study Helper only."""
    import requests

    key = str(cfg.get("ANTHROPIC_API_KEY")).strip()
    payload = {"model": str(cfg.get("STUDY_ANTHROPIC_MODEL")),
               "max_tokens": int(cfg.get("STUDY_MAX_TOKENS")),
               "system": TUTOR_SYSTEM_PROMPT, "messages": messages, "stream": True}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    try:
        with requests.post("https://api.anthropic.com/v1/messages", headers=headers,
                           json=payload, stream=True, timeout=60) as r:
            if r.status_code in (401, 403):
                raise StudyConfigError("Anthropic rejected the API key.")
            if r.status_code == 429:
                raise StudyBusy("The cloud provider is rate limiting. Try again shortly.")
            if r.status_code != 200:
                raise StudyUnavailable(f"Cloud provider error HTTP {r.status_code}")
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "content_block_delta":
                    piece = (data.get("delta") or {}).get("text")
                    if piece:
                        yield piece
                elif data.get("type") == "message_stop":
                    return
    except requests.exceptions.ConnectionError as exc:
        raise StudyUnavailable("Cannot reach the cloud provider.") from exc
    except requests.exceptions.Timeout as exc:
        raise StudyUnavailable("The cloud provider timed out.") from exc


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
class _Stream:
    """Iterator that owns a concurrency slot and always releases it."""

    def __init__(self, sem: threading.BoundedSemaphore, gen: Iterator[str]):
        self._sem, self._gen, self._released = sem, gen, False

    def __iter__(self):
        return self

    def __next__(self) -> str:
        try:
            return next(self._gen)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self._released:
            self._released = True
            try:
                close = getattr(self._gen, "close", None)
                if close:
                    close()
            finally:
                self._sem.release()

    def __del__(self):          # safety net if the caller forgets close()
        try:
            self.close()
        except Exception:
            pass


def ask_stream(question: str, history: Optional[list[dict]] = None) -> _Stream:
    """
    Start a streaming answer. Validates input and takes a capacity slot
    IMMEDIATELY (so StudyBusy / StudyInputError are raised before you send
    HTTP headers). Iterate to get text pieces; call .close() if you stop early.
    """
    q = validate_question(question)
    hist = validate_history(history)
    provider = _resolve_provider()
    sem = _acquire_or_busy()
    try:
        msgs = _messages(q, hist)
        gen = _anthropic_stream(msgs) if provider == "anthropic" else _ollama_stream(msgs)
        if provider == "anthropic":
            log.warning("Study Helper request sent to external provider (anthropic)")
        return _Stream(sem, gen)
    except BaseException:
        sem.release()
        raise


def ask(question: str, history: Optional[list[dict]] = None) -> str:
    """Non-streaming convenience wrapper. Returns the full answer text."""
    stream = ask_stream(question, history)
    try:
        return "".join(stream).strip()
    finally:
        stream.close()


def main() -> int:
    info = provider_info()
    print(f"Study Helper [{info['provider']} / {info['model']}] - {info['notice']}")
    print("Type 'exit' to quit.")
    history: list[dict] = []
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.lower() in {"exit", "quit", "q"}:
            return 0
        if not q:
            continue
        reply = []
        try:
            print("\nHelper: ", end="", flush=True)
            for piece in ask_stream(q, history):
                reply.append(piece)
                print(piece, end="", flush=True)
            print()
        except StudyError as exc:
            print(f"\n[error] {exc}")
            continue
        history += [{"role": "user", "content": q},
                    {"role": "assistant",
                     "content": "".join(reply)[:int(cfg.get("STUDY_MAX_HISTORY_CHARS"))] or "(no reply)"}]
        history = validate_history(history[-12:]) if history else history


if __name__ == "__main__":
    raise SystemExit(main())
