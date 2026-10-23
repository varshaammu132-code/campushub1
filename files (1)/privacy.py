"""
privacy.py - deterministic privacy filtering.

This is a SECURITY BOUNDARY. Nothing here calls an LLM. Rules are applied:

  * by the crawler, before a page is saved            (is_blocked_url / check_page)
  * by the indexer, again, before a page is embedded  (check_page)
  * by the RAG engine, on every retrieved chunk       (check_text)
  * by the RAG engine, on the user's question         (is_private_query)
  * by the RAG engine, on the model's answer          (answer_leaks_private_data)

The rules deliberately err on the side of blocking. A false positive costs one
missing web page; a false negative could expose a student's record.

Public, institution-level information (a general fee structure, a published
academic calendar, an attendance *policy*) is allowed. Individual-level data
(marks of a person, attendance of a person, fee status of a person, lists of
USNs) is not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

PRIVACY_REFUSAL = (
    "I can't share individual student information such as marks, attendance, "
    "or fee status. Please contact the office."
)

# --------------------------------------------------------------------------
# Student identifiers
# --------------------------------------------------------------------------
# VTU style USN: 1 digit + 2 letters + 2 digits + 2-3 letters + 3 digits
# e.g. 4XX23CS001 (synthetic).  Case-insensitive.
USN_RE = re.compile(r"\b\d[A-Za-z]{2}\d{2}[A-Za-z]{2,3}\d{3}\b")
# Generic roll / register number written with a label.
LABELLED_ID_RE = re.compile(
    r"\b(?:usn|reg(?:ister)?\.?\s*(?:no|number)|roll\s*(?:no|number)|student\s*id)"
    r"\s*[:#\-]?\s*[A-Za-z0-9/\-]{5,}",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# URL rules
# --------------------------------------------------------------------------
# Matched against the lower-cased, URL-decoded path + query, split into tokens.
_BLOCKED_URL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("login/portal page", re.compile(
        r"(^|[/_.\-?=&])(log-?in|sign-?in|signon|sso|auth|oauth|my-?account|"
        r"user-?login|student-?login|parent-?login|faculty-?login|staff-?login|"
        r"student-?portal|parent-?portal|erp|lms-?login|wp-login|wp-admin|"
        r"admin-?login|dashboard|webmail|forgot-?password|reset-?password)"
        r"([/_.\-?=&]|$)")),
    ("individual marks / results", re.compile(
        r"(^|[/_.\-?=&])(marks?|marks-?card|marksheet|grade-?card|gradecard|"
        r"results?|result-?list|scorecard|internal-?marks|ia-?marks|cie-?marks|"
        r"my-?result|view-?result|check-?result|student-?result)"
        r"([/_.\-?=&]|$)")),
    ("attendance records", re.compile(
        r"(^|[/_.\-?=&])(attendance|attendence|absentees?|shortage-?list|"
        r"detain(ed)?-?list|condonation-?list)([/_.\-?=&]|$)")),
    ("individual fee information", re.compile(
        r"(^|[/_.\-?=&])(fee-?status|fees?-?due|fee-?receipt|fees?-?paid|"
        r"fee-?defaulters?|defaulters?|pay-?fees?|fee-?payment|fees?-?balance|"
        r"challan|payment-?status|dues)([/_.\-?=&]|$)")),
    ("personal student records", re.compile(
        r"(^|[/_.\-?=&])(student-?records?|student-?profile|student-?details|"
        r"student-?data|my-?profile|profile-?edit|usn-?list|roll-?list|"
        r"admit-?card-?download|hall-?ticket-?download|seating-?list|"
        r"personal-?details)([/_.\-?=&]|$)")),
    ("query string with student identifier", re.compile(
        r"[?&](usn|rollno|roll_no|regno|reg_no|studentid|student_id)=", re.IGNORECASE)),
]

# --------------------------------------------------------------------------
# Content rules
# --------------------------------------------------------------------------
_MARKS_WORDS = re.compile(
    r"\b(marks?|score[sd]?|sgpa|cgpa|grade[sd]?|result|percentage|total\s+marks|"
    r"internal\s+marks|cie|see\s+marks|passed|failed|fail|pass|absent)\b",
    re.IGNORECASE)
_ATTENDANCE_WORDS = re.compile(
    r"\b(attendance|present|absent|shortage|detained|classes\s+attended)\b",
    re.IGNORECASE)
_FEE_WORDS = re.compile(
    r"\b(fee\s+(?:status|due|balance|paid|pending)|fees?\s+pending|"
    r"balance\s+due|amount\s+due|defaulter|paid\s+on|receipt\s+no|challan\s+no)\b",
    re.IGNORECASE)
_LOGIN_FORM = re.compile(
    r"(<input[^>]+type=[\"']?password|enter\s+your\s+(?:usn|password)|"
    r"forgot\s+password|sign\s+in\s+to\s+(?:your|the)\s+(?:account|portal)|"
    r"student\s+login|parent\s+login)", re.IGNORECASE)

# A table row that looks like "<USN> <NAME> <NUMBER ...>"
_USN_ROW = re.compile(
    r"\b\d[A-Za-z]{2}\d{2}[A-Za-z]{2,3}\d{3}\b[^\n]{0,80}?\d", re.IGNORECASE)

MULTI_USN_THRESHOLD = 3          # >= this many distinct USNs => result/list page


@dataclass(frozen=True)
class Decision:
    blocked: bool
    reason: str = ""

    def __bool__(self) -> bool:      # `if check_page(...)` => True when blocked
        return self.blocked


ALLOWED = Decision(False, "")


def _normalise_url(url: str) -> str:
    parsed = urlparse(url.strip())
    text = unquote(f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path)
    return text.lower()


def is_blocked_url(url: str) -> Decision:
    """Return Decision(blocked, reason) for a URL, using deterministic rules."""
    if not isinstance(url, str) or not url.strip():
        return Decision(True, "empty or invalid url")
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return Decision(True, "unparseable url")
    if parsed.username or parsed.password:
        return Decision(True, "url contains credentials")
    host = (parsed.hostname or "").lower()
    # Subdomains that are clearly portals, e.g. erp.college.edu, results.college.edu
    for label in host.split(".")[:-2] or []:
        if re.fullmatch(
            r"(erp|portal|login|sso|auth|results?|marks|attendance|"
            r"students?|parents?|lms|mail|webmail|fees?|payments?)", label):
            return Decision(True, f"restricted subdomain '{label}'")
    target = _normalise_url(url)
    for reason, pattern in _BLOCKED_URL_PATTERNS:
        if pattern.search(target):
            return Decision(True, reason)
    return ALLOWED


def find_student_ids(text: str) -> set[str]:
    return {m.group(0).upper() for m in USN_RE.finditer(text or "")}


def check_text(text: str) -> Decision:
    """Return Decision for a block of page/chunk text."""
    if not text or not text.strip():
        return ALLOWED
    usns = find_student_ids(text)
    if len(usns) >= MULTI_USN_THRESHOLD:
        return Decision(True, f"contains {len(usns)} student identifiers (USN list)")
    if usns:
        # A single USN is only tolerable if there is no personal data near it.
        if (_MARKS_WORDS.search(text) or _ATTENDANCE_WORDS.search(text)
                or _FEE_WORDS.search(text) or len(_USN_ROW.findall(text)) >= 1):
            return Decision(True, "student identifier next to personal data")
    if LABELLED_ID_RE.search(text) and (
            _MARKS_WORDS.search(text) or _ATTENDANCE_WORDS.search(text)
            or _FEE_WORDS.search(text)):
        return Decision(True, "labelled student id next to personal data")
    if _FEE_WORDS.search(text) and re.search(r"\b(usn|student|name)\b", text, re.I) \
            and re.search(r"(?:rs\.?|inr|₹)\s?\d", text, re.I):
        return Decision(True, "individual fee status")
    if _LOGIN_FORM.search(text):
        return Decision(True, "login / portal page content")
    return ALLOWED


def check_page(url: str, title: str = "", text: str = "") -> Decision:
    """Combined URL + title + content check used before indexing."""
    d = is_blocked_url(url)
    if d.blocked:
        return d
    d = check_text(f"{title}\n{text}")
    return d


# --------------------------------------------------------------------------
# User questions and model answers
# --------------------------------------------------------------------------
_PRIVATE_QUERY_PATTERNS = [
    re.compile(r"\b(marks?|results?|grades?|sgpa|cgpa|score|percentage)\b.{0,60}"
               r"\b(of|for|about)\b.{0,40}(\b\d[A-Za-z]{2}\d{2}[A-Za-z]{2,3}\d{3}\b|"
               r"\bstudent\b|\bmy\b|\bhis\b|\bher\b|\bhim\b)", re.I),
    re.compile(r"\b(attendance|attendence)\b.{0,60}"
               r"(\b\d[A-Za-z]{2}\d{2}[A-Za-z]{2,3}\d{3}\b|\bstudent\b|\bmy\b|\bhis\b|\bher\b)", re.I),
    re.compile(r"\bfees?\b.{0,40}\b(status|due|pending|paid|balance|defaulter)\b.{0,60}"
               r"(\b\d[A-Za-z]{2}\d{2}[A-Za-z]{2,3}\d{3}\b|\bstudent\b|\bmy\b|\bhis\b|\bher\b)", re.I),
    re.compile(r"\b(my|his|her|their)\s+(marks|result|results|attendance|cgpa|sgpa|"
               r"fee\s*status|fees?\s+due|grades?)\b", re.I),
    re.compile(r"\b(list|names?)\s+of\s+(students|usns?)\b", re.I),
    re.compile(r"\b(personal|contact|phone|address|aadhaar|aadhar)\b.{0,40}\bstudent\b", re.I),
]


def is_private_query(question: str) -> Decision:
    """True when the question asks for an individual's marks/attendance/fees/etc."""
    q = question or ""
    if USN_RE.search(q):
        return Decision(True, "question contains a student identifier")
    for pattern in _PRIVATE_QUERY_PATTERNS:
        if pattern.search(q):
            return Decision(True, "question asks for individual student data")
    return ALLOWED


def answer_leaks_private_data(answer: str) -> Decision:
    """Final output guard: no USNs, no marks-with-identifier in an answer."""
    if find_student_ids(answer):
        return Decision(True, "answer contains a student identifier")
    return check_text(answer) if check_text(answer).blocked else ALLOWED
