"""SMTP email + new-device alert dispatch. Never includes OTPs, passwords or tokens in alerts."""
import logging, os, smtplib, time
from email.message import EmailMessage
from datetime import datetime, timezone, timedelta
from database import tx, query

log = logging.getLogger("campusai.notify")
MODE = lambda: os.environ.get("TEACHER_ALERT_MODE", "faculty")  # faculty|admin|both|off
ADMIN_FALLBACK = lambda: os.environ.get("ADMIN_ALERT_EMAIL", "")
TZ = lambda: timezone(timedelta(minutes=int(os.environ.get("TZ_OFFSET_MIN", "330"))))  # IST default

def send_email(to: str, subject: str, body: str):
    """Single SMTP entry point (tests patch this). Raises on failure."""
    host = os.environ.get("SMTP_HOST")
    if not host: raise RuntimeError("SMTP not configured")
    m = EmailMessage(); m["From"] = os.environ.get("SMTP_FROM", "campusai@localhost")
    m["To"] = to; m["Subject"] = subject; m.set_content(body)
    with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587")), timeout=15) as s:
        s.starttls()
        if os.environ.get("SMTP_USER"): s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASS", ""))
        s.send_message(m)

def mask(email: str) -> str:
    u, _, d = email.partition("@"); return (u[:2] + "***@" + d)

def _recipients(student):
    mode = MODE(); out = []
    if mode == "off": return out
    fac = query("SELECT u.id, u.email FROM faculty_assignments f JOIN users u ON u.id=f.faculty_id"
                " WHERE f.student_id=?", (student["id"],))
    if mode in ("faculty", "both"):
        out += [r["email"] for r in fac]
    if mode in ("admin", "both") or (mode == "faculty" and not fac):
        if ADMIN_FALLBACK(): out.append(ADMIN_FALLBACK())
    return list(dict.fromkeys(out))

def create_alert(event_id, student_id):
    """In-app alert (always created unless mode=off), tied to the assigned faculty."""
    if MODE() == "off": return
    fac = query("SELECT faculty_id FROM faculty_assignments WHERE student_id=? LIMIT 1", (student_id,))
    with tx() as c:
        c.execute("INSERT INTO teacher_alerts(event_id,faculty_id,student_id,created_at) VALUES(?,?,?,?)",
                  (event_id, fac[0]["faculty_id"] if fac else None, student_id, int(time.time())))

def dispatch_new_device_alert(event_id, student_id, retries=2):
    """Run in a background task. Safe to call twice: already-sent recipients are skipped."""
    try:
        st = query("SELECT u.*, d.device_type, d.browser, d.os, e.created_at ev FROM security_events e"
                   " JOIN users u ON u.id=e.user_id JOIN devices d ON d.id=e.device_id WHERE e.id=?", (event_id,))[0]
        body = (f"Student: {st['name']}\nStudent ID: {st['student_id']}\nCollege email: {mask(st['email'])}\n"
                f"Department: {st['department']}\nDevice: {st['device_type']}\nBrowser: {st['browser']}\n"
                f"Operating system: {st['os']}\nLogin time: "
                f"{datetime.fromtimestamp(st['ev'], TZ()).strftime('%Y-%m-%d %H:%M:%S %z')}\n\n"
                "Action: Review this login if it appears unexpected.\n")
        for rcpt in _recipients(st):
            if query("SELECT 1 FROM notification_attempts WHERE event_id=? AND recipient=? AND status='sent'",
                     (event_id, rcpt)): continue
            status, err = "failed", ""
            for _ in range(retries + 1):
                try:
                    send_email(rcpt, "CampusAI Hub — New Device Login Alert", body); status = "sent"; break
                except Exception as e:  # SMTP failure must never crash the app
                    err = type(e).__name__
            with tx() as c:
                c.execute("INSERT INTO notification_attempts(event_id,recipient,status,error,created_at)"
                          " VALUES(?,?,?,?,?)", (event_id, rcpt, status, err, int(time.time())))
    except Exception:
        log.exception("alert dispatch failed")
