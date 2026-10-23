"""Hashing, device detection, CSRF helpers. Device metadata is approximate and spoofable."""
import hashlib, hmac, os, re, secrets, time
from database import tx, query, audit

SECRET = os.environ.get("APP_SECRET", "dev-only-change-me").encode()

def h(value: str) -> str:
    """Keyed hash; used for OTPs, session tokens, device tokens (never store plaintext)."""
    return hmac.new(SECRET, value.encode(), hashlib.sha256).hexdigest()

def new_token() -> str: return secrets.token_urlsafe(32)

def parse_ua(ua: str):
    ua = ua or ""
    browser = next((n for p, n in [("Edg/", "Edge"), ("OPR/", "Opera"), ("Chrome/", "Chrome"),
                                    ("Firefox/", "Firefox"), ("Safari/", "Safari")] if p in ua), "Unknown")
    os_ = next((n for p, n in [("Windows", "Windows"), ("Android", "Android"), ("iPhone", "iOS"),
                               ("iPad", "iOS"), ("Mac OS", "macOS"), ("Linux", "Linux")] if p in ua), "Unknown")
    dtype = "Mobile" if re.search(r"Mobile|Android|iPhone", ua) else "Desktop"
    return dtype, browser, os_

def check_device(c, user_id, device_token, ua, now):
    """Inside a tx. Returns (device_id, new_token_or_None, is_new). Recognized = valid, unrevoked
    registration cookie token. IP/UA are NOT used as identity."""
    if device_token:
        r = c.execute("SELECT id FROM devices WHERE user_id=? AND token_hash=? AND revoked=0",
                      (user_id, h(device_token))).fetchone()
        if r:
            c.execute("UPDATE devices SET last_seen=? WHERE id=?", (now, r["id"]))
            return r["id"], None, False
    tok = new_token(); dtype, br, os_ = parse_ua(ua)
    cur = c.execute("INSERT INTO devices(user_id,token_hash,device_type,browser,os,first_seen,last_seen)"
                    " VALUES(?,?,?,?,?,?,?)", (user_id, h(tok), dtype, br, os_, now, now))
    return cur.lastrowid, tok, True

def record_event(c, user_id, device_id, kind, dedupe_key, now):
    """Insert event; UNIQUE dedupe_key prevents duplicate alerts on retries. Returns id or None."""
    try:
        return c.execute("INSERT INTO security_events(user_id,device_id,kind,dedupe_key,created_at)"
                         " VALUES(?,?,?,?,?)", (user_id, device_id, kind, dedupe_key, now)).lastrowid
    except Exception:
        return None
