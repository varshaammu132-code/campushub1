"""OTP login, sessions, roles, CSRF."""
import json, os, re, secrets, time
from fastapi import HTTPException, Request
from database import tx, query, audit
from security import h, new_token, check_device, record_event

OTP_TTL, MAX_ATTEMPTS, COOLDOWN = 600, 5, 60
SESSION_TTL = int(os.environ.get("SESSION_TTL", 8 * 3600))
EMAIL_RE = re.compile(r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$")
ONE_ACTIVE = lambda: os.environ.get("ONE_ACTIVE_SESSION", "1") == "1"

def norm_email(e: str) -> str:
    e = (e or "").strip().lower()
    if len(e) > 254 or not EMAIL_RE.match(e): raise HTTPException(400, "Invalid email address")
    return e

def seed_roles(path="roles.json"):
    cfg = json.load(open(path)); now = int(time.time())
    with tx() as c:
        for u in cfg["users"]:
            c.execute("INSERT OR IGNORE INTO users(email,name,student_id,department,role,created_at)"
                      " VALUES(?,?,?,?,?,?)", (u["email"].lower(), u.get("name"), u.get("student_id"),
                                               u.get("department"), u["role"], now))
        for u in cfg["users"]:
            if u.get("teacher"):
                c.execute("INSERT OR IGNORE INTO faculty_assignments SELECT f.id, s.id FROM users f, users s"
                          " WHERE f.email=? AND s.email=?", (u["teacher"].lower(), u["email"].lower()))

def request_otp(email: str):
    """Always returns normally (generic response). Returns (code, email) only if an email should be sent."""
    email = norm_email(email); now = int(time.time())
    user = query("SELECT id FROM users WHERE email=?", (email,))
    with tx() as c:
        last = c.execute("SELECT created_at FROM otp_challenges WHERE email=? ORDER BY id DESC LIMIT 1",
                         (email,)).fetchone()
        if last and now - last["created_at"] < COOLDOWN: raise HTTPException(429, "Please wait before requesting again")
        if not user: 
            c.execute("INSERT INTO otp_challenges(email,code_hash,expires_at,used,created_at) VALUES(?,?,?,1,?)",
                      (email, "-", now, now)); return None  # unknown account: same response, no email
        c.execute("UPDATE otp_challenges SET used=1 WHERE email=? AND used=0", (email,))  # invalidate old
        code = f"{secrets.randbelow(10**6):06d}"
        c.execute("INSERT INTO otp_challenges(email,code_hash,expires_at,created_at) VALUES(?,?,?,?)",
                  (email, h(email + code), now + OTP_TTL, now))
    return code, email

def verify_otp(email, code, device_token, ua, confirm_switch=False):
    """Returns dict(session_token, csrf, device_token|None, new_device, event_id, user)."""
    email = norm_email(email); now = int(time.time()); generic = HTTPException(401, "Invalid or expired code")
    with tx() as c:
        ch = c.execute("SELECT * FROM otp_challenges WHERE email=? AND used=0 ORDER BY id DESC LIMIT 1", (email,)).fetchone()
        if not ch or ch["expires_at"] < now or ch["attempts"] >= MAX_ATTEMPTS:
            raise generic
        ok = secrets.compare_digest(ch["code_hash"], h(email + (code or "")))
        c.execute("UPDATE otp_challenges SET attempts=attempts+1, used=? WHERE id=?", (1 if ok else 0, ch["id"]))
        if not ok:
            c.execute("COMMIT"); c.execute("BEGIN IMMEDIATE")  # persist the attempt count
            raise generic
        user = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if not user: raise generic
        if ONE_ACTIVE() and user["role"] == "student":
            active = c.execute("SELECT 1 FROM sessions WHERE user_id=? AND revoked=0 AND expires_at>?",
                               (user["id"], now)).fetchone()
            if active and not confirm_switch:
                c.execute("UPDATE otp_challenges SET used=0 WHERE id=?", (ch["id"],))
                c.execute("COMMIT"); c.execute("BEGIN IMMEDIATE")
                raise HTTPException(409, "Another session is active. Confirm to switch devices.")
            c.execute("UPDATE sessions SET revoked=1 WHERE user_id=? AND revoked=0", (user["id"],))
            audit(c, user["id"], "session_switch", "previous session revoked", now)
        dev_id, dev_tok, is_new = check_device(c, user["id"], device_token, ua, now)
        tok, csrf = new_token(), new_token()
        c.execute("INSERT INTO sessions(user_id,token_hash,csrf_token,device_id,created_at,expires_at)"
                  " VALUES(?,?,?,?,?,?)", (user["id"], h(tok), csrf, dev_id, now, now + SESSION_TTL))
        ev = None
        if is_new and user["role"] == "student":
            ev = record_event(c, user["id"], dev_id, "new_device_login", f"nd:{user['id']}:{dev_id}", now)
        audit(c, user["id"], "login", "new_device" if is_new else "known_device", now)
    return dict(session_token=tok, csrf=csrf, device_token=dev_tok, new_device=is_new, event_id=ev, user=dict(user))

def current_session(request: Request, require_csrf=False):
    tok = request.cookies.get("session")
    if not tok: raise HTTPException(401, "Not authenticated")
    r = query("SELECT s.id sid, s.csrf_token, s.user_id, u.email, u.name, u.role FROM sessions s JOIN users u ON u.id=s.user_id"
              " WHERE s.token_hash=? AND s.revoked=0 AND s.expires_at>?", (h(tok), int(time.time())))
    if not r: raise HTTPException(401, "Not authenticated")
    s = dict(r[0])
    if require_csrf and request.method not in ("GET", "HEAD", "OPTIONS"):
        if not secrets.compare_digest(request.headers.get("x-csrf-token", ""), s["csrf_token"]):
            raise HTTPException(403, "CSRF check failed")
    return s

def require_role(request: Request, *roles):
    s = current_session(request, require_csrf=True)
    if s["role"] not in roles: raise HTTPException(403, "Forbidden")
    return s

def logout(sid):
    with tx() as c: c.execute("UPDATE sessions SET revoked=1 WHERE id=?", (sid,))
