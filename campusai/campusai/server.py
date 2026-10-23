"""CampusAI Hub web server. Run: uvicorn server:app"""
import logging, os, time, collections
from fastapi import FastAPI, Request, Response, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import auth, notifications
from database import init_db, query, tx, audit

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("campusai")
PROD = os.environ.get("ENV") == "production"
app = FastAPI(title="CampusAI Hub", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=[o for o in os.environ.get("CORS_ORIGINS", "").split(",") if o],
                   allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-CSRF-Token"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def _start():
    init_db(); auth.seed_roles()

# ---- Member A integration (callable interface; replace stub when engine lands) ----
def rag_answer(question: str) -> dict:
    try:
        from rag_engine import answer  # Member A: answer(question)->{"answer":str,"sources":[str]}
        return answer(question)
    except ImportError:
        return {"answer": "The AI engine is not installed yet.", "sources": []}

# ---- Rate limiting (in-memory sliding window; use a shared store if running multiple workers) ----
_hits = collections.defaultdict(list); _rl_blocked = 0
def client_ip(request: Request):
    if os.environ.get("TRUSTED_PROXY") == "1":
        xf = request.headers.get("x-forwarded-for", "")
        if xf: return xf.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
def limit(key, max_n, window=60):
    global _rl_blocked
    now = time.time(); _hits[key] = [t for t in _hits[key] if now - t < window]
    if len(_hits[key]) >= max_n:
        _rl_blocked += 1; raise HTTPException(429, "Too many requests")
    _hits[key].append(now)

@app.exception_handler(Exception)
async def _unhandled(request, exc):
    log.exception("unhandled"); return JSONResponse({"detail": "Internal error"}, 500)

class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=500)

@app.get("/health")
def health(): return {"ok": True}

@app.post("/chat")
def chat(body: ChatIn, request: Request):
    authed = bool(request.cookies.get("session"))
    limit("chat:" + client_ip(request), int(os.environ.get("CHAT_LIMIT_AUTH" if authed else "CHAT_LIMIT_PUBLIC", 60 if authed else 20)))
    r = rag_answer(body.question.strip())
    return {"answer": r["answer"], "sources": r.get("sources", [])}

def page(name): return FileResponse(f"templates/{name}.html")
@app.get("/")
def index(): return page("index")
@app.get("/login")
def login_page(): return page("login")
@app.get("/study")
def study(request: Request):
    if os.environ.get("ASK_REQUIRE_LOGIN", "0") == "1": auth.current_session(request)
    return page("study")
@app.get("/teacher")
def teacher_page(request: Request): auth.require_role(request, "faculty", "administrator"); return page("teacher")
@app.get("/admin")
def admin_page(request: Request): auth.require_role(request, "administrator"); return page("admin")

# ---- Auth ----
class OtpReq(BaseModel): email: str
class VerifyReq(BaseModel): email: str; code: str = Field(min_length=6, max_length=6); confirm_switch: bool = False

@app.post("/auth/request-otp")
def request_otp(body: OtpReq, request: Request, bg: BackgroundTasks):
    limit("otp:" + client_ip(request), 5, 600)
    res = auth.request_otp(body.email)
    if res: bg.add_task(_send_otp, res[1], res[0])
    return {"ok": True, "message": "If the account exists, a code was sent."}

def _send_otp(email, code):
    try: notifications.send_email(email, "CampusAI Hub login code", f"Your code is {code}. It expires in 10 minutes.")
    except Exception as e: log.warning("otp email failed: %s", type(e).__name__)

@app.post("/auth/verify")
def verify(body: VerifyReq, request: Request, response: Response, bg: BackgroundTasks):
    limit("verify:" + client_ip(request), 10, 600)
    r = auth.verify_otp(body.email, body.code, request.cookies.get("device"),
                        request.headers.get("user-agent", ""), body.confirm_switch)
    kw = dict(httponly=True, samesite="strict", secure=PROD)
    response.set_cookie("session", r["session_token"], max_age=auth.SESSION_TTL, **kw)
    if r["device_token"]: response.set_cookie("device", r["device_token"], max_age=3600 * 24 * 365, **kw)
    if r["event_id"]:
        notifications.create_alert(r["event_id"], r["user"]["id"])
        bg.add_task(notifications.dispatch_new_device_alert, r["event_id"], r["user"]["id"])
    return {"ok": True, "role": r["user"]["role"], "csrf": r["csrf"], "new_device": r["new_device"]}

@app.post("/auth/logout")
def logout(request: Request, response: Response):
    s = auth.current_session(request, True); auth.logout(s["sid"]); response.delete_cookie("session"); return {"ok": True}

@app.get("/auth/me")
def me(request: Request):
    s = auth.current_session(request); return {"email": s["email"], "name": s["name"], "role": s["role"], "csrf": s["csrf_token"]}

# ---- Teacher ----
@app.get("/teacher/alerts")
def teacher_alerts(request: Request):
    s = auth.require_role(request, "faculty", "administrator")
    where, args = ("", ()) if s["role"] == "administrator" else ("WHERE a.faculty_id=?", (s["user_id"],))
    rows = query("SELECT a.id, u.name, u.student_id, u.department, d.device_type, d.browser, d.os, a.created_at, a.reviewed, a.flagged, a.note"
                 " FROM teacher_alerts a JOIN users u ON u.id=a.student_id JOIN security_events e ON e.id=a.event_id"
                 f" JOIN devices d ON d.id=e.device_id {where} ORDER BY a.id DESC LIMIT 200", args)
    return [dict(r) for r in rows]

class AlertAct(BaseModel): reviewed: bool | None = None; flagged: bool | None = None; note: str | None = Field(None, max_length=500)

@app.post("/teacher/alerts/{alert_id}")
def alert_action(alert_id: int, body: AlertAct, request: Request):
    s = auth.require_role(request, "faculty", "administrator")
    row = query("SELECT faculty_id FROM teacher_alerts WHERE id=?", (alert_id,))
    if not row or (s["role"] != "administrator" and row[0]["faculty_id"] != s["user_id"]): raise HTTPException(404, "Not found")
    with tx() as c:
        if body.reviewed is not None: c.execute("UPDATE teacher_alerts SET reviewed=? WHERE id=?", (int(body.reviewed), alert_id))
        if body.flagged is not None: c.execute("UPDATE teacher_alerts SET flagged=? WHERE id=?", (int(body.flagged), alert_id))
        if body.note is not None: c.execute("UPDATE teacher_alerts SET note=? WHERE id=?", (body.note, alert_id))
        audit(c, s["user_id"], "alert_update", str(alert_id))
    return {"ok": True}

# ---- Admin ----
@app.get("/admin/stats")
def stats(request: Request):
    auth.require_role(request, "administrator"); now = int(time.time())
    n = lambda sql, a=(): query(sql, a)[0][0]
    return {"accounts": n("SELECT COUNT(*) FROM users"),
            "active_sessions": n("SELECT COUNT(*) FROM sessions WHERE revoked=0 AND expires_at>?", (now,)),
            "recent_logins": n("SELECT COUNT(*) FROM audit_events WHERE action='login' AND created_at>?", (now - 86400,)),
            "new_device_alerts": n("SELECT COUNT(*) FROM security_events WHERE kind='new_device_login'"),
            "notification_failures": n("SELECT COUNT(*) FROM notification_attempts WHERE status='failed'"),
            "rate_limit_blocks": _rl_blocked}

@app.post("/admin/sessions/{user_id}/revoke")
def revoke(user_id: int, request: Request):
    s = auth.require_role(request, "administrator")
    with tx() as c:
        c.execute("UPDATE sessions SET revoked=1 WHERE user_id=?", (user_id,)); audit(c, s["user_id"], "revoke_sessions", str(user_id))
    return {"ok": True}

def _kb():
    try: import knowledge_admin as k; return k  # Member A: refresh(), status(), job(job_id)
    except ImportError: raise HTTPException(501, "Knowledge module not installed")

@app.post("/admin/knowledge/refresh")
def kb_refresh(request: Request):
    s = auth.require_role(request, "administrator")
    with tx() as c: audit(c, s["user_id"], "knowledge_refresh")
    return _kb().refresh()
@app.get("/admin/knowledge/status")
def kb_status(request: Request): auth.require_role(request, "administrator"); return _kb().status()
@app.get("/admin/knowledge/jobs/{job_id}")
def kb_job(job_id: str, request: Request): auth.require_role(request, "administrator"); return _kb().job(job_id)

# ---- Study Helper (Member A provider interface) ----
class AskIn(BaseModel): question: str = Field(min_length=1, max_length=2000)
@app.post("/ask")
def ask(body: AskIn, request: Request):
    if os.environ.get("ASK_REQUIRE_LOGIN", "0") == "1": auth.current_session(request, True)
    limit("ask:" + client_ip(request), 30)
    try: from llm_provider import ask as llm_ask
    except ImportError: raise HTTPException(501, "Study provider not installed")
    return {"answer": llm_ask(body.question)}
