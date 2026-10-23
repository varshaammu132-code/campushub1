import re
import database, server, notifications
from fastapi.testclient import TestClient

UA = "Mozilla/5.0 (Windows NT 10.0) Chrome/120.0 Safari/537.36"
def login(client, sent, email="student@sjcit.ac.in", **extra):
    server._hits.clear()
    with database.tx() as d: d.execute("DELETE FROM otp_challenges WHERE email=?", (email,)) if False else None
    client.post("/auth/request-otp", json={"email": email})
    code = re.search(r"code is (\d{6})", sent[-1][2]).group(1)
    return client.post("/auth/verify", json={"email": email, "code": code, **extra}, headers={"user-agent": UA})

def fresh_cooldown():
    with database.tx() as d: d.execute("UPDATE otp_challenges SET created_at=created_at-120")

def test_health_and_chat_contract(env):
    c, _ = env
    assert c.get("/health").json() == {"ok": True}
    assert set(c.post("/chat", json={"question": "hi"}).json()) == {"answer", "sources"}
    assert c.post("/chat", json={"question": "x" * 501}).status_code == 422

def test_new_device_alerts_teacher_once(env):
    c, sent = env
    r = login(c, sent); assert r.status_code == 200 and r.json()["new_device"]
    mails = [m for m in sent if "New Device" in m[1]]
    assert len(mails) == 1 and mails[0][0] == "teacher@sjcit.ac.in"
    assert "st***@sjcit.ac.in" in mails[0][2] and "student@sjcit" not in mails[0][2]
    notifications.dispatch_new_device_alert(1, 3)  # retry must not duplicate
    assert len([m for m in sent if "New Device" in m[1]]) == 1

def test_recognized_device_no_alert(env):
    c, sent = env
    login(c, sent); c.post("/auth/logout", headers={"x-csrf-token": c.get("/auth/me").json()["csrf"]})
    fresh_cooldown(); n = len(sent); r = login(c, sent)
    assert r.status_code == 200 and not r.json()["new_device"]
    assert not [m for m in sent[n:] if "New Device" in m[1]]

def test_one_active_session_and_switch(env):
    c, sent = env; assert login(c, sent).status_code == 200
    c2 = TestClient(server.app); fresh_cooldown(); assert login(c2, sent).status_code == 409
    assert c.get("/auth/me").status_code == 200
    assert c2.post("/auth/verify", json={"email": "student@sjcit.ac.in", "code": re.search(r"code is (\d{6})", sent[-1][2]).group(1), "confirm_switch": True}).status_code == 200
    assert c.get("/auth/me").status_code == 401  # revoked server-side

def test_otp_attempt_limit_and_no_enumeration(env):
    c, sent = env
    c.post("/auth/request-otp", json={"email": "student@sjcit.ac.in"})
    code = re.search(r"code is (\d{6})", sent[-1][2]).group(1)
    for _ in range(5): assert c.post("/auth/verify", json={"email": "student@sjcit.ac.in", "code": "000000"}).status_code == 401
    server._hits.clear()
    assert c.post("/auth/verify", json={"email": "student@sjcit.ac.in", "code": code}).status_code == 401
    assert c.post("/auth/request-otp", json={"email": "nobody@sjcit.ac.in"}).json()["ok"]

def test_otp_expiry(env):
    c, sent = env; c.post("/auth/request-otp", json={"email": "student@sjcit.ac.in"})
    code = re.search(r"code is (\d{6})", sent[-1][2]).group(1)
    with database.tx() as d: d.execute("UPDATE otp_challenges SET expires_at=0")
    assert c.post("/auth/verify", json={"email": "student@sjcit.ac.in", "code": code}).status_code == 401

def test_roles_csrf_teacher_scope_admin(env):
    c, sent = env; login(c, sent)
    assert c.get("/admin/stats").status_code == 403 and c.get("/teacher/alerts").status_code == 403
    t = TestClient(server.app); login(t, sent, "teacher@sjcit.ac.in")
    assert len(t.get("/teacher/alerts").json()) == 1
    assert t.post("/teacher/alerts/1", json={"reviewed": True}).status_code == 403
    csrf = t.get("/auth/me").json()["csrf"]
    assert t.post("/teacher/alerts/1", json={"reviewed": True}, headers={"x-csrf-token": csrf}).status_code == 200
    a = TestClient(server.app); login(a, sent, "admin@sjcit.ac.in")
    assert a.get("/admin/stats").json()["new_device_alerts"] == 1

def test_smtp_failure_recorded_not_fatal(env, monkeypatch):
    c, sent = env
    def boom(*a): raise OSError("smtp down")
    monkeypatch.setattr(notifications, "send_email", boom)
    server._hits.clear(); c.post("/auth/request-otp", json={"email": "student@sjcit.ac.in"})  # OTP mail fails silently
    with database.tx() as d: d.execute("UPDATE otp_challenges SET code_hash=?", (__import__("security").h("student@sjcit.ac.in123456"),))
    assert c.post("/auth/verify", json={"email": "student@sjcit.ac.in", "code": "123456"}).status_code == 200
    assert database.query("SELECT COUNT(*) FROM notification_attempts WHERE status='failed'")[0][0] == 1
