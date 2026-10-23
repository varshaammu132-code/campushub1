import os, sys, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.update(APP_SECRET="test-secret", TEACHER_ALERT_MODE="faculty", ONE_ACTIVE_SESSION="1")
@pytest.fixture
def env(tmp_path, monkeypatch):
    import database, notifications, server, auth
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "t.db"))
    database.init_db(); auth.seed_roles(os.path.join(os.path.dirname(__file__), "..", "roles.json"))
    server._hits.clear(); sent = []
    monkeypatch.setattr(notifications, "send_email", lambda to, s, b: sent.append((to, s, b)))
    from fastapi.testclient import TestClient
    return TestClient(server.app), sent
