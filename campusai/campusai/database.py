"""SQLite layer: schema init + helpers. All SQL is parameterized."""
import os, sqlite3, threading
from contextlib import contextmanager

DB_PATH = os.environ.get("CAMPUSAI_DB", "campusai_auth.db")
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT, student_id TEXT,
  department TEXT, role TEXT NOT NULL DEFAULT 'student', created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS faculty_assignments(
  faculty_id INTEGER NOT NULL, student_id INTEGER NOT NULL, PRIMARY KEY(faculty_id, student_id));
CREATE TABLE IF NOT EXISTS otp_challenges(
  id INTEGER PRIMARY KEY, email TEXT NOT NULL, code_hash TEXT NOT NULL, expires_at INTEGER NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0, used INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS ix_otp_email ON otp_challenges(email, created_at);
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, token_hash TEXT UNIQUE NOT NULL,
  csrf_token TEXT NOT NULL, device_id INTEGER, created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_sess_user ON sessions(user_id, revoked);
CREATE TABLE IF NOT EXISTS devices(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, token_hash TEXT NOT NULL,
  device_type TEXT, browser TEXT, os TEXT, first_seen INTEGER NOT NULL,
  last_seen INTEGER NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_dev_user ON devices(user_id, token_hash);
CREATE TABLE IF NOT EXISTS security_events(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, device_id INTEGER, kind TEXT NOT NULL,
  dedupe_key TEXT UNIQUE, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS teacher_alerts(
  id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL, faculty_id INTEGER, student_id INTEGER NOT NULL,
  reviewed INTEGER NOT NULL DEFAULT 0, flagged INTEGER NOT NULL DEFAULT 0, note TEXT DEFAULT '',
  created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS notification_attempts(
  id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL, recipient TEXT NOT NULL,
  status TEXT NOT NULL, error TEXT, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS audit_events(
  id INTEGER PRIMARY KEY, actor_id INTEGER, action TEXT NOT NULL, detail TEXT, created_at INTEGER NOT NULL);
"""

def connect():
    c = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON"); c.execute("PRAGMA journal_mode=WAL")
    return c

@contextmanager
def tx():
    """Serialized write transaction (BEGIN IMMEDIATE) - prevents login races."""
    with _lock:
        c = connect()
        try:
            c.execute("BEGIN IMMEDIATE"); yield c; c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK"); raise
        finally:
            c.close()

def query(sql, args=()):
    c = connect()
    try: return c.execute(sql, args).fetchall()
    finally: c.close()

def init_db():
    c = connect(); c.executescript(SCHEMA); c.close()
    try: os.chmod(DB_PATH, 0o600)
    except OSError: pass

def audit(c, actor_id, action, detail="", now=None):
    import time
    c.execute("INSERT INTO audit_events(actor_id,action,detail,created_at) VALUES(?,?,?,?)",
              (actor_id, action, detail, now or int(time.time())))
