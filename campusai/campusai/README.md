# CampusAI Hub — web/security module (Member B)
Install: `./install_linux.sh` (Linux) or `install_windows.bat` (Windows).
Test: `python -m pytest -q`. Run: `uvicorn server:app --port 8000` (set `APP_SECRET`, `ENV=production` behind HTTPS).
Env: TEACHER_ALERT_MODE (faculty|admin|both|off), ADMIN_ALERT_EMAIL, ONE_ACTIVE_SESSION, ASK_REQUIRE_LOGIN, SMTP_*, CORS_ORIGINS, TRUSTED_PROXY, TZ_OFFSET_MIN.
Integration points for Member A: `rag_engine.answer(q)`, `knowledge_admin.refresh/status/job`, `llm_provider.ask(q)`.
Limits: device detection is approximate (cookie-based, spoofable); OTP login cannot stop credential sharing or a compromised device;
domain email != enrollment — keep `roles.json` as an approved roster.
