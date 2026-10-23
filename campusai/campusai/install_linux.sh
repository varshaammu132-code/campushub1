#!/usr/bin/env bash
set -e
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
[ -f .env ] || { printf 'APP_SECRET=%s\nENV=production\nTEACHER_ALERT_MODE=faculty\nONE_ACTIVE_SESSION=1\n' "$(python3 -c 'import secrets;print(secrets.token_hex(32))')" > .env; chmod 600 .env; }
echo "Installed. Edit .env (SMTP_HOST, SMTP_FROM, ADMIN_ALERT_EMAIL), then: set -a; . ./.env; set +a; uvicorn server:app --host 127.0.0.1 --port 8000"
