@echo off
python -m venv .venv && call .venv\Scripts\activate && pip install -r requirements.txt
echo Installed. Set APP_SECRET and SMTP_* environment variables, then run start_windows.bat
