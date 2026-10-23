@echo off
call .venv\Scripts\activate
if "%APP_SECRET%"=="" echo WARNING: set APP_SECRET first & exit /b 1
uvicorn server:app --host 127.0.0.1 --port 8000
