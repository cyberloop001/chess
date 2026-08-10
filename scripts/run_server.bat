@echo off
cd /d %~dp0\..
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
