@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
start "" ".venv\Scripts\python.exe" main.py
