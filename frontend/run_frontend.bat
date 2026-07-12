@echo off
setlocal
cd /d "%~dp0\.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" frontend\app.py --inbrowser
) else (
  python frontend\app.py --inbrowser
)
if errorlevel 1 pause
