@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_windows.ps1 first.
  exit /b 1
)
".venv\Scripts\python.exe" scripts\demo_webcam.py
