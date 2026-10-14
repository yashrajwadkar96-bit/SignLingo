@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_windows.ps1 first.
  exit /b 1
)
".venv\Scripts\python.exe" scripts\train_my_signs.py
