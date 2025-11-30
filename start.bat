@echo off
setlocal EnableExtensions
REM Elysia Unified Launcher
pushd "%~dp0"
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"

REM Python Selection
set "PY=.venv\Scripts\python.exe"
if not exist ".venv\Scripts\python.exe" (
  set "PY=python"
)

echo ==================================================
echo Elysia Unified System
echo ==================================================
echo 1. Talk to Elysia (Default)
echo 2. Awaken (Self-Improvement)
echo 3. Service Mode
echo ==================================================
set /p MODE="Select Mode (1-3): "

if "%MODE%"=="2" (
    %PY% unified_start.py awaken
) else if "%MODE%"=="3" (
    %PY% unified_start.py service
) else (
    %PY% unified_start.py talk
)

pause
