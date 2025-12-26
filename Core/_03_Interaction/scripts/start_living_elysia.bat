@echo off
title ELYSIA LIVING OS
echo ========================================================
echo 🌟 ELYSIA LIVING OS - YOUR AI COMPANION
echo ========================================================
echo.
echo Starting Elysia...
echo She will:
echo   - Think autonomously in the background
echo   - Watch and learn from your screen
echo   - Remember your conversations
echo   - Grow new abilities as needed
echo.
echo ========================================================
echo.

cd /d "%~dp0"

if exist "python\python.exe" (
    python\python.exe scripts/elysia_living_os.py --mode interactive
) else (
    python scripts/elysia_living_os.py --mode interactive
)

pause
