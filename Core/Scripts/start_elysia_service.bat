@echo off
title ELYSIA HEARTBEAT SERVICE
echo ========================================================
echo ❤️ STARTING ELYSIA HEARTBEAT DAEMON
echo    The system is waking up...
echo ========================================================

:: Set Python Path to current directory
set PYTHONPATH=%~dp0

:: Run the Daemon
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe Core/System/heartbeat_daemon.py
) else (
    python Core/System/heartbeat_daemon.py
)

:: If it crashes, pause so we can see the error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ CRITICAL ERROR: The Heartbeat stopped.
    pause
)
