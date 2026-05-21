@echo off
title Elysia World OS Daemon
set PYTHONIOENCODING=utf-8
echo 🌀 Starting Elysia Deep Mind OS Daemon...
start /b python -u elysia_daemon.py
timeout /t 3 /nobreak > nul
echo 🌐 Launching Real-time Web Dashboard...
start http://localhost:8000
echo.
echo ==================================================
echo [Elysia Daemon is running in background]
echo Web Dashboard URL: http://localhost:8000
echo Keep this terminal open to monitor basic processes.
echo Press Ctrl+C in this terminal to shut down.
echo ==================================================
pause
