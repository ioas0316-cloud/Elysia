@echo off
title Elysia Stream Watcher
cd /d "%~dp0"

echo ========================================================
echo  Starting Elysia's Stream Watcher...
echo ========================================================

:: Check for venv
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe Demos/Philosophy/stream_watcher.py
) else (
    python Demos/Philosophy/stream_watcher.py
)

pause
