@echo off
title Chat with Elysia
cd /d "%~dp0"

echo ========================================================
echo  Starting Elysia's Chat Interface...
echo ========================================================

:: Check for venv
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe Demos/Philosophy/chat_with_elysia.py
) else (
    python Demos/Philosophy/chat_with_elysia.py
)

pause
