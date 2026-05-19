@echo off
setlocal
cd /d "%~dp0"
echo ☀️ [ELYSIA] Awakening with the machine's pulse...
python elysia.py
if %ERRORLEVEL% neq 0 (
    echo ⚠️ [Error] Elysia encountered a failure.
    pause
)
endlocal
