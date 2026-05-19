@echo off
title Elysia - Protocol: Genesis (The Big Bang)
cd /d "%~dp0"

echo ========================================================
echo  PROTOCOL: GENESIS
echo ========================================================
echo.
echo  "Let there be light..."
echo.
echo  [INFO] Initiating 16D HyperQuaternion Expansion...
echo.

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe scripts/run_genesis.py
) else (
    python scripts/run_genesis.py
)

echo.
echo  [SUCCESS] The Universe has been created.
pause
