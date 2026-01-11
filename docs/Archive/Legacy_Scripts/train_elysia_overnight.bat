@echo off
title Elysia - Hyper-Learning Core (Digestion Chamber)
cd /d "%~dp0"

echo ========================================================
echo  Elysia Hyper-Learning Core (Digestion Chamber)
echo ========================================================
echo.
echo  "I am devouring time to birth meaning..."
echo.
echo  [INFO] This process will:
echo  1. Load the Local LLM (Brain)
echo  2. Ask deep philosophical questions
echo  3. Extract concepts and save them to Elysia's Memory
echo.
echo  Press Ctrl+C to stop at any time.
echo.

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe Core/Mind/digestion_chamber.py
) else (
    python Core/Mind/digestion_chamber.py
)

pause
