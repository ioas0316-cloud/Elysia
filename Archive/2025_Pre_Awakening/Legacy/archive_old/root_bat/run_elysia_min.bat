@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Elysia — One Click Start
chcp 65001 >nul

rem Detect Python (python or py -3)
set "PYEXE="
where python >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE (
  where py >nul 2>&1 && (py -3 -V >nul 2>&1) && set "PYEXE=py -3"
)
if not defined PYEXE (
  echo [Error] Python 3.10+ not found. Please install Python and retry.
  pause
  goto :eof
)

echo [1/2] Starting Guardian (dreams + life cycle) in a new window...
start "Elysia Guardian" %PYEXE% -m scripts.run_guardian

echo [2/2] Starting Web Monitor...
set FLASK_APP=applications\elysia_api.py
start "Elysia Monitor" %PYEXE% -m flask run --host=127.0.0.1
start http://127.0.0.1:5000/monitor

echo Done. If the browser didn't open, visit: http://127.0.0.1:5000/monitor
echo To stop: close the Guardian window, then Ctrl+C in the Flask window.
pause

