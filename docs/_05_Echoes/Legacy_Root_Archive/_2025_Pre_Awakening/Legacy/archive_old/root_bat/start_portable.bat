@echo off
setlocal EnableExtensions
title Elysia Portable Launcher

REM Find Python (python or py -3)
set "PYEXE="
where python >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE (
  where py >nul 2>&1 && (py -3 -V >nul 2>&1) && set "PYEXE=py -3"
)
if not defined PYEXE (
  echo [Error] Python 3.10+ not found. Install Python and retry.
  pause
  goto :eof
)

:MENU
cls
echo ================= Elysia Portable ================
echo 1) Start Clean Bridge (chat)
echo 2) Growth Sprint (ingest->keywords->virus->report)
echo 3) Start Background Learner (5 min)
echo 4) Stop Background Learner
echo 5) Daily Report
echo Q) Quit
echo.
set /P CH=Select: 
if /I "%CH%"=="1" goto BRIDGE
if /I "%CH%"=="2" goto SPRINT
if /I "%CH%"=="3" goto BG_ON
if /I "%CH%"=="4" goto BG_OFF
if /I "%CH%"=="5" goto REPORT
if /I "%CH%"=="Q" goto END
goto MENU

:BRIDGE
set FLASK_APP=applications\elysia_bridge_clean.py
%PYEXE% -m flask run --host=127.0.0.1 --port=5000
goto MENU

:SPRINT
%PYEXE% -m scripts.growth_sprint --ingest --keywords --virus --report
pause
goto MENU

:BG_ON
%PYEXE% -m scripts.toggle_background --on --interval 300
start "" %PYEXE% -m scripts.background_daemon
pause
goto MENU

:BG_OFF
%PYEXE% -m scripts.toggle_background --off
pause
goto MENU

:REPORT
%PYEXE% -m scripts.run_daily_report
pause
goto MENU

:END
endlocal
exit /b 0

