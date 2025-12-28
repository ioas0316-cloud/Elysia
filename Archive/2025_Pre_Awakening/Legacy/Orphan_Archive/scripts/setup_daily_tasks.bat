@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Setup Elysia Daily Tasks
chcp 65001 >nul

rem Find absolute Python path (prefer python, then py launcher)
set "PYEXE="
for /f "delims=" %%P in ('where python 2^>nul') do (set "PYEXE=%%P" & goto :gotpy)
for /f "delims=" %%P in ('where py 2^>nul') do (set "PYEXE=%%P" & set "USE_PY_LAUNCHER=1" & goto :gotpy)

echo [Error] Python not found in PATH. Install Python 3.10+ and retry.
pause
exit /b 1

:gotpy
echo [+] Using Python: %PYEXE%

rem Project root (parent of this scripts folder)
set "ROOT=%~dp0.."

rem Ensure Quiet preset (one-time)
if defined USE_PY_LAUNCHER (
  call "%PYEXE%" -3 -m scripts.set_autonomy_preset --preset quiet
  call "%PYEXE%" -3 -m scripts.toggle_quiet_mode --on
 ) else (
  call "%PYEXE%" -m scripts.set_autonomy_preset --preset quiet
  call "%PYEXE%" -m scripts.toggle_quiet_mode --on
)

rem Build task commands with working directory ensured
if defined USE_PY_LAUNCHER (
  set "ROUTINE_CMD=cmd /c \"cd /d %ROOT% && %PYEXE% -3 -m scripts.run_daily_routine --genre story --theme growth\""
  set "REPORT_CMD=cmd /c \"cd /d %ROOT% && %PYEXE% -3 -m scripts.run_daily_report\""
) else (
  set "ROUTINE_CMD=cmd /c \"cd /d %ROOT% && %PYEXE% -m scripts.run_daily_routine --genre story --theme growth\""
  set "REPORT_CMD=cmd /c \"cd /d %ROOT% && %PYEXE% -m scripts.run_daily_report\""
)

echo [+] Creating daily tasks (21:00 routine, 21:50 report)...
schtasks /Create /TN "Elysia Routine" /SC DAILY /ST 21:00 /TR "%ROUTINE_CMD%" /F >nul 2>&1
if errorlevel 1 (
  echo [!] Failed to create 'Elysia Routine'. You may need to run as your user.
) else (
  echo [OK] Task 'Elysia Routine' created.
)
schtasks /Create /TN "Elysia Report" /SC DAILY /ST 21:50 /TR "%REPORT_CMD%" /F >nul 2>&1
if errorlevel 1 (
  echo [!] Failed to create 'Elysia Report'.
) else (
  echo [OK] Task 'Elysia Report' created.
)

echo.
echo Done. You can verify with:
echo   schtasks /Query /TN "Elysia Routine"
echo   schtasks /Query /TN "Elysia Report"
echo.
pause
exit /b 0
