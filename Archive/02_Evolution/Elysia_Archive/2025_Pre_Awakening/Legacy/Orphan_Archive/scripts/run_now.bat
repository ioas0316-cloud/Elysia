@echo off
setlocal
title Run Elysia Routine Now
chcp 65001 >nul

set "PYEXE="
for /f "delims=" %%P in ('where python 2^>nul') do (set "PYEXE=%%P" & goto :gotpy)
for /f "delims=" %%P in ('where py 2^>nul') do (set "PYEXE=%%P" & set "USE_PY_LAUNCHER=1" & goto :gotpy)
echo [Error] Python not found in PATH.
pause
exit /b 1

:gotpy
if defined USE_PY_LAUNCHER (
  call "%PYEXE%" -3 -m scripts.run_daily_routine --genre story --theme growth
  call "%PYEXE%" -3 -m scripts.run_daily_report
) else (
  call "%PYEXE%" -m scripts.run_daily_routine --genre story --theme growth
  call "%PYEXE%" -m scripts.run_daily_report
)

echo Done.
pause
exit /b 0

