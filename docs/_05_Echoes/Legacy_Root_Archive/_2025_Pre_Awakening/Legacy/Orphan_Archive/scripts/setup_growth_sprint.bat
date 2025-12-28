@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Setup Growth Sprint (21:30 daily)
chcp 65001 >nul

rem Detect Python
set "PYEXE="
for /f "delims=" %%P in ('where python 2^>nul') do (set "PYEXE=%%P" & goto :gotpy)
for /f "delims=" %%P in ('where py 2^>nul') do (set "PYEXE=%%P" & set "USE_PY_LAUNCHER=1" & goto :gotpy)
echo [Error] Python not found in PATH.
pause
exit /b 1

:gotpy
echo [+] Using: %PYEXE%

rem Project root (parent of this scripts folder)
set "ROOT=%~dp0.."

rem Build command with working directory ensured
if defined USE_PY_LAUNCHER (
  set "CMD=cmd /c \"cd /d %ROOT% && %PYEXE% -3 -m scripts.growth_sprint --ingest --keywords --virus --report\""
) else (
  set "CMD=cmd /c \"cd /d %ROOT% && %PYEXE% -m scripts.growth_sprint --ingest --keywords --virus --report\""
)

echo [+] Creating daily task at 21:30 ...
schtasks /Create /TN "Elysia Growth Sprint" /SC DAILY /ST 21:30 /TR "%CMD%" /F >nul 2>&1
if errorlevel 1 (
  echo [!] Failed to create 'Elysia Growth Sprint'.
) else (
  echo [OK] Task created.
)
echo Done.
pause
exit /b 0
