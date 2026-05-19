@echo off
setlocal ENABLEDELAYEDEXPANSION
REM Launch Python bridge server + Godot project automatically

pushd %~dp0\..

if exist .venv\Scripts\python.exe (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo [run] using python: %PY%
if not exist tools\godot_bridge_server.py (
  echo [ERROR] tools\godot_bridge_server.py not found. Aborting.
  popd & exit /b 1
)

if not exist logs mkdir logs >nul 2>&1

echo [run] starting bridge server on ws://127.0.0.1:8765
REM Use cmd /c with nested quotes so redirection happens in the new process
start "Elysia Bridge" /min cmd /c ""%PY%" tools\godot_bridge_server.py --host 127.0.0.1 --port 8765 --rate 4 1^>logs\godot_bridge.out 2^>logs\godot_bridge.err"

REM small wait so the server can bind the port
ping -n 3 127.0.0.1 >nul

set "GODOT_EXE=ElysiaGodot\Godot_v4.5.1-stable_win64_console.exe"
if not exist "%GODOT_EXE%" (
  for %%G in (ElysiaGodot\Godot*.exe) do (
    set "GODOT_EXE=%%G"
    goto :found
  )
)

:found
if exist "%GODOT_EXE%" (
  echo [run] launching Godot: %GODOT_EXE%
  start "Elysia Godot" "%GODOT_EXE%" --path ElysiaGodot
  echo [run] Godot started. Logs: logs\godot_bridge.out / logs\godot_bridge.err
) else (
  echo [ERROR] Godot executable not found in "ElysiaGodot\".
  echo         Download Godot 4 Standard (portable) and place the EXE inside ElysiaGodot\
  echo         Then re-run scripts\run_worldviewer.bat
  popd & exit /b 1
)

popd
endlocal
