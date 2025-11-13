@echo off
setlocal EnableExtensions
REM Minimal, ASCII-safe launcher

REM fix working directory to this script location
pushd "%~dp0"

REM pick Python: prefer venv, else py -3, else python
set "PY=.venv\Scripts\python.exe"
if not exist ".venv\Scripts\python.exe" (
  set "PY=py -3"
)
where py >nul 2>&1 || (
  if "%PY%"=="py -3" (
    where python >nul 2>&1 && set "PY=python"
  )
)

REM show which interpreter
echo Using Python: %PY%

REM ensure pip and basic deps
call %PY% -m pip --disable-pip-version-check -q install --upgrade pip setuptools wheel
if exist requirements.txt (
  call %PY% -m pip -q install -r requirements.txt
) else (
  call %PY% -m pip -q install pygame-ce pyquaternion numpy
)

REM prefer packaged exe
if exist "dist\Elysia\Elysia.exe" (
  start "" "dist\Elysia\Elysia.exe"
  goto :done
)

REM else run scripts
if exist "ElysiaStarter\scripts\visualize_timeline.py" (
  call %PY% "ElysiaStarter\scripts\visualize_timeline.py"
) else (
  echo ERROR: Starter not found at ElysiaStarter\scripts\visualize_timeline.py
)

:done
popd
pause
endlocal
