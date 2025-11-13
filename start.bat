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
  rem include scipy (required by Project_Sophia.core.world)
  call %PY% -m pip -q install pygame-ce pyquaternion numpy scipy
)

REM prefer packaged exe
if exist "dist\Elysia\Elysia.exe" (
  start "" "dist\Elysia\Elysia.exe"
  goto :done
)

REM else run scripts
set "PYTHONPATH=%CD%"
if exist "ElysiaStarter\scripts\animated_event_visualizer.py" (
  call %PY% "ElysiaStarter\scripts\animated_event_visualizer.py"
) else if exist "ElysiaStarter\scripts\visualize_timeline.py" (
  call %PY% "ElysiaStarter\scripts\visualize_timeline.py"
) else (
  echo ERROR: Starter not found at ElysiaStarter\scripts\animated_event_visualizer.py or visualize_timeline.py
)

:done
popd
pause
endlocal
