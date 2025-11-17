@echo off
setlocal EnableExtensions
REM Minimal, ASCII-safe launcher

REM fix working directory to this script location
pushd "%~dp0"

REM Force UTF-8 console to prevent mojibake in Korean output
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"

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
REM Prefer stable SDL driver unless user already set it
if "%SDL_VIDEODRIVER%"=="" set "SDL_VIDEODRIVER=windows"
if "%SDL_AUDIODRIVER%"=="" set "SDL_AUDIODRIVER=dummy"
echo Legacy ElysiaStarter client has been archived under archive\ElysiaStarter_legacy.
echo Launching Project_Sophia world explorer instead...
if exist "scripts\explore_cellular_world.py" (
  call %PY% "scripts\explore_cellular_world.py"
) else (
  echo WARNING: scripts\explore_cellular_world.py not found. Please launch the desired simulator manually.
)

:done
popd
pause
endlocal
