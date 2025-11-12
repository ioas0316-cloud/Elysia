@echo off
chcp 65001 >nul

REM 1) venv 우선 실행
if exist ".venv\Scripts\python.exe" (
  echo [RUN] venv detected. Launching via Python...
  call .venv\Scripts\python.exe ElysiaStarter\scripts\elysia_start.py
  goto :end
)

REM 2) exe 실행
if exist "dist\Elysia\Elysia.exe" (
  echo [RUN] Launching packaged Elysia.exe...
  start "" "dist\Elysia\Elysia.exe"
  goto :end
)

echo [ERROR] 실행 대상이 없습니다.
echo - venv 또는 dist\Elysia\Elysia.exe 중 하나를 준비하세요.
echo - 빌드하려면: start_build.bat
:end
pause

