@echo off
chcp 65001 >nul

REM Prefer project venv Python if present; fallback to system py -3
set "PYTHON=.\.venv\Scripts\python.exe"
if not exist .\.venv\Scripts\python.exe (
  set "PYTHON=py -3"
)

echo [BUILD] Using interpreter: %PYTHON%
echo [BUILD] Installing/Updating build toolchain...
%PYTHON% -m pip install --upgrade pip setuptools wheel

REM Try pygame-ce first (wider wheel support). If it fails, try pygame.
%PYTHON% -m pip install pyinstaller pyquaternion numpy pygame-ce || %PYTHON% -m pip install pygame

echo [BUILD] Cleaning previous dist/build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__

echo [BUILD] Running PyInstaller with spec...
%PYTHON% -m PyInstaller elysia.spec

echo [DONE] Built: dist\Elysia\Elysia.exe
pause
