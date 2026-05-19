@echo off
title Elysia Sovereign Engine Installer
color 0b

echo ========================================================
echo   ELYSIA SOVEREIGN ENGINE (Project Optimization)
echo   Target: GTX 1060 3GB | Goal: Uncensored AI
echo ========================================================
echo.

echo [1/3] Checking System Tools...
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Git not found.
    echo [!] Switching to MANUAL MODE (Portable Version).
    echo.
    echo    I will open the download page for "ComfyUI_windows_portable.7z".
    echo    Please download it, extract it, and run "run_nvidia_gpu.bat".
    echo.
    pause
    start https://github.com/comfyanonymous/ComfyUI/releases/latest
    goto :END
)

echo [OK] Git found. Clones initiated.
echo.

echo [2/3] Downloading Engine (ComfyUI)...
if exist "ComfyUI" (
    echo [!] ComfyUI folder already exists. Skipping clone.
) else (
    git clone https://github.com/comfyanonymous/ComfyUI
)

echo.
echo [3/3] Configuration
echo    We need to install the dependencies.
echo    Please run 'install_dependencies.bat' inside the ComfyUI folder.
echo.

:END
echo ========================================================
echo   NEXT STEPS:
echo   1. Run ComfyUI (run_nvidia_gpu.bat)
echo   2. Tell Elysia: "Engine is running"
echo   3. Elysia will inject 'optimized_3gb.json' into it.
echo ========================================================
pause
