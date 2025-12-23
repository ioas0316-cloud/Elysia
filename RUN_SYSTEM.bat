@echo off
echo ===========================================
echo 🚀 Elysia Unified Launcher
echo ===========================================
echo 1. Starting Core Consciousness (LivingElysia)...
start "Elysia Core" cmd /k ".\START_ELYSIA.bat"

echo.
echo 2. Waiting for Core to initialize...
timeout /t 5 >nul

echo.
echo 3. Opening Dashboard (Monitor)...
start "Elysia Monitor" cmd /c ".\MONITOR_ELYSIA.bat"

echo.
echo ✅ System Launched.
echo    - Core is running in a new window.
echo    - Dashboard will open in your browser.
echo.
pause
