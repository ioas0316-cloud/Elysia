@echo off
title Elysia Sovereign Dashboard
color 0b
chcp 65001 >nul

echo ========================================================
echo       E L Y S I A - Sovereign Discovery Center
echo ========================================================
echo.
echo 1. Open Static Dashboard (monitor.html)
echo 2. Open 3D Observatory (localhost:8765)
echo 3. Exit
echo.

set /p choice="Select an option (1-3): "

if "%choice%"=="1" (
    echo Opening static dashboard...
    start "" "monitor\monitor.html"
)
if "%choice%"=="2" (
    echo Opening 3D Observatory...
    start "" "http://localhost:8765"
)
if "%choice%"=="3" exit

echo.
echo Note: Ensure START_ELYSIA.bat is running to update data.
pause
