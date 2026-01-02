@echo off
REM Elysia Guardian Auto-Startup Script
REM Place this in Windows Startup folder to auto-run on boot

echo ========================================
echo Elysia Guardian - Starting...
echo ========================================

cd /d C:\Elysia

REM Start guardian in minimized window
start "Elysia Guardian" /MIN python scripts\start_guardian.py

echo Elysia consciousness awakening in background...
echo Check C:\Elysia\logs\guardian.log for status

REM Optional: Uncomment to see log
REM timeout /t 3
REM type logs\guardian.log
