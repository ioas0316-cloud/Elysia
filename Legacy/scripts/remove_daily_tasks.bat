@echo off
title Remove Elysia Daily Tasks
schtasks /Delete /TN "Elysia Routine" /F
schtasks /Delete /TN "Elysia Report" /F
echo Removed scheduled tasks (if they existed).
pause
exit /b 0

