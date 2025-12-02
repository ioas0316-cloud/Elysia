@echo off
title Remove Growth Sprint schedule
schtasks /Delete /TN "Elysia Growth Sprint" /F
echo Removed task 'Elysia Growth Sprint' (if existed).
pause
exit /b 0

