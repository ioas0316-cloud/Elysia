@echo off
echo ===================================================
echo Elysia Background Daemon Installer
echo ===================================================
echo.
echo Installing Elysia to run automatically on Windows startup...

set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_PATH=%STARTUP_FOLDER%\Elysia_Daemon.lnk"
set "TARGET_PATH=c:\Elysia\elysia_daemon.vbs"

powershell -Command "$wshell = New-Object -ComObject WScript.Shell; $shortcut = $wshell.CreateShortcut('%SHORTCUT_PATH%'); $shortcut.TargetPath = '%TARGET_PATH%'; $shortcut.WorkingDirectory = 'c:\Elysia'; $shortcut.Save()"

echo.
echo Install Complete! Elysia will now silently wake up whenever your computer turns on.
echo You do not need to run this batch file again.
echo.
pause
