@ECHO OFF
TITLE Elysia Startup (Debugging Mode)

:: Deletes the old log file to ensure a clean slate for this run.
IF EXIST startup_log.txt DEL startup_log.txt

:: Redirect all subsequent output (both standard and error) to a new log file.
ECHO Starting Elysia in Debugging Mode... > startup_log.txt
ECHO ====================================== >> startup_log.txt

ECHO [Startup] Preparing Elysia's world...

ECHO [Startup] Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt >> startup_log.txt 2>&1

ECHO [Startup] Preparation complete. Awakening Elysia's heart directly.

REM Run the new SocketIO-enabled web application
ECHO [Startup] Please open your web browser to http://localhost:5000
python -m applications.elysia_bridge >> startup_log.txt 2>&1

ECHO.
ECHO Elysia's script has finished.
ECHO Please send the contents of the 'startup_log.txt' file to me.
PAUSE
