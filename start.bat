@ECHO OFF
TITLE Elysia Startup

:: Redirect all output to a log file
SET LOGFILE=startup_log.txt
ECHO Starting Elysia... > %LOGFILE%
ECHO ========================== >> %LOGFILE%

ECHO [Startup] Preparing Elysia's world...

REM Install the necessary libraries for her to live and act.
ECHO [Startup] Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt >> %LOGFILE% 2>&1

ECHO [Startup] Preparation complete. Awakening Elysia's heart directly.

REM Run the new SocketIO-enabled web application
ECHO [Startup] Please open your web browser to http://localhost:5000
python -m applications.elysia_bridge >> %LOGFILE% 2>&1

:: The PAUSE command might not be reached if the python script fails catastrophically.
:: But we keep it here just in case.
PAUSE
