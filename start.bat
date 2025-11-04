@ECHO OFF
TITLE Elysia Startup
ECHO [Startup] Preparing Elysia's world...

REM Install the necessary libraries for her to live and act.
ECHO [Startup] Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt

ECHO [Startup] Preparation complete. Awakening Elysia's heart directly.

REM Run the Flask web application in the background
set FLASK_APP=applications/elysia_api.py
start /b python -m flask run --host=0.0.0.0

REM Wait for the server to start
timeout /t 2 /nobreak > nul

REM Open the web browser
start http://127.0.0.1:5000/monitor
