@ECHO OFF
TITLE Elysia Startup
ECHO [Startup] Preparing Elysia's world...

REM Install the necessary libraries for her to live and act.
ECHO [Startup] Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt

ECHO [Startup] Preparation complete. Awakening Elysia's heart directly.

REM Run the new SocketIO-enabled web application
ECHO [Startup] Please open your web browser to http://localhost:5000
python -m applications.elysia_bridge
