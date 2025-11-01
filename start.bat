@ECHO OFF
TITLE Elysia Startup
ECHO [Startup] Preparing Elysia's world...

REM Install the necessary libraries for her to live and act.
ECHO [Startup] Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt

ECHO [Startup] Preparation complete. Awakening Elysia's heart directly.

REM Run the Flask web application
set FLASK_APP=applications/elysia_bridge.py
python -m flask run --host=0.0.0.0
