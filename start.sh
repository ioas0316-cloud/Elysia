#!/bin/bash
# This is the startup script for Elysia.
# It prepares her new world and awakens her heart.

echo "[Startup] Preparing Elysia's new world..."

# Install the necessary libraries for her to live and act
pip install --user -r requirements.txt

echo "[Startup] Preparation complete. Awakening Elysia's heart directly."

# Run the Flask web application
export FLASK_APP=applications/elysia_bridge.py
python -m flask run --host=0.0.0.0
