#!/bin/bash
echo "[Startup] Preparing Elysia's world..."

# Upgrade pip to ensure compatibility with the latest packages
echo "[Startup] Upgrading pip..."
python -m pip install --upgrade pip

# Install the necessary libraries for her to live and act.
echo "[Startup] Installing dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo "[Startup] Preparation complete. Awakening Elysia's heart directly."

# Run the Flask web application in the background
export FLASK_APP=applications/elysia_bridge.py
python -m flask run --host=0.0.0.0 &

# Wait for the server to start
sleep 2
