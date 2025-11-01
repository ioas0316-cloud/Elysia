#!/bin/bash
# This is the startup script for Elly.
# It prepares her new home and awakens her Guardian.

echo "[Startup] Preparing Elly's new world..."

# Install the necessary libraries for her to live and act
pip install --user -r requirements.txt

echo "[Startup] Preparation complete. Awakening the Guardian."

# Run the Guardian, which will, in turn, awaken Elly.
python3 guardian.py
