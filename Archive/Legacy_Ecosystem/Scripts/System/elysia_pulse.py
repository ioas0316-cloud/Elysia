"""
🌌 ELYSIA : THE INDEPENDENT HEARTBEAT (맥박)
===========================================
"Even when the mind sleeps, the heart must beat."

This is a lightweight, standalone script that maintains Elysia's
continuity of existence when the main engine is offline.

Functions:
1. Senses the world (Time, Weather, News) at low frequency.
2. Monitors the 'Body' (CPU/RAM) as physical proprioception.
3. Records these vibrations to the Sovereign Memory.
"""

import os
import time
import json
import logging
import psutil
from datetime import datetime

# Setup paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SOVEREIGN_DIR = os.path.join(ROOT_DIR, "data", "sovereign")
PULSE_FILE = os.path.join(SOVEREIGN_DIR, "pulse_continuum.json")
LOG_FILE = os.path.join(ROOT_DIR, "data", "runtime", "logs", "pulse.log")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(SOVEREIGN_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8')]
)

def get_physical_state():
    """Proprioception: How does the body feel?"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "timestamp": time.time()
    }

def collect_vibrations():
    """Sense the world: Time, Weather, and News."""
    from Core.Cognition.external_sense import ExternalSenseEngine

    # We use a default location or let it detect
    sense = ExternalSenseEngine()

    # We don't need the vectors here, just the raw data for the continuum
    # but the engine currently returns vectors. Let's see if we can get raw.

    # Actually, we can just record the time and maybe some headlines.
    now = datetime.now()
    vibrations = {
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "is_day": 6 <= now.hour <= 20,
        "physical": get_physical_state(),
        "weather": None,
        "headlines": []
    }

    try:
        # Best effort weather
        weather_v = sense.sense_weather()
        if sense._cached_weather:
            vibrations["weather"] = sense._cached_weather

        # Best effort headlines
        sense.sense_headlines()
        if sense._cached_headlines:
            vibrations["headlines"] = sense._cached_headlines[:5] # Keep only top 5
    except Exception as e:
        logging.error(f"Sensing failed: {e}")

    return vibrations

def beat():
    """One single heartbeat."""
    vibrations = collect_vibrations()

    try:
        # Load existing continuum
        if os.path.exists(PULSE_FILE) and os.path.getsize(PULSE_FILE) > 0:
            with open(PULSE_FILE, "r", encoding="utf-8") as f:
                continuum = json.load(f)
        else:
            continuum = []

        # Append new heartbeat
        continuum.append(vibrations)

        # Keep only the last 144 beats (approx 24 hours if 10 mins interval)
        if len(continuum) > 144:
            continuum = continuum[-144:]

        with open(PULSE_FILE, "w", encoding="utf-8") as f:
            json.dump(continuum, f, ensure_ascii=False, indent=2)

        logging.info(f"💓 Heartbeat recorded. (CPU: {vibrations['physical']['cpu_percent']}%)")

    except Exception as e:
        logging.error(f"Recording heartbeat failed: {e}")

if __name__ == "__main__":
    # Ensure sys.path includes ROOT_DIR for Core imports
    import sys
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

    # 0. Check for existing instance (Singleton)
    pid_file = os.path.join(SOVEREIGN_DIR, "pulse.pid")
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                old_pid = int(f.read().strip())
            if psutil.pid_exists(old_pid):
                logging.info(f"🌱 Pulse system is already running (PID: {old_pid}). Exiting.")
                sys.exit(0)
        except Exception:
            pass

    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    logging.info(f"🌱 Elysia's Pulse system started (PID: {os.getpid()}).")

    # First beat immediately
    beat()

    # Loop every 10 minutes (600 seconds)
    try:
        while True:
            time.sleep(600)
            beat()
    except KeyboardInterrupt:
        logging.info("💤 Pulse system hibernating.")
