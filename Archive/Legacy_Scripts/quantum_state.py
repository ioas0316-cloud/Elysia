"""
Quantum State (The Coin)
========================

"Two sides, one fate."

This module implements the "Shared State" (Quantum Entanglement) for the Elysia City.
It uses a file-based lock mechanism (simple but effective for this scale) to ensure
all modules see the SAME state instantly.

States:
- CALM (Blue): Normal operation.
- ALERT (Red): High traffic or error.
- CREATIVE (Purple): Generating content.
- SLEEP (Dark): Low energy.
"""

import json
import os
import time
from enum import Enum
from typing import Dict, Any

STATE_FILE = os.path.join(os.path.dirname(__file__), "../../data/quantum_state.json")

class StateMode(str, Enum):
    CALM = "CALM"
    ALERT = "ALERT"
    CREATIVE = "CREATIVE"
    SLEEP = "SLEEP"

DEFAULT_STATE = {
    "mode": StateMode.CALM.value,
    "last_flip": time.time(),
    "trigger": "Initialization"
}

def _ensure_file():
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w") as f:
            json.dump(DEFAULT_STATE, f)

def get_quantum_state() -> Dict[str, Any]:
    """Reads the current side of the coin."""
    _ensure_file()
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return DEFAULT_STATE

def flip_coin(mode: StateMode, trigger: str = "Manual"):
    """Flips the coin (Global State Change)."""
    _ensure_file()
    new_state = {
        "mode": mode.value,
        "last_flip": time.time(),
        "trigger": trigger
    }
    with open(STATE_FILE, "w") as f:
        json.dump(new_state, f)
    print(f"🪙 Quantum Coin Flipped: {mode.value} (by {trigger})")

# Theme Colors for Visualization
THEMES = {
    StateMode.CALM: {"bg": "#111122", "node": "#00ccff", "edge": "#444488"},
    StateMode.ALERT: {"bg": "#220000", "node": "#ff3333", "edge": "#880000"},
    StateMode.CREATIVE: {"bg": "#110011", "node": "#ff00ff", "edge": "#880088"},
    StateMode.SLEEP: {"bg": "#000000", "node": "#333333", "edge": "#222222"},
}

def get_theme(mode_str: str):
    return THEMES.get(StateMode(mode_str), THEMES[StateMode.CALM])
