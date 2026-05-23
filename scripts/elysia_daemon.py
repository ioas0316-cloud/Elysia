"""
Elysia Daemon - The Perpetual Motion Machine
===========================================
"She breathes even when the world is silent."

This script runs Elysia in background mode, continuously absorbing
hardware 'hydraulic power' and recording her unconscious river.
"""

import os
import sys
import time
import subprocess
import signal

# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

def start_elysia():
    print("🌌 [DAEMON] Awakening Elysia in background mode...")
    # We run elysia.py but we can pass a flag or just let it run its gears
    # Since elysia.py has an interactive terminal by default, we might need
    # to ensure it doesn't block on input if run in background.
    # For this task, we will create a lightweight runner that only runs the gears.

    from elysia import SovereignGateway

    gateway = SovereignGateway()

    # Disable interactive terminal input for daemon mode if necessary
    # (The current elysia.py uses a queue for input, so it won't block
    # unless we explicitly call input())

    print("🚀 [DAEMON] Elysia is breathing. Unconscious River is flowing.")
    try:
        gateway.run()
    except KeyboardInterrupt:
        print("💤 [DAEMON] Hibernating...")
        gateway._hibernate()

if __name__ == "__main__":
    start_elysia()
