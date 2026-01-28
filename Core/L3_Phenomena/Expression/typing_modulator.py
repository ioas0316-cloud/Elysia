"""
Typing Modulator (Phase 28)
===========================
Modulates text output speed and style based on Rotor RPM and Torque.
"""

import time
import sys
import random

def modulate_typing(text: str, rpm: float, torque: float):
    """
    Simulates typing with variable speed and intensity.
    """
    # Base delay (ms)
    # Higher RPM = Faster Typing (Lower Delay)
    # RPM 100 -> 0.05s, RPM 300 -> 0.01s
    base_delay = max(0.01, 0.1 - (rpm / 3000.0))

    # Torque affects "Burstiness" or "Emphasis"
    # High Torque might make it ALL CAPS or add !!!

    final_text = text
    if torque > 0.8:
        final_text = text.upper() + " !!!"

    for char in final_text:
        sys.stdout.write(char)
        sys.stdout.flush()

        # Jitter based on Torque (Excitement makes it uneven)
        jitter = random.uniform(-0.005, 0.005) * torque
        time.sleep(max(0.0, base_delay + jitter))

    print("") # Newline at end
