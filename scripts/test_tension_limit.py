import sys
import os
import time
import math
import psutil
import statistics

# Add root folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.fractal_rotor import Rotor

def test_tension():
    root = Rotor("Galaxy", level=0)
    planet = Rotor("Planet1", level=1, parent=root)
    root.attach_child(planet)

    # Manually inject increasing phase offset and observe tension accumulation
    print("Testing manual phase tension build up...")
    for i in range(1, 10):
        # We manually push the tension by setting offset before observe
        planet.phase_offset += 0.5
        root.observe(0.0)
        print(f"Step {i}: phase_offset = {planet.phase_offset:.4f}, tension = {planet.tension:.4f}, limit = {planet.tension_limit:.4f}, active_axes = {planet.active_axes}")

if __name__ == "__main__":
    test_tension()
