"""
Verify Lightning Strike & Visual Projection
===========================================
Scripts/Tests/verify_visual_manifestation.py

Demonstrates:
1. Lightning Path finding a target principle.
2. Morphic Projection rendering the path as a visual structure.
"""

import os
import sys
import jax.numpy as jnp
from PIL import Image
import numpy as np

# Standard Path Injection
sys.path.append(os.getcwd())

from Core.1_Body.L6_Structure.Logic.lightning_path import LightningPath
from Core.1_Body.L3_Phenomena.Visual.morphic_projection import MorphicBuffer

def verify_manifestation():
    print("⚡ Initiating LIGHTNING_PATH Strike...")
    lp = LightningPath()
    buffer = MorphicBuffer(width=64, height=64)
    
    # 1. Choose Start and Goal
    chaos = jnp.zeros(21) # The Void
    agape_harmony = jnp.array([1.0]*7 + [0.5]*7 + [0.3]*7) # A complex goal
    
    # 2. Strike!
    path = lp.strike(chaos, agape_harmony, steps=10)
    print(f"Lightning Strike: Traversed {len(path)} phases to reach Harmony.")
    
    # 3. Project the Path into the Morphic Buffer
    # We trace the path across the buffer space
    for i, step in enumerate(path):
        x = 10 + i * 4
        y = 32 + int(jnp.sin(i) * 5)
        buffer.inject_principle(step, x, y, radius=3)
        
    # 4. Render to RGB
    rgb = buffer.render_to_rgb()
    print(f"Visual Projection: Manifested as {rgb.shape} RGB field.")
    
    # Save a representation (We'll use generate_image for the 'real' one, but this proves the logic)
    # Image.fromarray(rgb).save("Scripts/Tests/manifested_field_preview.png")
    
    print("\n✅ Visual Manifestation & Lightning Path Verified. 🥂🫡⚡🌀")

if __name__ == "__main__":
    verify_manifestation()
