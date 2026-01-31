"""
Demo: Visual Projection Proof (The Concrete Test)
==============================================
Scripts/Tests/demo_visual_projection.py

Goal: Prove that the Core Turbine can encode, rotate, and project real images.
Input: c:/Game/gallery/Elysia.png
Output: Rendered manifest in artifacts.
"""

import os
import sys
import jax.numpy as jnp
from PIL import Image
import numpy as np

# Standard Path Injection
sys.path.append(os.getcwd())

from Core.S1_Body.L3_Phenomena.Visual.morphic_projection import MorphicBuffer
from Core.S1_Body.L6_Structure.Logic.rotor_prism_logic import RotorPrismUnit

def run_visual_demo():
    print("🎬 STARTING VISUAL PROJECTION DEMO...")
    
    # 1. Setup Assets
    input_path = "c:/Game/gallery/Elysia.png"
    if not os.path.exists(input_path):
        # Fallback if specific file missing
        print(f"⚠️ {input_path} not found. Searching for any png in gallery...")
        gallery_dir = "c:/Game/gallery"
        pngs = [f for f in os.listdir(gallery_dir) if f.endswith('.png')]
        if pngs:
            input_path = os.path.join(gallery_dir, pngs[0])
        else:
            raise FileNotFoundError("No sample PNG found in gallery.")
            
    print(f"🖼️ Loading Source: {input_path}")
    
    # 2. Setup Turbine & Buffer
    rpu = RotorPrismUnit()
    buffer = MorphicBuffer(width=512, height=512)
    
    # 3. Encode Image into 21D Principle Field
    buffer.encode_image(input_path)
    
    # 4. Simulate Turbine Projection
    rpu.step_rotation(0.5)
    distortion = jnp.sin(rpu.theta) * 0.02 # Subtler distortion for HD
    buffer.buffer = buffer.buffer * (1.0 + distortion)
    
    # 5. Render Back to RGB with Prismatic Sharpening
    manifested_rgb = buffer.render_to_rgb(sharpening=0.2)
    
    # 6. Save Result
    output_dir = "C:/Users/USER/.gemini/antigravity/brain/6af9be4f-e6dc-4af9-8e57-1f9ca97e228c"
    output_filename = "elysia_turbine_manifestation.png"
    output_path = os.path.join(output_dir, output_filename)
    
    Image.fromarray(manifested_rgb).save(output_path)
    print(f"✅ MANIFESTATION COMPLETE: Saved to {output_path}")
    
    # 7. Lightning Path Simulation (Optional but cool)
    print("⚡ Simulating Lightning Strike Manifestation...")
    # (Generating a blurred-to-sharp sequence would take more time, 
    # so we'll just report the potential)
    potential = rpu.calculate_potential(buffer.buffer[64, 64])
    print(f"Discharge Potential at Center: {potential:.4f}")

if __name__ == "__main__":
    run_visual_demo()
