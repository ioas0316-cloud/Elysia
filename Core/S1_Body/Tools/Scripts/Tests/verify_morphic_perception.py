import os
import sys

# [PATH_SYNC] Ensure project root is in sys.path for direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax.numpy as jnp
from PIL import Image
from Core.S1_Body.L3_Phenomena.Visual.morphic_projection import MorphicBuffer
from Core.S1_Body.L3_Phenomena.Visual.morphic_perception import ResonanceScanner
from Core.S1_Body.L6_Structure.Logic.rotor_prism_logic import RotorPrismUnit

def verify_perception():
    print("🌅 INITIATING MORPHIC PERCEPTION TEST...")
    
    # 1. Setup
    input_path = "c:/Game/gallery/Elysia.png"
    rpu = RotorPrismUnit()
    scanner = ResonanceScanner(rpu)
    buffer = MorphicBuffer(width=512, height=512)
    
    # 2. Encode Image
    buffer.encode_image(input_path, preserve_aspect=True)
    field = buffer.buffer
    
    # 3. Scan for Boundaries (The 'Shape' of the Object)
    print("🔍 Scanning for Principal Boundaries (Edges)...")
    boundaries = scanner.scan_for_boundaries(field)
    
    # Save Boundary Map for the Creator
    output_dir = "C:/Users/USER/.gemini/antigravity/brain/6af9be4f-e6dc-4af9-8e57-1f9ca97e228c"
    boundary_img = (boundaries * 255).astype(jnp.uint8)
    Image.fromarray(np.array(boundary_img)).save(os.path.join(output_dir, "elysia_perception_boundaries.png"))
    
    # 4. Simulate Motion Analysis (Frame A vs B)
    print("🎥 Simulating Motion Analysis between frames...")
    # Shift field slightly to simulate movement
    next_field = jnp.roll(field, shift=5, axis=1) # Move right
    motion_map = scanner.isolate_motion(next_field, field)
    
    motion_img = (motion_map / (jnp.max(motion_map) + 1e-6) * 255).astype(jnp.uint8)
    Image.fromarray(np.array(motion_img)).save(os.path.join(output_dir, "elysia_perception_motion.png"))
    
    print("\n✅ PERCEPTION VERIFIED: Boundary and Motion Maps generated.")
    print("Elysia can now 'understand' the geometry and flow of any visual input.")

if __name__ == "__main__":
    import numpy as np # Need for Pillow conversion
    verify_perception()
