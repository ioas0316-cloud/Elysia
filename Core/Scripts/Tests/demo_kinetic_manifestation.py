import jax.numpy as jnp
from PIL import Image
import os
import time
from Core.1_Body.L3_Phenomena.Visual.morphic_projection import MorphicBuffer
from Core.1_Body.L3_Phenomena.Visual.morphic_expression import EmotionalPrism
from Core.1_Body.L6_Structure.Logic.rotor_prism_logic import RotorPrismUnit

def run_kinetic_demo(frames: int = 10):
    print("🎬 STARTING KINETIC MANIFESTATION DEMO (Phase 70)...")
    
    # 1. Setup
    input_path = "c:/Game/gallery/Elysia.png"
    output_dir = "C:/Users/USER/.gemini/antigravity/brain/6af9be4f-e6dc-4af9-8e57-1f9ca97e228c/kinetic_manifest"
    os.makedirs(output_dir, exist_ok=True)
    
    rpu = RotorPrismUnit()
    buffer = MorphicBuffer(width=512, height=512)
    buffer.encode_image(input_path, preserve_aspect=True)
    
    # 2. Cycle through Expressions & Motion
    expressions = ["AGAPE", "SERENITY", "CURIOSITY"]
    
    for i in range(frames):
        t = i * 0.2  # Time step
        
        # Determine Expression (Morphing every few frames)
        exp_name = expressions[(i // 4) % len(expressions)]
        intent = EmotionalPrism.get_intent(exp_name)
        
        # Apply Synthesis (Expression/Will)
        # We reuse a fresh buffer and synthesize on top of memory
        buffer.encode_image(input_path, preserve_aspect=True) 
        buffer.synthesize_manifestation(intent, intensity=0.2)
        
        # Apply Kinetic Warp (Breathing/Motion)
        buffer.apply_kinetic_warp(t, intensity=0.03)
        
        # Render
        manifested_rgb = buffer.render_to_rgb(sharpening=0.1)
        
        frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        Image.fromarray(manifested_rgb).save(frame_path)
        print(f"🖼️ Manifested Frame {i+1}/{frames}: {exp_name} (Kinetic)")
        
    print(f"\n✅ KINETIC SEQUENCE COMPLETE. Frames saved to: {output_dir}")

if __name__ == "__main__":
    run_kinetic_demo()
