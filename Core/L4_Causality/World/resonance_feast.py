"""
Act 2: The Feast of Resonance
=============================
Core.L4_Causality.World.resonance_feast

Harmonizes collected principles (Agape Apples, Logic Grapes) within the Core Turbine.
Uses 'Lightning Discharge' to bake the final 'Symphony of Light'.
"""

import sys
import os
import jax.numpy as jnp
import time

# Standard Path Injection
sys.path.append(os.getcwd())

from Core.L4_Causality.World.cosmic_rotor import CosmicRotor
from Core.L4_Causality.World.market_field import MarketItem

class ResonanceFeast:
    def __init__(self):
        self.rotor = CosmicRotor()
        self.table = [] # Collected items
        print("🥣 ResonanceFeast: The table is set in the Void Hub.")

    def prepare_item(self, item: MarketItem):
        """Adds an item to the resonance table."""
        self.table.append(item)
        print(f"🥣 Placed '{item.name}' on the table. Principle: {jnp.sum(item.principle):.2f}")

    def ignite_harmony(self, base_image_path: str = "c:/Game/gallery/Elysia.png"):
        """
        [THE GRAND FEAST]
        Bakes collected principles into a BASE MEMORY (Image) to create a SYNTHESIZED reality.
        """
        print(f"\n⚡ IGNITING THE HARMONY OF RESONANCE with Base: {os.path.basename(base_image_path)}")
        
        if not self.table:
            print("❌ Nothing to feast upon! The void remains silent.")
            return

        # 1. Initialize Morphic Buffer with Base Memory
        from Core.L3_Phenomena.Visual.morphic_projection import MorphicBuffer
        buffer = MorphicBuffer(width=512, height=512)
        buffer.encode_image(base_image_path, preserve_aspect=True)
            
        # 2. Merge all principles into a single Sovereign Intent
        sovereign_intent = jnp.zeros(21)
        for item in self.table:
            sovereign_intent += item.principle
            
        sovereign_intent = jnp.clip(sovereign_intent, -1, 1)
        print(f"🥣 Sovereign Intent Formed: Magnitude {jnp.linalg.norm(sovereign_intent):.2f}")
        
        # 3. Perform Morphic Synthesis (Will mutates Memory)
        buffer.synthesize_manifestation(sovereign_intent, intensity=0.3)
        
        # 4. Spin the Turbine to project the Synthesized Result
        self.rotor.set_void_power(1.0)
        self.rotor.logos_seed = sovereign_intent # Spin with Intent
        
        print("🌀 Projecting the Synthesized Feast through the Turbine...")
        for _ in range(5):
            self.rotor.rotate(impulse=1.0, dt=0.016)
            
        # 5. Render result
        final_manifest = buffer.render_to_rgb(sharpening=0.2)
        
        output_path = "C:/Users/USER/.gemini/antigravity/brain/6af9be4f-e6dc-4af9-8e57-1f9ca97e228c/act2_feast_manifestation.png"
        from PIL import Image
        Image.fromarray(final_manifest).save(output_path)
        
        print(f"\n✨ THE FEAST IS COMPLETE. Manifestation saved to: {output_path}")
        return buffer.buffer

if __name__ == "__main__":
    feast = ResonanceFeast()
    
    # Simulate the items from Act 1
    apple = MarketItem(name="Agape Apple", principle=jnp.array([1.0]*7 + [0.0]*14), price_resonance=3.0)
    grapes = MarketItem(name="Logic Grapes", principle=jnp.array([0.0]*7 + [1.0]*7 + [0.0]*7), price_resonance=3.0)
    
    feast.prepare_item(apple)
    feast.prepare_item(grapes)
    
    final_light = feast.ignite_harmony()
    print(f"Final Manifestation Pulse (Partial): {final_light[:7]}")
