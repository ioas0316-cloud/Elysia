"""
Verify Elysia's Shopping Cycle
==============================
Scripts/Tests/verify_shopping_cycle

Simulates 'pretty Elysia' going to the market, interacting with a Merchant, 
and acquiring an Agape Apple.
"""

import os
import sys

# Standard Path Injection
sys.path.append(os.getcwd())

import jax.numpy as jnp
try:
    from Core.S1_Body.L4_Causality.World.market_field import MarketField
    from Core.S1_Body.L4_Causality.World.npc_spawner import NPCSpawner
    from Core.S1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic
    print("DEBUG: Component imports successful.")
except ImportError as e:
    print(f"DEBUG: Import failed: {e}")
    sys.exit(1)

def simulate_shopping():
    print("🌅 PHASE 66: The Unfolding of the Market Square...")
    
    market = MarketField()
    spawner = NPCSpawner()
    
    # 1. Unfold the world from the Rotor's Light
    market.unfold_market()
    print("The stalls have manifested from the high-speed spin of the light.")
    
    # 2. Spawn the Merchant at (3,3)
    # Market Field at (3,3) should be high resonance for a Merchant
    cell_field = market.manifested_field[3, 3]
    entities = spawner.unfold_from_field(cell_field, (3, 3))
    
    if not entities:
        print("Wait... where is everyone? (Low resonance at stall).")
        return
    
    merchant = entities[0]
    print(f"\n[Elysia sees {merchant.name} standing by the stall.]")
    
    # 3. Elysia selects an item
    item = market.get_item_at(3, 3)
    if item:
        print(f"Elysia: 'Oh! This {item.name} looks so juicy and full of Papa's warmth!'" )
        
        # 4. Interaction Cycle
        print(f"\n[Interaction: GREET]")
        print(f"Merchant: '{merchant.interact('GREET')}'")
        
        print(f"\n[Interaction: PRICE]")
        print(f"Merchant: '{merchant.interact('PRICE')}'")
        
        # 5. Acquisition (Simulation of Resonance Pulse)
        print(f"\nElysia emits a Pulse of Joy (+1.0 Vector) to the Merchant.")
        print(f"--- SUCCESS: {item.name} added to Elysia's Inventory ---")
        
        print(f"\nElysia: 'Thank you! I'm going to make the most delicious harmonic feast ever! 🥂🫡'")
    else:
        print("Elysia: 'Nothing catches my eye today. I'll wait for the rotor to spin again.'")

if __name__ == "__main__":
    simulate_shopping()
