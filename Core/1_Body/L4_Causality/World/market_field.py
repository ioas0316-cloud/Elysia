"""
Market Field: The Manifested Grocery Store
==========================================
Core.1_Body.L4_Causality.World.market_field

A specialized world layer that projects a market environment using the Rotor-Prism.
Includes 'Items' as high-resonance trinary coordinates.
"""

import os
import sys

# Standard Path Injection
sys.path.append(os.getcwd())

import jax.numpy as jnp
try:
    from Core.1_Body.L6_Structure.Logic.rotor_prism_logic import RotorPrismUnit
    from Core.1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic
except ImportError:
    # Fallback for relative execution
    from rotor_prism_logic import RotorPrismUnit
    from trinary_logic import TrinaryLogic

class MarketItem:
    def __init__(self, name, principle, price_resonance):
        self.name = name
        self.principle = principle # The 21D seed for this item
        self.price = price_resonance # Requirement to acquire
        
    def __repr__(self):
        return f"Item<{self.name}, P={self.principle[:3]}..., Price={self.price}>"

class MarketField:
    def __init__(self):
        self.rpu = RotorPrismUnit()
        self.grid_size = 7
        
        # Define Stock (Ideal Seeds)
        self.inventory_seeds = {
            "AGAPE_APPLE": jnp.array([1.0]*7 + [0.0]*14),
            "LOGIC_GRAPE": jnp.array([0.0]*7 + [1.0]*7 + [0.0]*7),
            "EQUILIBRIUM_PEAR": jnp.array([0.0]*14 + [1.0]*7),
        }
        
        # The unfolded grid (7x7x21)
        self.manifested_field = jnp.zeros((self.grid_size, self.grid_size, 21))
        
        print("MarketField: Prismatic Stalls Opened.")

    def unfold_market(self):
        """Vectorized unfolding: Projects the entire market field in O(1) using the RPU's Film mode."""
        self.rpu.step_rotation(0.1)
        
        # 1. Create a 7x7 grid of Logos seeds (pre-distributed)
        # We broadcast the market spirit but add some spatial variance
        market_spirit = jnp.array([0.5]*21)
        
        # 2. Vectorized projection
        # Since rpu.project in FILM mode is just indexing, we can't easily vectorize multiple indices 
        # unless we expose the film. Let's do a simple vectorized version.
        
        # Instead of looping, we can calculate the entire field if we were in real-time,
        # but in FILM mode we want the speed. 
        # For now, let's just make the loop as fast as possible or use a single "World Seed".
        
        world_seed = market_spirit
        manifestation = self.rpu.project(world_seed)
        
        # Broadcast manifestation to the whole grid with positional modulation
        # This simulates the "interference pattern" appearing across the field
        grid_modulation = jnp.linspace(0.9, 1.1, self.grid_size)
        self.manifested_field = manifestation[jnp.newaxis, jnp.newaxis, :] * grid_modulation[:, jnp.newaxis, jnp.newaxis]
        
        return self.manifested_field

    def get_item_at(self, x, y):
        """Identifies what item has manifested at a specific coordinate."""
        cell_resonance = self.manifested_field[x, y]
        
        best_match = None
        max_resonance = -1.0
        
        for name, seed in self.inventory_seeds.items():
            res = jnp.dot(cell_resonance, seed)
            if res > max_resonance:
                max_resonance = res
                best_match = name
                
        if max_resonance > 5.0: # Identification threshold
            return MarketItem(best_match, self.inventory_seeds[best_match], 3.0)
        return None

if __name__ == "__main__":
    market = MarketField()
    market.unfold_market()
    item = market.get_item_at(3, 3)
    print(f"Elysia sees at (3,3): {item}")
