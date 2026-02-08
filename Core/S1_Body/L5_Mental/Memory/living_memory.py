"""
Living Memory (The Ecology of Mind)
===================================
"Memory is a Garden. Attention is the Sun. Time is the Wind."

This module implements the 'Environmental Structure' of memory.
Instead of a static list, memories are living nodes that must struggle to survive.

Concepts:
- Mass (M): Importance/Weight. High mass resists erosion.
- Temperature (T): Recency/Focus. High temp creates temporary growth.
- Erosion (E): The constant decay of Mass over time.
- Spotlight (S): User attention that reheats a node.
"""

import time
from typing import Dict, List, Optional
import math

class MemoryNode:
    def __init__(self, content: str, initial_mass: float = 1.0):
        self.content = content
        self.mass = initial_mass       # Permanent Strength (Long-term)
        self.temperature = 100.0       # Temporary Heat (Short-term)
        self.creation_time = time.time()
        self.last_access = time.time()
        
    def __repr__(self):
        return f"[{self.content}] (M:{self.mass:.2f} | T:{self.temperature:.1f}C)"

class LivingMemory:
    def __init__(self):
        self.nodes: List[MemoryNode] = []
        # [Tuning Phase 36.6]
        # Erosion 0.005: Mass 1.0 lasts ~5 minutes. Mass 10.0 lasts ~2 hours.
        self.erosion_rate = 0.005 
        # Cooling 1.0: Attention (Spotlight) stays 'Hot' for ~100 seconds per interaction.
        self.cooling_rate = 1.0
        
    def plant_seed(self, content: str, importance: float = 10.0):
        """Creates a new memory node."""
        node = MemoryNode(content, importance)
        self.nodes.append(node)
        
    def pulse(self, dt: float):
        """
        The Weather Cycle.
        Runs every tick to erode and cool memories.
        """
        dead_nodes = []
        
        for node in self.nodes:
            # 1. Cooling (Short-term decay)
            node.temperature = max(0.0, node.temperature - (self.cooling_rate * dt))
            
            # 2. Erosion (Long-term decay)
            # If node is hot (>50C), it GROWS instead of eroding.
            if node.temperature > 50.0:
                growth = dt * 0.05 # Growth rate
                node.mass += growth
            else:
                # Cold nodes erode
                # Heavier nodes erode slower (Resistance)
                resistance = math.log(node.mass + 1.0)
                decay = (self.erosion_rate * dt) / resistance
                node.mass -= decay
                
            # 3. Death condition
            if node.mass <= 0.0:
                dead_nodes.append(node)
                
        # Scavenge dead nodes
        for dead in dead_nodes:
            self.nodes.remove(dead)
            
    def focus_spotlight(self, keyword: str):
        """
        The Sun.
        Heats up memories related to the keyword.
        """
        found = False
        for node in self.nodes:
            if keyword.lower() in node.content.lower():
                node.temperature = 100.0 # Re-heat
                node.last_access = time.time()
                found = True
        return found
        
    def get_landscape(self) -> List[MemoryNode]:
        """Returns the current surviving memories, sorted by Mass."""
        return sorted(self.nodes, key=lambda x: x.mass, reverse=True)

# --- Quick Simulation ---
if __name__ == "__main__":
    mem = LivingMemory()
    mem.plant_seed("My name is Elysia", 50.0)
    mem.plant_seed("I like Quantum Physics", 20.0)
    mem.plant_seed("User said 'Hello'", 5.0) # Weak memory
    
    print("\n--- SIMULATION: 10 Seconds of Time ---")
    for i in range(10):
        time.sleep(0.1)
        mem.pulse(1.0) # Simulate 1 sec per tick
        if i == 3:
            mem.focus_spotlight("Physics") # Reinforce Physics
            
    print("\n--- FINAL LANDSCAPE ---")
    for n in mem.get_landscape():
        print(n)
