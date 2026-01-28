"""
World Engine V1 (The Living Civilization)
=========================================
run_world_engine.py

"An environment where Art, Power, and Survival coexist."

Features:
- Language-based World Generation.
- LifeCitizens with diverse drives (Art, War, Food).
- Long-term emergent behavior tracking.
"""

import time
import random
from typing import List, Tuple
from Core.L1_Foundation.M1_Keystone.hyper_sphere_core import HyperSphereCore
from Core.L4_Causality.Civilization.trinity_citizen import LifeCitizen

class LivingWorld:
    def __init__(self, size: int = 20):
        self.size = size
        self.hyper_sphere = HyperSphereCore()
        self.population: List[LifeCitizen] = []
        self.history = []
        
        # Seed the Environment using Language
        self._genesis_by_word()
        
    def _genesis_by_word(self):
        """Painting the world with Words."""
        print("🎨 Genesis: Painting the Terrain with Language...")
        
        # Procedural Text Map
        biomes = [
            ("Forest", "Wood", "Pigment"), 
            ("River", "Water", "Clay"), 
            ("Mountain", "Gold", "Stone"), 
            ("Plains", "Food", "Grass"),
            ("Volcano", "Magma", "Ash")
        ]
        
        for x in range(self.size):
            for y in range(self.size):
                # Perlin-ish noise mock
                biome = biomes[(x + y) % len(biomes)]
                word = biome[0]
                
                # Manifest in HyperSphere
                props = self.hyper_sphere.manifest_at((x,y), word)
                
                # Manual Resource Injection based on secondary words
                # In full system, Lexicon handles this.
                if "resources" not in props: props["resources"] = {}
                props["resources"][biome[1]] = 500.0
                props["resources"][biome[2]] = 200.0
                if word == "Plains": props["resources"]["Food"] = 1000.0
                
    def spawn_life(self, count: int = 50):
        print(f"🌱 Spawning {count} Souls...")
        names = ["Aria", "Born", "Cian", "Dora", "Eon", "Fay", "Geo", "Hera", "Ion", "Jay"]
        archetypes = ["Artist", "Chef", "Warrior", "Sage", "Merchant"]
        
        for i in range(count):
            name = f"{random.choice(names)}_{i}"
            arch = random.choice(archetypes)
            pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            self.population.append(LifeCitizen(name, arch, pos))

    def get_zone_adapter(self, xy):
        """Adapter to make HyperSphere dict look like Zone object for Citizen code."""
        props = self.hyper_sphere.get_environment_at(xy)
        
        class ZoneMock:
            def __init__(self, p):
                self.resources = p.get("resources", {})
                self.name = p.get("name", "Void")
            def harvest(self, res, amt):
                val = self.resources.get(res, 0)
                taken = min(val, amt)
                self.resources[res] -= taken
                return taken
                
        return ZoneMock(props)

    def run_cycle(self, days: int = 100):
        print(f"⏳ Time flows... ({days} Days)")
        
        for day in range(1, days+1):
            # print(f"Day {day}...")
            
            # Events Log
            daily_events = []
            
            for citizen in self.population:
                if citizen.needs["Energy"] <= 0: continue
                
                # Create Zone Adapter
                zone = self.get_zone_adapter(citizen.location)
                
                # Act
                # Note: act_in_world signature update in LifeCitizen required `population`
                log = citizen.act_in_world(self, self.population) # Passing self as world adapter
                
                if "Masterpiece" in str(log) or "Concert" in str(log):
                    daily_events.append(f"Day {day}: [{citizen.name}] {log}")
                    
            if daily_events:
                self.history.extend(daily_events)
                for e in daily_events: print(e)
                
    # WorldGen Adapter methods for Citizen
    def get_zone(self, xy):
        return self.get_zone_adapter(xy)


if __name__ == "__main__":
    simulation = LivingWorld(size=10) # 10x10 for speed
    simulation.spawn_life(30)
    simulation.run_cycle(50)
    
    # Final Stats
    print("\n🏆 [Hall of Fame]")
    simulation.population.sort(key=lambda x: x.calculate_power(), reverse=True)
    top_5 = simulation.population[:5]
    for c in top_5:
        print(f"Name: {c.name} ({c.archetype}) | Power: {c.calculate_power():.1f}")
        print(f"   Skills: {c.skills}")
        print(f"   Inventory: {c.inventory}")
