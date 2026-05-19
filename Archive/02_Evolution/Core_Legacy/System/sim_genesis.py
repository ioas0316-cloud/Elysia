"""
Genesis Simulation (The Rise of Structure)
==========================================
sim_genesis.py

Runs the simulation of 30 agents struggling and thriving in an uneven world.
Observes EMERGENT BEHAVIOR:
- Do they cluster in Valleys? (Cities)
- Does one agent hoard Gold? (Kings)
- Do agents starve? (Selection)
"""

import time
import random
from Core.Cognition.genesis import WorldGen
from Core.Cognition.trinity_citizen import Survivor

def run_genesis():
    print(">>> 🌋 GENESIS: Awakening the World Spirit...")
    
    # 1. Create the World
    world = WorldGen(width=10, height=10) # 100 Zones
    
    # 2. Spawn Survivors (The First Men)
    population = []
    archetypes = ["Warrior", "Explorer", "Sage", "Worker"]
    
    for i in range(30):
        name = f"Human_{i}"
        arch = random.choice(archetypes)
        # Random spawn
        start_pos = (random.randint(0,9), random.randint(0,9))
        citizen = Survivor(name, arch, start_pos)
        population.append(citizen)
        
    print(f"👻 Spawning {len(population)} souls into the void...")
    
    # 3. The Time Loop (History)
    days = 20
    history_log = []
    
    for day in range(1, days + 1):
        print(f"\n☀️  [Day {day}] Sun rises.")
        
        # Agent Actions
        for citizen in population:
            # Alive check
            if citizen.needs["Energy"] <= 0:
                # Dead
                continue
                
            action_log = citizen.act_in_world(world)
            # Log significant events
            if "Migrating" in action_log or "Wealth" in action_log:
                # print(f"   [{citizen.name}] {action_log}")
                pass
                
        # World Regeneration
        for zone in world.grid.values():
            zone.regenerate()
            
        # Snapshot: Who is powerful?
        # Sort by Power
        population.sort(key=lambda s: s.calculate_power(), reverse=True)
        top_dog = population[0]
        history_log.append(f"Day {day} Ruler: {top_dog.name} ({top_dog.archetype}) - Gold: {top_dog.inventory['Gold']:.0f}, Food: {top_dog.inventory['Food']:.0f}")
        
    # 4. Final Analysis (The Emerged Structure)
    print("\n📜 [The Chronicles of Genesis]")
    for entry in history_log:
        print(entry)
        
    print("\n📊 [Demographics]")
    survivors = [c for c in population if c.needs["Energy"] > 0]
    print(f"Survival Rate: {len(survivors)} / 30")
    
    # Clustering Analysis
    locations = {}
    for s in survivors:
        if s.location not in locations: locations[s.location] = 0
        locations[s.location] += 1
        
    print("\n🏙️  [Emergent Cities] (Zones with > 2 citizens)")
    for loc, count in locations.items():
        if count > 2:
            zone = world.get_zone(loc)
            print(f" - {zone.name} ({zone.coords}): {count} Citizens. (Type: {zone.name.split('_')[0]})")
            
    # Inequality Analysis
    gold_counts = [s.inventory["Gold"] for s in survivors]
    if survivors:
        avg_gold = sum(gold_counts) / len(survivors)
        max_gold = max(gold_counts)
        print(f"\n💰 [Economy]")
        print(f"Average Wealth: {avg_gold:.1f}")
        print(f"Richest King: {max_gold:.1f} (Gini Coeff Risk: High)")

if __name__ == "__main__":
    run_genesis()
