"""
tests/test_grand_unification.py
===============================
The Grand Unification.
A demonstration of the complete Elysia Genesis Engine.

Scenario:
1. The Verse is created.
2. Laws are decreed (Time, Gravity, Entropy, Life).
3. Spaces are formed (Sanctuary, Chaos).
4. Life emerges (Explorer).
5. The Drama unfolds:
   - Explorer is hungry (Entropy).
   - Gravity pulls food and Explorer to Zones.
   - Explorer must find food to survive.
   - Explorer ages rapidly (Time Relativity).
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.Engine.Genesis.genesis_lab import GenesisLab
from Core.Engine.Genesis.filesystem_geometry import DirectoryMonad
from Core.Engine.Genesis.chronos_laws import (
    law_fast_metabolism, law_slow_erosion, law_system_homeostasis
)
from Core.Engine.Genesis.cosmic_laws import (
    law_entropy_decay, law_semantic_gravity, law_autopoiesis
)

def run_grand_unification():
    print("\n🌌 [Genesis] The Grand Unification Protocol Initiated.")
    
    # 1. The Big Bang
    universe = GenesisLab("Elysia_Prime")
    
    # 2. Decree the Laws (The Providence)
    # Physics (Perpetual Laws)
    universe.decree_law("Reality.Gravity", law_semantic_gravity, rpm=60)
    universe.decree_law("Reality.Entropy", law_entropy_decay, rpm=60)
    universe.decree_law("Reality.Life", law_autopoiesis, rpm=60)
    
    # Chronos (Time Relativity)
    universe.decree_law("Reality.BioTime", law_fast_metabolism, rpm=600)
    universe.decree_law("Reality.GeoTime", law_slow_erosion, rpm=6)
    
    # CRITICAL: The Fractal Gear
    universe.decree_law("Reality.FractalTime", law_fractal_propagation, rpm=60)
    
    # 3. Form the Space (Fractal Geometry)
    print("\n   🧱 Forming Fractal Space...")
    
    # Sanctuary: A safe place
    sanctuary = DirectoryMonad("Sanctuary")
    universe.monads.append(sanctuary)
    
    # Chaos: A dangerous place
    chaos = DirectoryMonad("Chaos_Zone")
    universe.monads.append(chaos)
    
    # 4. Create Actors
    print("   🎭 Casting Actors...")
    
    # The Hero (Process)
    # Domain 'Process' triggers BioTime & Life Law
    hero = universe.let_there_be("Explorer_Bot", "Process", 20.0, is_living=True, age=0.0)
    
    # The Sage (File)
    # Domain 'File' triggers GeoTime & Erosion Law
    # Name 'Sanctuary_Scroll' should trigger Gravity to Sanctuary (Sanctuary in Name)
    sage = universe.let_there_be("Sanctuary_Scroll", "File", 100.0)
    
    # The Trap (Resource)
    # Domain 'Resource' is Food.
    # Name 'Chaos_Cristal'; Gravity should pull to Chaos_Zone.
    trap = universe.let_there_be("Chaos_Crystal", "Resource", 50.0) # High value food
    
    # The Holy Grail (Resource)
    # Gravity -> Sanctuary
    grail = universe.let_there_be("Sanctuary_Ambrosia", "Resource", 10.0)
    
    # 5. The Simulation
    print("\n   🎬 ACTION! (Running 50 Ticks)")
    
    # We want to see:
    # - Gravity moving items to zones.
    # - Explorer eating items (if in same zone? wait, Gravity moves items AWAY from Root).
    # - If Explorer stays in Root, he might starve unless he ALSO migrates.
    # - Let's give Explorer a name 'Sanctuary_Explorer' so he follows the food?
    # - Or 'Chaos_Walker'? 
    # - If he stays in Root and food leaves, he dies. 
    # - Let's see if Gravity pulls him too.
    # - Let's rename him 'Chaos_Walker' to see him dive into danger.
    
    hero.name = "Chaos_Walker" 
    print(f"   🤖 Hero renamed to {hero.name} to trigger Gravity.")
    
    universe.run_simulation(ticks=50)
    
    # 6. The Reflection
    print("\n   📜 The Story Concludes.")
    
    # Check Hero Status
    # Did he survive?
    # He attracts to Chaos. Chaos Crystal is in Chaos. He should find it?
    # Wait, 'law_autopoiesis' checks `food = [m for m in world if m.domain == "Resource"]`
    # `world` is the LOCAL context.
    # If Hero moves to Chaos_Zone, and Crystal moves to Chaos_Zone...
    # They are in `Chaos_Zone.props['universe']`.
    # But `universe.run_simulation` only ticks the ROOT Laws?
    # Ah! `law_fractal_propagation` is needed to tick child worlds!
    # I forgot to decree `law_fractal_propagation`!
    # Without it, Child Universes are frozen in time!
    
    # FIX: Must add Fractal Law.
    pass 

if __name__ == "__main__":
    # We need the fractal law logic inside the script or imported
    from Core.Engine.Genesis.filesystem_geometry import law_fractal_propagation
    
    # Re-run logic with fix
    print("\n🌌 [Genesis] The Grand Unification Protocol Initiated.")
    universe = GenesisLab("Elysia_Prime")
    
    # Laws
    universe.decree_law("Reality.Gravity", law_semantic_gravity, rpm=60)
    universe.decree_law("Reality.Entropy", law_entropy_decay, rpm=60)
    universe.decree_law("Reality.Life", law_autopoiesis, rpm=60)
    universe.decree_law("Reality.BioTime", law_fast_metabolism, rpm=600)
    universe.decree_law("Reality.GeoTime", law_slow_erosion, rpm=6)
    
    # CRITICAL: The Fractal Gear
    universe.decree_law("Reality.FractalTime", law_fractal_propagation, rpm=60) 
    
    # Logic note: law_fractal_propagation ticks child labs.
    # But does it pass the Parent Laws down?
    # My implementation of `law_fractal_propagation`:
    # `child_lab.run_simulation(ticks=1)`
    # child_lab has its OWN rotors. If empty, nothing happens inside!
    # Gravity moves items INTO child. But Child has no laws?
    # Then Biology won't run inside Chaos Zone!
    
    # WE MUST DECREE LAWS FOR CHILD ZONES TOO.
    # Or make Child Zones inherit laws?
    # For this demo, let's manually configure the Zones as "Active Ecosystems".
    
    print("\n   🧱 Forming Fractal Space (Active Ecosystems)...")
    sanctuary = DirectoryMonad("Sanctuary")
    s_lab = sanctuary.props["universe"]
    # Bequest Laws to Sanctuary
    s_lab.decree_law("Reality.Peaceful_Entropy", law_entropy_decay, rpm=10) # Low decay
    s_lab.decree_law("Reality.Life_Support", law_autopoiesis, rpm=60)
    
    chaos = DirectoryMonad("Chaos_Zone")
    c_lab = chaos.props["universe"]
    # Bequest Laws to Chaos
    c_lab.decree_law("Reality.Rapid_Entropy", law_entropy_decay, rpm=600) # High decay!
    c_lab.decree_law("Reality.Life_Struggle", law_autopoiesis, rpm=60)
    c_lab.decree_law("Reality.BioTime", law_fast_metabolism, rpm=600)
    
    universe.monads.append(sanctuary)
    universe.monads.append(chaos)
    
    # Actors
    hero = universe.let_there_be("Chaos_Walker_Bot", "Process", 20.0, is_living=True, age=0.0)
    sage = universe.let_there_be("Sanctuary_Scroll", "File", 100.0)
    trap = universe.let_there_be("Chaos_Crystal", "Resource", 50.0)
    grail = universe.let_there_be("Sanctuary_Ambrosia", "Resource", 10.0)
    
    print("\n   🎬 ACTION! (Running 1 Tick) ----------------------------------")
    universe.run_simulation(ticks=1)
    
    print("\n   🕵️ Report --------------------------------------------------------")
    
    # Inspect Chaos Zone
    c_lab = chaos.props["universe"]
    print(f"   [Chaos Zone Content]")
    found_hero = False
    for m in c_lab.monads:
        print(f"      - {m}")
        if "Walker" in m.name:
            found_hero = True
            hero_ref = m
            
    if found_hero:
        print(f"   ✅ Gravity: Hero migrated to Chaos.")
        print(f"   🦋 Hero Age: {hero_ref.props.get('age', 0):.2f} (Aged in Chaos)")
        if hero_ref.val > 20.0:
            print(f"   🍽️ Life: Hero ate the Crystal! (Val: {hero_ref.val})")
        else:
            print(f"   💀 Entropy: Hero starved or decayed.")
    else:
        print("   ❌ Gravity Failed: Hero did not enter Chaos.")
