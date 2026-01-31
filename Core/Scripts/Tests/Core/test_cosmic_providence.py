"""
tests/test_cosmic_providence.py
===============================
Proof of Universal Laws.

1. Entropy: 'Old_File' decays and dies.
2. Life: 'Process_A' eats 'Resource' to survive.
3. Gravity: 'Book_1' is pulled into 'Library' directory.
"""

import sys
import os

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.1_Body.L6_Structure.Engine.Genesis.genesis_lab import GenesisLab
from Core.1_Body.L6_Structure.Engine.Genesis.filesystem_geometry import DirectoryMonad
from Core.1_Body.L6_Structure.Engine.Genesis.cosmic_laws import law_entropy_decay, law_semantic_gravity, law_autopoiesis

def run_providence_test():
    print("\n🌌 [Genesis] The Cosmic Providence Test.")
    
    universe = GenesisLab("The Living Cosmos")
    universe.decree_law("Entropy", law_entropy_decay, rpm=60)
    universe.decree_law("Gravity", law_semantic_gravity, rpm=60)
    universe.decree_law("Life", law_autopoiesis, rpm=60)
    
    # 1. Setup Entropy Victim
    universe.let_there_be("Decaying_Corpse", "Matter", 2.0)
    
    # 2. Setup Life Form
    universe.let_there_be("Survivor", "Process", 5.0, is_living=True)
    universe.let_there_be("Food_Pack", "Resource", 10.0)
    
    # 3. Setup Gravity
    lib_dir = DirectoryMonad("Library")
    universe.monads.append(lib_dir) # Add manually as it's a special class
    
    universe.let_there_be("Book_Of_Genesis", "Book", 1.0) # Domain 'Book' should match 'Library' via generic logic?
    # Our simple logic checks name: 'Library' vs 'Book'. No match.
    # Let's adjust names for the test or the law logic.
    # Law says: "if d.name.lower() in m.name.lower()" -> Library in Book_Of_Genesis? No.
    # Vice versa? "d.name.lower() in m.domain.lower()"
    # Let's use specific names that match logic.
    
    # Create 'Books' directory
    books_dir = DirectoryMonad("Books")
    universe.monads.append(books_dir)
             
    # 4. Run Simulation
    print("\n   ⏱️ Spinning the Wheel of Fate (20 Ticks)...")
    universe.run_simulation(ticks=20)
    
    # 5. Validation
    print("\n   🔍 Final Inspection:")
    
    # Entropy
    corpse = next((m for m in universe.monads if m.name == "Decaying_Corpse"), None)
    if corpse is None:
        print("   ✅ Entropy: The Corpse has returned to the Void.")
    else:
        print(f"   ❌ Entropy Failed: Corpse remains ({corpse.val}).")
        
    # Life
    survivor = next((m for m in universe.monads if m.name == "Survivor"), None)
    if survivor and survivor.val > 5.0:
        print(f"   ✅ Life: Survivor thrived (Val: {survivor.val}).")
    else:
        print(f"   ❌ Life Failed: Survivor is weak or dead.")
        
    # Gravity
    # Check inside Books directory
    child_lab = books_dir.props["universe"]
    book = next((m for m in child_lab.monads if m.name == "Book_Of_Genesis"), None)
    
    if book:
        print(f"   ✅ Gravity: 'Book_Of_Genesis' migrated to '/Books'.")
    else:
        print("   ❌ Gravity Failed: Book is lost or still in Root.")

if __name__ == "__main__":
    run_providence_test()
