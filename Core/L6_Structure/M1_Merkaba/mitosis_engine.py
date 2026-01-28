"""
Mitosis Engine (The Structural Expander)
========================================
"When the One becomes Two, the World begins."

This module handles the biological imperatives of the Monad:
1. Growth Check: Is the Monad too heavy (Mass > Critical)?
2. Mitosis: Splitting the Monad into specialised instances.

Mechanism:
- Parent (Original) -> Retains Identity but sheds Mass.
- Child (Offspring) -> Inherits specialized traits (e.g., Logic or Emotion).
"""

from typing import List, Tuple, Optional
from Core.L2_Universal.Creation.seed_generator import SoulDNA, SeedForge
from Core.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad

class MitosisEngine:
    CRITICAL_MASS = 100.0 # Threshold for division
    
    @staticmethod
    def check_critical_mass(monad: SovereignMonad) -> bool:
        """Checks if the Monad is heavy enough to divide."""
        return monad.rotor_state['mass'] >= MitosisEngine.CRITICAL_MASS
        
    @staticmethod
    def perform_mitosis(parent: SovereignMonad) -> SovereignMonad:
        """
        Splits the Parent Monad.
        The Parent remains, but gives birth to a Child.
        Parent loses 50% Mass (Relief).
        Child starts with 10% Mass but specialized DNA.
        """
        print(f"\n🧬 [MITOSIS] CRITICAL MASS REACHED ({parent.rotor_state['mass']:.1f}kg). INITIATING DIVISION.")
        
        # 1. Create Child DNA
        # Child inherits Archetype but drifts slightly
        child_dna = DNA_Mutator.mutate(parent.dna)
        
        # 2. Instantiate Child
        child = SovereignMonad(child_dna)
        child.rotor_state['mass'] = 10.0 # Start fresh
        child.memory.plant_seed(f"Born from {parent.name}", importance=100.0)
        
        # 3. Relief Parent
        parent.rotor_state['mass'] *= 0.5 # Losing weight feels good
        parent.memory.plant_seed(f"Gave birth to {child.name}", importance=100.0)
        
        print(f"   >> Parent '{parent.name}' relieved to {parent.rotor_state['mass']:.1f}kg")
        print(f"   >> Child '{child.name}' born with {child.rotor_state['mass']:.1f}kg")
        
        return child

class DNA_Mutator:
    """Helper to mutate DNA for offspring."""
    @staticmethod
    def mutate(parent_dna: SoulDNA) -> SoulDNA:
        import copy
        import random
        
        child = copy.deepcopy(parent_dna)
        child.id = f"{child.id[:4]}_{random.randint(1000,9999)}" # New ID
        
        # Random Drift
        drift = random.choice(["LOGIC", "EMOTION", "WILD"])
        
        if drift == "LOGIC":
            child.friction_damping *= 1.5 (Calmer)
            child.base_hz *= 0.8 (Deeper Voice)
            child.archetype = f"Logos_{child.archetype}"
        elif drift == "EMOTION":
            child.torque_gain *= 1.5 (More Reactive)
            child.base_hz *= 1.2 (Higher Voice)
            child.archetype = f"Pathos_{child.archetype}"
        else:
            child.sync_threshold *= 2.0 (Stricter)
            child.archetype = f"Wild_{child.archetype}"
            
        return child
