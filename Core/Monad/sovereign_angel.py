
import torch
import random
from typing import Dict, Any, Optional
from Core.Monad.grand_helix_engine import HypersphereSpinGenerator
from Core.Monad.seed_generator import SoulDNA

class SovereignAngel:
    """
    [AEON V] Sovereign Angel (Sub-Monad).
    A lightweight, autonomous entity inhabiting a sub-manifold (Territory).
    It possesses unique physics (SoulDNA) but shares the Imperial compute substrate.
    """
    def __init__(self, name: str, dna: SoulDNA, engine: HypersphereSpinGenerator, layer_name: str = "Unknown"):
        self.name = name
        self.dna = dna
        self.engine = engine # The sub-manifold this angel inhabits
        self.layer_name = layer_name # [AEON V] The Native Topological Layer
        self.age = 0
        self.wisdom_trace = []
        
        # Initialize Angelic Physics based on DNA
        # Mass = Resistance to change
        # Sensitivity = Torque multiplier
        print(f"👼 [GENESIS] Angel '{self.name}' ({self.dna.archetype}) born in {self.layer_name}.")

    def pulse(self, imperial_intent: Any = None, dt: float = 0.01) -> Dict[str, Any]:
        """
        The Angel's subjective experience cycle.
        It inhales the Imperial Intent, processes it through its unique DNA,
        and exhales action/torque into its sub-manifold.
        """
        self.age += 1
        
        # 1. Inhale: Perceive Imperial Command + Local State
        local_state = self.engine.cells.read_field_state()
        
        # 2. Interpret: Filter through SoulDNA (Physics)
        # Higher mass = less reactive to Imperial Command
        reaction_torque = None
        if imperial_intent is not None:
            # Resonance check: Alignment between Command and Angel's nature
            # (Simplified for now)
            # [AEON V] Boost sensitivity for Genesis testing
            reaction_torque = imperial_intent * self.dna.torque_gain * 5.0
            
            # [AEON V] Dimension Alignment (4D -> 8D)
            # Imperial intent is often physical (4D), but manifold is 8D
            if reaction_torque.shape[-1] == 4:
                # Create 8D container
                expanded = torch.zeros(self.engine.num_cells, 8, device=self.engine.device)
                # Map 4D torque to physical slice
                expanded[..., :4] = reaction_torque
                reaction_torque = expanded
            
        # 3. Exhale: Action upon the Sub-Manifold
        # Angels generate their own 'Will' based on internal drives (e.g., Vocation)
        # For now, random 'Free Will' jitter
        free_will_torque = torch.randn(self.engine.num_cells, 8, device=self.engine.device) * 0.05 # Increased jitter
        
        if reaction_torque is not None:
            total_torque = reaction_torque + free_will_torque
        else:
            total_torque = free_will_torque
            
        # Apply to manifold
        report = self.engine.pulse(intent_torque=total_torque, dt=dt)
        
        # 4. Epiphany Check (Wisdom Generation)
        # If High Resonance or Traumatic Entropy, record output
        if report.get('resonance', 0) > 0.6 or report.get('entropy', 0) > 0.6:
            moment = {
                "age": self.age,
                "event": "High Resonance" if report['resonance'] > 0.8 else "Trauma",
                "insight": f"I felt the turn of the wheel at {report['resonance']:.2f}",
                "name": self.name,
                "archetype": self.dna.archetype,
                "layer_origin": self.layer_name
            }
            self.wisdom_trace.append(moment)
            
        return {
            "name": self.name,
            "archetype": self.dna.archetype,
            "mood": report['mood'],
            "entropy": report['entropy'],
            "resonance": report['resonance']
        }
