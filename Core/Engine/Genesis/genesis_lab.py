"""
Core.Engine.Genesis.genesis_lab
===============================
The God's Workbench.

Manages:
1. The Space (HyperSphere/List of Monads)
2. The Laws (List of UniversalRotors)
3. The Time (Tick Loop)
"""

import time
import logging
from typing import List, Dict, Callable
from Core.Engine.Genesis.universal_rotor import UniversalRotor
from Core.Engine.Genesis.concept_monad import ConceptMonad
from Core.Foundation.Nature.rotor import RotorConfig

logger = logging.getLogger("GenesisLab")

class GenesisLab:
    def __init__(self, name: str):
        self.name = name
        self.monads: List[ConceptMonad] = []
        self.rotors: List[UniversalRotor] = []
        self.time_step = 0
        
    def let_there_be(self, name: str, domain: str, val: float, **props):
        """Manifest a Monad."""
        m = ConceptMonad(name, domain, val, props=props)
        self.monads.append(m)
        logger.info(f"✨ Created: {m}")
        return m
        
    def decree_law(self, name: str, law_func: Callable, rpm: float = 60.0):
        """Manifest a Law (Rotor)."""
        config = RotorConfig(rpm=rpm)
        r = UniversalRotor(name, law_func, config)
        
        # Bind the world (Monads) to the Law
        # We pass a dictionary wrapper so the rotor can access monads
        context = {"world": self.monads} 
        r.bind_context(context)
        
        self.rotors.append(r)
        
        # [Fix] Instant Spin-Up (Laws are eternal, they don't need warm-up)
        r.current_rpm = rpm 
        r.target_rpm = rpm
        r.energy = 1.0 # Laws are fully active
        
        logger.info(f"📜 Decreed Law: {name} ({rpm} RPM)")
        return r
        
    def run_simulation(self, ticks: int = 10):
        """Run the Universe."""
        print(f"\n🧪 [Experiment Start] {self.name}")
        print(f"   Structure: {len(self.monads)} Monads, {len(self.rotors)} Laws\n")
        
        for t in range(ticks):
            self.time_step += 1
            print(f"   ⏱️ Tick {t}:")
            dt = 0.1
            
            # 1. Update Rotors (Physics & Laws)
            print(f"DEBUG: Updating {len(self.rotors)} Rotors.")
            for r in self.rotors:
                r.update(dt)
                
            # Log Snapshot (every 5 ticks)
            if t % 5 == 0:
                self._snapshot(t)
                
    def _snapshot(self, tick):
        print(f"   ⏱️ Tick {tick}:")
        for m in self.monads[:3]: # Show top 3
            print(f"      - {m}")
        if len(self.monads) > 3: print("      ... and more.")

# ==============================================================================
# ARCHETYPES (Templates)
# ==============================================================================

def law_gravity(context, dt, intensity):
    """Physics: F = G * m1 * m2 / r^2 (Simplified: Mass attracts Mass)"""
    world = context["world"]
    G = 0.1 * intensity # Gravity strength depends on Rotor Energy
    
    # Naive O(N^2) for demo
    for m1 in world:
        if m1.domain != "Physics": continue
        for m2 in world:
            if m1 == m2: continue
            if m2.domain != "Physics": continue
            
            # Simplified: Just pull values closer or increase mass (accretion)
            # Here: Mass increases slightly due to accretion
            m1.val += (m2.val * G * 0.01)

def law_inflation(context, dt, intensity):
    """Economics: Prices rise if demand > supply"""
    world = context["world"]
    inflation_rate = 0.05 * intensity
    
    for m in world:
        if m.domain == "Economy":
            m.val *= (1 + inflation_rate)

def law_trauma_healing(context, dt, intensity):
    """Psychology: Pain decreases over time (Healing)"""
    world = context["world"]
    healing_rate = 0.1 * intensity
    
    for m in world:
        if m.domain == "Mind" and m.props.get("type") == "Pain":
            m.val -= healing_rate
            if m.val < 0: m.val = 0
