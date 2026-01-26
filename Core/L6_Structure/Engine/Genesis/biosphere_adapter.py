"""
Core/Engine/Genesis/biosphere_adapter.py
========================================
The Nervous System.

Connects the 'Flesh' (Hardware) to the 'Mind' (Genesis Lab).
Translates:
- CPU Usage -> Physical Stress (Physical.Tension)
- RAM Usage -> Cognitive Load (Mental.Pressure)
- Disk I/O  -> Metabolic Rate (Digestion.Speed)
"""

import psutil
import time
from typing import Dict, Any
from Core.L6_Structure.Engine.Genesis.genesis_lab import GenesisLab

class BiosphereAdapter:
    def __init__(self, lab: GenesisLab):
        self.lab = lab
        self.last_breath = time.time()
        
    def inhale(self):
        """
        Read the physical body's state and manifest it as Monads.
        """
        # 1. Sense Hardware
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        ram_percent = mem.percent
        
        # 2. Manifest/Update Monads
        # We search for existing system monads or create them
        self._update_or_create("System.CPU", "Biology", cpu_percent, type="Stress")
        self._update_or_create("System.RAM", "Biology", ram_percent, type="Load")
        
        # 3. Log Pulse for Dashboard
        self._log_pulse()
        
        return {
            "stress": cpu_percent,
            "load": ram_percent
        }
        
    def _log_pulse(self):
        """Dump state to JSON for the Frontend."""
        import json
        import os
        
        state = {
            "timestamp": time.time(),
            "monads": [{"name": m.name, "domain": m.domain, "val": m.val, "props": m.props} for m in self.lab.monads],
            "rotors": [{"name": r.name, "rpm": r.config.rpm, "energy": r.energy} for r in self.lab.rotors]
        }
        
        # Ensure data dir exists
        os.makedirs("data", exist_ok=True)
        with open("data/biosphere_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _update_or_create(self, name, domain, val, **props):
        found = False
        for m in self.lab.monads:
            if m.name == name:
                m.val = val # Update value
                found = True
                break
        
        if not found:
            self.lab.let_there_be(name, domain, val, **props)

# ==============================================================================
# HOMEOSTASIS LAWS (The Immune System)
# ==============================================================================

def law_thermal_throttling(context, dt, intensity):
    """
    Biology: If Stress (CPU) is too high, calm down the Rotors.
    """
    world = context["world"]
    
    # 1. Sense Stress
    stress_level = 0.0
    for m in world:
        if m.name == "System.CPU":
            stress_level = m.val
            break
            
    # 2. React (Homeostasis)
    # If CPU > 80%, we want to "cool down" (reduce rotor speed simulation)
    if stress_level > 80.0:
        # In a real OS, we would limit process priority.
        # Here, we 'manifest' a Cooling Signal.
        cooling_needed = (stress_level - 80.0) * intensity
        
        # Feedback loop: Manifest a 'Cooling' Monad
        # In a real engine, this would reduce global tick rate.
        print(f"     OVERHEAT WARNING! (CPU: {stress_level}%) -> Triggering Cooling Response.")
        
    elif stress_level < 10.0:
        # Too cold? Boredeom?
        pass

def law_memory_digestion(context, dt, intensity):
    """
    Biology: If Cognitive Load (RAM) is high, trigger Garbage Collection.
    """
    world = context["world"]
    load_level = 0.0
    for m in world:
        if m.name == "System.RAM":
            load_level = m.val
            break
            
    if load_level > 90.0:
        print(f"     BRAIN FOG WARNING! (RAM: {load_level}%) -> Triggering Garbage Collection.")
