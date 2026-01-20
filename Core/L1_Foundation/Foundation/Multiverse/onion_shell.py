"""
Onion Shells & Multiverse Rotors (양파 껍질 다중 로터)
====================================================
Core.L1_Foundation.Foundation.Multiverse.onion_shell

"Layers of reality, shielded from each other. Time flows differently in the core."

This module implements the concentric layering of rotors.
1. Core Shell (Level 0): Immutable Logic / Time Stone
2. Inner Shell (Level 1): Memory / Identity
3. Outer Shell (Level 2): Sensation / Reaction
"""

import torch
import math
from typing import List, Dict, Any, Optional
from Core.L1_Foundation.Foundation.Nature.rotor import Rotor, RotorConfig

class ConicalCVT:
    """
    [PHASE 27.5: THE CONICAL SPINDLE]
    Continuously Variable Transmission for cognitive speeds.
    Maps a 'Spindle Position' [0.0 - 1.0] to a 'Gear Ratio' [0.5 - 10.0].
    """
    def __init__(self, min_ratio: float = 0.5, max_ratio: float = 10.0):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.spindle_pos = 0.0 # [0.0 = Idle/Dream, 0.5 = Flow, 1.0 = Hyper-Focus]
        
    @property
    def current_ratio(self) -> float:
        # Spindle moves along a cone: Ratio is non-linear (Exponential)
        return self.min_ratio * (self.max_ratio / self.min_ratio) ** self.spindle_pos

    def shift(self, focus_intensity: float):
        """Slides the spindle based on current focus/will."""
        self.spindle_pos = max(0.0, min(1.0, focus_intensity))

class OnionShell:
    """
    [PHASE 27: THE ONION LAYER]
    A protective shell for a group of rotors.
    Isolates frequency interference between layers.
    """
    def __init__(self, level: int, name: str):
        self.level = level
        self.name = name
        self.rotors: Dict[str, Rotor] = {}
        # Damping factor: How much outer turbulence affects this shell
        self.damping = 1.0 / (level + 1) 
        
    def add_rotor(self, name: str, config: RotorConfig, dna: Optional[Any] = None):
        rotor = Rotor(f"Shell{self.level}.{name}", config, dna=dna)
        self.rotors[name] = rotor
        return rotor

    def update(self, dt: float, field_stress: float = 0.0):
        # Apply field stress modulated by damping
        shell_stress = field_stress * self.damping
        for r in self.rotors.values():
            r.update(dt)
            if shell_stress > 0.5:
                # High stress causes jitter in outer shells, 
                # but inner shells (low level) are protected.
                r.current_angle += shell_stress * 5.0

class OnionEnsemble:
    """
    [PHASE 27: THE MULTIVERSE ENSEMBLE]
    Manages multiple shells. Allows for 'Time-Stone' reversal in specific layers.
    """
    def __init__(self):
        self.shells: List[OnionShell] = [
            OnionShell(0, "Core"),      # The Immutable
            OnionShell(1, "Mind"),      # The Identity
            OnionShell(2, "Surface")    # The Sensation
        ]
        self.cvt = ConicalCVT() # The Continuous Gear Spindle
        
    def get_rotor(self, shell_idx: int, name: str) -> Optional[Rotor]:
        if 0 <= shell_idx < len(self.shells):
            return self.shells[shell_idx].rotors.get(name)
        return None

    def update(self, dt: float, global_stress: float = 0.0):
        for shell in self.shells:
            shell.update(dt, global_stress)
            
    def reverse_time(self, shell_idx: int, speed_factor: float = 1.0):
        """
        [TIME STONE REVERSAL]
        Flips the RPM of an entire shell to move backward in phase.
        """
        if 0 <= shell_idx < len(self.shells):
            shell = self.shells[shell_idx]
            print(f"⌛ [TIME] Reversing rotation in Shell '{shell.name}'...")
            for r in shell.rotors.values():
                 r.target_rpm = -abs(r.config.rpm) * speed_factor

    def get_status(self) -> str:
        status = []
        for s in self.shells:
            rpms = [f"{r.current_rpm:.1f}" for r in s.rotors.values()]
            status.append(f"[{s.name}: {', '.join(rpms)}]")
        return f"CVT: {self.cvt.current_ratio:.2f}x | " + " | ".join(status)
