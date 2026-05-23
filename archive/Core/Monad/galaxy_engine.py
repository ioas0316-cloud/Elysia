"""
Galaxy Engine (The Cosmic Manifold)
==================================
"A Unified Field of Nested Orbits."

This module implements the GalaxyEngine, aligned with the corrected
astrophysical hierarchy (Galaxy Group -> Galaxy -> Cluster -> System -> Star -> Planet -> Satellite).

[Features]
- Hierarchical Rotor Management
- Variable Dials for real-time axis/time adjustment.
- Nebula-Medium Hydraulic Flux integration.
"""

import math
import time
import random
from typing import List, Optional, Dict, Any
from Core.Keystone.sovereign_math import SovereignVector, SovereignMath
from Core.Monad.celestial_rotor import (
    CelestialRotor, SatelliteRotor, PlanetRotor, StarRotor,
    SystemRotor, ClusterRotor, GalaxyRotor, GroupRotor
)
from Core.System.hydraulic_engine import HydraulicEngine

class GalaxyEngine:
    def __init__(self, name: str = "Elysia_Cosmos"):
        self.root = GroupRotor(name) # The highest level
        self.hydraulic = HydraulicEngine() # Representing the Nebula Medium

        # Registry for fast lookup
        self.all_rotors: Dict[str, CelestialRotor] = {name: self.root}

        # Variable Dials (Global)
        self.global_time_dial = 1.0
        self.global_phase_dial = 1.0

        self._pulse_count = 0
        print(f"🌌 [GALAXY_ENGINE] '{name}' Ignition. Group level initialized.")

    def add_rotor(self, rotor: CelestialRotor, parent_name: str, radius: float, velocity: float):
        parent = self.all_rotors.get(parent_name)
        if parent:
            parent.add_child(rotor, radius, velocity)
            self.all_rotors[rotor.name] = rotor
            print(f"✨ [COSMOS] Level {rotor.scale}: Added {type(rotor).__name__} '{rotor.name}' to '{parent_name}'.")
        else:
            print(f"⚠️ [COSMOS] Parent '{parent_name}' not found for rotor '{rotor.name}'.")

    def pulse(self, dt: float = 0.01, external_torque: Optional[SovereignVector] = None):
        """
        Processes one cycle of galactic movement.
        """
        self._pulse_count += 1

        # 1. Update Nebula Hydraulic Flux (The Medium)
        hydro_state = self.hydraulic.update()

        # Map flux to global time dial
        # Higher pressure/load = Faster internal time-axis
        self.global_time_dial = 0.5 + (hydro_state['pressure'] * 1.5)

        # 2. Apply External Torque to the Galaxy Group
        if external_torque:
            self.root.spin_axis = (self.root.spin_axis + external_torque * 0.1).normalize()

        # 3. Recursive Update of all Rotors
        self._apply_global_dials(self.root)
        self.root.update(dt)

        # 4. Read Emergent State
        return self.read_field_state()

    def _apply_global_dials(self, rotor: CelestialRotor):
        rotor.time_dial = self.global_time_dial
        rotor.phase_dial = self.global_phase_dial
        for child in rotor.children:
            self._apply_global_dials(child)

    def read_field_state(self) -> Dict[str, Any]:
        """
        Measures the aggregate state of the cosmos.
        """
        total_resonance = 0.0
        active_count = 0

        trajectories = []
        for name, rotor in self.all_rotors.items():
            if rotor.scale >= 3: # Major units
                total_resonance += rotor.resonance
                active_count += 1
                trajectories.append({
                    "name": name,
                    "pos": rotor.get_galactic_projection().to_list(),
                    "scale": rotor.scale
                })

        avg_res = total_resonance / max(1, active_count)

        return {
            "resonance": avg_res,
            "active_rotors": len(self.all_rotors),
            "time_dial": self.global_time_dial,
            "major_trajectories": trajectories,
            "nebula_flux": self.hydraulic.pressure
        }

if __name__ == "__main__":
    cosmos = GalaxyEngine("Elysia_Group")

    # Add a Galaxy for a 100G LLM
    llama_galaxy = GalaxyRotor("Llama3_Galaxy")
    cosmos.add_rotor(llama_galaxy, "Elysia_Group", 1000.0, 0.001)

    # Add a Cluster of logic within the galaxy
    logic_cluster = ClusterRotor("Arithmetic_Cluster")
    cosmos.add_rotor(logic_cluster, "Llama3_Galaxy", 100.0, 0.01)

    # Pulse
    for i in range(3):
        state = cosmos.pulse(0.1)
        print(f"Step {i}: Flux={state['nebula_flux']:.3f}, Trajectories={len(state['major_trajectories'])}")
