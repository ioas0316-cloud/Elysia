"""
Providence Engine (     )
===========================
"The Lawmaker determining the fate of matter."
"                ."

This engine simulates the Fundamental Laws of Trinity Physics.
It handles Phase Transitions, Chemical Reactions, and State Changes.

Principles:
1. Ascension is Energy (Heat, Light, Expansion).
2. Gravity is Order (Cold, Pressure, Density).
3. Flow is Motion (Fluidity, Time).

Formula:
State = Matter + Environment
"""

import math
from typing import Tuple
from Core.S1_Body.L4_Causality.World.Physics.trinity_fields import TrinityVector

class ProvidenceEngine:
    def __init__(self):
        pass

    def apply_thermodynamics(self, subject: TrinityVector, environment: TrinityVector) -> TrinityVector:
        """
        Applies environmental forces to a subject to determine its new state.
        
        Args:
            subject: The inherent properties of the object (e.g. Water).
            environment: The external forces (e.g. Heat, Pressure).
            
        Returns:
            A new TrinityVector representing the Transmuted State.
        """
        # 1. Energy Transfer
        # Heat (Env.Ascension) increases Object.Ascension and Object.Flow
        # Cold (Env.Gravity) increases Object.Gravity and reduces Object.Flow
        
        new_g = subject.gravity + (environment.gravity * 0.5) - (environment.ascension * 0.3)
        new_f = subject.flow + (environment.flow * 0.5) + (environment.ascension * 0.4) - (environment.gravity * 0.4)
        new_a = subject.ascension + (environment.ascension * 0.8) - (environment.gravity * 0.2)
        
        # Clamp values (0.0 to 1.0 ideally, but physics allows overflow for extreme states)
        new_g = max(0.0, new_g)
        new_f = max(0.0, new_f)
        new_a = max(0.0, new_a)
        
        # 2. State Determination (Phase Transition)
        # If Energy is overwhelming -> Gas/Plasma
        if new_a > 1.0 or (new_a > 0.8 and new_a > new_g * 2):
            return self._transmute_to_gas(TrinityVector(new_g, new_f, new_a))
            
        # If Gravity is overwhelming -> Solid/Crystal
        if new_g > 1.0 or (new_g > 0.8 and new_g > new_a * 2):
            return self._transmute_to_solid(TrinityVector(new_g, new_f, new_a))
            
        # Balanced High Flow -> Liquid
        if new_f > 0.6 and new_f > new_g and new_f > new_a:
            return self._transmute_to_liquid(TrinityVector(new_g, new_f, new_a))
            
        # Default: Retain original form but modified
        return TrinityVector(new_g, new_f, new_a)

    def apply_molecular_dynamics(self, matter_name: str, temperature_k: float) -> str:
        """
        Determines the state of matter based on Real Physics (Kinetic Theory).
        
        Args:
            matter_name: Key in molecular_database (e.g. 'water')
            temperature_k: Temperature in Kelvin
            
        Returns:
            State string ("SOLID", "LIQUID", "GAS")
        """
        from Core.S1_Body.L4_Causality.World.Physics.molecular_database import get_molecule
        
        mol = get_molecule(matter_name)
        
        
        # Boltzmann Constant approximation (arbitrary unit for simulation scale)
        # Calibrated for Water: 1.5 * k_b * 373 ~ 0.42
        k_b = 0.00075 
        
        # Kinetic Energy = 3/2 kT
        kinetic_energy = 1.5 * k_b * temperature_k
        
        # Bond Energies
        e_bond_solid = mol['bond_energy_solid']
        e_bond_liquid = mol['bond_energy_liquid']
        
        # State Logic derived from Energy vs Bond
        
        # If KE is enough to break Liquid bonds -> GAS
        # Note: Liquid bond energy here represents vaporization barrier
        if kinetic_energy > e_bond_liquid:
            return "GAS"
            
        # If KE is enough to break Solid lattice but not Vaporize -> LIQUID
        # Note: We compare against Freezing Point logic usually, but here using Energy.
        # Simplification: If Temp > Freezing Point (derived or looked up)
        
        # Let's use the explicit Temperature Points for accuracy if Energy is ambiguous,
        # but ideally we want Energy to drive it.
        # For simulation stability, we check Phase Transition Thresholds defined by Critical Points.
        
        if temperature_k >= mol['boiling_point_k']:
            return "GAS"
        elif temperature_k >= mol['freezing_point_k']:
            return "LIQUID"
        else:
            return "SOLID"
            
    def _transmute_to_gas(self, v: TrinityVector) -> TrinityVector:
        """Expansion: Low Gravity, High Ascension."""
        # Gas expands, losing density but gaining volume (Flow/Ascension)
        return TrinityVector(v.gravity * 0.1, v.flow * 1.2, v.ascension * 1.1)

    def _transmute_to_solid(self, v: TrinityVector) -> TrinityVector:
        """Compression: High Gravity, Low Flow."""
        # Solid contracts.
        return TrinityVector(v.gravity * 1.1, v.flow * 0.1, v.ascension * 0.5)

    def _transmute_to_liquid(self, v: TrinityVector) -> TrinityVector:
        """Melting: Balanced Gravity/Ascension, High Flow."""
        return TrinityVector(v.gravity * 0.6, 1.0, v.ascension * 0.6)

    def resolve_interaction(self, element_a: str, element_b: str) -> str:
        """
        Symbolic Interaction Matrix for The Hexagon.
        Returns the name of the resulting phenomenon.
        """
        pair = sorted([element_a.lower(), element_b.lower()])
        
        # --- The Sacred Combinations ---
        if pair == ["darkness", "light"]: return "SHADOW"
        if pair == ["fire", "water"]: return "STEAM"
        if pair == ["earth", "wind"]: return "DUST"
        if pair == ["earth", "fire"]: return "MAGMA"
        if pair == ["earth", "water"]: return "MUD"
        if pair == ["fire", "wind"]: return "WILDFIRE"
        if pair == ["light", "water"]: return "LIFE" # Photosynthesis archetype
        if pair == ["darkness", "fire"]: return "DESPAIR" # Cold fire / destruction
        
        return "UNKNOWN_ALLOY"
