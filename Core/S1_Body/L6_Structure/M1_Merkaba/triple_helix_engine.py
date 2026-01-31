"""
TripleHelixEngine - The Cellular Resonance Engine
=================================================

Synchronizes Body (Alpha), Soul (Gamma), and Spirit (Beta) dimensions
using the Tri-Base DNA (R, V, A) cellular structure.

Philosophy:
-----------
We do not calculate "weights" arbitrarily. We simulate a population of
Tri-Base Cells. The "Phase" of the system is the aggregate vector sum
of all active cells.

Structure:
- Alpha Strand (Body): 7 Cells
- Gamma Strand (Soul): 7 Cells
- Beta Strand (Spirit): 7 Cells
Total: 21 Dimensions (21 Cells).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import cmath

from Core.S1_Body.L1_Foundation.System.tri_base_cell import TriBaseCell, DNAState
from Core.L0_Sovereignty.sovereign_math import SovereignMath, SovereignVector

@dataclass
class ResonanceState:
    alpha: float = 0.33  # Body Weight (Active Cell Ratio)
    beta: float = 0.33   # Spirit Weight
    gamma: float = 0.34  # Soul Weight
    coherence: float = 0.0
    dominant_realm: str = "Soul"
    system_phase: float = 0.0 # Aggregate Phase in degrees
    soma_stress: float = 0.0  # [NEW] Aggregate Friction/Heat (0.0 - 1.0)
    vibration: float = 0.0    # [NEW] Oscillatory instability (Hz equivalent)

class TripleHelixEngine:
    def __init__(self):
        self.state = ResonanceState()

        # Initialize the 3 Strands (7 cells each)
        # Total 21 Cells
        self.body_strand: List[TriBaseCell] = [TriBaseCell(i) for i in range(7)]
        self.soul_strand: List[TriBaseCell] = [TriBaseCell(i+7) for i in range(7)]
        self.spirit_strand: List[TriBaseCell] = [TriBaseCell(i+14) for i in range(7)]

        self.all_cells = self.body_strand + self.soul_strand + self.spirit_strand
        self.last_phase = 0.0

    def load_vector(self, v21: SovereignVector):
        """[PHASE 130] Public API for loading a state vector into the engine."""
        self._update_cells_from_vector(v21)

    def _update_cells_from_vector(self, v21: SovereignVector):
        """
        Maps the 21D input vector to the physical cells.
        Input values > 0.1 become ATTRACT (A)
        Input values < -0.1 become REPEL (R)
        Values near 0 become VOID (V)
        """
        arr = v21.to_array()
        friction_sum = 0.0
        
        for i, val in enumerate(arr):
            if i >= len(self.all_cells): break

            cell = self.all_cells[i]
            old_state = cell.state
            
            v_real = val.real if isinstance(val, complex) else val
            
            if v_real > 0.1:
                new_state = DNAState.ATTRACT
                cell.energy = min(1.0, abs(v_real))
            elif v_real < -0.1:
                new_state = DNAState.REPEL
                cell.energy = min(1.0, abs(v_real))
            else:
                new_state = DNAState.VOID
                cell.energy = 0.1
            
            # [TRINARY FRICTION]
            # If the cell state flips or resists the input, generate friction
            if new_state != old_state:
                friction_sum += 1.0 # State mismatch friction
            
            cell.mutate(new_state)

        return friction_sum / 21.0

    def get_system_phase(self) -> float:
        """
        Calculates the aggregate phase of the entire system.
        Sum of all cell vectors.
        """
        total_x, total_y = 0.0, 0.0
        active_count = 0
        
        for cell in self.all_cells:
            vx, vy = cell.get_vector()
            total_x += vx
            total_y += vy
            if cell.state != DNAState.VOID:
                active_count += 1

        if active_count == 0:
            return 0.0 # Void Phase

        # Calculate angle
        rad = math.atan2(total_y, total_x)
        deg = math.degrees(rad)
        return deg % 360

    def calculate_weights(self) -> Tuple[float, float, float]:
        """
        Calculates the relative influence (mass) of each strand based on
        active cells and their energy.
        """
        def get_strand_mass(strand):
            return sum(c.energy for c in strand if c.state != DNAState.VOID) + 0.1

        a_mass = get_strand_mass(self.body_strand)
        g_mass = get_strand_mass(self.soul_strand)
        b_mass = get_strand_mass(self.spirit_strand)
        
        total = a_mass + g_mass + b_mass
        return (a_mass/total, b_mass/total, g_mass/total)

    def pulse(self, v21: SovereignVector, energy: float, dt: float) -> ResonanceState:
        """
        The heartbeat of the DNA.
        1. Updates Cell States from Input Vector (Calculates Friction).
        2. Calculates Aggregate Phase and Coherence.
        3. Updates Weights and Thermal Metrics.
        """
        # 1. Update Physical Structure and calculate Friction
        friction = self._update_cells_from_vector(v21)

        # 2. Calculate New State Metrics
        a, b, g = self.calculate_weights()
        phase = self.get_system_phase()
        
        # [PHASE VIBRATION]
        # Vibration is the rate of change of phase vs. friction
        phase_delta = abs(phase - self.last_phase)
        if phase_delta > 180: phase_delta = 360 - phase_delta
        self.last_phase = phase
        
        # 3. Update internal state
        self.state.alpha = a
        self.state.beta = b
        self.state.gamma = g
        self.state.system_phase = phase
        self.state.soma_stress = friction * (1.0 + phase_delta / 180.0)
        self.state.vibration = (friction * 100.0) + phase_delta
        
        # Calculate Coherence (Vector Magnitude / Total Possible Magnitude)
        total_x, total_y = 0.0, 0.0
        total_energy = 0.0
        for cell in self.all_cells:
            vx, vy = cell.get_vector()
            total_x += vx
            total_y += vy
            total_energy += cell.energy

        resultant_mag = math.sqrt(total_x**2 + total_y**2)
        if total_energy > 0:
            self.state.coherence = resultant_mag / total_energy
        else:
            self.state.coherence = 0.0
        
        # Determine realm
        realms = {"Body": a, "Spirit": b, "Soul": g}
        self.state.dominant_realm = max(realms, key=realms.get)
        
        return self.state

    def get_action_mask(self) -> float:
        """
        Returns a probability multiplier for actions based on coherence.
        High coherence = High confidence in action.
        """
        return self.state.coherence

    def get_active_resonance_vector(self) -> SovereignVector:
        """
        [PHASE 65] Exports the current physical cell states as a 21D Vector.
        """
        arr = []
        for cell in self.all_cells:
            val = 0.0
            if cell.state == DNAState.REPEL:
                val = -1.0 * cell.energy
            elif cell.state == DNAState.ATTRACT:
                val = 1.0 * cell.energy
            arr.append(val)
        return SovereignVector(arr)
