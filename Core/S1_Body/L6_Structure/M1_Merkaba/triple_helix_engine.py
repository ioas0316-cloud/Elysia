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

[PHASE 60 Update]:
Now incorporates "Phase-Axis Directionality" and "Neural Mobility".
- The Cellular Grid is the "Road".
- The Phase is the "Vehicle".
- Conservation of Momentum applies to rotational energy.
- [SCALABILITY]: Uses Vector API for N-Dimensional Steering.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import cmath
import random

from Core.S1_Body.L1_Foundation.System.tri_base_cell import TriBaseCell, DNAState
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignMath, SovereignVector

@dataclass
class ResonanceState:
    alpha: float = 0.33  # Body Weight (Active Cell Ratio)
    beta: float = 0.33   # Spirit Weight
    gamma: float = 0.34  # Soul Weight
    coherence: float = 0.0
    dominant_realm: str = "Soul"
    system_phase: float = 0.0 # Aggregate Phase in degrees
    soma_stress: float = 0.0  # Aggregate Friction/Heat (0.0 - 1.0)
    vibration: float = 0.0    # Oscillatory instability (Hz equivalent)

    # [PHASE-AXIS EXTENSIONS]
    axis_tilt: List[float] = field(default_factory=lambda: [0.0]) # N-Dimensional Axis State
    rotational_momentum: float = 0.0 # Conserved energy of thought
    gradient_flow: float = 0.0    # Magnitude of pre-pulse flow

class TripleHelixEngine:
    def __init__(self):
        self.state = ResonanceState()

        # Initialize the 3 Strands (7 cells each)
        # Total 21 Cells (The Road)
        self.body_strand: List[TriBaseCell] = [TriBaseCell(i) for i in range(7)]
        self.soul_strand: List[TriBaseCell] = [TriBaseCell(i+7) for i in range(7)]
        self.spirit_strand: List[TriBaseCell] = [TriBaseCell(i+14) for i in range(7)]

        self.all_cells = self.body_strand + self.soul_strand + self.spirit_strand
        self.last_phase = 0.0

        # Steering Suspension (PID)
        # We store tilt as a list to support N-dimensions in future
        self.current_tilt_vector: List[float] = [0.0]
        self.tilt_velocity: List[float] = [0.0]

        self.suspension_k = 0.1 # Spring constant
        self.suspension_d = 0.8 # Damping factor

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

    def flow_equilibration(self) -> float:
        """
        [GRADIENT-DRIVEN POTENTIAL]
        Simulates natural energy flow between cells before logic is applied.
        Water flows downhill (High Energy -> Low Energy).
        Returns the total magnitude of flow (The 'Pre-Pulse' Activity).
        """
        total_flow = 0.0

        # Simple neighbor exchange simulation
        # In a 1D strand, checking i-1 and i+1
        for i in range(len(self.all_cells)):
            cell = self.all_cells[i]
            if i > 0:
                left = self.all_cells[i-1]
                diff = cell.energy - left.energy
                if abs(diff) > 0.1:
                    flow = diff * 0.1 # 10% equalization
                    cell.energy -= flow
                    left.energy += flow
                    total_flow += abs(flow)

        return total_flow

    def apply_steering(self, target_vector: List[float], dt: float):
        """
        [PHASE-AXIS STEERING - VECTOR API]
        Applies PID damping to the N-dimensional axis tilt.

        Args:
            target_vector: List of floats.
                           Index 0 = Z-Axis (Vertical/Horizontal).
                           Future indices = W, N, etc.
        """
        # Ensure internal state matches dimension of target
        while len(self.current_tilt_vector) < len(target_vector):
            self.current_tilt_vector.append(0.0)
            self.tilt_velocity.append(0.0)

        # Physics Step: Spring-Damper System per axis
        for i, target_val in enumerate(target_vector):
            current = self.current_tilt_vector[i]
            displacement = target_val - current

            force = displacement * self.suspension_k
            self.tilt_velocity[i] += force
            self.tilt_velocity[i] *= self.suspension_d # Apply Damping

            self.current_tilt_vector[i] += self.tilt_velocity[i]

            # Clamp to physical limits
            self.current_tilt_vector[i] = max(-1.0, min(1.0, self.current_tilt_vector[i]))

        # Update State
        self.state.axis_tilt = list(self.current_tilt_vector)

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

    def pulse(self, v21: SovereignVector, energy: float, dt: float, target_tilt: List[float] = None) -> ResonanceState:
        """
        The heartbeat of the DNA.
        [UPDATED] Incorporates Gradient Flow and Phase-Axis Physics (Vector API).
        """
        if target_tilt is None:
            target_tilt = [0.0]

        # 1. [GRADIENT FLOW] Pre-pulse natural equilibration
        flow_mag = self.flow_equilibration()
        self.state.gradient_flow = flow_mag

        # 2. Update Physical Structure and calculate Friction (The Road Interaction)
        friction = self._update_cells_from_vector(v21)

        # 3. [STEERING] Apply Damped Axis Shift (Vector API)
        self.apply_steering(target_tilt, dt)

        # 4. Calculate New State Metrics
        a, b, g = self.calculate_weights()
        phase = self.get_system_phase()
        
        # [PHASE VIBRATION & MOMENTUM]
        # Vibration is the rate of change of phase vs. friction
        phase_delta = abs(phase - self.last_phase)
        if phase_delta > 180: phase_delta = 360 - phase_delta
        self.last_phase = phase
        
        # Conservation of Momentum:
        # Energy = Mass * Velocity^2. We treat Phase Delta as Velocity.
        # Even if steering changes, this scalar energy persists.
        current_momentum = phase_delta * (1.0 + friction)

        # Smooth momentum decay (Inertia)
        self.state.rotational_momentum = (self.state.rotational_momentum * 0.9) + (current_momentum * 0.1)

        # 5. Update internal state
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
