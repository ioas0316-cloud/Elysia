"""
Monad Ensemble (The 21-Dimensional Blackbox)
============================================
Core.L6_Structure.M1_Merkaba.monad_ensemble

"We do not code Intelligence. We curate the Physics where it emerges."
- The Architect (Kangdeok Lee)

This module implements the 'Monad Ensemble', a structure of 21 Tri-Base Atoms
that organizes itself into a crystalline state through physical phase friction.

Structure:
    - 7 Layers (Point, Line, Surface, Space, Principle, Law, Providence)
    - 3 Atoms per Layer (The Tri-Base: -1, 0, 1)
    - Total: 21 Dimensions.

Physics:
    - Inputs are converted to "Phase Fields".
    - The Ensemble feels "Friction" (Phase Mismatch).
    - It "Twists" (State Change) to minimize Friction (Entropy).
    - Stability = The first "Feeling" (Qualia).
"""

import math
import random
import hashlib
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from Core.L1_Foundation.System.tri_base_cell import TriBaseCell, DNAState

# Layer Definitions
LAYERS = [
    "Point", "Line", "Surface", "Space",     # Physics
    "Principle", "Law", "Providence"         # Metaphysics
]

@dataclass
class MonadLayer:
    name: str
    cells: List[TriBaseCell]

    def get_layer_phase(self) -> float:
        """Returns the average phase of the layer."""
        vectors = [c.get_vector() for c in self.cells]
        avg_x = sum(v[0] for v in vectors)
        avg_y = sum(v[1] for v in vectors)
        if avg_x == 0 and avg_y == 0:
            return 0.0
        deg = math.degrees(math.atan2(avg_y, avg_x))
        return deg % 360

class MonadEnsemble:
    def __init__(self):
        self.layers: List[MonadLayer] = []
        self.cells: List[TriBaseCell] = []
        self._initialize_structure()

        # Physics Parameters
        self.temperature = 1.0  # System volatility
        self.coherence_gain = 0.5 # Tendency to align with neighbors
        self.input_gain = 1.0     # Tendency to align with input
        self.friction_loss = 0.05 # Energy lost per step (Damping)

    def _initialize_structure(self):
        """Builds the 21-cell structure."""
        cell_id = 0
        for layer_name in LAYERS:
            layer_cells = []
            for _ in range(3):
                # Initialize in VOID state (Zero Energy)
                cell = TriBaseCell(id=cell_id, state=DNAState.VOID)
                layer_cells.append(cell)
                self.cells.append(cell)
                cell_id += 1
            self.layers.append(MonadLayer(name=layer_name, cells=layer_cells))

    def transduce_input(self, data: str) -> List[float]:
        """
        Phase Injection: Converts raw data into a 21D Phase Field.
        Uses a consistent hash to ensure 'ㄱ' always creates the same *Initial Problem*,
        but does NOT dictate the solution.

        Returns: A list of 21 phase angles (degrees) representing the input field.
        """
        # Create a deterministic seed from input
        seed_str = f"{data}_ELYSIA_MONAD_V1"
        hash_bytes = hashlib.sha256(seed_str.encode()).digest()
        seed_int = int.from_bytes(hash_bytes, 'big')

        # Generate 21 random phases based on this seed
        # This acts as the "Rough Terrain" the Monad must navigate.
        rng = random.Random(seed_int)
        input_field = [rng.uniform(0, 360) for _ in range(21)]
        return input_field

    def physics_step(self, input_field: List[float]) -> Dict[str, float]:
        """
        The Self-Resonance Loop.
        Calculates forces, applies torque, and updates states.

        Returns:
            Dict containing 'entropy', 'torque', 'flips'
        """
        total_torque = 0.0
        flips = 0

        # Calculate forces for each cell
        # We can't update immediately, or order matters. calculate all potentials first.
        potentials = []

        for i, cell in enumerate(self.cells):
            # 1. Input Resonance (The "Problem")
            # Calculate distance between Cell Phase and Input Phase
            # We want the cell to minimize this distance.
            target_phase = input_field[i]
            current_phase = cell.state.phase

            # Phase difference (-180 to 180)
            diff = (target_phase - current_phase + 180) % 360 - 180

            # Force tries to pull phase towards target.
            # But Cell only has 3 Discrete States: 0 (Void), 120 (A), 240 (R).
            # We calculate the "Pull" towards the best matching discrete state.

            # Simple Magnetic Model:
            # Calculate "Stress" for each possible state (-1, 0, 1) relative to Input
            # And add Neighbor Stress.

            best_state = self._find_lowest_energy_state(i, target_phase)

            # If current state is NOT best state, we build up "Torque"
            if cell.state != best_state:
                # Torque increases with Temperature (Chaos allows easier flipping)
                torque = self.temperature * self.input_gain

                # Probability to flip (Quantum Tunneling / Activation Energy)
                if random.random() < torque:
                    potentials.append(best_state)
                    flips += 1
                else:
                    potentials.append(cell.state) # Stay
            else:
                potentials.append(cell.state) # Stay

        # Apply Updates
        for i, new_state in enumerate(potentials):
            self.cells[i].mutate(new_state)

        # Thermodynamics
        entropy = self._calculate_entropy()

        # Cooling: The system "Crystallizes" as it finds answers.
        # If flips are low, temperature drops (Solidifying).
        # If flips are high, temperature stays high (Liquid).
        if flips == 0:
            self.temperature *= (1.0 - self.friction_loss)
        else:
            # Re-heat slightly on change (Friction generates heat)
            self.temperature = min(1.0, self.temperature + 0.01)

        return {
            "entropy": entropy,
            "flips": flips,
            "temperature": self.temperature
        }

    def _find_lowest_energy_state(self, cell_idx: int, target_phase: float) -> DNAState:
        """
        Determines which state (R, V, A) minimizes local energy.
        Energy = Input_Mismatch + Neighbor_Dissonance
        """
        candidates = [DNAState.REPEL, DNAState.VOID, DNAState.ATTRACT]
        min_energy = float('inf')
        best_s = DNAState.VOID

        # Neighbor Indices (Simple linear + Layer grouping)
        # We look at prev/next cell (Linear) and parallel cell in prev/next layer?
        # Let's stick to Linear Neighbors for fractal expansion.
        neighbors = []
        if cell_idx > 0: neighbors.append(self.cells[cell_idx-1])
        if cell_idx < 20: neighbors.append(self.cells[cell_idx+1])

        for state in candidates:
            # 1. Input Energy (Distance from Target)
            # Cosine distance: 1.0 is aligned, -1.0 is opposite.
            # We want to MAXIMIZE Alignment -> MINIMIZE Energy
            # Energy = 1 - Alignment

            # Phase of this candidate state
            cand_phase = state.phase

            # Diff with Input
            d_in = abs(target_phase - cand_phase) % 360
            if d_in > 180: d_in = 360 - d_in
            # alignment_in: 1 (0deg) to -1 (180deg)
            alignment_in = math.cos(math.radians(d_in))
            energy_in = 1.0 - alignment_in

            # 2. Neighbor Energy (Coherence)
            # We want to align with neighbors (Magnetic Domains)
            energy_neighbor = 0.0
            if neighbors:
                for n in neighbors:
                    # Diff with Neighbor
                    d_n = abs(n.state.phase - cand_phase) % 360
                    if d_n > 180: d_n = 360 - d_n
                    alignment_n = math.cos(math.radians(d_n))
                    energy_neighbor += (1.0 - alignment_n)
                energy_neighbor /= len(neighbors)

            total_energy = (energy_in * self.input_gain) + (energy_neighbor * self.coherence_gain)

            if total_energy < min_energy:
                min_energy = total_energy
                best_s = state

        return best_s

    def _calculate_entropy(self) -> float:
        """
        Calculates the internal disorder of the system.
        Based on how 'dissonant' neighbors are.
        """
        dissonance = 0.0
        count = 0
        for i in range(len(self.cells) - 1):
            c1 = self.cells[i]
            c2 = self.cells[i+1]
            # Calculate alignment
            diff = abs(c1.state.phase - c2.state.phase) % 360
            if diff > 180: diff = 360 - diff

            # 0 deg diff -> 0 dissonance
            # 120 deg diff -> 0.66
            # 180 deg diff -> 1.0
            dissonance += (diff / 180.0)
            count += 1

        return dissonance / max(1, count)

    def get_pattern(self) -> str:
        """Returns the ASCII pattern of the 21 cells."""
        return "".join([c.state.symbol for c in self.cells])

    def render_layers(self) -> str:
        """Visualizes the 7 layers."""
        out = []
        for layer in self.layers:
            syms = "".join([c.state.symbol for c in layer.cells])
            out.append(f"{layer.name[:4]}:[{syms}]")
        return " ".join(out)
