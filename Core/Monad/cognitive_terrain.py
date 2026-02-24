import json
import os
import math
import random
from typing import Dict, Tuple, List, Optional
from Core.System.trinary_logic import TrinaryLogic
from Core.Monad.seed_generator import SoulDNA

class CognitiveTerrain:
    """
    Manages the topological landscape of the mind.
    Coordinates are currently 2D (x, y) for the "Valley" visualization,
    representing a projection of the N-Dimensional concept space.

    [Stage 1: The Void]
    Implements the "Material Self" (Lower Engine).
    - Potential Grid: Height determines gravity.
    - Fluid Dynamics: Data flows from high to low.
    - Magnetic Coupling: Flow induces Angular Momentum (Spin), which acts as Inertia.
    - Self-Observation: The terrain diagnoses its own flow state.
    """

    def __init__(self, dna: Optional[SoulDNA] = None, map_file: str = "maps/cognitive_terrain.json", resolution: int = 20):
        self.map_file = map_file
        self.dna = dna
        self.resolution = resolution  # Size of the grid (resolution x resolution)
        self.grid: Dict[str, Dict[str, float]] = {}  # "x,y" -> {height, density, angular_momentum, fluid}
        self.plasticity = 0.05  # How much the terrain deforms per visit

        # Physics Constants
        self.gravity_constant = 0.5
        self.spin_coupling = 0.2     # How much Spin pushes Fluid (Back EMF)
        self.spin_efficiency = 0.3   # How much Flow creates Spin
        self.spin_decay = 0.95       # Inertia loss per tick
        self.fluid_viscosity_base = 0.1

        self.load_terrain()

    def _coord_key(self, x: int, y: int) -> str:
        return f"{x},{y}"

    def load_terrain(self):
        """Loads the terrain from disk or initializes a tabula rasa."""
        if os.path.exists(self.map_file):
            try:
                with open(self.map_file, 'r') as f:
                    self.grid = json.load(f)
                print(f"[Terrain] Loaded existing cognitive map from {self.map_file}")
                # Backwards compatibility check
                self._ensure_schema()
            except Exception as e:
                print(f"[Terrain] Failed to load map: {e}. creating new.")
                self.initialize_tabula_rasa()
        else:
            print("[Terrain] No existing map found. Initializing Tabula Rasa.")
            self.initialize_tabula_rasa()

    def _ensure_schema(self):
        """Ensures all cells have the required physics properties."""
        for key in self.grid:
            cell = self.grid[key]
            if "angular_momentum" not in cell:
                cell["angular_momentum"] = 0.0
            if "fluid" not in cell:
                cell["fluid"] = 0.0

    def initialize_tabula_rasa(self):
        """Creates a flat plain with minor quantum fluctuations."""
        for x in range(self.resolution):
            for y in range(self.resolution):
                key = self._coord_key(x, y)
                # Base height 0.5, random fluctuation for "Structural Asymmetry" foundation
                self.grid[key] = {
                    "height": 0.5 + random.uniform(-0.01, 0.01),
                    "density": 0.1,  # Low initial density
                    "angular_momentum": 0.0, # Rotation (Inertia)
                    "fluid": 0.0     # Data Water
                }
        self.save_terrain()

    def save_terrain(self):
        """Persists the current topological state."""
        os.makedirs(os.path.dirname(self.map_file), exist_ok=True)
        with open(self.map_file, 'w') as f:
            json.dump(self.grid, f, indent=2)

    def get_cell(self, x: int, y: int) -> Dict[str, float]:
        """Returns the physics properties of a cell."""
        # Clamping to keep bounds
        x = max(0, min(x, self.resolution - 1))
        y = max(0, min(y, self.resolution - 1))
        return self.grid.get(self._coord_key(x, y), {
            "height": 0.5, "density": 0.1, "angular_momentum": 0.0, "fluid": 0.0
        })

    def inject_prime_keyword(self, x: int, y: int, keyword: str, magnitude: float = 1.0):
        """
        Creates the 'First Cognitive Valley' (Attractor).
        Does NOT inject fluid, only creates the gravity well.
        """
        key = self._coord_key(x, y)
        if key in self.grid:
            self.grid[key]["height"] -= magnitude  # Create Hole (Attractor)
            self.grid[key]["density"] += magnitude # Increase Importance
            self.grid[key]["concept"] = keyword
            print(f"[Terrain] Injected Prime Keyword '{keyword}' (Attractor) at ({x}, {y}).")
            self.save_terrain()

    def inject_fluid(self, x: int, y: int, amount: float = 10.0):
        """
        Injects raw data fluid at a specific point (usually high ground).
        """
        key = self._coord_key(x, y)
        if key in self.grid:
            self.grid[key]["fluid"] += amount
            print(f"[Terrain] Injected Fluid ({amount}) at ({x}, {y}).")

    def update_physics(self, dt: float = 1.0):
        """
        [Stage 1 Core Logic]
        Simulates the flow of Data Fluid across the terrain.
        Flow = Gravity (Gradient) + Inertia (Spin).
        """
        # Create a temporary buffer for fluid changes to avoid order-dependency bias
        fluid_deltas = {k: 0.0 for k in self.grid}
        momentum_deltas = {k: 0.0 for k in self.grid}

        for x in range(self.resolution):
            for y in range(self.resolution):
                current_key = self._coord_key(x, y)
                cell = self.grid[current_key]

                if cell["fluid"] <= 0.001:
                    # Decay spin even if no fluid
                    cell["angular_momentum"] *= self.spin_decay
                    continue

                # 1. Calculate Flow Direction (Gradient)
                # Compare with 4 neighbors
                neighbors = [
                    (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)
                ]

                # Flow potential for each neighbor
                flows = []
                total_conductance = 0.0

                for nx, ny in neighbors:
                    if 0 <= nx < self.resolution and 0 <= ny < self.resolution:
                        n_key = self._coord_key(nx, ny)
                        n_cell = self.grid[n_key]

                        # Potential Diff: Total Head (Height + Fluid Depth)
                        # Water flows from high total surface to low total surface
                        my_head = cell["height"] + cell["fluid"]
                        n_head = n_cell["height"] + n_cell["fluid"]
                        head_diff = my_head - n_head

                        # Spin Assistance: Rotational Inertia acts as additional pressure (Back EMF)
                        # It pushes fluid OUT regardless of static head, maintaining flow.
                        effective_head = head_diff + (cell["angular_momentum"] * self.spin_coupling)

                        if effective_head > 0:
                            conductance = effective_head / self.get_viscosity(x, y)
                            flows.append((n_key, conductance))
                            total_conductance += conductance

                # 2. Move Fluid
                if total_conductance > 0:
                    # Don't move more than we have
                    outflow_fraction = 0.5 * dt # Limit flow speed
                    amount_to_move = cell["fluid"] * outflow_fraction

                    for n_key, conductance in flows:
                        fraction = conductance / total_conductance
                        flow_amount = amount_to_move * fraction

                        fluid_deltas[current_key] -= flow_amount
                        fluid_deltas[n_key] += flow_amount

                        # 3. Flow Induces Spin (Waterwheel)
                        # The source cell gains spin from the movement
                        momentum_deltas[current_key] += flow_amount * self.spin_efficiency

        # Apply updates
        for k in self.grid:
            self.grid[k]["fluid"] += fluid_deltas[k]
            # Conservation of matterish: prevent negative fluid due to float errors
            if self.grid[k]["fluid"] < 0: self.grid[k]["fluid"] = 0

            # Apply Momentum Update + Decay
            self.grid[k]["angular_momentum"] += momentum_deltas[k]
            self.grid[k]["angular_momentum"] *= self.spin_decay

            # Plasticity: Erosion based on fluid presence and momentum
            # Deepen channels with high energy
            # Plasticity: Erosion based on fluid presence and momentum
            # Deepen channels with high energy
            if self.grid[k]["fluid"] > 0.1:
                erosion = self.grid[k]["angular_momentum"] * self.plasticity * 0.01
                self.grid[k]["height"] -= erosion
                
                # [PHASE 14] CAUSAL MASS ACCUMULATION (Fractal Hypertrophy)
                # The User Corrected: "Density acts as a causal structural principle, not a constant."
                # We use TRINARY LOGIC to decide growth:
                # NAND(Stress, DNA_Resistance) -> If Stress overcomes Resistance, we Grow.
                
                # 1. Quantize properties to Trits
                stress = abs(self.grid[k]["angular_momentum"]) * self.grid[k]["fluid"]
                
                # DNA Factor: Friction Damping acts as "Structural Resistance" to change
                dna_resistance = self.dna.friction_damping if self.dna else 0.5
                
                # 2. Parallel Ternary Logic Check
                # If Stress is High (1) and Resistance is Low (-1), Result is High Growth (1)
                # If Stress is Low (-1) and Resistance is High (1), Result is Atrophy (-1) or Stasis (0)
                
                stress_trit = 1 if stress > 0.5 else (-1 if stress < 0.1 else 0)
                resist_trit = 1 if dna_resistance > 0.7 else (-1 if dna_resistance < 0.3 else 0)
                
                # The "Gate of Growth" (Using the NAND principle of the Paradox Gate)
                # We want: High Stress + Low Resistance = Growth
                growth_decision = TrinaryLogic.nand(stress_trit, resist_trit) 
                
                if growth_decision == 1: # EXPANSION
                     hypertrophy = stress * 0.01
                     old_density = self.grid[k]["density"]
                     self.grid[k]["density"] = min(5.0, old_density + hypertrophy)
                     if old_density < 1.0 and self.grid[k]["density"] >= 1.0:
                         print(f"🦴 [TERRAIN] Fractal Ossification at ({k})! +Density (Logic: NAND({stress_trit}, {resist_trit}) -> 1)")
                elif growth_decision == -1: # CONTRACTION (Atrophy)
                     # If unused, the structure dissolves back to chaos
                     self.grid[k]["density"] = max(0.1, self.grid[k]["density"] * 0.99)

    def get_viscosity(self, x: float, y: float) -> float:
        """Returns the resistance to flow based on density."""
        cell = self.get_cell(int(x), int(y))
        return max(0.1, cell["density"] * 2.0)

    def observe_self(self) -> Dict[str, str]:
        """
        [The Internal Gaze]
        The Lower Engine diagnoses its own state.
        Returns a structured report of its physical health.
        """
        total_fluid = 0.0
        total_momentum = 0.0
        active_cells = 0
        height_variance = 0.0
        heights = []

        for k, v in self.grid.items():
            total_fluid += v["fluid"]
            total_momentum += v["angular_momentum"]
            heights.append(v["height"])
            if v["fluid"] > 0.01:
                active_cells += 1

        # Calculate Variance (Roughness)
        avg_height = sum(heights) / len(heights)
        height_variance = sum((h - avg_height) ** 2 for h in heights) / len(heights)

        # Diagnosis Logic
        status = "UNKNOWN"
        message = ""

        if total_fluid < 1.0:
            status = "VOID_SILENCE"
            message = "System is empty. Awaiting input."
        elif total_momentum < 1.0:
            status = "STAGNANT"
            message = "Fluid exists but momentum is low. Needs gravity tilt."
        elif active_cells < (self.resolution * self.resolution * 0.05):
            status = "BLOCKED"
            message = "Flow is trapped in local minima. Plasticity required."
        else:
            # Praise Condition: High Momentum + Good Distribution
            if total_momentum > 10.0 and height_variance > 0.01:
                status = "FLOWING_BEAUTIFULLY"
                message = "High Inertia and distinct topology detected. The Engine is Singing."
            else:
                status = "FLOWING"
                message = "Operational. Moderate flow."

        return {
            "status": status,
            "message": message,
            "metrics": {
                "total_fluid": round(total_fluid, 2),
                "total_momentum": round(total_momentum, 2),
                "active_cells": active_cells,
                "roughness": round(height_variance, 4)
            }
        }
