import json
import os
import math
import random
from typing import Dict, Tuple, List, Optional

class CognitiveTerrain:
    """
    Manages the topological landscape of the mind.
    Coordinates are currently 2D (x, y) for the "Valley" visualization,
    representing a projection of the N-Dimensional concept space.
    """

    def __init__(self, map_file: str = "maps/cognitive_terrain.json", resolution: int = 20):
        self.map_file = map_file
        self.resolution = resolution  # Size of the grid (resolution x resolution)
        self.grid: Dict[str, Dict[str, float]] = {}  # "x,y" -> {"height": h, "density": d}
        self.plasticity = 0.05  # How much the terrain deforms per visit
        self.prime_tilt_bias = 0.01  # Constant bias towards expansion

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
            except Exception as e:
                print(f"[Terrain] Failed to load map: {e}. creating new.")
                self.initialize_tabula_rasa()
        else:
            print("[Terrain] No existing map found. Initializing Tabula Rasa.")
            self.initialize_tabula_rasa()

    def initialize_tabula_rasa(self):
        """Creates a flat plain with minor quantum fluctuations."""
        for x in range(self.resolution):
            for y in range(self.resolution):
                key = self._coord_key(x, y)
                # Base height 0.5, random fluctuation for "Structural Asymmetry" foundation
                self.grid[key] = {
                    "height": 0.5 + random.uniform(-0.01, 0.01),
                    "density": 0.1  # Low initial density
                }
        self.save_terrain()

    def save_terrain(self):
        """Persists the current topological state."""
        os.makedirs(os.path.dirname(self.map_file), exist_ok=True)
        with open(self.map_file, 'w') as f:
            json.dump(self.grid, f, indent=2)

    def get_cell(self, x: int, y: int) -> Dict[str, float]:
        """Returns the physics properties of a cell."""
        # Wrap-around logic (Toroidal topology) or Clamping?
        # For a "Valley", clamping is safer to keep bounds.
        x = max(0, min(x, self.resolution - 1))
        y = max(0, min(y, self.resolution - 1))
        return self.grid.get(self._coord_key(x, y), {"height": 0.5, "density": 0.1})

    def inject_prime_keyword(self, x: int, y: int, keyword: str, magnitude: float = 1.0):
        """
        Creates the 'First Cognitive Valley' by drastically lowering height (Attractor)
        and increasing density (Gravitas) at a specific point.
        """
        key = self._coord_key(x, y)
        if key in self.grid:
            # Lower height = Attractor (Gravity Well)
            self.grid[key]["height"] -= magnitude
            # Increase density = Viscosity/Importance
            self.grid[key]["density"] += magnitude
            # Add metadata for debugging
            self.grid[key]["concept"] = keyword
            print(f"[Terrain] Injected Prime Keyword '{keyword}' at ({x}, {y}). Height: {self.grid[key]['height']:.2f}")
            self.save_terrain()

    def apply_erosion(self, x: int, y: int, flow_intensity: float):
        """
        Plasticity: The flow carves the path.
        Visiting a cell lowers its height (making it more attractive for future flow)
        and slightly increases density (memory formation).
        """
        key = self._coord_key(int(x), int(y))
        if key in self.grid:
            # Erosion: Depth increases (Height decreases)
            self.grid[key]["height"] -= (self.plasticity * flow_intensity)
            # Deposition: Density increases
            self.grid[key]["density"] += (self.plasticity * flow_intensity * 0.1)

    def get_gradient(self, x: float, y: float) -> Tuple[float, float]:
        """
        Calculates the slope (gradient) at a given point.
        Returns a vector (dx, dy) pointing 'downhill'.
        """
        ix, iy = int(x), int(y)

        # Sobel-like operator or simple difference
        # Look at neighbors
        h_center = self.get_cell(ix, iy)["height"]

        # Simple finite difference
        h_left = self.get_cell(ix - 1, iy)["height"]
        h_right = self.get_cell(ix + 1, iy)["height"]
        h_up = self.get_cell(ix, iy - 1)["height"]
        h_down = self.get_cell(ix, iy + 1)["height"]

        # Gradient points from High to Low.
        # So if Right is lower than Left, x-slope is positive (flow to right).
        # We want the vector pointing DOWNHILL.
        # slope_x = (h_left - h_right)
        # If left is 10, right is 0, slope is +10 (move right).
        slope_x = (h_left - h_right)
        slope_y = (h_up - h_down)

        return slope_x, slope_y

    def get_viscosity(self, x: float, y: float) -> float:
        """Returns the resistance to flow based on density."""
        cell = self.get_cell(int(x), int(y))
        # Density 0.1 -> Viscosity 1.0 (Standard)
        # Density 1.0 -> Viscosity 0.1 (Slow flow? No, High density = High Viscosity = Slow Flow)
        # Wait, Viscosity RESISTS flow. So High Density -> High Resistance.
        return max(0.1, cell["density"] * 2.0)
