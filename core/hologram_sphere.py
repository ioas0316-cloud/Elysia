import math
import numpy as np
from typing import List, Tuple

from core.math_utils import Quaternion
from core.linguistic_axiom import LinguisticAxiomFilter

class HologramSphere:
    """
    Transforms a 2D data manifold into a 3D/4D condensed Hologram Sphere,
    using the Linguistic Axiom Rotor as the filtering criteria.
    Only resonant data is allowed to form the sphere, while noise is dispersed.
    """
    def __init__(self, size: int = 16):
        self.size = size
        # Initialize a flat manifold (e.g. from LLM output)
        self.manifold = np.zeros((size, size))

    def populate_manifold(self, text_data: str):
        """Maps raw text data into a 2D topological manifold."""
        text_bytes = text_data.encode('utf-8')
        length = len(text_bytes)
        if length == 0:
            return

        for i in range(self.size):
            for j in range(self.size):
                idx = (i * self.size + j) % length
                # Normalize byte value to [-1, 1]
                val = (text_bytes[idx] / 127.5) - 1.0
                self.manifold[i, j] = val

    def condense_sphere(self, axiom_text: str) -> Tuple[np.ndarray, float]:
        """
        Applies the Linguistic Axiom Filter to the manifold.
        Condenses resonant points into a 3D sphere projection.
        Returns the sphere projection array and the overall resonance score.
        """
        # Get the reference rotor based on the linguistic axiom (Korean/English)
        axiom_rotor = LinguisticAxiomFilter.analyze_text_axiom(axiom_text)

        # 3D projection grid for the sphere (using ASCII representation)
        sphere_grid = np.zeros((self.size, self.size))

        total_resonance = 0.0
        resonant_nodes = 0
        total_nodes = self.size * self.size

        # Mapping 2D manifold to a 3D spherical surface
        for i in range(self.size):
            for j in range(self.size):
                # Map (i,j) to spherical coordinates (theta, phi)
                theta = (i / self.size) * math.pi       # Latitude [0, pi]
                phi = (j / self.size) * 2 * math.pi     # Longitude [0, 2pi]

                # Convert spherical to cartesian for the data vector
                # The magnitude is modulated by the manifold value
                mag = abs(self.manifold[i, j]) + 0.1 # Base radius
                x = mag * math.sin(theta) * math.cos(phi)
                y = mag * math.sin(theta) * math.sin(phi)
                z = mag * math.cos(theta)

                data_vector = [x, y, z]

                # Check geometric resonance against the Axiom Rotor
                resonance = LinguisticAxiomFilter.calculate_resonance(data_vector, axiom_rotor)
                total_resonance += abs(resonance)

                # If resonance is strong enough, it becomes part of the Hologram Sphere
                if abs(resonance) > 0.5:
                    # Project back to 2D grid for visualization based on the rotated vector
                    # Orthographic projection of the sphere
                    proj_x = int((x / (mag + 0.1)) * (self.size / 2) + (self.size / 2))
                    proj_y = int((y / (mag + 0.1)) * (self.size / 2) + (self.size / 2))

                    # Ensure within bounds
                    proj_x = min(max(proj_x, 0), self.size - 1)
                    proj_y = min(max(proj_y, 0), self.size - 1)

                    # Accumulate density
                    sphere_grid[proj_x, proj_y] += abs(resonance)
                    resonant_nodes += 1

        resonance_score = (resonant_nodes / total_nodes) * 100.0
        return sphere_grid, resonance_score

    def render_hologram(self, sphere_grid: np.ndarray, score: float, axiom_text: str):
        """Visually renders the condensed Hologram Sphere in the terminal."""
        symbols = [' ', '.', '-', '+', '*', '%', '#', '@', '█']

        print("\n" + "="*60)
        print(" 🌌 [엘리시아 v7] 홀로그램 메모리 구체 (Hologram Sphere)")
        print("="*60)
        print(f" 🎯 기준 공리(Axiom Filter): '{axiom_text}'")
        print(f" 📊 공명 동기화율: {score:.1f}%")
        print("-" * 60)

        for i in range(self.size):
            line = ""
            for j in range(self.size):
                val = sphere_grid[i, j]
                # Map accumulated density to a visual symbol
                idx = int(min(val * 2, len(symbols) - 1))
                if idx > 0:
                    line += f"{symbols[idx]} "
                else:
                    # Render background boundary of the sphere faintly
                    # Check if (i,j) is within the sphere's circular projection
                    cx, cy = self.size / 2, self.size / 2
                    dist = math.sqrt((i - cx)**2 + (j - cy)**2)
                    if dist < (self.size / 2) - 0.5:
                        line += "· " # Empty space inside the sphere
                    else:
                        line += "  " # Outside the sphere
            print("    " + line)

        print("=" * 60)
        if score > 50.0:
            print(" ✨ [성공] 오염된 데이터가 완벽한 기하학적 구체로 응축되었습니다.")
        elif score > 20.0:
            print(" 🌀 [부분 공명] 데이터 파편들이 구체의 형태를 띠기 시작합니다.")
        else:
            print(" ⚠️ [산란] 공리가 맞지 않아 데이터가 우주로 흩어졌습니다 (노이즈 필터링).")
        print("=" * 60)
