"""
Generative Phase Molding Engine
================================
Real-time generative physical/structural emergence engine driven by interface boundary
continuous phase interference from the 3x3x3 Isotropic Wave Lattice Memory.

Key capabilities:
1. Isotropic Field to Physical Density Mapping: Converts phase spectra and resonance across lattice nodes
   into 3D continuous density field and volumetric stress topology.
2. Structural Simplex Rigidity Calculation: Evaluates topological rigidity and structural stability
   using triangular simplex constraints derived from wave interference patterns.
3. Generative Molding & Shape Emergence: Generates 3D structural meshes and optimal geometric forms
   (Generative Molding / Morphing) aligned with the lattice macro-rotor reference frame.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class GenerativeMoldingEngine:
    """
    Generative Molding Engine that transforms isotropic wave resonance fields
    into physical/structural material density, stress topology, and emergent 3D geometry.
    """

    def __init__(self, resolution: int = 16):
        """
        Initialize the molding engine.

        Args:
            resolution: Grid resolution along each axis for continuous volumetric rendering/density field.
        """
        self.resolution = resolution
        self.voxel_grid = np.zeros((resolution, resolution, resolution), dtype=float)
        self.stress_tensor_field = np.zeros((resolution, resolution, resolution, 3), dtype=float)

    def mold_structure_from_lattice(self, lattice_nodes: Dict[Tuple[int, int, int], Dict[str, Any]], macro_rotor_phase: float = 0.0) -> np.ndarray:
        """
        Map 3x3x3 wave lattice state onto high-resolution 3D volumetric material density field.

        Args:
            lattice_nodes: Node dictionary from FractalCubeLatticeEngine.
            macro_rotor_phase: Current phase alignment of the macro rotor axis.

        Returns:
            3D numpy array of shape (resolution, resolution, resolution) representing material density [0, 1].
        """
        self.voxel_grid.fill(0.0)

        # Grid range mapping [-1, 1] to index [0, resolution-1]
        lin_space = np.linspace(-1.0, 1.0, self.resolution)
        grid_x, grid_y, grid_z = np.meshgrid(lin_space, lin_space, lin_space, indexing='ij')

        for coord, node in lattice_nodes.items():
            nx, ny, nz = coord
            amplitude = node.get("amplitude", 1.0)
            phase = node.get("phase", 0.0)
            spectrum = node.get("spectrum", np.array([1.0]))
            energy = float(np.linalg.norm(spectrum) * amplitude)

            if energy <= 1e-6:
                continue

            # Distance in normalized 3D spatial field from voxel points to node coordinate
            dx = grid_x - nx
            dy = grid_y - ny
            dz = grid_z - nz
            dist = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6

            # Isotropic Gaussian-radial basis wave interference
            wave_field = energy * np.exp(-1.5 * dist**2) * np.cos(dist * np.pi - phase + macro_rotor_phase)
            self.voxel_grid += wave_field

        # Normalize density field to [0, 1] via sigmoid-like phase activation
        self.voxel_grid = 1.0 / (1.0 + np.exp(-self.voxel_grid))
        return self.voxel_grid

    def compute_stress_topology(self) -> np.ndarray:
        """
        Compute volumetric stress gradient topology (3D vector field) across the molded density field.
        Represents principal forces and load distribution along simplex rigidity paths.

        Returns:
            3D gradient vector field of shape (resolution, resolution, resolution, 3)
        """
        # Calculate numerical spatial gradients along x, y, z
        grad_x, grad_y, grad_z = np.gradient(self.voxel_grid)

        self.stress_tensor_field[..., 0] = grad_x
        self.stress_tensor_field[..., 1] = grad_y
        self.stress_tensor_field[..., 2] = grad_z

        return self.stress_tensor_field

    def generate_emergent_mesh(self, density_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Extract emergent 3D structural topology (nodes and tetrahedral/triangular simplex connections)
        where material density exceeds the threshold.

        Returns:
            Dict with 'vertices' (List of 3D coordinates), 'simplex_edges' (List of index pairs),
            'total_structural_mass', and 'rigidity_score'.
        """
        indices = np.argwhere(self.voxel_grid >= density_threshold)
        if len(indices) == 0:
            return {
                "vertices": [],
                "simplex_edges": [],
                "total_structural_mass": 0.0,
                "rigidity_score": 0.0
            }

        # Convert index back to normalized [-1, 1] world coordinates
        lin_space = np.linspace(-1.0, 1.0, self.resolution)
        vertices = [
            (float(lin_space[ix]), float(lin_space[iy]), float(lin_space[iz]))
            for ix, iy, iz in indices
        ]

        # Calculate structural connections between nearby high-density nodes (Simplex Triangulation)
        simplex_edges = []
        num_vertices = len(vertices)

        # Subsample if vertex count is large to maintain efficiency
        max_vertices_to_check = min(num_vertices, 100)
        check_indices = np.random.choice(num_vertices, max_vertices_to_check, replace=False) if num_vertices > 100 else list(range(num_vertices))

        for i_idx, i in enumerate(check_indices):
            v1 = np.array(vertices[i])
            for j in check_indices[i_idx + 1:]:
                v2 = np.array(vertices[j])
                dist = np.linalg.norm(v1 - v2)
                # Form simplex connection if within spatial resonance radius
                if 0.1 <= dist <= 0.5:
                    simplex_edges.append((i, j))

        total_structural_mass = float(np.sum(self.voxel_grid[self.voxel_grid >= density_threshold]))
        rigidity_score = float(len(simplex_edges) / (len(vertices) + 1e-5))

        return {
            "vertices": vertices,
            "simplex_edges": simplex_edges,
            "total_structural_mass": total_structural_mass,
            "rigidity_score": rigidity_score
        }
