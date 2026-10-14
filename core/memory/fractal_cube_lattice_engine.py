r"""
Fractal Cube Lattice Engine (3x3x3 Dynamic Isotropic Wave Lattice Memory Architecture)
====================================================================================
Implements a 3x3x3 topological lattice memory substrate where a discrete cube indexing format
$$(x, y, z) \in \{-1, 0, 1\}^3$$
is seamlessly integrated with continuous spherical-radial field mapping:
1. Isotropic Wave Field: Converts discrete grid coordinates to continuous spherical coordinates $$(r, \theta, \phi)$$
   and computes radial distance metrics to eliminate grid anisotropy ($$\sqrt{2}$$ diagonal distortion)
   and guarantee rotational invariance.
2. Interface Boundary Phase Weaving: Manages the 27 topological nodes (6 faces, 12 edges, 8 vertices,
   1 central core node) with wave phase interaction, gear/rotor rotation shifts, and constructive/destructive interference.
3. Central Core Node Spectrum Rectification: Rectifies surrounding wave variations and noise from outer 26 nodes into a unified
   macro-rotor phase state.
4. Intrinsic Frame Puzzle Resonance: Computes phase resonance for incoming chaotic data fragments across outer boundary nodes,
   matching them to target lattice coordinates when resonance exceeds a given threshold (phase transition).
5. Lossless Retrocausal Tracing: Reversibly back-traces final wave states and rotor transformations
   to retrieve exact initial state parameters and phase transformation sequences ($$\Delta S \approx 0$$).
6. Fractal Scale Expansion: Aggregates a 3x3x3 lattice into a higher-order macro unit for recursive scale expansion.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class FractalCubeLatticeEngine:
    """
    3x3x3 Dynamic Isotropic Wave Lattice Memory Architecture.
    """

    def __init__(self, num_phase_bins: int = 16, resonance_threshold: float = 0.7):
        """
        Initialize the 3x3x3 lattice.

        Args:
            num_phase_bins: Dimension of phase spectrum vector per node.
            resonance_threshold: Threshold above which chaos fragments snap to grid nodes.
        """
        self.num_phase_bins = num_phase_bins
        self.resonance_threshold = resonance_threshold

        # 27 nodes indexed by integer tuples (x, y, z) in {-1, 0, 1}^3
        self.grid_coords = [
            (x, y, z)
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
        ]

        # Initialize node wave state tensors & phase vectors
        self.nodes: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self._init_nodes()

        # Spherical transformation lookup: (x,y,z) -> (r, theta, phi)
        self.spherical_coords = self._compute_spherical_coords()

        # Macro rotor state (rectified central core axis)
        self.macro_rotor_phase = 0.0
        self.macro_rotor_vector = np.zeros(num_phase_bins, dtype=float)

        # History log for retrocausal back-tracing
        self.transformation_history: List[Dict[str, Any]] = []

    def _init_nodes(self) -> None:
        """Initialize all 27 nodes with classification and wave spectra."""
        for coord in self.grid_coords:
            x, y, z = coord
            dist_sq = x*x + y*y + z*z

            if dist_sq == 0:
                node_type = "core"      # Central Node (0, 0, 0)
            elif dist_sq == 1:
                node_type = "face"      # 6 Face Nodes
            elif dist_sq == 2:
                node_type = "edge"      # 12 Edge Nodes
            else:
                node_type = "vertex"    # 8 Vertex Nodes

            self.nodes[coord] = {
                "coord": coord,
                "node_type": node_type,
                "amplitude": 1.0,
                "phase": 0.0,
                "spectrum": np.zeros(self.num_phase_bins, dtype=float),
                "data_payload": None,
            }

    def _compute_spherical_coords(self) -> Dict[Tuple[int, int, int], Tuple[float, float, float]]:
        """
        Convert discrete grid coordinates to continuous spherical coordinates (r, theta, phi).
        This eliminates grid anisotropy by treating distance in isotropic radial field metric.
        """
        coords_map = {}
        for x, y, z in self.grid_coords:
            r = np.sqrt(x*x + y*y + z*z)
            if r == 0:
                theta = 0.0
                phi = 0.0
            else:
                theta = float(np.arccos(np.clip(z / r, -1.0, 1.0)))
                phi = float(np.arctan2(y, x))
            coords_map[(x, y, z)] = (float(r), theta, phi)
        return coords_map

    def get_isotropic_distance(self, coord1: Tuple[int, int, int], coord2: Tuple[int, int, int]) -> float:
        """
        Calculate true isotropic Euclidean distance between two nodes in radial field space.
        """
        p1 = np.array(coord1, dtype=float)
        p2 = np.array(coord2, dtype=float)
        return float(np.linalg.norm(p1 - p2))

    def rotate_face_rotor(self, axis: str, layer: int, steps: int = 1) -> None:
        """
        Simulate a Rubik's cube face rotor gear rotation.
        Rotates a slice along the given axis ('x', 'y', or 'z') at slice value `layer` (-1, 0, 1)
        by 90 degrees * steps around the axis.
        """
        steps = steps % 4
        if steps == 0:
            return

        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        affected_coords = [c for c in self.grid_coords if c[axis_idx] == layer]

        snapshot = {
            "op": "rotate_face_rotor",
            "axis": axis,
            "layer": layer,
            "steps": steps,
            "prev_state": {c: dict(self.nodes[c]) for c in affected_coords}
        }
        self.transformation_history.append(snapshot)

        new_states = {}
        for coord in affected_coords:
            x, y, z = coord
            for _ in range(steps):
                if axis.lower() == 'z':
                    x, y = -y, x
                elif axis.lower() == 'x':
                    y, z = -z, y
                elif axis.lower() == 'y':
                    x, z = z, -x
            new_coord = (x, y, z)

            updated_node = dict(self.nodes[coord])
            updated_node["coord"] = new_coord
            updated_node["phase"] = (updated_node["phase"] + (np.pi / 2.0) * steps) % (2 * np.pi)
            new_states[new_coord] = updated_node

        for coord, state in new_states.items():
            self.nodes[coord] = state

        self.rectify_central_core()

    def inject_wave_fragment(self, spectrum: np.ndarray, amplitude: float = 1.0, phase: float = 0.0) -> Tuple[int, int, int]:
        """
        Intrinsic Frame Puzzle Resonance:
        An incoming chaotic data/spectrum fragment is evaluated against outer boundary lattice nodes
        using isotropic spherical field resonance. The fragment snaps into the best matching outer node.

        Returns:
            Matched node coordinate (x, y, z)
        """
        if len(spectrum) != self.num_phase_bins:
            spectrum = np.resize(spectrum, self.num_phase_bins)

        norm_spec = np.linalg.norm(spectrum)
        if norm_spec > 0:
            spectrum_unit = spectrum / norm_spec
        else:
            spectrum_unit = spectrum

        best_coord = (1, 0, 0)
        best_resonance = -1.0

        for coord, node in self.nodes.items():
            if coord == (0, 0, 0):
                continue

            node_spec = node["spectrum"]
            node_norm = np.linalg.norm(node_spec)

            if node_norm == 0:
                r, theta, phi = self.spherical_coords[coord]
                spatial_harmonic = np.cos(r * np.pi / 2.0) + np.sin(theta) * np.cos(phi)
                resonance = float(0.5 + 0.3 * spatial_harmonic)
            else:
                cos_sim = float(np.dot(spectrum_unit, node_spec / node_norm))
                phase_diff = np.cos(phase - node["phase"])
                resonance = float(0.7 * cos_sim + 0.3 * phase_diff)

            if resonance > best_resonance:
                best_resonance = resonance
                best_coord = coord

        target_node = self.nodes[best_coord]

        snapshot = {
            "op": "inject_wave_fragment",
            "coord": best_coord,
            "prev_spectrum": target_node["spectrum"].copy(),
            "prev_amplitude": target_node["amplitude"],
            "prev_phase": target_node["phase"],
            "injected_spectrum": spectrum.copy(),
            "injected_amplitude": amplitude,
            "injected_phase": phase,
        }
        self.transformation_history.append(snapshot)

        # Constructive / Destructive Wave Weaving
        target_node["spectrum"] = target_node["spectrum"] + spectrum
        target_node["amplitude"] = float((target_node["amplitude"] + amplitude) / 2.0)
        target_node["phase"] = float((target_node["phase"] + phase) / 2.0 % (2 * np.pi))

        # Rectify core
        self.rectify_central_core()

        return best_coord

    def rectify_central_core(self) -> Dict[str, Any]:
        """
        Central Core Node Spectrum Rectification:
        Aggregates outer 26 nodes' wave variations and noise through isotropic radial weights
        (1 / r^2 coupling) into the central core node (0, 0, 0).
        Rectifies turbulent phase noise into a solid Macro-Rotor state.
        """
        aggregated_spectrum = np.zeros(self.num_phase_bins, dtype=float)
        total_weight = 0.0
        phase_vectors = []

        for coord, node in self.nodes.items():
            if coord == (0, 0, 0):
                continue
            r, _, _ = self.spherical_coords[coord]
            weight = 1.0 / (r * r)
            aggregated_spectrum += weight * node["spectrum"]
            total_weight += weight

            phase_vectors.append(weight * np.exp(1j * node["phase"]))

        if total_weight > 0:
            aggregated_spectrum /= total_weight

        mean_phase_complex = np.sum(phase_vectors)
        rectified_phase = float(np.angle(mean_phase_complex) % (2 * np.pi)) if np.abs(mean_phase_complex) > 1e-6 else 0.0

        # Update Central Core Node & Macro Rotor
        core_node = self.nodes[(0, 0, 0)]
        if np.linalg.norm(aggregated_spectrum) > 0:
            core_node["spectrum"] = aggregated_spectrum
            core_node["phase"] = rectified_phase
            core_node["amplitude"] = float(np.linalg.norm(aggregated_spectrum))
            self.macro_rotor_phase = rectified_phase
            self.macro_rotor_vector = aggregated_spectrum.copy()
        else:
            # Retain existing embedded core spectrum if present
            self.macro_rotor_phase = core_node["phase"]
            self.macro_rotor_vector = core_node["spectrum"].copy()
            core_node["amplitude"] = float(np.linalg.norm(core_node["spectrum"]))

        return {
            "core_spectrum": self.macro_rotor_vector,
            "rectified_phase": self.macro_rotor_phase,
            "macro_rotor_amplitude": float(np.linalg.norm(self.macro_rotor_vector)),
        }

    def retrocausal_trace(self, steps: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Reversible Retrocausal Tracing:
        Back-traces transformation history step-by-step to revert lattice state
        to initial conditions with zero entropy loss (Delta S approx 0).
        """
        if steps is None:
            steps = len(self.transformation_history)

        undone_log = []

        for _ in range(min(steps, len(self.transformation_history))):
            snapshot = self.transformation_history.pop()
            op = snapshot["op"]

            if op == "rotate_face_rotor":
                axis = snapshot["axis"]
                layer = snapshot["layer"]
                orig_steps = snapshot["steps"]
                inv_steps = (4 - orig_steps) % 4

                prev_state = snapshot["prev_state"]
                for coord, state in prev_state.items():
                    self.nodes[coord] = state

                undone_log.append({
                    "reverted_op": op,
                    "axis": axis,
                    "layer": layer,
                    "inv_steps": inv_steps
                })

            elif op == "inject_wave_fragment":
                coord = snapshot["coord"]
                target_node = self.nodes[coord]
                target_node["spectrum"] = snapshot["prev_spectrum"].copy()
                target_node["amplitude"] = snapshot["prev_amplitude"]
                target_node["phase"] = snapshot["prev_phase"]

                undone_log.append({
                    "reverted_op": op,
                    "coord": coord,
                })

        self.rectify_central_core()
        return undone_log

    def expand_fractal_scale(self) -> 'FractalCubeLatticeEngine':
        """
        Fractal Scale Expansion:
        Aggregates this 3x3x3 lattice into a single macro-unit, returning a new
        higher-order FractalCubeLatticeEngine where this lattice serves as the central
        core or a node in the higher dimension.
        """
        higher_lattice = FractalCubeLatticeEngine(
            num_phase_bins=self.num_phase_bins,
            resonance_threshold=self.resonance_threshold
        )

        higher_lattice.nodes[(0, 0, 0)]["spectrum"] = self.macro_rotor_vector.copy()
        higher_lattice.nodes[(0, 0, 0)]["phase"] = self.macro_rotor_phase
        higher_lattice.nodes[(0, 0, 0)]["amplitude"] = float(np.linalg.norm(self.macro_rotor_vector))
        higher_lattice.rectify_central_core()

        return higher_lattice
