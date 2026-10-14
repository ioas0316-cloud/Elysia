"""
Prismatic Axis Spectrum Engine: First Principles of Continuous Wave Perception & Cognition
========================================================================================

Adheres strictly to the 4 Continuities of the Causal Field & Chromatic Dynamics:
1. Spatiotemporal Wave & Rotational Rotor: Continuous rotational wave field on a fundamental spindle axis.
2. Prismatic Bifurcation (Root -> Branch -> Leaf): Fundamental white light waves refract
   through phase-frequency prisms into child spectral branches (Red, Yellow, Blue, Green),
   embedding structural hierarchy into observation space without statistical labeling.
3. Multi-Layered Wave Memory Stack: Spectral components weave into an interlocking multi-layered
   wave memory stack. Cross-relational interference patterns persist even if specific layers are gated.
4. Phase Resonance Search (Wave Collapse): Information retrieval is executed as constructive
   phase resonance rather than brute-force linear index scanning.
5. Phase-Locked Focus & Noise Suppression: Selective attention boosts target frequency amplitudes
   and phase-locks matching components while destructively interfering noise.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class SpectralNode:
    """Represents a node in the Prismatic Spectrum Hierarchy."""
    node_id: str
    parent_id: Optional[str]
    frequency: float
    phase: float
    amplitude: float
    threshold: float
    chromatic_signature: np.ndarray  # [Red(Flux), Blue(Order), Yellow(Entropy)]
    depth: int


@dataclass
class ResonanceResult:
    """Result of Phase Resonance Search (Wave Collapse)."""
    target_node_id: str
    resonance_score: float
    constructive_energy: float
    destructive_cancellation: float
    collapsed_state: float  # Continuous or discrete projection


class WaveMemoryLayer:
    """A single layer in the Multi-Layered Wave Memory Stack."""

    def __init__(self, layer_id: str, color_channel: str, spatial_dim: int = 64):
        self.layer_id = layer_id
        self.color_channel = color_channel
        self.spatial_dim = spatial_dim
        self.spiral_phase_grid = np.zeros((spatial_dim, spatial_dim), dtype=np.float64)
        self.is_active: bool = True

        # Initialize rotor phase spiral
        y, x = np.ogrid[-1:1:complex(0, spatial_dim), -1:1:complex(0, spatial_dim)]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        self.base_spiral = np.sin(5 * r - theta)

    def write_wave_pattern(self, frequency: float, phase: float, amplitude: float):
        """Imprints a wave pattern onto the memory layer."""
        if not self.is_active:
            return
        pattern = amplitude * np.cos(self.base_spiral * frequency + phase)
        self.spiral_phase_grid += pattern

    def set_active_state(self, active: bool):
        """Gates or enables this memory layer."""
        self.is_active = active


class MultiLayeredWaveMemoryStack:
    """
    Multi-Layered Wave Memory Stack:
    Integrates spectral components across Red (Flux), Blue (Order), and Yellow (Entropy) layers.
    Even if individual layers are deactivated, cross-layer relational interference persists.
    """

    def __init__(self, spatial_dim: int = 64):
        self.spatial_dim = spatial_dim
        self.layers: Dict[str, WaveMemoryLayer] = {
            "red_flux_layer": WaveMemoryLayer("red_flux_layer", "Red", spatial_dim),
            "blue_order_layer": WaveMemoryLayer("blue_order_layer", "Blue", spatial_dim),
            "yellow_entropy_layer": WaveMemoryLayer("yellow_entropy_layer", "Yellow", spatial_dim),
        }

    def imprint_node(self, node: SpectralNode):
        """Imprints a spectral node onto corresponding chromatic wave memory layers."""
        chroma = node.chromatic_signature
        self.layers["red_flux_layer"].write_wave_pattern(node.frequency, node.phase, node.amplitude * chroma[0])
        self.layers["blue_order_layer"].write_wave_pattern(node.frequency, node.phase, node.amplitude * chroma[1])
        self.layers["yellow_entropy_layer"].write_wave_pattern(node.frequency, node.phase, node.amplitude * chroma[2])

    def compute_cross_relational_field(self) -> np.ndarray:
        """
        Computes the interlocked relational interference field across active layers.
        Reconstructs macro relational state even if some layers are gated off.
        """
        composite_field = np.zeros((self.spatial_dim, self.spatial_dim), dtype=np.float64)
        active_count = 0
        for layer in self.layers.values():
            if layer.is_active:
                composite_field += layer.spiral_phase_grid
                active_count += 1
        if active_count > 0:
            composite_field /= active_count
        return composite_field


class PrismaticAxisSpectrumEngine:
    r"""
    [Prismatic Axis Spectrum Engine]

    Models continuous rotational wave fields, threshold projections for discrete emergence,
    prismatic scale/frequency bifurcation, multi-layered wave memory stack,
    phase resonance search, and phase-locked focus.
    """

    def __init__(
        self,
        base_frequency: float = 10.0,
        num_bifurcation_levels: int = 3,
        branches_per_node: int = 2,
        seed: int = 42
    ):
        np.random.seed(seed)
        self.base_frequency = base_frequency
        self.num_levels = num_bifurcation_levels
        self.branches_per_node = branches_per_node
        self.nodes: Dict[str, SpectralNode] = {}
        self.memory_stack = MultiLayeredWaveMemoryStack(spatial_dim=64)
        self.time: float = 0.0

        # Build initial Prismatic Root-Branch-Leaf tree
        self._build_prismatic_tree()
        self._imprint_memory_stack()

    def _build_prismatic_tree(self):
        """Constructs the root-branch prismatic hierarchy from fundamental white light wave."""
        root = SpectralNode(
            node_id="root_white_light",
            parent_id=None,
            frequency=self.base_frequency,
            phase=0.0,
            amplitude=1.0,
            threshold=0.0,
            chromatic_signature=np.array([0.333, 0.333, 0.334], dtype=np.float64),
            depth=0
        )
        self.nodes[root.node_id] = root

        # Recursive bifurcation through Prismatic Refraction
        queue = [root]
        while queue:
            parent = queue.pop(0)
            if parent.depth >= self.num_levels:
                continue

            for b in range(self.branches_per_node):
                child_id = f"{parent.node_id}_b{b+1}"

                # Prismatic phase shift & frequency harmonic dispersion
                child_freq = parent.frequency * (1.5 ** (b + 1))
                child_phase = (parent.phase + (b + 1) * (np.pi / 3.0)) % (2 * np.pi)
                child_amplitude = parent.amplitude * 0.7071  # Energy conservation (1/sqrt(2))

                # Chromatic shift: Flux (Red), Order (Blue), Entropy (Yellow)
                if b % 3 == 0:
                    chromatic = np.array([0.6, 0.3, 0.1])  # Red-dominant
                elif b % 3 == 1:
                    chromatic = np.array([0.1, 0.6, 0.3])  # Blue-dominant
                else:
                    chromatic = np.array([0.3, 0.1, 0.6])  # Yellow-dominant

                child_node = SpectralNode(
                    node_id=child_id,
                    parent_id=parent.node_id,
                    frequency=child_freq,
                    phase=child_phase,
                    amplitude=child_amplitude,
                    threshold=float(np.sin(child_phase) * 0.2),
                    chromatic_signature=chromatic,
                    depth=parent.depth + 1
                )
                self.nodes[child_id] = child_node
                queue.append(child_node)

    def _imprint_memory_stack(self):
        """Imprints all spectral nodes into the multi-layered wave memory stack."""
        for node in self.nodes.values():
            self.memory_stack.imprint_node(node)

    def evaluate_continuous_wave(self, t: float, spatial_x: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the continuous fundamental wave field \Phi(t, x) as a superposition of all spectral modes.
        \Phi(t, x) = \sum_{node} A_n \cdot \cos(2\pi \cdot f_n \cdot t - x \cdot f_n / c + \phi_n)
        """
        phi = np.zeros_like(spatial_x, dtype=np.float64)
        c = 343.0  # Phase propagation speed
        for node in self.nodes.values():
            wave = node.amplitude * np.cos(2 * np.pi * node.frequency * t - (spatial_x * node.frequency / c) + node.phase)
            phi += wave
        return phi

    def project_discrete_states(
        self,
        continuous_wave: np.ndarray,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Projects a continuous wave field into discrete binary 0/1 states via a threshold axis.
        Shows 0 and 1 are boundary projections on a continuous wave axis.
        """
        discrete_binary = (continuous_wave >= threshold).astype(int)
        continuous_margin = continuous_wave - threshold

        # Calculate transition boundary density (where continuous wave crosses threshold)
        zero_crossings = np.where(np.diff(discrete_binary) != 0)[0]

        return {
            "discrete_binary": discrete_binary,
            "continuous_wave": continuous_wave,
            "threshold": threshold,
            "continuous_margin": continuous_margin,
            "zero_crossing_count": int(len(zero_crossings)),
            "zero_crossing_indices": zero_crossings.tolist()
        }

    def determine_prismatic_relationship(self, node_id_a: str, node_id_b: str) -> Dict[str, Any]:
        """
        Directly derives topological and phase relationships between two concepts/nodes
        without statistical labeling or explicit memorization.
        """
        if node_id_a not in self.nodes or node_id_b not in self.nodes:
            raise ValueError("Nodes must exist in the prismatic spectrum.")

        nA = self.nodes[node_id_a]
        nB = self.nodes[node_id_b]

        # Phase difference & Frequency ratio
        phase_diff = abs((nA.phase - nB.phase + np.pi) % (2 * np.pi) - np.pi)
        freq_ratio = max(nA.frequency, nB.frequency) / (min(nA.frequency, nB.frequency) + 1e-9)

        # Common Root Ancestor Search
        ancestors_a = self._get_ancestor_path(node_id_a)
        ancestors_b = self._get_ancestor_path(node_id_b)

        common_root = None
        for a in ancestors_a:
            if a in ancestors_b:
                common_root = a
                break

        # Phase coherence (1 = in phase, 0 = orthogonal / anti-phase)
        phase_coherence = float(np.cos(phase_diff))

        # Chromatic similarity (Dot product of Red/Blue/Yellow signatures)
        chromatic_sim = float(np.dot(nA.chromatic_signature, nB.chromatic_signature) / (
            np.linalg.norm(nA.chromatic_signature) * np.linalg.norm(nB.chromatic_signature) + 1e-9
        ))

        return {
            "node_a": node_id_a,
            "node_b": node_id_b,
            "common_root": common_root,
            "phase_difference_rad": float(phase_diff),
            "phase_coherence": phase_coherence,
            "frequency_ratio": float(freq_ratio),
            "chromatic_similarity": chromatic_sim,
            "structural_affinity": float((phase_coherence + 1.0) / 2.0 * chromatic_sim)
        }

    def _get_ancestor_path(self, node_id: str) -> List[str]:
        path = []
        curr = node_id
        while curr is not None and curr in self.nodes:
            path.append(curr)
            curr = self.nodes[curr].parent_id
        return path

    def execute_phase_resonance_search(
        self,
        query_frequency: float,
        query_phase: float,
        tolerance_freq: float = 5.0
    ) -> List[ResonanceResult]:
        """
        Executes instantaneous Phase Resonance Search (Wave Collapse).
        Constructive interference boosts matching target nodes while mismatching nodes self-cancel.
        """
        results = []
        for node in self.nodes.values():
            freq_diff = abs(node.frequency - query_frequency)
            phase_diff = abs((node.phase - query_phase + np.pi) % (2 * np.pi) - np.pi)

            # Constructive interference factor: cos(delta_phi) * Gaussian(delta_freq)
            freq_match = np.exp(- (freq_diff ** 2) / (2.0 * (tolerance_freq ** 2)))
            constructive = node.amplitude * np.cos(phase_diff) * freq_match
            destructive = node.amplitude * (1.0 - np.cos(phase_diff)) * (1.0 - freq_match)

            resonance_score = float(constructive - 0.5 * destructive)

            # Continuous collapse projection
            collapsed = 1.0 if resonance_score > 0.3 else (0.0 if resonance_score < -0.1 else 0.5)

            results.append(ResonanceResult(
                target_node_id=node.node_id,
                resonance_score=resonance_score,
                constructive_energy=float(max(0.0, constructive)),
                destructive_cancellation=float(max(0.0, destructive)),
                collapsed_state=collapsed
            ))

        # Sort by highest resonance score
        results.sort(key=lambda x: x.resonance_score, reverse=True)
        return results

    def apply_phase_locked_focus(
        self,
        target_node_id: str,
        boost_factor: float = 2.5,
        noise_suppression_factor: float = 0.1
    ) -> Dict[str, Any]:
        """
        Applies Phase-Locked Focus & Attentive Noise Cancellation.
        Boosts amplitude and phase-locks target node, while suppressing unaligned noise frequencies.
        """
        if target_node_id not in self.nodes:
            raise ValueError(f"Target node {target_node_id} not found.")

        target = self.nodes[target_node_id]
        target_phase = target.phase

        modified_nodes = {}
        total_energy_before = sum(n.amplitude ** 2 for n in self.nodes.values())

        for node_id, node in self.nodes.items():
            phase_diff = abs((node.phase - target_phase + np.pi) % (2 * np.pi) - np.pi)

            if node_id == target_node_id or node.parent_id == target_node_id:
                # Target and immediate children: Boost amplitude and phase lock
                node.amplitude *= boost_factor
                node.phase = target_phase  # Perfect phase alignment
            else:
                # Unaligned noise nodes: Suppress amplitude via destructive alignment
                alignment = np.cos(phase_diff)
                if alignment < 0.5:
                    node.amplitude *= noise_suppression_factor

            modified_nodes[node_id] = node.amplitude

        total_energy_after = sum(n.amplitude ** 2 for n in self.nodes.values())

        return {
            "focused_target": target_node_id,
            "target_new_amplitude": target.amplitude,
            "target_phase_locked_rad": target.phase,
            "total_energy_before": float(total_energy_before),
            "total_energy_after": float(total_energy_after),
            "energy_efficiency_ratio": float(total_energy_after / (total_energy_before + 1e-9))
        }

    def step_continuous_time(self, dt: float = 0.01):
        """Advances phase field over time according to intrinsic node frequencies."""
        self.time += dt
        for node in self.nodes.values():
            node.phase = (node.phase + 2 * np.pi * node.frequency * dt) % (2 * np.pi)
