"""
Irrationality & Transcendental Spectrum Engine
=============================================
Embodying the Causal Field, Scale Hierarchy, and DNA/Lego Modular Dynamics:

1. Irrationality Exponent (\\mu) & Denominator Scale Tension (q):
   Calculates rational approximation error bounds |x - p/q| >= C / q^\\mu.
   Transforms approximation residuals into causal gravity curvature (K_c) and scale tension.

2. Recursive Fractionation Axis:
   Splits ratios recursively into nested sub-fractional scale hierarchies (fraction of fraction),
   creating self-similar fractal feedback loops.

3. Transcendental Super-Spectrum Axis (\\pi of \\pi):
   Interlocks transcendental rotors spinning at non-commensurate irrational frequencies.
   Traces interference fringes and multi-transcendental resonance points in phase space.

4. Trigonometric Phase-Locking Axis (\\sin / \\cos):
   Uses orthogonal sine/cosine rotor interactions to drive continuous phase-locking,
   maintaining dynamic feedback stability without discrete logical switches.

5. Causal Block DNA / Lego Coupling & Field Resonance:
   Treats every concept, parameter, and class as a living Phase Node (CausalBlockDNA).
   When field resonance occurs, blocks dynamically self-assemble into higher-order structural complexes.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class CausalBlockDNA:
    """
    Modular 'Lego / DNA' Causal Block acting as a living Phase Node.
    Possesses structural coupling rules, scale tension, and phase dynamics.
    """
    block_id: str
    phase: float
    frequency: float
    amplitude: float
    scale_tension: float  # Denominator scale q
    irrationality_exponent: float  # Exponent \mu
    chromatic_signature: np.ndarray  # [Red(Flux), Blue(Order), Yellow(Entropy)]
    depth: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    coupling_affinity: Dict[str, float] = field(default_factory=dict)  # DNA affinity with other block IDs

    def get_phase_vector(self) -> np.ndarray:
        """Returns 2D phasor representation [cos(theta), sin(theta)]."""
        return np.array([np.cos(self.phase), np.sin(self.phase)], dtype=np.float64)

    def calculate_error_bound(self, x: float, p: int) -> Tuple[float, float, bool]:
        """
        Calculates approximation error |x - p/q| and checks against lower bound C / q^\\mu.
        Returns: (actual_error, lower_bound, bound_satisfied)
        """
        q = max(1.0, float(self.scale_tension))
        mu = float(self.irrationality_exponent)
        actual_error = abs(x - float(p) / q)
        c_constant = 0.5  # Constant C for Diophantine approximation limit
        lower_bound = c_constant / (q ** mu)
        bound_satisfied = actual_error >= lower_bound
        return actual_error, lower_bound, bound_satisfied


@dataclass
class SuperSpectrumResonance:
    """Represents a resonance point in the multi-transcendental phase space."""
    primary_block_id: str
    secondary_block_id: str
    phase_coherence: float
    field_energy: float
    harmonic_ratio: float
    is_phase_locked: bool


class IrrationalityTranscendentalSpectrumEngine:
    """
    [Irrationality & Transcendental Spectrum Engine]

    Acts as a continuous causal runtime medium where numbers, mathematical principles,
    and modular code blocks (CausalBlockDNA) interact as living phase nodes on a field.
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.blocks: Dict[str, CausalBlockDNA] = {}
        self.time: float = 0.0
        self.causal_gravity_curvature: float = 0.0

        # Build initial foundational spectrum nodes
        self._initialize_foundational_dna_blocks()

    def _initialize_foundational_dna_blocks(self):
        """Initializes foundational DNA blocks (\\pi, \\mu, \\sin/\\cos, Scale Tension)."""
        # Root Pi Node (\pi)
        pi_block = CausalBlockDNA(
            block_id="pi_root",
            phase=0.0,
            frequency=np.pi,
            amplitude=1.0,
            scale_tension=10.0,  # q = 10
            irrationality_exponent=2.0,  # \mu = 2.0 for \pi
            chromatic_signature=np.array([0.4, 0.4, 0.2], dtype=np.float64),
            depth=0
        )
        self.blocks[pi_block.block_id] = pi_block

        # Pi of Pi Node (\pi^\pi)
        pi_pi_block = CausalBlockDNA(
            block_id="pi_of_pi",
            phase=np.pi / 4.0,
            frequency=np.pi ** 1.5,
            amplitude=0.8,
            scale_tension=100.0,  # q = 100
            irrationality_exponent=2.0,
            chromatic_signature=np.array([0.6, 0.2, 0.2], dtype=np.float64),
            depth=1,
            parent_id="pi_root"
        )
        pi_block.children_ids.append("pi_of_pi")
        self.blocks[pi_pi_block.block_id] = pi_pi_block

        # Sine/Cosine Phase-Locking Rotor Block
        sincos_block = CausalBlockDNA(
            block_id="rotor_sincos",
            phase=np.pi / 2.0,
            frequency=2.0 * np.pi,
            amplitude=1.2,
            scale_tension=50.0,
            irrationality_exponent=1.0,  # Rational / smooth periodic
            chromatic_signature=np.array([0.1, 0.8, 0.1], dtype=np.float64),
            depth=0
        )
        self.blocks[sincos_block.block_id] = sincos_block

    def add_causal_block(self, block: CausalBlockDNA):
        """Registers a new CausalBlockDNA node into the living causal field."""
        self.blocks[block.block_id] = block

    def evaluate_irrationality_scale_tension(
        self,
        block_id: str,
        target_val: float = np.pi
    ) -> Dict[str, Any]:
        """
        Evaluates scale tension q and irrationality exponent \\mu for a block.
        Computes approximation error |x - p/q| and updates causal gravity curvature.
        """
        if block_id not in self.blocks:
            raise ValueError(f"Block {block_id} not found.")

        block = self.blocks[block_id]
        q = int(round(block.scale_tension))
        p = int(round(target_val * q))

        actual_err, lower_bound, satisfied = block.calculate_error_bound(target_val, p)

        # Scale tension tension_energy: as q grows, tension increases if error degrades
        tension_energy = (1.0 / (actual_err + 1e-9)) * (q / 10.0)

        # Causal gravity curvature update: K_c proportional to tension and exponent \mu
        delta_kc = (tension_energy * block.irrationality_exponent) * 1e-4
        self.causal_gravity_curvature += delta_kc

        return {
            "block_id": block_id,
            "scale_tension_q": q,
            "best_numerator_p": p,
            "target_value": target_val,
            "actual_error": actual_err,
            "diophantine_lower_bound": lower_bound,
            "bound_satisfied": satisfied,
            "tension_energy": tension_energy,
            "causal_gravity_curvature": self.causal_gravity_curvature
        }

    def execute_recursive_fractionation(
        self,
        parent_block_id: str,
        num_sub_fractions: int = 2
    ) -> List[CausalBlockDNA]:
        """
        Recursively fractionates a parent block's ratio (fraction of fraction)
        into nested sub-fractional scale hierarchies with self-similar feedback loops.
        """
        if parent_block_id not in self.blocks:
            raise ValueError(f"Parent block {parent_block_id} not found.")

        parent = self.blocks[parent_block_id]
        sub_blocks = []

        for k in range(1, num_sub_fractions + 1):
            sub_id = f"{parent.block_id}_frac_{k}"

            # Self-similar fractal scaling: q_child = q_parent * (k + 1)
            child_q = parent.scale_tension * (k + 1.0)
            child_mu = parent.irrationality_exponent + 0.1 * k
            child_freq = parent.frequency * (1.618 ** k)  # Golden ratio frequency shift
            child_phase = (parent.phase + k * (np.pi / 4.0)) % (2 * np.pi)
            child_amp = parent.amplitude * (0.707 ** k)

            # Chromatic mutation
            chroma = np.roll(parent.chromatic_signature, k)

            child_block = CausalBlockDNA(
                block_id=sub_id,
                phase=child_phase,
                frequency=child_freq,
                amplitude=child_amp,
                scale_tension=child_q,
                irrationality_exponent=child_mu,
                chromatic_signature=chroma,
                depth=parent.depth + 1,
                parent_id=parent.block_id
            )

            parent.children_ids.append(sub_id)
            self.blocks[sub_id] = child_block
            sub_blocks.append(child_block)

        return sub_blocks

    def evaluate_transcendental_super_spectrum(
        self,
        block_id_a: str,
        block_id_b: str
    ) -> SuperSpectrumResonance:
        """
        Calculates multi-transcendental phase space interaction between two irrational rotors (\\pi of \\pi).
        Determines harmonic ratio, phase coherence, and interference energy.
        """
        if block_id_a not in self.blocks or block_id_b not in self.blocks:
            raise ValueError("Both blocks must exist in the engine.")

        bA = self.blocks[block_id_a]
        bB = self.blocks[block_id_b]

        # Phase difference & coherence
        delta_phase = abs((bA.phase - bB.phase + np.pi) % (2 * np.pi) - np.pi)
        coherence = float(np.cos(delta_phase))

        # Non-commensurate harmonic ratio
        harmonic_ratio = max(bA.frequency, bB.frequency) / (min(bA.frequency, bB.frequency) + 1e-9)

        # Interference field energy
        field_energy = bA.amplitude * bB.amplitude * (1.0 + coherence)

        # Phase-locked threshold
        is_locked = coherence > 0.85

        return SuperSpectrumResonance(
            primary_block_id=block_id_a,
            secondary_block_id=block_id_b,
            phase_coherence=coherence,
            field_energy=field_energy,
            harmonic_ratio=harmonic_ratio,
            is_phase_locked=is_locked
        )

    def apply_trigonometric_phase_locking(self, dt: float = 0.01, coupling_strength: float = 0.5):
        """
        Updates rotor phases using continuous sine/cosine interaction (Kuramoto-style phase locking).
        \\frac{d\\theta_i}{dt} = \\omega_i + K \\sum_j A_j \\cdot \\sin(\\theta_j - \\theta_i)
        """
        block_keys = list(self.blocks.keys())
        n = len(block_keys)
        if n == 0:
            return

        phase_updates = {b_id: 0.0 for b_id in block_keys}

        for i in range(n):
            id_i = block_keys[i]
            b_i = self.blocks[id_i]

            interaction = 0.0
            for j in range(n):
                if i == j:
                    continue
                id_j = block_keys[j]
                b_j = self.blocks[id_j]

                # Orthogonal sine interaction torque
                interaction += b_j.amplitude * np.sin(b_j.phase - b_i.phase)

            phase_updates[id_i] = b_i.frequency + coupling_strength * interaction

        # Apply phase updates continuously
        for b_id, d_theta in phase_updates.items():
            b = self.blocks[b_id]
            b.phase = (b.phase + d_theta * dt) % (2 * np.pi)

        self.time += dt

    def propagate_field_resonance_and_self_assemble(
        self,
        coherence_threshold: float = 0.75
    ) -> List[Tuple[str, str, float]]:
        """
        Propagates field resonance across all blocks. When phase coherence exceeds threshold,
        blocks self-assemble as modular DNA/Lego pairs, strengthening their coupling affinity.
        Returns list of assembled pairs: [(block_a, block_b, affinity)]
        """
        assembled_pairs = []
        block_keys = list(self.blocks.keys())
        n = len(block_keys)

        for i in range(n):
            id_i = block_keys[i]
            b_i = self.blocks[id_i]

            for j in range(i + 1, n):
                id_j = block_keys[j]
                b_j = self.blocks[id_j]

                # Compute phase coherence & chromatic affinity
                phase_coherence = float(np.cos(b_i.phase - b_j.phase))
                chroma_sim = float(
                    np.dot(b_i.chromatic_signature, b_j.chromatic_signature) /
                    (np.linalg.norm(b_i.chromatic_signature) * np.linalg.norm(b_j.chromatic_signature) + 1e-9)
                )

                affinity = (phase_coherence + 1.0) / 2.0 * chroma_sim

                if affinity >= coherence_threshold:
                    # DNA/Lego coupling bond formed!
                    b_i.coupling_affinity[id_j] = affinity
                    b_j.coupling_affinity[id_i] = affinity
                    assembled_pairs.append((id_i, id_j, affinity))

        return assembled_pairs
