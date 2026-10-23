"""
Elysia Unified System Architecture (통섭적 시스템 아키텍처)
===========================================================
A 5-Stage Wavefield Computing Paradigm bridging fundamental physical principles
to scalable, self-correcting cognitive architecture.

1. Stage 1: Basic Computational Unit Redefinition (Scalar Bit -> Complex Phasor Field)
2. Stage 2: Prism Bifurcation Operator (N-Phase Spectral Projection & Causal Branching)
3. Stage 3: Multi-Layered Wave Memory & Dynamic Macro-Rotor Stack (Dynamic Chunking)
4. Stage 4: Retrocausal Execution Loop & Phase Error Self-Correction (Reversible Mapping & Retro-trace)
5. Stage 5: Recursive Fractal Scaling & Rapid Wave Collapse (Fractal Cubes & Global Resonance)
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np


# =============================================================================
# STAGE 1: Complex Phasor Field & Mini Rotors
# =============================================================================

class ComplexPhasor:
    """
    Represents a dynamic mini rotor with magnitude (Amplitude A),
    phase angle (phi in radians), and rotational frequency (omega).

    Mathematical form: Z = A * exp(i * phi)
    """
    def __init__(self, amplitude: float = 1.0, phase: float = 0.0, frequency: float = 1.0):
        self.amplitude = float(amplitude)
        self.phase = float(phase % (2 * np.pi))
        self.frequency = float(frequency)

    @property
    def complex_value(self) -> complex:
        return self.amplitude * np.exp(1j * self.phase)

    @classmethod
    def from_complex(cls, z: complex, frequency: float = 1.0) -> 'ComplexPhasor':
        amp = float(np.abs(z))
        phase = float(np.angle(z))
        return cls(amplitude=amp, phase=phase, frequency=frequency)

    def rotate(self, delta_phase: float) -> 'ComplexPhasor':
        """Applies phase rotation in time/space."""
        new_phase = (self.phase + delta_phase) % (2 * np.pi)
        return ComplexPhasor(self.amplitude, new_phase, self.frequency)

    def step_time(self, dt: float) -> 'ComplexPhasor':
        """Advances rotor in time according to its intrinsic frequency."""
        return self.rotate(self.frequency * dt)

    def __add__(self, other: 'ComplexPhasor') -> 'ComplexPhasor':
        """Interference addition of two phasors."""
        z_sum = self.complex_value + other.complex_value
        avg_freq = 0.5 * (self.frequency + other.frequency)
        return ComplexPhasor.from_complex(z_sum, frequency=avg_freq)

    def __repr__(self) -> str:
        return f"Phasor(A={self.amplitude:.4f}, phi={self.phase:.4f} rad, freq={self.frequency:.2f}Hz)"


class PhasorField:
    """
    Multi-dimensional vector field of complex phasors stored as vectorized NumPy arrays.
    Carries rotational momentum, amplitude envelope, and phase dynamics.
    """
    def __init__(self, shape: Tuple[int, ...], field: Optional[np.ndarray] = None, frequencies: Optional[np.ndarray] = None):
        self.shape = shape
        if field is not None:
            assert field.shape == shape, f"Field shape mismatch: {field.shape} vs {shape}"
            self.field = field.astype(np.complex128)
        else:
            # Default initialization with uniform magnitude and zero phase
            self.field = np.ones(shape, dtype=np.complex128)

        if frequencies is not None:
            assert frequencies.shape == shape, f"Frequencies shape mismatch: {frequencies.shape} vs {shape}"
            self.frequencies = frequencies.astype(np.float64)
        else:
            self.frequencies = np.ones(shape, dtype=np.float64)

    @property
    def amplitudes(self) -> np.ndarray:
        return np.abs(self.field)

    @property
    def phases(self) -> np.ndarray:
        return np.angle(self.field)

    def rotate(self, delta_phase: np.ndarray | float) -> 'PhasorField':
        """Rotates phase across the entire field."""
        new_field = self.field * np.exp(1j * delta_phase)
        return PhasorField(self.shape, field=new_field, frequencies=self.frequencies.copy())

    def step_time(self, dt: float) -> 'PhasorField':
        """Steps field through time using frequency spectrum."""
        return self.rotate(self.frequencies * dt)

    def superposition(self, other: 'PhasorField') -> 'PhasorField':
        """Interference superposition with another phasor field."""
        assert self.shape == other.shape, "Shape mismatch for field superposition"
        new_field = self.field + other.field
        avg_freq = 0.5 * (self.frequencies + other.frequencies)
        return PhasorField(self.shape, field=new_field, frequencies=avg_freq)

    def calculate_coherence(self) -> float:
        """Calculates global order parameter (Kuramoto phase order parameter R)."""
        phases = self.phases.flatten()
        order_param = np.abs(np.mean(np.exp(1j * phases)))
        return float(order_param)


# =============================================================================
# STAGE 2: Prism Bifurcation Operator (N-Phase Spectral Projection)
# =============================================================================

class CausalBranchTag:
    """Metadata tag recording causal lineage from parent field to sub-spectrum."""
    def __init__(self, parent_id: str, branch_index: int, phase_shift: float, threshold_passed: bool):
        self.parent_id = parent_id
        self.branch_index = branch_index
        self.phase_shift = phase_shift
        self.threshold_passed = threshold_passed

    def __repr__(self) -> str:
        return (f"CausalBranchTag(parent='{self.parent_id}', branch={self.branch_index}, "
                f"shift={self.phase_shift:.3f}rad, passed={self.threshold_passed})")


class PrismBifurcationOperator:
    """
    Algorithmic prism that refracts an incoming root phasor field into
    N distinct phase-symmetric sub-spectra, automatically tagging parent-child causal relationships.
    """
    def __init__(self, num_phases: int = 3, threshold: float = 0.1):
        self.num_phases = num_phases
        self.threshold = threshold

    def refract(self, root_field: PhasorField, root_id: str = "root") -> List[Tuple[PhasorField, CausalBranchTag]]:
        """
        Bifurcates root field into N symmetrical sub-spectra with causal tagging.
        """
        sub_spectra = []
        base_angle_step = 2.0 * np.pi / self.num_phases

        for k in range(self.num_phases):
            phase_shift = k * base_angle_step
            # Refract field by phase shift operator e^(i * k * 2pi / N)
            refracted_raw = root_field.field * np.exp(1j * phase_shift)

            # Apply thresholding mask (prismatic activation)
            amp_mask = np.abs(refracted_raw) >= self.threshold
            refracted_masked = np.where(amp_mask, refracted_raw, 0.0 + 0.0j)

            passed = bool(np.any(amp_mask))
            branch_tag = CausalBranchTag(
                parent_id=root_id,
                branch_index=k,
                phase_shift=phase_shift,
                threshold_passed=passed
            )

            # Frequencies scale harmonically by branch index + 1
            branch_freqs = root_field.frequencies * (1.0 + 0.5 * k)
            branch_field = PhasorField(root_field.shape, field=refracted_masked, frequencies=branch_freqs)

            sub_spectra.append((branch_field, branch_tag))

        return sub_spectra


# =============================================================================
# STAGE 3: Multi-Layered Wave Memory & Dynamic Macro-Rotor Stack
# =============================================================================

class MacroRotor:
    """
    A promoted macro-node formed by phase interference and chunking of stable
    lower-layer wave patterns.
    """
    def __init__(self, rotor_id: str, layer_level: int, aggregated_phasor: ComplexPhasor, constituent_count: int):
        self.rotor_id = rotor_id
        self.layer_level = layer_level
        self.aggregated_phasor = aggregated_phasor
        self.constituent_count = constituent_count
        self.spatiotemporal_harmonics: List[float] = []
        # Volumetric S^3 Quaternion Tensor extension: [q_r, q_i, q_j, q_k]
        self.quaternion_q: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Solid Angle in Steradians [0, 4\pi]
        self.solid_angle_sr: float = 4.0 * np.pi

    def add_harmonic_residual(self, harmonic_freq: float) -> None:
        self.spatiotemporal_harmonics.append(harmonic_freq)

    def __repr__(self) -> str:
        return (f"MacroRotor(id='{self.rotor_id}', L={self.layer_level}, "
                f"A={self.aggregated_phasor.amplitude:.3f}, phi={self.aggregated_phasor.phase:.3f}, "
                f"count={self.constituent_count})")


class DynamicMacroRotorStack:
    """
    Self-growing wave memory structure that aggregates stable lower-layer wave patterns
    into macro-rotors via interference phase-locking (Dynamic Chunking).
    Memory is kept as a woven spatiotemporal field rather than overwritten table rows.
    """
    def __init__(self, max_layers: int = 4, chunking_coherence_threshold: float = 0.65):
        self.max_layers = max_layers
        self.chunking_threshold = chunking_coherence_threshold
        # Layers map level -> list of MacroRotors
        self.layers: Dict[int, List[MacroRotor]] = {lvl: [] for lvl in range(max_layers)}
        self.field_residuals: List[PhasorField] = []

        from core.physics.multiscale_volumetric_coupler import MultiScaleVolumetricPhaseCoupler
        self.volumetric_coupler = MultiScaleVolumetricPhaseCoupler()

    def process_and_chunk(self, incoming_field: PhasorField) -> List[MacroRotor]:
        """
        Evaluates interference pattern coherence in incoming field.
        If coherence passes threshold, dynamically chunks into L+1 MacroRotor.
        """
        coherence = incoming_field.calculate_coherence()
        promoted_rotors = []

        # Retain residual field in spatiotemporal memory stack
        self.field_residuals.append(incoming_field)
        if len(self.field_residuals) > 10:
            self.field_residuals.pop(0)

        if coherence >= self.chunking_threshold:
            # Aggregate field into complex sum phasor
            total_complex = np.sum(incoming_field.field)
            mean_freq = float(np.mean(incoming_field.frequencies))

            macro_phasor = ComplexPhasor.from_complex(total_complex, frequency=mean_freq)
            rotor_count = len(self.layers[1]) + 1

            macro_rotor = MacroRotor(
                rotor_id=f"macro_L1_{rotor_count}",
                layer_level=1,
                aggregated_phasor=macro_phasor,
                constituent_count=int(incoming_field.field.size)
            )

            # Extract high-order harmonics
            harmonics = np.fft.fftfreq(incoming_field.field.size, d=1.0 / (mean_freq + 1e-5))
            macro_rotor.spatiotemporal_harmonics = list(np.abs(harmonics[:3]))

            # S^3 Quaternion mapping from complex phase
            phase = macro_phasor.phase
            half_p = phase * 0.5
            macro_rotor.quaternion_q = np.array([
                np.cos(half_p),
                np.sin(half_p) * 0.57735,
                np.sin(half_p) * 0.57735,
                np.sin(half_p) * 0.57735
            ], dtype=np.float32)

            self.layers[1].append(macro_rotor)
            promoted_rotors.append(macro_rotor)

            # Check if Level 1 rotors can be chunked into Level 2
            if len(self.layers[1]) >= 3:
                self._check_higher_layer_chunking(target_level=2)

        return promoted_rotors

    def _check_higher_layer_chunking(self, target_level: int) -> None:
        if target_level >= self.max_layers:
            return

        prev_rotors = self.layers[target_level - 1]
        sum_z = sum(r.aggregated_phasor.complex_value for r in prev_rotors)
        phases = [r.aggregated_phasor.phase for r in prev_rotors]
        coherence = float(np.abs(np.mean(np.exp(1j * np.array(phases)))))

        if coherence >= self.chunking_threshold:
            mean_freq = float(np.mean([r.aggregated_phasor.frequency for r in prev_rotors]))
            mega_phasor = ComplexPhasor.from_complex(sum_z, frequency=mean_freq)
            mega_rotor = MacroRotor(
                rotor_id=f"macro_L{target_level}_{len(self.layers[target_level]) + 1}",
                layer_level=target_level,
                aggregated_phasor=mega_phasor,
                constituent_count=sum(r.constituent_count for r in prev_rotors)
            )
            self.layers[target_level].append(mega_rotor)


# =============================================================================
# STAGE 4: Retrocausal Execution Loop & Phase Error Self-Correction
# =============================================================================

class ReversibleWaveTransformation:
    """
    Encapsulates a forward wave transformation f(Z) and its analytical inverse f^(-1)(Z),
    ensuring lossless reversibility (Delta S approx 0).
    """
    def __init__(self, name: str, forward_fn, inverse_fn):
        self.name = name
        self.forward_fn = forward_fn
        self.inverse_fn = inverse_fn

    def apply_forward(self, field: PhasorField, **kwargs) -> PhasorField:
        return self.forward_fn(field, **kwargs)

    def apply_inverse(self, field: PhasorField, **kwargs) -> PhasorField:
        return self.inverse_fn(field, **kwargs)


class RetrocausalExecutionLoop:
    """
    4th Stage Retrocausal Pipeline.
    Verifies lossless causality by reverse-tracing final computational outputs back to initial field conditions.
    Also performs geometric phase error self-correction by overlapping field with reference order.
    """
    def __init__(self, entropy_tolerance: float = 1e-6):
        self.entropy_tolerance = entropy_tolerance

    @staticmethod
    def create_phase_rotation_transform(delta_phase: float) -> ReversibleWaveTransformation:
        def forward(field: PhasorField, **kwargs) -> PhasorField:
            return field.rotate(delta_phase)

        def inverse(field: PhasorField, **kwargs) -> PhasorField:
            return field.rotate(-delta_phase)

        return ReversibleWaveTransformation("PhaseRotation", forward, inverse)

    @staticmethod
    def create_amplitude_scaling_transform(scale_factor: float) -> ReversibleWaveTransformation:
        assert scale_factor != 0, "Scale factor cannot be zero for reversible mapping"

        def forward(field: PhasorField, **kwargs) -> PhasorField:
            new_field = field.field * scale_factor
            return PhasorField(field.shape, field=new_field, frequencies=field.frequencies.copy())

        def inverse(field: PhasorField, **kwargs) -> PhasorField:
            new_field = field.field / scale_factor
            return PhasorField(field.shape, field=new_field, frequencies=field.frequencies.copy())

        return ReversibleWaveTransformation("AmplitudeScaling", forward, inverse)

    def retrocausal_trace(
        self,
        initial_field: PhasorField,
        final_field: PhasorField,
        transformations: List[ReversibleWaveTransformation],
        transform_kwargs_list: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, float, PhasorField]:
        """
        Traces back from final_field to initial_field using inverse transformations in reverse order.
        Returns:
            (is_causally_valid, entropy_loss_Delta_S, recovered_initial_field)
        """
        if transform_kwargs_list is None:
            transform_kwargs_list = [{} for _ in transformations]

        current_field = final_field
        # Reverse execution path
        for transform, kwargs in zip(reversed(transformations), reversed(transform_kwargs_list)):
            current_field = transform.apply_inverse(current_field, **kwargs)

        # Calculate entropy loss Delta S as mean relative difference in field state
        diff = np.abs(current_field.field - initial_field.field)
        initial_norm = np.mean(np.abs(initial_field.field)) + 1e-12
        entropy_loss = float(np.mean(diff) / initial_norm)

        is_valid = entropy_loss <= self.entropy_tolerance
        return is_valid, entropy_loss, current_field

    def detect_and_rectify_phase_errors(
        self,
        distorted_field: PhasorField,
        reference_field: PhasorField
    ) -> Tuple[PhasorField, np.ndarray, float]:
        """
        Self-Correction via Geometric Phase Matching (Moire Overlap Principle).
        Overlaps distorted field with reference field to reveal local phase distortion / mismatch.
        Applies targeted phase rectification to distorted modes without disrupting global macro order.

        Returns:
            (rectified_field, mismatch_map, global_distortion_level)
        """
        assert distorted_field.shape == reference_field.shape, "Shape mismatch for geometric phase matching"

        # Interference product Z_dist * conj(Z_ref) reveals phase relative difference
        interference_product = distorted_field.field * np.conj(reference_field.field)
        phase_mismatch = np.angle(interference_product)  # Mismatch angle delta_phi

        # Mismatch intensity map (0 = perfect alignment, >0 = phase distortion)
        mismatch_map = np.abs(phase_mismatch)
        global_distortion = float(np.mean(mismatch_map))

        # Targeted Phase Tuning: subtract phase mismatch angle
        rectified_complex = distorted_field.field * np.exp(-1j * phase_mismatch)
        rectified_field = PhasorField(
            distorted_field.shape,
            field=rectified_complex,
            frequencies=distorted_field.frequencies.copy()
        )

        return rectified_field, mismatch_map, global_distortion


# =============================================================================
# STAGE 5: Recursive Fractal Scaling & Wave Collapse
# =============================================================================

class FractalResonanceUnit:
    """
    Self-similar unit (e.g., 3x3 or 3x3x3 grid) that serves as a single node in a higher-dimensional field.
    Encapsulates recursive phase dynamics.
    """
    def __init__(self, unit_id: str, dimension: int = 3, depth: int = 1):
        self.unit_id = unit_id
        self.dimension = dimension
        self.depth = depth
        shape = tuple([dimension] * 3) if dimension == 3 else tuple([dimension] * 2)

        self.local_field = PhasorField(shape)
        self.child_units: List['FractalResonanceUnit'] = []

        # Recursively construct child units if depth > 1
        if depth > 1:
            num_children = dimension ** 2
            for i in range(num_children):
                child = FractalResonanceUnit(
                    unit_id=f"{unit_id}_c{i}",
                    dimension=dimension,
                    depth=depth - 1
                )
                self.child_units.append(child)

    def aggregate_macro_phasor(self) -> ComplexPhasor:
        """Collects resonant state from local field and child units."""
        local_z = np.sum(self.local_field.field)
        if self.child_units:
            child_z = sum(child.aggregate_macro_phasor().complex_value for child in self.child_units)
            total_z = local_z + child_z
        else:
            total_z = local_z

        return ComplexPhasor.from_complex(total_z, frequency=float(np.mean(self.local_field.frequencies)))

    def wave_collapse(self, coupling_strength: float = 0.5, max_steps: int = 20) -> float:
        """
        Performs rapid wave collapse via global phase synchronization across all fractal levels.
        Returns final global coherence R.
        """
        for _ in range(max_steps):
            # Phase synchronization across local field cells
            mean_phase = np.angle(np.mean(np.exp(1j * self.local_field.phases)))
            phase_diff = mean_phase - self.local_field.phases

            # Kuramoto-like coupling update
            new_field = self.local_field.field * np.exp(1j * coupling_strength * np.sin(phase_diff))
            self.local_field = PhasorField(self.local_field.shape, field=new_field, frequencies=self.local_field.frequencies)

            # Synchronize child units recursively
            for child in self.child_units:
                child.wave_collapse(coupling_strength=coupling_strength, max_steps=1)

        return self.local_field.calculate_coherence()


# =============================================================================
# UNIFIED SYSTEM ARCHITECTURE (5-Stage Orchestrator)
# =============================================================================

class UnifiedSystemArchitecture:
    """
    Top-level Orchestrator integrating all 5 Stages of Wavefield Computing:
    Stage 1: Complex Phasor Field
    Stage 2: Prism Bifurcation Operator
    Stage 3: Dynamic Macro-Rotor Stack
    Stage 4: Retrocausal Execution Loop & Self-Correction
    Stage 5: Recursive Fractal Scaling
    """
    def __init__(self, num_prism_phases: int = 3, fractal_depth: int = 2):
        self.stage1_field: Optional[PhasorField] = None
        self.stage2_prism = PrismBifurcationOperator(num_phases=num_prism_phases)
        self.stage3_memory = DynamicMacroRotorStack()
        self.stage4_retro = RetrocausalExecutionLoop()
        self.stage5_fractal = FractalResonanceUnit(unit_id="root_fractal", depth=fractal_depth)

    def initialize_root_field(self, shape: Tuple[int, ...], random_seed: Optional[int] = None) -> PhasorField:
        """Initializes fundamental continuous phasor field."""
        if random_seed is not None:
            np.random.seed(random_seed)

        amps = np.random.uniform(0.5, 1.5, size=shape)
        phases = np.random.uniform(0, 2 * np.pi, size=shape)
        freqs = np.random.uniform(0.8, 1.2, size=shape)

        field_complex = amps * np.exp(1j * phases)
        self.stage1_field = PhasorField(shape, field=field_complex, frequencies=freqs)
        return self.stage1_field

    def execute_full_cycle(self) -> Dict[str, Any]:
        """
        Runs complete 5-stage pipeline and returns diagnostic summary.
        """
        assert self.stage1_field is not None, "Initialize root field before execution"

        results: Dict[str, Any] = {}

        # --- Stage 1: Continuous Phasor Field ---
        s1_coherence = self.stage1_field.calculate_coherence()
        results["Stage1"] = {
            "shape": self.stage1_field.shape,
            "initial_coherence": s1_coherence
        }

        # --- Stage 2: Prism Bifurcation Operator ---
        refracted_branches = self.stage2_prism.refract(self.stage1_field, root_id="core_root")
        results["Stage2"] = {
            "num_branches": len(refracted_branches),
            "branch_tags": [tag for _, tag in refracted_branches]
        }

        # --- Stage 3: Multi-Layered Wave Memory & Dynamic Chunking ---
        chunked_rotors = []
        for branch_field, _ in refracted_branches:
            chunked = self.stage3_memory.process_and_chunk(branch_field)
            chunked_rotors.extend(chunked)

        results["Stage3"] = {
            "chunked_rotors_count": len(chunked_rotors),
            "layer_stats": {lvl: len(rotors) for lvl, rotors in self.stage3_memory.layers.items()}
        }

        # --- Stage 4: Retrocausal Execution Loop & Self-Correction ---
        rot_transform = RetrocausalExecutionLoop.create_phase_rotation_transform(np.pi / 4)
        scale_transform = RetrocausalExecutionLoop.create_amplitude_scaling_transform(1.5)
        transforms = [rot_transform, scale_transform]

        # Apply forward transformations
        transformed_field = rot_transform.apply_forward(self.stage1_field)
        final_field = scale_transform.apply_forward(transformed_field)

        # Retrocausal validation trace
        valid, entropy_loss, _ = self.stage4_retro.retrocausal_trace(
            self.stage1_field, final_field, transforms
        )

        # Geometric Phase Self-Correction Test
        noise_phase = np.random.normal(0, 0.2, size=self.stage1_field.shape)
        distorted = self.stage1_field.rotate(noise_phase)
        rectified, mismatch_map, global_distortion = self.stage4_retro.detect_and_rectify_phase_errors(
            distorted, self.stage1_field
        )

        results["Stage4"] = {
            "retrocausal_valid": valid,
            "entropy_loss_Delta_S": entropy_loss,
            "pre_correction_distortion": global_distortion,
            "post_correction_coherence": rectified.calculate_coherence()
        }

        # --- Stage 5: Recursive Fractal Scaling & Wave Collapse ---
        initial_fractal_rotor = self.stage5_fractal.aggregate_macro_phasor()
        final_fractal_coherence = self.stage5_fractal.wave_collapse(coupling_strength=0.8, max_steps=15)
        final_fractal_rotor = self.stage5_fractal.aggregate_macro_phasor()

        results["Stage5"] = {
            "initial_fractal_rotor": initial_fractal_rotor,
            "final_fractal_rotor": final_fractal_rotor,
            "final_fractal_coherence": final_fractal_coherence
        }

        return results
