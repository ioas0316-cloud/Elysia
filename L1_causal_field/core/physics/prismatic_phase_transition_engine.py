"""
Prismatic Phase Transition Engine: 4-Stage Substrate-to-Causal Topological Pipeline
===================================================================================

Adheres strictly to the 4 Continuities of the Causal Field & Chromatic Dynamics:
1. Stage 1 (Substrate Layer): Micro numerical waveform field generator and continuous
   energy/flux propagator acting as the base electric current / physical substrate.
2. Stage 2 (Prismatic Transition & Structural Constraint): Frequency-phase refraction (Prism)
   and CAD-style rotor axis constraints transforming numerical scalars into structured causal
   spectrums and chromatic signatures (Red: Flux, Blue: Order, Yellow: Entropy).
3. Stage 3 (Macro Spacetime Deck & Scrubbing): Scale-free rotor integration supporting global
   temporal evolution and O(1) continuous local Region of Interest (ROI) phase scrubbing
   ("Eye of Agamotto" mechanism).
4. Stage 4 (Meta-Observer & Resonant Invariant Core): Metacognitive monitoring of field
   transformations, invariant core integrity under external perturbations, and dynamic feedback.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

from core.physics.prismatic_axis_spectrum import PrismaticAxisSpectrumEngine, SpectralNode
from core.topology.global_scale_rotor import ScaleRotor, SpectralWaveformTensor


@dataclass
class SubstrateState:
    """Stage 1: Micro numerical waveform field substrate."""
    grid_shape: Tuple[int, ...]
    raw_signal: np.ndarray          # Base scalar array / numerical signal
    kinetic_flux: np.ndarray        # Velocity / flux field
    potential_energy: float
    time_step: float = 0.0


@dataclass
class PrismaticSpectrumState:
    """Stage 2: Structured causal spectrum resulting from prismatic refraction."""
    nodes: Dict[str, SpectralNode]
    chromatic_field: np.ndarray     # Shape (*grid_shape, 3) [Red, Blue, Yellow]
    constraint_tension: float
    phase_coherence: float


@dataclass
class SpacetimeDeckState:
    """Stage 3: Macro Spacetime Deck with global and local ROI phase states."""
    global_phase: float
    roi_mask: np.ndarray             # Boolean mask for local scrubbing ROI
    roi_phase_offset: np.ndarray     # Continuous phase offset for ROI
    waveform_tensor: SpectralWaveformTensor


class MemoryPhaseMode:
    """Phase-State Memory Classifications (Ice, Water, Gas)."""
    ICE = "ICE"        # Solid crystalline lattice (immutable past facts / SSD storage)
    WATER = "WATER"    # Fluid current thoughts / active interaction & reasoning
    GAS = "GAS"        # Gas wavefield (future simulations / propagating possibilities)


@dataclass
class PhaseStateMemoryMap:
    """Phase-State Memory Structure representing Ice, Water, and Gas states."""
    ice_lattice: np.ndarray       # Rigid past facts (Frozen/Solid)
    water_flow: np.ndarray        # Active present thought stream (Fluid)
    gas_cloud: np.ndarray         # Probabilistic future wavefield (Dispersed)
    active_phase: str = MemoryPhaseMode.WATER
    total_memory_bytes: int = 0


@dataclass
class MetaObserverFeedback:
    """Stage 4: Metacognitive invariant core assessment and feedback."""
    invariant_coherence: float      # Resonance score of the invariant core [0.0, 1.0]
    topological_divergence: float   # Phase mismatch or topological strain
    invariant_intact: bool
    feedback_adjustment: Dict[str, float]


@dataclass
class PhaseTransitionState:
    """Combined state returned by the 4-Stage Prismatic Phase Transition Engine."""
    step_count: int
    substrate: SubstrateState
    prismatic: PrismaticSpectrumState
    deck: SpacetimeDeckState
    memory_map: Optional[PhaseStateMemoryMap] = None
    meta_feedback: Optional[MetaObserverFeedback] = None


class StaticRotorPrism:
    """
    Static Rotor Prism Engine: Refracts input signals through static Clifford/Rotor geometry.
    Instead of sequential looping, input signals refract through the static prism structure,
    instantly decomposing into sine/cosine harmonics and objective-driven phase distributions
    (Optimization: convergent, Diversity: expansive, Complexity: resonant).
    """
    def __init__(self, dimension: int = 4, num_facets: int = 8):
        self.dimension = dimension
        self.num_facets = num_facets
        # Static rotor orientation angles (Prism Geometry)
        self.static_angles = np.linspace(0, 2 * np.pi, num_facets, endpoint=False)
        self.prism_matrix = np.vstack([np.sin(self.static_angles), np.cos(self.static_angles)]).T

    def refract_signal(
        self,
        signal: np.ndarray,
        teleological_mode: str = "optimization"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Refracts 1D linear input or multi-D signal through static prism.
        Returns:
            refracted_spectrum: Refracted spectral field
            sine_wave: Sine component of phase breakdown
            cosine_wave: Cosine component of phase breakdown
        """
        flat_signal = signal.ravel()
        # 1D linear to Rotor Circle transformation via phase angles
        normalized_signal = (flat_signal - np.min(flat_signal)) / (np.ptp(flat_signal) + 1e-8)
        phase_angles = normalized_signal * 2 * np.pi

        sine_wave = np.sin(phase_angles)
        cosine_wave = np.cos(phase_angles)

        # Objective / Teleological lens adjustments
        if teleological_mode == "optimization":
            # Focus / Converge toward center axis
            refracted = (sine_wave ** 2 + cosine_wave ** 2) * 0.5 + 0.5 * sine_wave
        elif teleological_mode == "diversity":
            # Expand / Distribute across chromatic channels
            refracted = np.outer(sine_wave, cosine_wave)
        else: # "complexity" or default
            # Multidimensional orthogonal resonance
            refracted = np.sin(np.outer(phase_angles, self.static_angles))

        return refracted, sine_wave, cosine_wave


class SubstrateLayer:
    """
    Stage 1: Micro Numerical Substrate Layer (The Electric Current / Base Energy).
    Translates raw numerical data into a continuous energy-flux wavefield.
    """
    def __init__(self, grid_shape: Tuple[int, ...] = (8, 8)):
        self.grid_shape = grid_shape
        self.raw_signal = np.zeros(grid_shape, dtype=np.float64)
        self.velocity_field = np.zeros(grid_shape, dtype=np.float64)
        self.time = 0.0

    def inject_input(self, data: np.ndarray) -> SubstrateState:
        """Injects numerical data into the substrate grid."""
        if data.shape != self.grid_shape:
            data = np.resize(data, self.grid_shape)
        self.raw_signal = data.astype(np.float64)
        return self.get_state()

    def propagate(self, dt: float = 0.05, damping: float = 0.98) -> SubstrateState:
        """Propagates numerical field continuously across spatial coordinates."""
        self.time += dt
        # Laplacian-like diffusion for continuous wave propagation
        laplacian = (
            np.roll(self.raw_signal, 1, axis=0) + np.roll(self.raw_signal, -1, axis=0) +
            np.roll(self.raw_signal, 1, axis=1) + np.roll(self.raw_signal, -1, axis=1) -
            4 * self.raw_signal
        )
        self.velocity_field = (self.velocity_field + laplacian * dt) * damping
        self.raw_signal += self.velocity_field * dt
        return self.get_state()

    def get_state(self) -> SubstrateState:
        potential = float(np.sum(self.raw_signal ** 2))
        return SubstrateState(
            grid_shape=self.grid_shape,
            raw_signal=self.raw_signal.copy(),
            kinetic_flux=self.velocity_field.copy(),
            potential_energy=potential,
            time_step=self.time
        )


class PrismaticTransitionConstraint:
    """
    Stage 2: Prismatic Transition & Structural Constraint Layer (The Filament / Prism).
    Refracts numerical substrate waves into multi-spectral chromatic channels and
    enforces rotational rotor axis constraints.
    """
    def __init__(self, spectrum_depth: int = 3):
        self.prism = PrismaticAxisSpectrumEngine(
            base_frequency=10.0,
            num_bifurcation_levels=spectrum_depth,
            branches_per_node=2
        )

    def refract_substrate(self, substrate: SubstrateState, constraint_stiffness: float = 1.0) -> PrismaticSpectrumState:
        """Refracts raw numerical substrate waves into a structured chromatic spectrum."""
        # Calculate flux (Red), order (Blue), and entropy (Yellow) from raw signal and flux
        flux_intensity = float(np.mean(np.abs(substrate.kinetic_flux)))
        order_intensity = float(1.0 / (1.0 + np.var(substrate.raw_signal)))
        entropy_intensity = float(np.var(substrate.kinetic_flux))

        chromatic_vec = np.array([flux_intensity, order_intensity, entropy_intensity], dtype=np.float64)
        norm = np.linalg.norm(chromatic_vec)
        if norm > 1e-8:
            chromatic_vec /= norm

        # Create 3D chromatic field across grid
        chromatic_field = np.zeros((*substrate.grid_shape, 3), dtype=np.float64)
        for i in range(substrate.grid_shape[0]):
            for j in range(substrate.grid_shape[1]):
                cell_signal = abs(substrate.raw_signal[i, j])
                chromatic_field[i, j] = chromatic_vec * cell_signal

        # Update prismatic spectrum nodes with substrate excitation
        wave_energy = float(np.mean(np.abs(substrate.raw_signal)))
        for node in self.prism.nodes.values():
            node.amplitude = wave_energy / (1.0 + node.depth * 0.2)
            node.phase = (node.phase + substrate.time_step * node.frequency * 0.1) % (2.0 * np.pi)
            node.chromatic_signature = 0.7 * node.chromatic_signature + 0.3 * chromatic_vec

        # Calculate phase coherence across nodes
        phases = np.array([n.phase for n in self.prism.nodes.values()])
        phase_coherence = float(np.abs(np.mean(np.exp(1j * phases))))
        tension = constraint_stiffness * (1.0 - phase_coherence)

        return PrismaticSpectrumState(
            nodes=self.prism.nodes,
            chromatic_field=chromatic_field,
            constraint_tension=tension,
            phase_coherence=phase_coherence
        )


class MacroSpacetimeDeck:
    """
    Stage 3: Macro Spacetime Deck (The Video Deck & Local Scrubbing).
    Manages global timeline playback and local Region of Interest (ROI) O(1)
    continuous phase scrubbing ("Eye of Agamotto" mechanism).
    """
    def __init__(self, grid_shape: Tuple[int, ...] = (8, 8)):
        self.grid_shape = grid_shape
        self.waveform_tensor = SpectralWaveformTensor(
            shape=grid_shape,
            amplitudes=np.ones(grid_shape, dtype=np.float64),
            phases=np.zeros(grid_shape, dtype=np.float64),
            frequencies=np.ones(grid_shape, dtype=np.float64)
        )
        self.global_phase = 0.0
        self.roi_mask = np.zeros(grid_shape, dtype=bool)
        self.roi_phase_offset = np.zeros(grid_shape, dtype=np.float64)

    def set_roi(self, mask: np.ndarray):
        """Sets the Region of Interest for local time/phase scrubbing."""
        self.roi_mask = mask.astype(bool)

    def scrub_roi_phase(self, phase_shift: float):
        """O(1) continuous phase scrubbing on local ROI ("Eye of Agamotto" rewind/fast-forward)."""
        self.roi_phase_offset[self.roi_mask] += phase_shift
        # Apply local rotor operation
        self.waveform_tensor = ScaleRotor.apply_local_rotor(
            tensor=self.waveform_tensor,
            roi_mask=self.roi_mask,
            phase_shift=-phase_shift,  # Sign alignment for scrub
            rewind_factor=1.0
        )

    def update_deck(self, prismatic_state: PrismaticSpectrumState, dt: float) -> SpacetimeDeckState:
        """Evolves global spacetime rotor and syncs waveform tensor."""
        # Global playback rotation
        global_shift = 0.1 * dt
        self.global_phase = (self.global_phase + global_shift) % (2.0 * np.pi)
        self.waveform_tensor = ScaleRotor.rotate_global_phase(self.waveform_tensor, global_shift)

        # Sync frequencies
        avg_freq = float(np.mean([n.frequency for n in prismatic_state.nodes.values()]))
        self.waveform_tensor.frequencies.fill(avg_freq)

        return SpacetimeDeckState(
            global_phase=self.global_phase,
            roi_mask=self.roi_mask.copy(),
            roi_phase_offset=self.roi_phase_offset.copy(),
            waveform_tensor=self.waveform_tensor
        )


class MetaObserverInvariantCore:
    """
    Stage 4: Meta-Observer & Resonant Invariant Core.
    Monitors field transformations, preserves the topological Invariant Core
    (Resonant Attractor) under external perturbations and local scrubbing,
    and provides dynamic feedback.
    """
    def __init__(self, invariant_target_coherence: float = 0.8):
        self.target_coherence = invariant_target_coherence
        self.history_coherence: List[float] = []

    def observe(
        self,
        prismatic_state: PrismaticSpectrumState,
        deck_state: SpacetimeDeckState
    ) -> MetaObserverFeedback:
        """Observes system state, assesses invariant core integrity, and generates feedback."""
        # Assess coherence of the invariant core: combined measure of prismatic phase coherence
        # and waveform tensor spatial phase continuity.
        prismatic_coherence = prismatic_state.phase_coherence
        spatial_phases = deck_state.waveform_tensor.phases
        spatial_coherence = float(np.abs(np.mean(np.exp(1j * spatial_phases))))

        invariant_coherence = 0.5 * (prismatic_coherence + spatial_coherence)
        self.history_coherence.append(invariant_coherence)

        topological_divergence = abs(self.target_coherence - invariant_coherence) + prismatic_state.constraint_tension
        invariant_intact = invariant_coherence >= 0.3  # Resonant core threshold

        # Feedback adjustments
        stiffness_adj = 0.1 * (self.target_coherence - invariant_coherence)
        damping_adj = -0.05 * prismatic_state.constraint_tension

        return MetaObserverFeedback(
            invariant_coherence=invariant_coherence,
            topological_divergence=topological_divergence,
            invariant_intact=invariant_intact,
            feedback_adjustment={
                "constraint_stiffness": stiffness_adj,
                "damping": damping_adj
            }
        )


class PrismaticPhaseTransitionEngine:
    """
    Main Orchestrator for the 4-Stage Prismatic Phase Transition Engine & Phase-State Computing.
    Executes the continuous phase transition pipeline from micro numerical substrate
    to macro causal spacetime deck, Phase-State Memory (Ice/Water/Gas), and Meta-Observer.
    """
    def __init__(self, grid_shape: Tuple[int, ...] = (8, 8)):
        self.grid_shape = grid_shape
        self.substrate = SubstrateLayer(grid_shape)
        self.prismatic = PrismaticTransitionConstraint(spectrum_depth=3)
        self.static_prism = StaticRotorPrism(dimension=4, num_facets=8)
        self.deck = MacroSpacetimeDeck(grid_shape)
        self.meta_observer = MetaObserverInvariantCore(invariant_target_coherence=0.75)
        self.step_count = 0
        self.constraint_stiffness = 1.0

        # Phase-State Memory (Ice, Water, Gas)
        self.memory_map = PhaseStateMemoryMap(
            ice_lattice=np.zeros(grid_shape, dtype=np.float64),
            water_flow=np.zeros(grid_shape, dtype=np.float64),
            gas_cloud=np.zeros(grid_shape, dtype=np.float64),
            active_phase=MemoryPhaseMode.WATER,
            total_memory_bytes=0
        )

    def transition_memory_phase(
        self,
        target_mode: str,
        teleological_purpose: str = "optimization"
    ) -> PhaseStateMemoryMap:
        """
        Executes physical-topological phase-state transition of internal memory blocks
        between Ice (crystalline past), Water (fluid thought), and Gas (probabilistic future).
        Controlled by top-level teleological purpose without disk swapping.
        """
        self.memory_map.active_phase = target_mode

        if target_mode == MemoryPhaseMode.ICE:
            # Crystallize water flow into rigid ice lattice
            self.memory_map.ice_lattice = np.where(
                np.abs(self.memory_map.water_flow) > 0.1,
                np.sign(self.memory_map.water_flow) * 1.0,
                self.memory_map.ice_lattice
            )
            # Damped flow
            self.memory_map.water_flow *= 0.1
            self.memory_map.gas_cloud *= 0.05

        elif target_mode == MemoryPhaseMode.WATER:
            # Melt ice or condense gas into active thought flow
            refracted, s, c = self.static_prism.refract_signal(
                self.memory_map.ice_lattice + self.memory_map.gas_cloud,
                teleological_mode=teleological_purpose
            )
            if refracted.shape != self.grid_shape:
                refracted = np.resize(refracted, self.grid_shape)
            self.memory_map.water_flow = 0.5 * self.memory_map.water_flow + 0.5 * refracted

        elif target_mode == MemoryPhaseMode.GAS:
            # Evaporate active water flow into wide probabilistic future wavefield
            laplacian = (
                np.roll(self.memory_map.water_flow, 1, axis=0) + np.roll(self.memory_map.water_flow, -1, axis=0) +
                np.roll(self.memory_map.water_flow, 1, axis=1) + np.roll(self.memory_map.water_flow, -1, axis=1) -
                4 * self.memory_map.water_flow
            )
            self.memory_map.gas_cloud = np.abs(laplacian) * 2.0

        # Estimate footprint in bytes (< 100MB footprint requirement)
        bytes_used = (
            self.memory_map.ice_lattice.nbytes +
            self.memory_map.water_flow.nbytes +
            self.memory_map.gas_cloud.nbytes
        )
        self.memory_map.total_memory_bytes = bytes_used
        return self.memory_map

    def step(
        self,
        numerical_input: Optional[np.ndarray] = None,
        dt: float = 0.1,
        roi_scrub_delta: float = 0.0,
        teleological_mode: str = "optimization"
    ) -> PhaseTransitionState:
        """
        Executes one full cycle of the 4-stage phase transition pipeline.
        """
        self.step_count += 1

        # Stage 1: Substrate propagation
        if numerical_input is not None:
            sub_state = self.substrate.inject_input(numerical_input)
        else:
            sub_state = self.substrate.propagate(dt=dt)

        # Refract through Static Rotor Prism
        refracted_signal, sine_w, cos_w = self.static_prism.refract_signal(
            sub_state.raw_signal,
            teleological_mode=teleological_mode
        )

        # Update Active Memory (Water state)
        if refracted_signal.shape == self.grid_shape:
            self.memory_map.water_flow = 0.8 * self.memory_map.water_flow + 0.2 * refracted_signal
        else:
            resized_refracted = np.resize(refracted_signal, self.grid_shape)
            self.memory_map.water_flow = 0.8 * self.memory_map.water_flow + 0.2 * resized_refracted

        # Stage 2: Prismatic refraction & structural constraint
        prism_state = self.prismatic.refract_substrate(sub_state, constraint_stiffness=self.constraint_stiffness)

        # Stage 3: Macro Spacetime Deck & Scrubbing
        if abs(roi_scrub_delta) > 1e-8:
            self.deck.scrub_roi_phase(roi_scrub_delta)
        deck_state = self.deck.update_deck(prism_state, dt=dt)

        # Stage 4: Meta-Observation & Invariant Core Feedback
        meta_feedback = self.meta_observer.observe(prism_state, deck_state)

        # Apply adaptive feedback to constraint stiffness
        self.constraint_stiffness = float(np.clip(
            self.constraint_stiffness + meta_feedback.feedback_adjustment.get("constraint_stiffness", 0.0),
            0.1, 5.0
        ))

        # Memory footprint check
        self.transition_memory_phase(self.memory_map.active_phase, teleological_purpose=teleological_mode)

        return PhaseTransitionState(
            step_count=self.step_count,
            substrate=sub_state,
            prismatic=prism_state,
            deck=deck_state,
            memory_map=self.memory_map,
            meta_feedback=meta_feedback
        )
