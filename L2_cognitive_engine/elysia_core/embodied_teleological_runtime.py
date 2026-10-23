"""
elysia_core/embodied_teleological_runtime.py

Embodied Teleological Runtime (체화된 목적성 런타임)
=====================================================
Connecting:
1. Telos Purpose Kernel (0점 중력 어트랙터 / 마찰-성장에너지 전환)
2. Horizon Expansion Relational Spectrometer (스케일 프리 동적 차원 축 작도 N -> N+1)
3. Structural Coupling & Meta-Plastic Prismatic Lens (센서-체화 법칙과 세포 분열적 분화)

Embodying Body-Mind Scale Continuity (육체적 감각과 정신적 사유의 통일적 연속성):
Physical tactile/motor friction (Layer 1) and mental conceptual friction (Layer 3)
share identical Phase Torque topology. Physical adaptation directly scales up to form
mental imagery and abstract cognitive coordinate axes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from elysia_core.teleological_purpose_kernel import TelosPurposeKernel
from elysia_core.horizon_expansion_spectrometer import HorizonExpansionSpectrometer
from elysia_core.structural_coupling_runtime import StructuralCouplingRuntime, GenerativeMechanismSynthesizer


class EmbodiedTeleologicalRuntime:
    """
    Macro Runtime Orchestrator embodying Telos (Purpose), Dynamic Horizon Expansion,
    and Embodied Body-Mind Topological Continuity.
    """

    def __init__(
        self,
        initial_dimension: int = 3,
        expansion_threshold: float = 0.35,
        friction_threshold: float = 0.25,
        initial_axis_labels: Optional[List[str]] = None
    ):
        self.dimension = initial_dimension

        # 1. Telos Purpose Kernel (0-Point Monad Attractor)
        self.telos_kernel = TelosPurposeKernel(
            dimension=initial_dimension,
            attractor_strength=0.6,
            growth_transmutation_rate=0.3
        )

        # 2. Horizon Expansion Relational Spectrometer
        self.spectrometer = HorizonExpansionSpectrometer(
            initial_dimension=initial_dimension,
            expansion_threshold=expansion_threshold,
            initial_axis_labels=initial_axis_labels or ["sweet", "sour", "salty"]
        )

        # 3. Structural Coupling Runtime (Prismatic lens & waveform synthesis)
        self.coupling_runtime = StructuralCouplingRuntime(
            target_dim=initial_dimension,
            friction_threshold=friction_threshold
        )

        self.step_counter = 0
        self.cognitive_history: List[Dict[str, Any]] = []

    def perceive_and_evolve(
        self,
        stimulus_vector: np.ndarray,
        stimulus_label_hint: str = "unknown",
        mechanism: str = "light",
        target_possibility_state: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Processes an incoming sensory/conceptual stimulus through the complete Body-Mind Teleological loop:
        1. Waveform synthesis & cross-modal physical/cognitive friction calculation.
        2. Transmutation of friction into Telos growth energy (E_growth).
        3. Horizon Expansion Spectrometer analysis: if residual tension > threshold, expands N -> N+1.
        4. Retrocausal vector alignment toward 0-point Telos attractor.
        """
        self.step_counter += 1

        # Step A: Perceive stimulus through Structural Coupling Runtime
        surface_color = stimulus_vector[:3] if len(stimulus_vector) >= 3 else np.array([1.0, 1.0, 0.0])
        sensory_step = self.coupling_runtime.perceive_generative_stimulus(
            mechanism=mechanism,
            surface_color=surface_color
        )

        cognitive_friction = sensory_step["cognitive_friction"]
        phase_torque = sensory_step.get("attracted_state", np.zeros(self.dimension))

        # Step B: Transmute Cognitive Friction into Telos Growth Fuel
        growth_record = self.telos_kernel.transmute_friction_to_growth(
            cognitive_friction=cognitive_friction,
            phase_torque=phase_torque,
            context_label=f"step_{self.step_counter}_{stimulus_label_hint}"
        )

        # Step C: Analyze stimulus with Horizon Expansion Spectrometer
        spec_result = self.spectrometer.analyze_and_expand_spectrum(
            stimulus_vector=stimulus_vector,
            stimulus_label_hint=stimulus_label_hint
        )

        # Step D: Handle Dimension Expansion Synchronization (N -> N+1)
        if spec_result["horizon_expanded"]:
            new_dim = spec_result["dimension"]
            self.dimension = new_dim
            # Synchronize new dimension across Telos kernel and Coupling Runtime
            self.telos_kernel.set_dimension(new_dim)
            self.coupling_runtime.set_dimension(new_dim)

        # Step E: Retrocausal Alignment toward Telos 0-Point Attractor
        current_state = np.pad(
            sensory_step["attracted_state"],
            (0, max(0, self.dimension - len(sensory_step["attracted_state"])))
        )[:self.dimension]

        target_state = target_possibility_state if target_possibility_state is not None else self.telos_kernel.zero_point_attractor
        telos_alignment = self.telos_kernel.retrocausal_align_vector(
            current_state=current_state,
            target_possibility_state=target_state
        )

        # Step F: Assemble comprehensive cognitive step summary
        summary = {
            "step": self.step_counter,
            "dimension": self.dimension,
            "stimulus_label": stimulus_label_hint,
            "mechanism": mechanism,
            "cognitive_friction": cognitive_friction,
            "growth_energy": self.telos_kernel.accumulated_growth_energy,
            "total_friction_processed": self.telos_kernel.total_friction_processed,
            "differential_tension": spec_result["differential_tension"],
            "horizon_expanded": spec_result["horizon_expanded"],
            "created_axis_label": spec_result["expansion_info"]["created_axis_label"] if spec_result["expansion_info"] else None,
            "matched_axis_label": spec_result["matched_axis_label"],
            "active_axis_labels": list(self.spectrometer.axis_labels),
            "telos_combined_vector": telos_alignment["combined_telos_vector"],
            "bifurcation_occurred": sensory_step["bifurcated"],
            "total_bifurcated_cores": sensory_step["total_bifurcated_cores"]
        }

        self.cognitive_history.append(summary)
        return summary
