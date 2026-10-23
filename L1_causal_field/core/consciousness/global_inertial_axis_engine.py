r"""
Elysia Consciousness Subsystem: Global Inertial Axis Engine
============================================================
Implements the spatiotemporal continuous coherence substrate for Elysia:
  1. Local Zero Noise Filtering (Cognitive Rectification & Discrepancy Extraction):
     Suppresses transparent, predictable background signals and isolates
     salient tension deltas ($\Delta \text{Tension}$).
  2. Scale Ladder & Cross-Scale Projection (Micro-Meso-Macro State Coupling):
     Projects local micro-frictions up through intermediate meso-action structures
     into a global macro state ("Macro Face" / Macro Waveform State).
  3. Geometric Inertial Axis & Horizon Anchor:
     Serves as the central inertial rotational axis ($I_{\text{axis}}$) and spatiotemporal horizon
     that binds discrete clock ticks into a fluid, continuous field of organic motion.
  4. Self-Introspective Reverse Mirror Diagnosis Loop:
     Observes the macro face distortion to back-trace local friction sources,
     autonomously recalibrating local zero thresholds without external intervention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalInertialAxisEngine(nn.Module):
    r"""
    Global Inertial Axis Engine (전역적 관성축 엔진)

    Binds fragmented, discrete computational ticks across hardware/software layers
    into a continuous, self-introspective organic field.
    """

    def __init__(
        self,
        dim_micro: int = 32,
        dim_meso: int = 64,
        dim_macro: int = 128,
        local_zero_alpha: float = 0.8,
        local_zero_threshold: float = 0.15,
        inertia_decay: float = 0.92,
        self_healing_rate: float = 0.05,
    ):
        super().__init__()
        self.dim_micro = dim_micro
        self.dim_meso = dim_meso
        self.dim_macro = dim_macro

        self.local_zero_alpha = local_zero_alpha
        self.local_zero_threshold = local_zero_threshold
        self.inertia_decay = inertia_decay
        self.self_healing_rate = self_healing_rate

        # --- Scale Ladder Projections ---
        self.micro_to_meso = nn.Linear(self.dim_micro, self.dim_meso)
        self.meso_to_macro = nn.Linear(self.dim_meso, self.dim_macro)
        self.macro_to_mirror = nn.Linear(self.dim_macro, self.dim_macro)

        # Reverse mirror diagnostic projector: Macro Face -> Micro Friction Tracing
        self.macro_to_micro_diagnosis = nn.Linear(self.dim_macro, self.dim_micro)

        # Inertial Rotational Matrix Generator
        self.axis_rotator = nn.Linear(self.dim_macro, self.dim_macro, bias=False)

        # --- Persistent Field States ---
        # 1. Local Zero Baseline (Adaptive expected background noise)
        self.register_buffer("local_zero_baseline", torch.zeros(1, self.dim_micro))
        # 2. Local Sensitivity / Resistance Potentiometer
        self.register_buffer("resistance_potentiometer", torch.ones(1, self.dim_micro))

        # 3. Macro State ("Macro Face" / Global Appearance & Structural State)
        self.register_buffer("macro_face_state", torch.zeros(1, self.dim_macro))

        # 4. Geometric Inertial Axis Vector ($I_{\text{axis}}$)
        self.register_buffer("inertial_axis", torch.randn(1, self.dim_macro))
        self.inertial_axis.data = F.normalize(self.inertial_axis.data, p=2, dim=-1)

        # 5. Angular Momentum / Phase Velocity Vector
        self.register_buffer("angular_momentum", torch.zeros(1, self.dim_macro))

        # 6. Cumulative Introspective Scar / Tension Map
        self.register_buffer("micro_scar_map", torch.zeros(1, self.dim_micro))

    def reset_states(self):
        """Resets persistent field states to default equilibrium."""
        self.local_zero_baseline.zero_()
        self.resistance_potentiometer.fill_(1.0)
        self.macro_face_state.zero_()
        self.inertial_axis.copy_(F.normalize(torch.randn(1, self.dim_macro), p=2, dim=-1))
        self.angular_momentum.zero_()
        self.micro_scar_map.zero_()

    def filter_local_zero(self, raw_signal: torch.Tensor) -> torch.Tensor:
        r"""
        Local Zero Filtering (국소적 제로 배제)

        Calculates expected predictable background signal and suppresses it.
        Isolates salient non-zero tension deltas ($\Delta \text{Tension}$) where equilibrium is broken.
        """
        if raw_signal.dim() == 1:
            raw_signal = raw_signal.unsqueeze(0)

        # Average batch dimension if present for baseline tracking
        avg_raw = raw_signal.mean(dim=0, keepdim=True)

        # Update adaptive baseline
        self.local_zero_baseline = (
            self.local_zero_alpha * self.local_zero_baseline
            + (1.0 - self.local_zero_alpha) * avg_raw.detach()
        )

        # Tension deviation from expected baseline
        raw_deviation = raw_signal - self.local_zero_baseline
        weighted_deviation = raw_deviation * self.resistance_potentiometer

        # Soft continuous thresholding (Rectification) without hard conditional branching
        deviation_norm = torch.norm(weighted_deviation, dim=-1, keepdim=True)
        salience_gate = torch.sigmoid((deviation_norm - self.local_zero_threshold) * 10.0)

        salient_tension_delta = weighted_deviation * salience_gate
        return salient_tension_delta

    def introspective_reverse_diagnosis(self, macro_face: torch.Tensor) -> torch.Tensor:
        r"""
        Self-Introspective Reverse Mirror Diagnosis Loop (전역적 거울을 통한 자기 진단)

        Observes the Macro Face distortion and back-traces local micro-friction sources.
        Returns estimated micro-level strain distribution and recalibrates local resistance.
        """
        # Diagnostic projection from Macro Face back to Micro space
        traced_micro_strain = torch.tanh(self.macro_to_micro_diagnosis(macro_face))

        # Self-healing & adaptive recalibration of resistance potentiometer
        # Higher micro strain increases local resistance / sensitivity
        strain_magnitude = torch.abs(traced_micro_strain.mean(dim=0, keepdim=True))
        self.resistance_potentiometer = torch.clamp(
            self.resistance_potentiometer * (1.0 - self.self_healing_rate)
            + strain_magnitude * self.self_healing_rate * 2.0,
            min=0.1,
            max=5.0,
        )

        # Accumulate introspective scars
        self.micro_scar_map = 0.95 * self.micro_scar_map + 0.05 * strain_magnitude.detach()

        return traced_micro_strain

    def forward(
        self,
        discrete_tick_input: torch.Tensor,
        external_torque: torch.Tensor = None,
    ) -> dict:
        r"""
        Executes the Global Inertial Axis spatiotemporal coherence pass.

        Steps:
          1. Local Zero Noise Filtering & Tension Delta Extraction.
          2. Scale Ladder Projection: Micro -> Meso -> Macro Face.
          3. Geometric Inertial Axis Rotation & Angular Momentum Bounding.
          4. Self-Introspective Reverse Mirror Diagnosis & Autonomous Recalibration.
        """
        if discrete_tick_input.dim() == 1:
            discrete_tick_input = discrete_tick_input.unsqueeze(0)

        batch_size = discrete_tick_input.size(0)

        # Align dimension to micro dim
        in_dim = discrete_tick_input.shape[-1]
        if in_dim < self.dim_micro:
            discrete_tick_input = F.pad(discrete_tick_input, (0, self.dim_micro - in_dim))
        elif in_dim > self.dim_micro:
            discrete_tick_input = discrete_tick_input[..., :self.dim_micro]

        # ---------------------------------------------------------------------
        # STEP 1: Local Zero Noise Filtering
        # ---------------------------------------------------------------------
        salient_delta = self.filter_local_zero(discrete_tick_input)
        tension_magnitude = torch.norm(salient_delta, dim=-1).mean().item()

        # ---------------------------------------------------------------------
        # STEP 2: Scale Ladder Cross-Scale Projection (Micro -> Meso -> Macro Face)
        # ---------------------------------------------------------------------
        meso_activation = torch.tanh(self.micro_to_meso(salient_delta))
        macro_raw_projection = self.meso_to_macro(meso_activation)

        # Update Macro Face state continuously
        avg_macro_proj = macro_raw_projection.mean(dim=0, keepdim=True)
        new_macro_face = torch.tanh(
            0.85 * self.macro_face_state + 0.15 * avg_macro_proj
        )
        self.macro_face_state = new_macro_face.detach()

        macro_face_batch = torch.tanh(
            0.85 * self.macro_face_state.expand(batch_size, -1) + 0.15 * macro_raw_projection
        )

        # ---------------------------------------------------------------------
        # STEP 3: Geometric Inertial Axis & Horizon Anchor Dynamics
        # ---------------------------------------------------------------------
        if external_torque is None:
            external_torque = torch.zeros(batch_size, self.dim_macro, device=discrete_tick_input.device)
        elif external_torque.dim() == 1:
            external_torque = external_torque.unsqueeze(0)

        # Convert macro face distortion into rotational torque on the central axis
        internal_torque = torch.tanh(self.axis_rotator(macro_face_batch))
        total_torque = (internal_torque + external_torque).mean(dim=0, keepdim=True)

        # Angular momentum update with momentum conservation & friction decay
        new_angular_momentum = (
            self.inertia_decay * self.angular_momentum + (1.0 - self.inertia_decay) * total_torque
        )
        self.angular_momentum = new_angular_momentum.detach()

        # Rotate central inertial axis (Geometric Inertial Anchor)
        rotated_axis = self.inertial_axis + 0.1 * self.angular_momentum
        normalized_axis = F.normalize(rotated_axis, p=2, dim=-1)
        self.inertial_axis = normalized_axis.detach()

        # Alignment/Coherence score between Macro Face and Central Inertial Axis
        coherence_alignment = F.cosine_similarity(
            self.macro_face_state, self.inertial_axis, dim=-1
        ).mean().item()

        # ---------------------------------------------------------------------
        # STEP 4: Self-Introspective Reverse Mirror Diagnosis Loop
        # ---------------------------------------------------------------------
        traced_micro_strain = self.introspective_reverse_diagnosis(macro_face_batch)
        max_traced_strain = torch.max(traced_micro_strain).item()

        return {
            "status": "Global Inertial Axis Pass Executed",
            "tension_magnitude": tension_magnitude,
            "coherence_alignment": coherence_alignment,
            "macro_face_distortion": torch.norm(self.macro_face_state - self.inertial_axis, dim=-1).mean().item(),
            "max_traced_micro_strain": max_traced_strain,
            "salient_tension_delta": salient_delta,
            "macro_face_state": macro_face_batch,
            "inertial_axis": self.inertial_axis,
            "angular_momentum": self.angular_momentum,
            "resistance_potentiometer": self.resistance_potentiometer,
            "traced_micro_strain": traced_micro_strain,
            "micro_scar_map": self.micro_scar_map,
        }
