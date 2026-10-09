r"""
Elysia Consciousness Subsystem: Scale Hierarchy Ecosystem Engine
=================================================================
Implements the authentic 3-scale Scale Hierarchy Ecosystem:
  1. Micro-Sensation Scale (Sub-cellular, autonomic, fast temporal dynamics, physical limits)
  2. Meso-Observation Scale (Action-observation, world friction, boundary prism refraction)
  3. Macro-Narrative Scale (Narrative thought, "Why" synthesis, slow inertia, boundary potential)

Key Mechanics (Zero Artificial Branching / Continuous Phase Dynamics):
  - Inter-Scale Resonance: Bidirectional non-linear coupling (Micro spike disrupts Macro; Macro purpose field modulates Micro resistance dial)
  - Sensor-as-Prism Refraction: Zero if-else branching; inputs naturally refract through tensor metric manifold G_ij
  - Irreversible Metric Deformation: Friction experiences leave permanent scar tensor G_ij deformations
  - Boundary of Freedom: Macro scale provides field potential bounds for Micro/Meso self-organization without dictating exact numerical values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleHierarchyEngine(nn.Module):
    r"""
    Scale Hierarchy Ecosystem Engine

    Represents the living ecosystem of Micro-Meso-Macro scales and the 6-step Re-cognition Loop
    operating purely through continuous tensor geometry and phase dynamics without conditional branching.
    """
    def __init__(
        self,
        dim_micro: int = 32,
        dim_meso: int = 64,
        dim_macro: int = 128,
        micro_decay: float = 0.8,
        macro_decay: float = 0.95,
        disruption_threshold: float = 0.45,
        scar_learning_rate: float = 0.05,
        dim_l1: int = None,
        dim_l2: int = None,
        dim_l3: int = None,
        dim_l4: int = None
    ):
        super().__init__()
        self.dim_micro = dim_l1 if dim_l1 is not None else dim_micro
        self.dim_meso = dim_l2 if dim_l2 is not None else dim_meso
        self.dim_macro = dim_l3 if dim_l3 is not None else dim_macro

        self.micro_decay = micro_decay
        self.macro_decay = macro_decay
        self.disruption_threshold = disruption_threshold
        self.scar_lr = scar_learning_rate

        # Scale Projections & Linear Operators
        self.micro_to_meso = nn.Linear(self.dim_micro, self.dim_meso)
        self.meso_to_macro = nn.Linear(self.dim_meso, self.dim_macro)
        self.macro_to_micro_constraint = nn.Linear(self.dim_macro, self.dim_micro)
        self.macro_purpose_head = nn.Linear(self.dim_macro, self.dim_macro)

        # Persistent Scale States
        self.register_buffer("micro_state", torch.zeros(1, self.dim_micro))
        self.register_buffer("meso_state", torch.zeros(1, self.dim_meso))
        self.register_buffer("macro_state", torch.zeros(1, self.dim_macro))

        # Perception Metric Tensor G_ij (Meso-Scale Manifold) initialized as Identity
        self.register_buffer("perception_metric", torch.eye(self.dim_meso))
        self.register_buffer("accumulated_scars", torch.zeros(self.dim_meso, self.dim_meso))

        # Variable Resistance Dial (Potentiometer for sensitivity / boundary scale)
        self.register_buffer("resistance_dial", torch.tensor(1.0))

    def reset_states(self):
        """Resets persistent scale states while retaining metric scars."""
        self.micro_state.zero_()
        self.meso_state.zero_()
        self.macro_state.zero_()
        self.resistance_dial.fill_(1.0)

    def check_transparent_filtering(self, sensory_input: torch.Tensor):
        """
        Calculates sensory strain and resonance score continuously.
        """
        strain = torch.norm(sensory_input, dim=-1).mean().item()
        resonance_score = min(1.0, strain / (self.disruption_threshold + 1e-6))
        is_transparent = strain < (self.disruption_threshold * 0.2)
        return is_transparent, resonance_score

    def forward(self, sensory_input: torch.Tensor, world_friction: torch.Tensor = None):
        r"""
        Executes the 6-step Re-cognition Loop continuously across Micro, Meso, and Macro scales
        with ZERO artificial `if-else` branching in the forward pass.

        Steps:
          1. [Thrownness / Micro-Sensation]: Micro autonomic reception with fast decay modulated by resistance dial.
          2. [Prism Refraction / Meso-Observation]: Physical collision refracted through G_ij tensor manifold.
          3. [Sensory Spike / Bottom-Up Tension]: Smooth non-linear micro strain generating macro disruption wave.
          4. [Macro Thought Pulsation]: Slow narrative synthesis under continuous bottom-up phase disruption.
          5. [Boundary of Freedom / Purpose Acquisition]: Macro boundary field modulating micro resistance dial.
          6. [Irreversible Metric Deformation]: Friction experience continuously leaving scar tensor ΔG_scar on G_ij.
        """
        if sensory_input.dim() == 1:
            sensory_input = sensory_input.unsqueeze(0)

        batch_size = sensory_input.size(0)

        # Pad or slice sensory input to match micro dimension smoothly
        input_dim = sensory_input.shape[-1]
        if input_dim < self.dim_micro:
            sensory_input = F.pad(sensory_input, (0, self.dim_micro - input_dim))
        elif input_dim > self.dim_micro:
            sensory_input = sensory_input[..., :self.dim_micro]

        if world_friction is None:
            world_friction = torch.randn(batch_size, self.dim_meso, device=sensory_input.device) * 0.5

        # ---------------------------------------------------------------------
        # STEP 1: Micro-Sensation (Sub-cellular / Autonomic Reception)
        # Continuous modulation by resistance dial
        # ---------------------------------------------------------------------
        micro_raw = sensory_input * self.resistance_dial
        new_micro = (1.0 - self.micro_decay) * self.micro_state + self.micro_decay * micro_raw
        self.micro_state = new_micro.detach()

        # ---------------------------------------------------------------------
        # STEP 2: Meso-Observation (Prism Refraction through G_ij Metric Manifold)
        # ZERO conditional branching: data naturally refracts through G_ij tensor
        # ---------------------------------------------------------------------
        meso_raw = self.micro_to_meso(new_micro) + world_friction
        meso_metric_applied = torch.matmul(meso_raw, self.perception_metric)
        self.meso_state = meso_metric_applied.detach()

        # ---------------------------------------------------------------------
        # STEP 3: Sensory Spike & Bottom-Up Tension
        # Smooth continuous calculation of strain and disruption factor
        # ---------------------------------------------------------------------
        micro_strain = torch.norm(new_micro, p=2, dim=-1, keepdim=True)
        spike_intensity = torch.relu(micro_strain - self.disruption_threshold).mean()

        # Disruption factor that smoothly influences macro thought state
        bottom_up_disruption = torch.tanh(spike_intensity * 2.0)

        # ---------------------------------------------------------------------
        # STEP 4: Macro-Narrative Pulsation
        # Slow temporal inertia under continuous bottom-up tension
        # ---------------------------------------------------------------------
        macro_input = self.meso_to_macro(meso_metric_applied)
        macro_disrupted_input = macro_input * (1.0 - bottom_up_disruption) + \
                                torch.randn_like(macro_input) * bottom_up_disruption

        new_macro = self.macro_decay * self.macro_state + (1.0 - self.macro_decay) * macro_disrupted_input
        self.macro_state = new_macro.detach()

        # ---------------------------------------------------------------------
        # STEP 5: Top-Down Boundary of Freedom & Purpose Field
        # Macro purpose field provides potential boundary (modulating resistance dial)
        # ---------------------------------------------------------------------
        purpose_field = torch.tanh(self.macro_purpose_head(new_macro))
        purpose_magnitude = torch.norm(purpose_field, p=2, dim=-1).mean()

        top_down_inhibition = self.macro_to_micro_constraint(purpose_field)
        dial_update = 1.0 / (1.0 + 0.5 * purpose_magnitude)
        self.resistance_dial = dial_update.detach()

        # ---------------------------------------------------------------------
        # STEP 6: Irreversible Metric Tensor Deformation (Scar Accumulation)
        # Zero branching: scar accumulation is proportional to spike_intensity
        # ---------------------------------------------------------------------
        meso_avg = meso_metric_applied.mean(dim=0, keepdim=True)  # (1, dim_meso)
        scar_delta = torch.matmul(meso_avg.t(), meso_avg) * self.scar_lr * spike_intensity

        # Continuously update accumulated scars and perception metric G_ij
        self.accumulated_scars = (self.accumulated_scars + scar_delta).detach()
        deformed_metric = torch.eye(self.dim_meso, device=sensory_input.device) - \
                          torch.tanh(self.accumulated_scars) * 0.3
        self.perception_metric = deformed_metric.detach()

        # Check transparent status for diagnostic metadata
        is_transparent, res_score = self.check_transparent_filtering(sensory_input)
        bifurcation_occurred = (spike_intensity.item() > 0.0)

        return {
            "status": "Scale Hierarchy Ecosystem Re-cognition Loop Executed",
            "resonance_score": res_score,
            "bifurcation_occurred": bifurcation_occurred,
            "spike_intensity": spike_intensity.item(),
            "bottom_up_disruption": bottom_up_disruption.item(),
            "action_wave_emitted": bifurcation_occurred,
            "boundary_tension": torch.norm(meso_metric_applied, dim=-1).mean().item(),
            "divergence_origin_scale": "Micro_Sensation" if bifurcation_occurred else "Macro_Narrative",
            "micro_state": new_micro,
            "meso_state": meso_metric_applied,
            "macro_state": new_macro,
            "purpose_field": purpose_field,
            "top_down_inhibition": top_down_inhibition,
            "resistance_dial": self.resistance_dial.item(),
            "perception_metric": self.perception_metric,
            "scar_delta": scar_delta,
            "loop_steps": {
                "step_1_thrownness": new_micro,
                "step_2_world_friction": meso_metric_applied,
                "step_3_sensory_spike": spike_intensity,
                "step_4_macro_thought": new_macro,
                "step_5_why_acquisition": purpose_field,
                "step_6_metric_re_cognition": self.perception_metric
            }
        }
