import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any, Optional

# Attempt loading native C++ Extension module
try:
    import autonomic_tension_cpp
except ImportError:
    autonomic_tension_cpp = None


class PyFallbackAutonomicTensionController:
    """Python fallback implementation of AutonomicTensionController if C++ module is absent."""
    def __init__(self, kp: float, ki: float, kd: float, th: float):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.threshold = th
        self.integral_error = 0.0
        self.prev_error = 0.0

    def step(
        self,
        micro_phase_error: torch.Tensor,
        wave_field: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, float, float]:
        current_error = torch.norm(micro_phase_error).item()
        self.integral_error += current_error * dt
        derivative_error = (current_error - self.prev_error) / dt if dt > 0 else 0.0
        u_pid = (self.Kp * current_error) + (self.Ki * self.integral_error) + (self.Kd * derivative_error)
        self.prev_error = current_error

        sympathetic_weight = 1.0 / (1.0 + math.exp(-max(min(u_pid - self.threshold, 50.0), -50.0)))
        parasympathetic_weight = 1.0 - sympathetic_weight

        suppressed_field = wave_field * (1.0 - sympathetic_weight * 0.85)

        scale_diff = torch.zeros_like(wave_field)
        if wave_field.requires_grad and wave_field.grad_fn is not None:
            try:
                grads = torch.autograd.grad([wave_field.sum()], [wave_field], create_graph=True, retain_graph=True)
                if grads and grads[0] is not None:
                    scale_diff = grads[0]
            except Exception:
                scale_diff = torch.zeros_like(wave_field)

        relaxed_field = suppressed_field + (parasympathetic_weight * 0.1 * scale_diff)
        return relaxed_field, sympathetic_weight, parasympathetic_weight


class MacroValueManifold(nn.Module):
    """거시 가치 지형 V(S_max): Macro scale objective & belief value manifold."""
    def __init__(self, channels: int):
        super().__init__()
        self.value_network = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_network(state)


class HJBTopDownAttentionPipeline(nn.Module):
    """HJB 역전파 기반 하향식 어텐션 파이프라인 (Top-Down Attention Pipeline)."""
    def __init__(self, channels: int, temp: float = 0.5):
        super().__init__()
        self.channels = channels
        self.tau_temp = temp
        self.sigma_inv = nn.Parameter(torch.eye(channels))

    def forward(
        self,
        macro_manifold: MacroValueManifold,
        current_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. HJB Top-down value gradient computation
        state_clone = current_state.clone().detach().requires_grad_(True)
        value_potential = macro_manifold(state_clone)

        value_gradient = torch.autograd.grad(
            outputs=value_potential.sum(),
            inputs=state_clone,
            create_graph=True,
            retain_graph=True
        )[0]

        # 2. Optimal attention control amount u* = -\Sigma^{-1} \nabla_\Psi V
        u_star = -torch.einsum('bchw,cd->bdhw', value_gradient, self.sigma_inv)

        # 3. Sensory observation mask formulation & 4D Spacetime Sculpting
        attention_energy = torch.norm(u_star, dim=1, keepdim=True)
        B, C, H, W = attention_energy.shape
        flat_energy = attention_energy.view(B, -1)
        sensory_mask = F.softmax(flat_energy / self.tau_temp, dim=-1).view(B, 1, H, W)

        sculpted_observation = current_state * sensory_mask
        return sculpted_observation, sensory_mask


class BiologicalCognitiveEngine(nn.Module):
    """자율신경 장력 제어와 HJB 어텐션이 결합된 통합 생체 인지 엔진 (Biological Cognitive Engine)."""
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width

        # Subsystems
        self.macro_manifold = MacroValueManifold(channels)
        self.attention_pipeline = HJBTopDownAttentionPipeline(channels)

        # Autonomic controller setup with C++ extension or PyFallback
        if autonomic_tension_cpp is not None and hasattr(autonomic_tension_cpp, "AutonomicTensionController"):
            self.autonomic_controller = autonomic_tension_cpp.AutonomicTensionController(1.2, 0.1, 0.4, 0.5)
        else:
            self.autonomic_controller = PyFallbackAutonomicTensionController(1.2, 0.1, 0.4, 0.5)

    def step(
        self,
        raw_sensory_wave: torch.Tensor,
        prev_state: torch.Tensor,
        dt: float = 0.016
    ) -> Dict[str, Any]:
        # Step A: HJB Top-down attention sculpting
        sculpted_wave, sensory_mask = self.attention_pipeline(self.macro_manifold, raw_sensory_wave)

        # Step B: Micro phase error e(t)
        micro_phase_error = sculpted_wave - prev_state

        # Step C: Autonomic tension control step (Sympathetic/Parasympathetic switching)
        relaxed_wave, symp_weight, parasymp_weight = self.autonomic_controller.step(
            micro_phase_error,
            sculpted_wave,
            dt
        )

        # Step D: Parasympathetic relaxation mode value manifold re-perception/integration
        if parasymp_weight > 0.5:
            manifold_loss = F.mse_loss(self.macro_manifold(relaxed_wave), relaxed_wave.detach())
            manifold_loss.backward()

        return {
            "next_state": relaxed_wave.detach(),
            "sensory_mask": sensory_mask,
            "sympathetic_weight": symp_weight,
            "parasympathetic_weight": parasymp_weight,
            "phase_error_norm": torch.norm(micro_phase_error).item()
        }
