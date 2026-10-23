import numpy as np
from typing import Tuple, Union, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class ElysiaFluxWeakeningController:
    """
    Field Oriented Control (FOC) Flux-Weakening Controller for AI Engine Memory Protection.
    Under GPU/VRAM pressure, dynamically scales D-axis (Core Context Vector) to prevent OOM
    while maintaining Q-axis (Task Momentum / Generation Speed) at 100%.
    """
    def __init__(
        self,
        vram_limit_mb: float = 3072.0,
        safety_margin: float = 0.15,
        lambda_gain: float = 0.005,
        min_flux_floor: float = 0.15
    ):
        self.vmax = vram_limit_mb
        self.safety_margin = safety_margin
        self.v_margin = vram_limit_mb * (1.0 - safety_margin)
        self.lambda_k = lambda_gain
        self.min_flux_floor = min_flux_floor

    def compute_flux_weakening_factor(self, current_vram_mb: float) -> float:
        """
        Calculates D-axis flux weakening attenuation factor (gamma_d).
        """
        if current_vram_mb <= self.v_margin:
            return 1.0  # Normal state: 100% context flux preserved

        v_err = current_vram_mb - self.v_margin
        gamma_d = float(np.exp(-self.lambda_k * v_err))
        return max(gamma_d, self.min_flux_floor)

    def apply_foc_scaling(
        self,
        d_axis_context: Union["torch.Tensor", np.ndarray],
        q_axis_momentum: Union["torch.Tensor", np.ndarray],
        current_vram_mb: float
    ) -> Tuple[Union["torch.Tensor", np.ndarray], Union["torch.Tensor", np.ndarray]]:
        """
        Applies dynamic flux weakening scaling to D-Q isolated vectors.
        """
        gamma_d = self.compute_flux_weakening_factor(current_vram_mb)

        # D-axis (Core Context) receives flux attenuation
        compressed_d_context = d_axis_context * gamma_d

        # Q-axis (Task Momentum) is 100% preserved
        preserved_q_momentum = q_axis_momentum

        return compressed_d_context, preserved_q_momentum

    @staticmethod
    def clarke_park_transform(
        input_abc: Union["torch.Tensor", np.ndarray],
        angles: Union["torch.Tensor", np.ndarray],
        gamma_d: float = 1.0
    ) -> Union["torch.Tensor", np.ndarray]:
        """
        3-phase (a, b, c) to D-Q-0 Clifford Rotor transformation.
        """
        SQRT3_INV_2 = 0.86602540378
        ONE_THIRD = 0.33333333333

        if HAS_TORCH and isinstance(input_abc, torch.Tensor):
            a = input_abc[..., 0]
            b = input_abc[..., 1]
            c = input_abc[..., 2]

            v_alpha = a - 0.5 * (b + c)
            v_beta = SQRT3_INV_2 * (b - c)
            v_zero = ONE_THIRD * (a + b + c)

            sin_t = torch.sin(angles)
            cos_t = torch.cos(angles)

            v_d = (v_alpha * cos_t + v_beta * sin_t) * gamma_d
            v_q = -v_alpha * sin_t + v_beta * cos_t

            return torch.stack([v_d, v_q, v_zero], dim=-1)
        else:
            a = input_abc[..., 0]
            b = input_abc[..., 1]
            c = input_abc[..., 2]

            v_alpha = a - 0.5 * (b + c)
            v_beta = SQRT3_INV_2 * (b - c)
            v_zero = ONE_THIRD * (a + b + c)

            sin_t = np.sin(angles)
            cos_t = np.cos(angles)

            v_d = (v_alpha * cos_t + v_beta * sin_t) * gamma_d
            v_q = -v_alpha * sin_t + v_beta * cos_t

            return np.stack([v_d, v_q, v_zero], axis=-1)

    @staticmethod
    def inverse_park_clarke_transform(
        dq0: Union["torch.Tensor", np.ndarray],
        angles: Union["torch.Tensor", np.ndarray]
    ) -> Union["torch.Tensor", np.ndarray]:
        """
        D-Q-0 to 3-phase (a, b, c) inverse Park-Clarke transformation.
        Inverse scaling factor 2/3 converts 1.5 * alpha back to original 3-phase amplitude.
        """
        TWO_THIRDS = 0.6666666666666666
        SQRT3_INV_3 = 0.5773502691896257

        if HAS_TORCH and isinstance(dq0, torch.Tensor):
            v_d = dq0[..., 0]
            v_q = dq0[..., 1]
            v_zero = dq0[..., 2]

            sin_t = torch.sin(angles)
            cos_t = torch.cos(angles)

            v_alpha = v_d * cos_t - v_q * sin_t
            v_beta  = v_d * sin_t + v_q * cos_t

            a = TWO_THIRDS * v_alpha + v_zero
            b = -1.0/3.0 * v_alpha + SQRT3_INV_3 * v_beta + v_zero
            c = -1.0/3.0 * v_alpha - SQRT3_INV_3 * v_beta + v_zero

            return torch.stack([a, b, c], dim=-1)
        else:
            v_d = dq0[..., 0]
            v_q = dq0[..., 1]
            v_zero = dq0[..., 2]

            sin_t = np.sin(angles)
            cos_t = np.cos(angles)

            v_alpha = v_d * cos_t - v_q * sin_t
            v_beta  = v_d * sin_t + v_q * cos_t

            a = TWO_THIRDS * v_alpha + v_zero
            b = -1.0/3.0 * v_alpha + SQRT3_INV_3 * v_beta + v_zero
            c = -1.0/3.0 * v_alpha - SQRT3_INV_3 * v_beta + v_zero

            return np.stack([a, b, c], axis=-1)


class ElysiaPIClosedLoopController:
    """
    FOC Closed-Loop Proportional-Integral (PI) Cognitive Controller.
    Corrects cognitive latent momentum Q towards target goal Q* without heavy backpropagation.
    """
    def __init__(self, kp: float = 2.0, ki: float = 0.1, dt: float = 0.01, max_integral: float = 20.0):
        self.kp = kp
        self.ki = ki
        self.dt = dt
        self.max_integral = max_integral
        self.integral_error = 0.0

    def step(
        self,
        target_q: Union[float, np.ndarray],
        current_q: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Calculates error e = Q* - Q and updates cognitive control voltage/momentum.
        """
        error = target_q - current_q
        self.integral_error += error * self.dt
        # Anti-windup clamping
        if self.max_integral > 0:
            self.integral_error = np.clip(self.integral_error, -self.max_integral, self.max_integral)

        control_signal = self.kp * error + self.ki * self.integral_error
        updated_q = current_q + control_signal * self.dt
        return updated_q, error
