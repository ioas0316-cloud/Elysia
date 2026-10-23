import torch
from torch.autograd import Function

try:
    import elysia_phase_cuda
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False


class ElysiaPhaseCouplingFunction(Function):
    @staticmethod
    def forward(ctx, slow_phase, fast_phase, slow_omega, fast_omega, metric,
                dt=0.001, K_slow=0.15, K_fast=0.30, M_mod=2.00, alpha_fb=0.08):
        if not HAS_CUDA_EXT or not slow_phase.is_cuda:
            # Vectorized PyTorch CPU/fallback implementation
            TWO_PI = 2.0 * torch.pi
            num_nodes = slow_phase.size(0)
            spatial_influence = torch.exp(-metric)

            slow_diff = slow_phase.unsqueeze(1) - slow_phase.unsqueeze(0)
            fast_diff = fast_phase.unsqueeze(1) - fast_phase.unsqueeze(0)

            sum_slow = torch.sum(spatial_influence * torch.sin(slow_diff), dim=1)
            sum_fast = torch.sum(spatial_influence * torch.sin(fast_diff), dim=1)

            top_down = M_mod * torch.cos(slow_phase)
            bottom_up = alpha_fb * torch.sin(fast_phase - slow_phase)

            d_slow = slow_omega + (K_slow * sum_slow) + bottom_up
            d_fast = fast_omega + top_down + (K_fast * sum_fast)

            slow_out = torch.remainder(slow_phase + d_slow * dt + TWO_PI, TWO_PI)
            fast_out = torch.remainder(fast_phase + d_fast * dt + TWO_PI, TWO_PI)

            ctx.save_for_backward(slow_phase, fast_phase, slow_omega, fast_omega, metric)
            ctx.params = (dt, K_slow, K_fast, M_mod, alpha_fb)
            return slow_out, fast_out

        slow_out = torch.empty_like(slow_phase)
        fast_out = torch.empty_like(fast_phase)

        elysia_phase_cuda.forward(
            slow_out, fast_out,
            slow_phase, fast_phase, slow_omega, fast_omega, metric,
            dt, K_slow, K_fast, M_mod, alpha_fb
        )

        ctx.save_for_backward(slow_phase, fast_phase, slow_omega, fast_omega, metric)
        ctx.params = (dt, K_slow, K_fast, M_mod, alpha_fb)

        return slow_out, fast_out

    @staticmethod
    def backward(ctx, grad_slow_out, grad_fast_out):
        slow_phase, fast_phase, slow_omega, fast_omega, metric = ctx.saved_tensors
        dt, K_slow, K_fast, M_mod, alpha_fb = ctx.params

        if not HAS_CUDA_EXT or not slow_phase.is_cuda:
            # Fallback PyTorch autograd gradient computation
            grad_slow_in = grad_slow_out * (1.0 - dt * alpha_fb * torch.cos(fast_phase - slow_phase)) \
                           - grad_fast_out * dt * M_mod * torch.sin(slow_phase)
            grad_fast_in = grad_fast_out * 1.0 + grad_slow_out * dt * alpha_fb * torch.cos(fast_phase - slow_phase)

            spatial_influence = torch.exp(-metric)
            slow_diff = slow_phase.unsqueeze(0) - slow_phase.unsqueeze(1)
            fast_diff = fast_phase.unsqueeze(0) - fast_phase.unsqueeze(1)

            grad_metric = -dt * spatial_influence * (
                grad_slow_out.unsqueeze(1) * K_slow * torch.sin(slow_diff) +
                grad_fast_out.unsqueeze(1) * K_fast * torch.sin(fast_diff)
            )

            grad_slow_omega = grad_slow_out * dt
            grad_fast_omega = grad_fast_out * dt

            return grad_slow_in, grad_fast_in, grad_slow_omega, grad_fast_omega, grad_metric, None, None, None, None, None

        grad_slow_in = torch.empty_like(slow_phase)
        grad_fast_in = torch.empty_like(fast_phase)
        grad_metric = torch.empty_like(metric)

        elysia_phase_cuda.backward(
            grad_slow_in, grad_fast_in, grad_metric,
            grad_slow_out.contiguous(), grad_fast_out.contiguous(),
            slow_phase, fast_phase, metric,
            dt, K_slow, K_fast, M_mod, alpha_fb
        )

        grad_slow_omega = grad_slow_out * dt
        grad_fast_omega = grad_fast_out * dt

        return grad_slow_in, grad_fast_in, grad_slow_omega, grad_fast_omega, grad_metric, None, None, None, None, None


class ElysiaPhaseCouplingLayer(torch.nn.Module):
    def __init__(self, dt=0.001, K_slow=0.15, K_fast=0.30, M_mod=2.00, alpha_fb=0.08):
        super().__init__()
        self.dt = dt
        self.K_slow = K_slow
        self.K_fast = K_fast
        self.M_mod = M_mod
        self.alpha_fb = alpha_fb

    def forward(self, slow_phase, fast_phase, slow_omega, fast_omega, metric):
        return ElysiaPhaseCouplingFunction.apply(
            slow_phase, fast_phase, slow_omega, fast_omega, metric,
            self.dt, self.K_slow, self.K_fast, self.M_mod, self.alpha_fb
        )
