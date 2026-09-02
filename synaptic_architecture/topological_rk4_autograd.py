import torch
import torch.nn as nn
from torch.autograd import Function

# Try importing the compiled CUDA C++ extension if available
try:
    import topological_rk4_cuda
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False


def pytorch_rk4_step(s, W, stress_grad, W0, dt, tau_s, tau_w, k_elastic, lambda_w):
    """
    Pure PyTorch fallback implementation for 2-timescale RK4 integration.
    Performs step computation without illegal in-place modifications during autograd.
    """
    def compute_grads(curr_s, curr_w):
        # 1. Fast dynamics gradient: ds/dt = (W*s - s - k_elastic * stress_grad) / tau_s
        Ws = torch.matmul(curr_w, curr_s)
        grad_s = (Ws - curr_s - k_elastic * stress_grad) / tau_s

        # 2. Slow dynamics gradient: dW/dt = (0.5 * s*s^T - lambda*(W - W0) + Torque) / tau_w
        hebbian = 0.5 * torch.outer(curr_s, curr_s)
        decay = lambda_w * (curr_w - W0)
        torque = -k_elastic * torch.outer(stress_grad, curr_s)
        grad_w = (hebbian - decay + torque) / tau_w

        return grad_s, grad_w

    # RK4 Integration stages
    k1_s, k1_w = compute_grads(s, W)

    temp_s1 = s + 0.5 * dt * k1_s
    temp_w1 = W + 0.5 * dt * k1_w
    k2_s, k2_w = compute_grads(temp_s1, temp_w1)

    temp_s2 = s + 0.5 * dt * k2_s
    temp_w2 = W + 0.5 * dt * k2_w
    k3_s, k3_w = compute_grads(temp_s2, temp_w2)

    temp_s3 = s + dt * k3_s
    temp_w3 = W + dt * k3_w
    k4_s, k4_w = compute_grads(temp_s3, temp_w3)

    s_next = s + (dt / 6.0) * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s)
    W_next = W + (dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)

    return s_next, W_next


class TopologicalRK4Function(Function):
    @staticmethod
    def forward(ctx, s, W, stress_grad, W0, dt, tau_s, tau_w, k_elastic, lambda_w):
        """
        [Forward Pass]
        Executes 2-timescale RK4 integration via CUDA C++ kernel or PyTorch fallback.
        Creates tensor clones to preserve autograd graph safety.
        """
        s_next = s.clone()
        W_next = W.clone()

        if s.is_cuda and HAS_CUDA_EXT:
            topological_rk4_cuda.step(
                s_next, W_next, stress_grad, W0,
                float(dt), float(tau_s), float(tau_w), float(k_elastic), float(lambda_w)
            )
        else:
            s_next, W_next = pytorch_rk4_step(
                s, W, stress_grad, W0,
                float(dt), float(tau_s), float(tau_w), float(k_elastic), float(lambda_w)
            )

        ctx.save_for_backward(s, W, stress_grad, W0)
        ctx.dt = dt
        ctx.tau_s = tau_s
        ctx.tau_w = tau_w
        ctx.k_elastic = k_elastic
        ctx.lambda_w = lambda_w

        return s_next, W_next

    @staticmethod
    def backward(ctx, grad_s_next, grad_W_next):
        """
        [Backward Pass]
        Calculates 2-timescale Vector-Jacobian Product (VJP) using adjoint sensitivity.
        """
        s, W, stress_grad, W0 = ctx.saved_tensors
        dt = ctx.dt
        tau_s = ctx.tau_s
        tau_w = ctx.tau_w
        k_elastic = ctx.k_elastic
        lambda_w = ctx.lambda_w

        with torch.enable_grad():
            s_in = s.detach().requires_grad_(True)
            W_in = W.detach().requires_grad_(True)
            sg_in = stress_grad.detach().requires_grad_(True)
            W0_in = W0.detach().requires_grad_(True)

            # 1. Fast Dynamics: ds/dt = (W*s - s - k*stress_grad) / tau_s
            f_s = (torch.matmul(W_in, s_in) - s_in - k_elastic * sg_in) / tau_s

            # 2. Slow Dynamics: dW/dt = (0.5*s*s^T - lambda*(W - W0) - k*stress_grad*s^T) / tau_w
            hebbian = 0.5 * torch.outer(s_in, s_in)
            decay = lambda_w * (W_in - W0_in)
            torque = -k_elastic * torch.outer(sg_in, s_in)
            f_w = (hebbian - decay + torque) / tau_w

            # Single-step RK4 approximation VJP
            s_approx = s_in + dt * f_s
            W_approx = W_in + dt * f_w

            grads = torch.autograd.grad(
                outputs=(s_approx, W_approx),
                inputs=(s_in, W_in, sg_in, W0_in),
                grad_outputs=(grad_s_next, grad_W_next),
                allow_unused=True,
                retain_graph=False
            )

        grad_s, grad_W, grad_stress_grad, grad_W0 = grads

        return grad_s, grad_W, grad_stress_grad, grad_W0, None, None, None, None, None


class TopologicalRK4Layer(nn.Module):
    """
    PyTorch nn.Module layer wrapping TopologicalRK4Function
    """
    def __init__(self, dt=0.01, tau_s=0.1, tau_w=10.0, k_elastic=0.2, lambda_w=0.01):
        super().__init__()
        self.dt = dt
        self.tau_s = tau_s
        self.tau_w = tau_w
        self.k_elastic = k_elastic
        self.lambda_w = lambda_w

    def forward(self, s, W, stress_grad, W0):
        return TopologicalRK4Function.apply(
            s, W, stress_grad, W0,
            self.dt, self.tau_s, self.tau_w, self.k_elastic, self.lambda_w
        )
