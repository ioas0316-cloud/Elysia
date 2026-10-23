import torch
import torch.nn as nn
from elysia_engine.core.cmplr_fallback import CliffordPhaseLockKernelFallback

try:
    from elysia_engine._C import cmplr as cmplr_cuda
    HAS_CUDA_EXTENSION = True
except ImportError:
    HAS_CUDA_EXTENSION = False


class CliffordPhaseLockKernel(nn.Module):
    """
    Clifford Multivector Phase-Lock Relaxation (CMPLR) Kernel.
    Auto-dispatches between high-performance CUDA C++ extension and Pure PyTorch CPU Fallback.
    """
    def __init__(self, Psi, row_ptr, col_ind, K_tensors, eps=1e-8):
        super().__init__()
        self.Psi = nn.Parameter(Psi) if not isinstance(Psi, nn.Parameter) else Psi
        self.register_buffer("row_ptr", row_ptr)
        self.register_buffer("col_ind", col_ind)
        self.K_tensors = nn.Parameter(K_tensors) if not isinstance(K_tensors, nn.Parameter) else K_tensors
        self.eps = eps

        # Internal CPU fallback instance
        self.fallback = CliffordPhaseLockKernelFallback(self.Psi, self.row_ptr, self.col_ind, self.K_tensors, eps=self.eps)

    def step(self, dt=0.01):
        """Executes single phase-lock relaxation step with auto-dispatching."""
        if self.Psi.is_cuda and HAS_CUDA_EXTENSION:
            cmplr_cuda.step(self.Psi, self.row_ptr, self.col_ind, self.K_tensors, dt)
            return self.Psi
        else:
            return self.fallback.step(dt=dt)

    def forward(self, dt=0.01):
        return self.step(dt=dt)
