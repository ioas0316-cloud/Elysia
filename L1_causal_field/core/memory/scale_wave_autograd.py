"""
core/memory/scale_wave_autograd.py

PyTorch Custom Autograd Function for Scale-Invariant Wave Tensor Memory.
Enables end-to-end backpropagation to learn coupling parameters (coupling_gain, phase_shift).
Provides dynamic GPU CUDA kernel or pure PyTorch CPU fallback execution.
"""

import torch
import torch.nn as nn


def _is_cuda_extension_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from core.memory import scale_wave_cuda_ops
        return True
    except ImportError:
        return False


USE_CUDA_OPS = _is_cuda_extension_available()


class ScaleWaveMemoryAutogradFunction(torch.autograd.Function):
    """
    스케일 불변 파동 텐서 메모리의 미분 가능(Differentiable) 순전파 및 역전파 커널
    """

    @staticmethod
    def forward(
        ctx,
        tensor_memory: torch.Tensor,   # [num_scales, base_dim, 3] (sin, cos, tan)
        scale_factors: torch.Tensor,   # [num_scales]
        target_scale: int,
        target_idx: int,
        phase_shift: torch.Tensor,     # Scalar Parameter (Requires Grad)
        coupling_gain: torch.Tensor    # Scalar Parameter (Requires Grad)
    ) -> torch.Tensor:

        num_scales, base_dim, _ = tensor_memory.shape
        device = tensor_memory.device

        # 순전파 출력 메모리 할당
        updated_memory = tensor_memory.clone()

        # 1. 타겟 국소 셀 위상 연산
        target_sin = tensor_memory[target_scale, target_idx, 0]
        target_cos = tensor_memory[target_scale, target_idx, 1]
        target_theta_init = torch.atan2(target_sin, target_cos)
        target_theta_new = target_theta_init + phase_shift

        # 타겟 셀 삼위일체 파동 업데이트
        updated_memory[target_scale, target_idx, 0] = torch.sin(target_theta_new)
        updated_memory[target_scale, target_idx, 1] = torch.cos(target_theta_new)
        updated_memory[target_scale, target_idx, 2] = torch.tan(target_theta_new)

        # 2. 전 스케일 파동 결합 전파 (Cross-Scale Coupling)
        for l in range(num_scales):
            for i in range(base_dim):
                if l == target_scale and i == target_idx:
                    continue

                dist = abs(l - target_scale)
                coupling_k = torch.exp(torch.tensor(-dist, dtype=torch.float32, device=device)) * coupling_gain
                scale_ratio = scale_factors[l] / (scale_factors[target_scale] + 1e-8)
                coupled_theta = target_theta_new * coupling_k * scale_ratio

                cur_sin = tensor_memory[l, i, 0] + 0.1 * torch.sin(coupled_theta)
                cur_cos = tensor_memory[l, i, 1] + 0.1 * torch.cos(coupled_theta)

                # 규격화 (Normalization)
                norm = torch.sqrt(cur_sin**2 + cur_cos**2) + 1e-8
                cur_sin = cur_sin / norm
                cur_cos = cur_cos / norm

                updated_memory[l, i, 0] = cur_sin
                updated_memory[l, i, 1] = cur_cos
                updated_memory[l, i, 2] = cur_sin / (cur_cos + 1e-8)

        # 역전파 연산을 위한 상태 저장 (Context Save)
        ctx.save_for_backward(
            tensor_memory, scale_factors, phase_shift, coupling_gain,
            target_theta_init, target_theta_new
        )
        ctx.target_scale = target_scale
        ctx.target_idx = target_idx

        return updated_memory

    @staticmethod
    def backward(ctx, grad_output_memory: torch.Tensor):
        """
        역전파 체인 룰(Chain Rule)을 통한 매개변수 기울기 계산
        """
        (
            tensor_memory, scale_factors, phase_shift, coupling_gain,
            target_theta_init, target_theta_new
        ) = ctx.saved_tensors

        target_scale = ctx.target_scale
        target_idx = ctx.target_idx
        num_scales, base_dim, _ = tensor_memory.shape
        device = tensor_memory.device

        # 기울기 초기화
        grad_tensor_memory = torch.zeros_like(tensor_memory)
        grad_phase_shift = torch.tensor(0.0, device=device)
        grad_coupling_gain = torch.tensor(0.0, device=device)

        # 1. 국소 파동 세포 위상 미분 연산자 \delta_{l,i} = \partial L / \partial \theta_{l,i}
        delta_memory = torch.zeros((num_scales, base_dim), device=device)
        for l in range(num_scales):
            for i in range(base_dim):
                d_sin = grad_output_memory[l, i, 0]
                d_cos = grad_output_memory[l, i, 1]
                d_tan = grad_output_memory[l, i, 2]

                sin_val = tensor_memory[l, i, 0]
                cos_val = tensor_memory[l, i, 1]

                # \delta_{l,i} = d_sin * cos - d_cos * sin + d_tan * sec^2(theta)
                sec2_val = 1.0 / (cos_val**2 + 1e-6)
                delta_memory[l, i] = d_sin * cos_val - d_cos * sin_val + d_tan * sec2_val

        # 2. 타겟 국소 셀에 의한 phase_shift 기울기 누적
        grad_phase_shift = grad_phase_shift + delta_memory[target_scale, target_idx]

        # 3. 스케일 결합 파동 전파에 의한 coupling_gain 및 phase_shift 기울기 역전파
        for l in range(num_scales):
            for i in range(base_dim):
                if l == target_scale and i == target_idx:
                    continue

                dist = abs(l - target_scale)
                scale_ratio = (scale_factors[l] / (scale_factors[target_scale] + 1e-8)).item()
                decay = torch.exp(torch.tensor(-dist, dtype=torch.float32, device=device)).item()

                delta_l_i = delta_memory[l, i]

                # \partial L / \partial \eta 누적
                grad_coupling_gain = grad_coupling_gain + delta_l_i * 0.1 * decay * scale_ratio * torch.sin(target_theta_new)

                # \partial L / \partial \Delta\phi 누적
                grad_phase_shift = grad_phase_shift + delta_l_i * 0.1 * coupling_gain * decay * scale_ratio * torch.cos(target_theta_new)

                # 원본 텐서 메모리로 입력 기울기 전파
                grad_tensor_memory[l, i, 0] = grad_tensor_memory[l, i, 0] + grad_output_memory[l, i, 0]
                grad_tensor_memory[l, i, 1] = grad_tensor_memory[l, i, 1] + grad_output_memory[l, i, 1]

        return grad_tensor_memory, None, None, None, grad_phase_shift, grad_coupling_gain


class TrainableScaleWaveMemory(nn.Module):
    """경사하강법(Gradient Descent)으로 위상 결합 매개변수를 자동 최적화하는 모듈"""

    def __init__(self, num_scales: int = 4, base_dim: int = 16, device: str = "cpu"):
        super().__init__()
        self.num_scales = num_scales
        self.base_dim = base_dim
        self.device = torch.device(device)

        scale_factors = [1.0 / (2.0**l) for l in range(num_scales)]
        self.register_buffer("scale_factors", torch.tensor(scale_factors, dtype=torch.float32, device=self.device))

        # 버퍼: 파동 메모리 상태
        self.register_buffer("tensor_memory", torch.zeros((num_scales, base_dim, 3), dtype=torch.float32, device=self.device))
        self._initialize_trinity_lattice()

        # 학습 가능한 파동 결합 매개변수 (nn.Parameter)
        self.phase_shift = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=self.device))
        self.coupling_gain = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device))

    def _initialize_trinity_lattice(self):
        for l in range(self.num_scales):
            lam = self.scale_factors[l].item()
            for i in range(self.base_dim):
                theta = lam * (i + 1) * (3.14159265 / 4.0)
                sin_v = torch.sin(torch.tensor(theta))
                cos_v = torch.cos(torch.tensor(theta))
                tan_v = torch.tan(torch.tensor(theta)) if torch.abs(cos_v) > 1e-3 else torch.sign(sin_v) * 1e3
                self.tensor_memory[l, i, 0] = sin_v
                self.tensor_memory[l, i, 1] = cos_v
                self.tensor_memory[l, i, 2] = tan_v

    def forward(self, target_scale: int, target_idx: int) -> torch.Tensor:
        """미분 가능한 순전파 실행"""
        updated_mem = ScaleWaveMemoryAutogradFunction.apply(
            self.tensor_memory,
            self.scale_factors,
            target_scale,
            target_idx,
            self.phase_shift,
            self.coupling_gain
        )
        return updated_mem
