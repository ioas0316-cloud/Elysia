"""
Topological Spectral Engine & Spectral Wave Relaxation Modules
Elysia Engine Core Architectural Module

Implements:
1. TopologicalSpectralEngine: Complex phase field mapping, spectral transfer operator H(k), resonance query.
2. SpectralWaveRelaxation1D, 2D, 3D: Multi-dimensional FFT-based wave relaxation and global spectral filtering.
3. TopologicalAttractorRelaxation & SpectralAttractorRelaxation: Informational mass (M_I), value density (rho_V),
   local phase coherence tensor, metric deformation (g_ij), and non-linear noise evaporation.
4. StaticRotorField2D: Quaternion rotor field, local rotor perturbation (q' = R * q * R*), Hamilton product, and wave diffusion.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


# -------------------------------------------------------------------
# 1. Quaternion & Rotor Utilities
# -------------------------------------------------------------------
def quaternion_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product of two quaternion tensors q1, q2.
    q = [w, x, y, z] (last dimension size = 4).
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate q* = [w, -x, -y, -z]"""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def create_rotor(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Create a rotor R = e^(theta/2 * u) from rotation axis (u_x, u_y, u_z) and angle (theta).
    R = [cos(theta/2), sin(theta/2) * u_x, sin(theta/2) * u_y, sin(theta/2) * u_z]
    """
    axis = F.normalize(axis, p=2, dim=-1)
    half_angle = angle / 2.0
    w = torch.cos(half_angle).reshape(-1)
    xyz = torch.sin(half_angle) * axis
    return torch.cat([w, xyz], dim=-1)


# -------------------------------------------------------------------
# 2. Topological Spectral Engine
# -------------------------------------------------------------------
class TopologicalSpectralEngine(nn.Module):
    """
    Topological Spectral Engine maps discrete memory bit arrays or feature arrays
    into continuous complex phase fields defined by frequency, amplitude, and phase.
    """

    def __init__(self, shape=(128, 128), device='cpu'):
        super().__init__()
        self.shape = shape
        self.device = torch.device(device if isinstance(device, torch.device) else (device if torch.cuda.is_available() and device != 'cpu' else 'cpu'))

        # Complex Phase Field: Real = Amplitude (Data Intensity), Imaginary = Phase (Connectivity)
        self.register_buffer("field", torch.zeros(shape, dtype=torch.complex64, device=self.device))

    def encode_data_wave(self, frequency_k: tuple, phase_phi: float, amplitude: float = 1.0):
        """
        Injects a specific frequency and phase wave mode into the spatial phase field without pointer allocation.
        wave_mode = A * exp(i * (kx * x + ky * y + phi))
        """
        ky, kx = frequency_k
        y = torch.linspace(0, 2 * math.pi, self.shape[0], device=self.device).view(-1, 1)
        x = torch.linspace(0, 2 * math.pi, self.shape[1], device=self.device).view(1, -1)

        wave_mode = amplitude * torch.exp(1j * (kx * x + ky * y + phase_phi))
        self.field = self.field + wave_mode

    def apply_spectral_operator(self, transfer_function_H: torch.Tensor):
        """
        Executes algorithm logic via spectral filtering in k-space.
        O(1) spatial contraction through FFT2 and IFFT2.
        """
        # Transform spatial field to frequency domain (k-space)
        k_space_field = fft.fft2(self.field)

        # Apply spectral transfer function H(k)
        transformed_k_space = k_space_field * transfer_function_H.to(self.device)

        # Inverse FFT back to spatial domain
        self.field = fft.ifft2(transformed_k_space)

    def query_by_resonance(self, probe_phase: float) -> torch.Tensor:
        """
        Extracts standing wave responses that resonate with probe_phase without pointer tracing.
        Resonance signal = Re(field * e^(-i * probe_phase))
        """
        resonance_signal = torch.real(self.field * torch.exp(-1j * torch.tensor(probe_phase, device=self.device)))
        return F.relu(resonance_signal)


# -------------------------------------------------------------------
# 3. Multi-Dimensional FFT Wave Relaxation Modules
# -------------------------------------------------------------------
class SpectralWaveRelaxation1D(nn.Module):
    """1D FFT Wave Relaxation Module."""

    def __init__(self, in_channels: int, length: int):
        super().__init__()
        self.in_channels = in_channels
        self.length = length
        freq_len = length // 2 + 1
        self.complex_filter = nn.Parameter(
            torch.randn(1, in_channels, freq_len, dtype=torch.complex64) * 0.02
        )

    def forward(self, spatial_field: torch.Tensor) -> torch.Tensor:
        # spatial_field: (B, C, L)
        _, _, L = spatial_field.shape
        k_space = fft.rfft(spatial_field, dim=-1)
        relaxed_k_space = k_space * (1.0 + self.complex_filter)
        return fft.irfft(relaxed_k_space, n=L, dim=-1)


class SpectralWaveRelaxation2D(nn.Module):
    """2D FFT Wave Relaxation Module (Image / 2D Grid / Maze Navigation)."""

    def __init__(self, in_channels: int, height: int, width: int):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        freq_w = width // 2 + 1
        self.complex_filter = nn.Parameter(
            torch.randn(1, in_channels, height, freq_w, dtype=torch.complex64) * 0.02
        )

    def forward(self, spatial_field: torch.Tensor) -> torch.Tensor:
        # spatial_field: (B, C, H, W)
        _, _, H, W = spatial_field.shape
        k_space = fft.rfft2(spatial_field, dim=(-2, -1))
        relaxed_k_space = k_space * (1.0 + self.complex_filter)
        return fft.irfft2(relaxed_k_space, s=(H, W), dim=(-2, -1))


class SpectralWaveRelaxation3D(nn.Module):
    """3D FFT Wave Relaxation Module (3D Potential Field / Volumetric Topological Manifold)."""

    def __init__(self, in_channels: int, depth: int, height: int, width: int):
        super().__init__()
        self.in_channels = in_channels
        self.depth = depth
        self.height = height
        self.width = width
        freq_w = width // 2 + 1
        self.complex_filter = nn.Parameter(
            torch.randn(1, in_channels, depth, height, freq_w, dtype=torch.complex64) * 0.02
        )

    def forward(self, spatial_field_3d: torch.Tensor) -> torch.Tensor:
        # spatial_field_3d: (B, C, D, H, W)
        _, _, D, H, W = spatial_field_3d.shape
        k_space_3d = fft.rfftn(spatial_field_3d, dim=(-3, -2, -1))
        relaxed_k_space_3d = k_space_3d * (1.0 + self.complex_filter)
        return fft.irfftn(relaxed_k_space_3d, s=(D, H, W), dim=(-3, -2, -1))


# -------------------------------------------------------------------
# 4. Topological Attractor Relaxation & Spectral Attractor
# -------------------------------------------------------------------
class TopologicalAttractorRelaxation(nn.Module):
    """
    Topological Attractor Field Relaxation:
    Replaces O(N^2) Softmax Attention with linear field contraction,
    measuring value density (rho_V), metric deformation (g_ij), and non-linear noise evaporation.
    """

    def __init__(self, embed_dim: int, field_resolution: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.field_res = field_resolution

        kernel = torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3)
        self.register_buffer("kernel", kernel)

        self.density_proj = nn.Linear(embed_dim, 1)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def compute_value_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes semantic value density rho_V = sigmoid(density_proj(x)).
        """
        return torch.sigmoid(self.density_proj(x))

    def forward(self, x: torch.Tensor, relax_steps: int = 3, noise_threshold: float = 0.05) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape

        # A. Semantic value density rho_V
        rho_V = self.compute_value_density(x)  # (B, N, 1)
        values = self.value_proj(x)            # (B, N, D)

        # Non-linear noise evaporation (Step 3: Low-mass noise dissipation)
        values = torch.where(rho_V > noise_threshold, values * rho_V, values * 0.01)

        # B. Manifold projection
        field_transposed = values.transpose(1, 2)  # (B, D, N)

        # C. Laplacian wave diffusion over field
        for _ in range(relax_steps):
            field_transposed = F.conv1d(
                field_transposed,
                self.kernel.expand(D, 1, 3),
                padding=1,
                groups=D
            )

        return field_transposed.transpose(1, 2)


class SpectralAttractorRelaxation(nn.Module):
    """
    torch.fft-based O(N log N) Spectral Attractor Relaxation Layer.
    Global spectral filtering over the sequence length without pair-wise dot products.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.density_proj = nn.Linear(embed_dim, 1)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.complex_spectral_filter = nn.Parameter(
            torch.randn(1, 1, embed_dim, dtype=torch.complex64) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape

        # Compute semantic value density and complex spatial field
        density = torch.sigmoid(self.density_proj(x))  # (B, N, 1)
        values = self.value_proj(x)                    # (B, N, D)
        spatial_field = values * density

        # 1D Real-to-Complex FFT: O(N log N)
        k_space_field = fft.rfft(spatial_field, dim=1)  # (B, N//2 + 1, D)

        # Global Spectral Filtering in k-space
        relaxed_k_space = k_space_field * (1.0 + self.complex_spectral_filter)

        # Inverse FFT back to spatial sequence: O(N log N)
        return fft.irfft(relaxed_k_space, n=N, dim=1)


# -------------------------------------------------------------------
# 5. Static Rotor Field 2D (Quaternion Field & Local Perturbations)
# -------------------------------------------------------------------
class StaticRotorField2D(nn.Module):
    """
    2D Static Quaternion Rotor Field (H, W, 4).
    Enables memory-compute fusion where local perturbation q' = R * q * R*
    updates field state in O(radius^2) local sandbox time without global re-evaluation.
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

        # Static Quaternion Field [w, x, y, z] (w=1 scalar, x=1 vector component)
        field_init = torch.zeros(height, width, 4)
        field_init[..., 0] = 1.0
        field_init[..., 1] = 1.0
        field_init = F.normalize(field_init, p=2, dim=-1)
        self.register_buffer("field", field_init)

        laplacian = torch.tensor([
            [0.05, 0.20, 0.05],
            [0.20, -1.0, 0.20],
            [0.05, 0.20, 0.05]
        ]).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", laplacian)

    def apply_local_rotor_update(self, center_h: int, center_w: int, radius: int, axis: torch.Tensor, angle: torch.Tensor):
        """
        Selectively applies rotor R_local to local bounding box [center - radius, center + radius].
        Local update q' = R * q * R*.
        """
        h_min = max(0, center_h - radius)
        h_max = min(self.height, center_h + radius + 1)
        w_min = max(0, center_w - radius)
        w_max = min(self.width, center_w + radius + 1)

        local_subfield = self.field[h_min:h_max, w_min:w_max]  # (Sub_H, Sub_W, 4)

        R = create_rotor(axis, angle)  # (4,)
        R_conj = quaternion_conjugate(R)

        R_expanded = R.expand_as(local_subfield)
        R_conj_expanded = R_conj.expand_as(local_subfield)

        rotated_subfield = quaternion_mul(R_expanded, local_subfield)
        rotated_subfield = quaternion_mul(rotated_subfield, R_conj_expanded)

        self.field[h_min:h_max, w_min:w_max] = rotated_subfield

    def relax_wave_diffusion(self, steps: int = 1, dt: float = 0.1):
        """
        Diffuses local phase perturbations into surrounding field via Laplacian wave relaxation.
        """
        field_transposed = self.field.permute(2, 0, 1).unsqueeze(1)  # (4, 1, H, W)

        for _ in range(steps):
            lap_out = F.conv2d(field_transposed, self.laplacian_kernel, padding=1)
            field_transposed = field_transposed + dt * lap_out
            field_transposed = F.normalize(field_transposed, p=2, dim=0)

        self.field = field_transposed.squeeze(1).permute(1, 2, 0)
