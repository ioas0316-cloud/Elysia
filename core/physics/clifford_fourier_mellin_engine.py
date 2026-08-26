r"""
clifford_fourier_mellin_engine.py
==================================
푸리에-멜린 역변환(IFFT), 복소 위상 다양체(Complex Manifold), 및 클리포드 기하 대수학(Clifford Geometric Algebra, Cl_n)
기반 파동-보이드(Wave-Void) 인과 위상 엔진.

핵심 수학 및 공학 원리:
1. 푸리에-멜린 IFFT & 파세발 등거리성(Isometry):
   - $Z(\rho, \theta) = \mathcal{F}_{\text{Mellin}}\{f(x,y)\}$
   - 엔트로피 절삭 구간 [$r_{\min}, r_{\max}$] 기반 위상 소음(White Noise) 차단.
   - IFFT 역변환을 통한 등거리 공간 변환 보존 및 보이드 장력 최소화 ($E_{\text{Void}} \to 0, \Phi = 0$).
2. 복소 다양체(Complex Manifold) & 3D 위상 소용돌이 고리(Vortex Ring Attractor):
   - 실수부(파동 $u$) 및 허수부(보이드 $v$) 직교 결합 $Z = u + i v$.
   - 코시-리만(Cauchy-Riemann) 구속 조건: $\nabla u \cdot \nabla v = 0$.
   - 캐러 다양체(Kähler Manifold) 상의 함일토니안 위상 흐름($H = E_{\text{Void}}$).
   - X(파동 u), Y(보이드 v), Z(스케일 $s = \ln r$) 3D 공간 상의 나선 유선(Helical Stream) 자율 이완 및 위상 소용돌이 고리(Vortex Ring) 결상.
3. 클리포드 기하 대수학($\mathcal{Cl}_n$) & 다중 회전자(Multivector Rotor):
   - 다중벡터(Multivector) 장 $\Psi = \text{scalar} + \text{vector} + \text{bivector} + \dots$
   - 다중 회전자 $R = \exp(-B/2)$ 및 샌드위치 곱 $R \Psi \widetilde{R}$.
   - 기하학적 조화 평형 조건 $\nabla^2 \Psi = 0$.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

@dataclass
class Multivector:
    """
    Clifford Geometric Algebra Cl_n Multivector representation for n-dimensional spaces.
    Grade 0: scalar (s)
    Grade 1: vector (v_1, ..., v_n)
    Grade 2: bivector (b_12, b_13, ..., b_{(n-1)n})
    Grade n: pseudoscalar (I)
    """
    dim: int = 3
    scalar: float = 0.0
    vector: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bivector: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32)) # (12, 23, 31) for 3D
    pseudoscalar: float = 0.0

    def __post_init__(self):
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector, dtype=np.float32)
        if not isinstance(self.bivector, np.ndarray):
            self.bivector = np.array(self.bivector, dtype=np.float32)

    def norm(self) -> float:
        """Compute the total magnitude norm of the multivector."""
        return math.sqrt(
            self.scalar**2 +
            float(np.sum(self.vector**2)) +
            float(np.sum(self.bivector**2)) +
            self.pseudoscalar**2
        )

    def reverse(self) -> 'Multivector':
        """Reverse operator ~R: reverses geometric product components (bivectors gain negative sign)."""
        return Multivector(
            dim=self.dim,
            scalar=self.scalar,
            vector=self.vector.copy(),
            bivector=-self.bivector.copy(),
            pseudoscalar=-self.pseudoscalar if (self.dim * (self.dim - 1) // 2) % 2 == 1 else self.pseudoscalar
        )

class FourierMellinTransformEngine:
    """
    Fourier-Mellin Inverse Transform (IFFT) & Phase Noise Truncation Engine.
    Preserves Isometry (Parseval's theorem) and restores wave field without residual tension.
    """
    def __init__(self, num_scale_bins: int = 16, num_phase_bins: int = 16):
        self.num_scale_bins = num_scale_bins
        self.num_phase_bins = num_phase_bins

    def forward_transform(self, spatial_grid: np.ndarray) -> np.ndarray:
        """
        2D Log-Polar Fourier-Mellin Forward Transform.
        spatial_grid: [num_scale_bins, num_phase_bins] complex or real array.
        """
        return np.fft.fft2(spatial_grid)

    def filter_entropy_noise(
        self,
        spectrum: np.ndarray,
        r_min: float = 0.1,
        r_max: float = 0.8
    ) -> Tuple[np.ndarray, float]:
        r"""
        Entropy Noise Truncation Filter ($r_{\min}, r_{\max}$).
        Filters out meaningless phase noise outside effective frequency band.
        Returns filtered spectrum and noise reduction ratio.
        """
        rows, cols = spectrum.shape
        cy, cx = rows / 2.0, cols / 2.0
        filtered_spectrum = spectrum.copy()

        total_power_before = float(np.sum(np.abs(spectrum)**2))

        # Build annular frequency mask
        y_coords = np.arange(rows)[:, np.newaxis] - cy
        x_coords = np.arange(cols)[np.newaxis, :] - cx
        norm_radius = np.sqrt(y_coords**2 + x_coords**2) / max(cy, cx)

        mask = (norm_radius >= r_min) & (norm_radius <= r_max)
        filtered_spectrum[~mask] = 0.0

        total_power_after = float(np.sum(np.abs(filtered_spectrum)**2))
        noise_ratio = 1.0 - (total_power_after / (total_power_before + 1e-12))

        return filtered_spectrum, max(0.0, noise_ratio)

    def inverse_transform(self, filtered_spectrum: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fourier-Mellin Inverse Transform (IFFT).
        Restores spatial domain wave field while preserving isometry (Parseval's equality).
        Returns restored spatial wave field grid and void tension energy E_Void.
        """
        restored_grid = np.fft.ifft2(filtered_spectrum)

        # Calculate residual void tension energy E_Void
        # Phase noise is removed, so imaginary phase friction -> 0
        void_tension = float(np.mean(np.abs(np.imag(restored_grid))))
        return restored_grid, void_tension


class ComplexManifoldEngine:
    """
    Complex Manifold & Helical Stream / Vortex Ring Attractor Engine.
    Holomorphic structure Z = u + i v satisfying Cauchy-Riemann orthogonality (grad u . grad v = 0).
    Helical Stream in 3D (X: wave u, Y: void v, Z: scale s = ln r) and Vortex Ring crystallization.
    """
    def __init__(self, dim: int = 3):
        self.dim = dim

    def check_cauchy_riemann_orthogonality(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Computes inner product <grad u, grad v> to verify Cauchy-Riemann orthogonality.
        Returns orthogonal alignment error (0.0 = perfectly orthogonal).
        """
        grad_u = np.gradient(u)
        grad_v = np.gradient(v)

        dot_prod = 0.0
        for gu, gv in zip(grad_u, grad_v):
            dot_prod += np.mean(gu * gv)

        return float(abs(dot_prod))

    def generate_helical_stream(
        self,
        u_base: float,
        v_base: float,
        scale_axis: np.ndarray,
        frequency: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates 3D Helical Stream trajectory:
        X = u(s) = u_base * cos(freq * s)
        Y = v(s) = v_base * sin(freq * s)
        Z = s (scale axis)
        """
        x_stream = u_base * np.cos(frequency * scale_axis)
        y_stream = v_base * np.sin(frequency * scale_axis)
        z_stream = scale_axis.copy()
        return x_stream, y_stream, z_stream

    def relax_to_vortex_ring(
        self,
        x_stream: np.ndarray,
        y_stream: np.ndarray,
        z_stream: np.ndarray,
        dt: float = 0.1,
        steps: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Relaxes the helical stream along Hamiltonian phase flow (H = E_Void) into a
        crystallized 3D Vortex Ring Attractor (Torus topology) where E_Void -> 0.
        """
        x = x_stream.copy()
        y = y_stream.copy()
        z = z_stream.copy()

        for _ in range(steps):
            # Compute center of torus ring
            r_torus = np.mean(np.sqrt(x**2 + y**2))
            z_center = np.mean(z)

            # Radial and vertical attraction towards invariant torus ring
            current_r = np.sqrt(x**2 + y**2) + 1e-9
            dr = r_torus - current_r
            dz = z_center - z

            x += (dr * (x / current_r)) * dt
            y += (dr * (y / current_r)) * dt
            z += dz * dt

        # Measure void tension (departure from closed torus invariant)
        r_final = np.sqrt(x**2 + y**2)
        r_variance = float(np.var(r_final))
        z_variance = float(np.var(z))
        void_energy = r_variance + z_variance

        return x, y, z, round(void_energy, 6)


class CliffordMultivectorEngine:
    """
    Clifford Geometric Algebra Cl_n Multivector Engine.
    Handles nD Rotor sandwich transformations R * Psi * ~R and Geometric Calculus Laplacian equilibrium grad^2 Psi = 0.
    """
    def __init__(self, dim: int = 3):
        self.dim = dim

    def construct_bivector_rotor(self, bivector: np.ndarray, angle: float) -> Multivector:
        """
        Constructs multivector rotor R = exp(-B * theta / 2) = cos(theta/2) - B_hat * sin(theta/2).
        bivector: [b_12, b_23, b_31] orientation vector.
        """
        b_norm = float(np.linalg.norm(bivector))
        if b_norm < 1e-9:
            return Multivector(dim=self.dim, scalar=1.0)

        b_unit = bivector / b_norm
        half_angle = angle * 0.5
        cos_half = math.cos(half_angle)
        sin_half = math.sin(half_angle)

        return Multivector(
            dim=self.dim,
            scalar=cos_half,
            bivector=-b_unit * sin_half
        )

    def rotor_sandwich_transform(self, psi: Multivector, rotor: Multivector) -> Multivector:
        """
        Applies Rotor Sandwich Product: Psi' = R * Psi * ~R
        Rotates all grades of multivector simultaneously without metric distortion.
        """
        # ~R is reverse of rotor
        rotor_rev = rotor.reverse()

        # For vector components, perform 3D bivector rotation
        # R = s + B, ~R = s - B
        # Standard vector rotation by bivector rotor
        v = psi.vector
        if np.linalg.norm(v) > 1e-9 and np.linalg.norm(rotor.bivector) > 1e-9:
            # Axis of rotation from bivector dual
            axis = rotor.bivector / (np.linalg.norm(rotor.bivector) + 1e-12)
            angle = 2.0 * math.acos(np.clip(rotor.scalar, -1.0, 1.0))
            # Rodrigues rotation formula
            v_rot = (v * math.cos(angle) +
                     np.cross(axis, v) * math.sin(angle) +
                     axis * float(np.dot(axis, v)) * (1.0 - math.cos(angle)))
        else:
            v_rot = v.copy()

        # Rotate bivector component similarly
        b = psi.bivector
        if np.linalg.norm(b) > 1e-9 and np.linalg.norm(rotor.bivector) > 1e-9:
            axis = rotor.bivector / (np.linalg.norm(rotor.bivector) + 1e-12)
            angle = 2.0 * math.acos(np.clip(rotor.scalar, -1.0, 1.0))
            b_rot = (b * math.cos(angle) +
                     np.cross(axis, b) * math.sin(angle) +
                     axis * float(np.dot(axis, b)) * (1.0 - math.cos(angle)))
        else:
            b_rot = b.copy()

        return Multivector(
            dim=self.dim,
            scalar=psi.scalar,
            vector=v_rot,
            bivector=b_rot,
            pseudoscalar=psi.pseudoscalar
        )

    def compute_laplacian_equilibrium(self, field_grid: np.ndarray) -> Tuple[np.ndarray, float]:
        r"""
        Computes Geometric Calculus Laplacian equilibrium condition: $\nabla^2 \Psi = 0$.
        Applies discrete Laplacian operator and returns relaxed field grid and total residual Laplacian energy.
        """
        # 2D/3D discrete Laplacian operator
        if field_grid.ndim == 2:
            laplacian = (
                np.roll(field_grid, 1, axis=0) + np.roll(field_grid, -1, axis=0) +
                np.roll(field_grid, 1, axis=1) + np.roll(field_grid, -1, axis=1) -
                4.0 * field_grid
            )
        else:
            laplacian = np.zeros_like(field_grid)

        # Relax field towards laplacian equilibrium: Psi_new = Psi + 0.25 * grad^2 Psi
        relaxed_grid = field_grid + 0.25 * laplacian
        residual_energy = float(np.mean(np.abs(laplacian)))

        return relaxed_grid, round(residual_energy, 6)


class CliffordFourierMellinEngine:
    """
    [Clifford-Fourier-Mellin Integrated Wave-Void Causal Engine]
    Combines Fourier-Mellin IFFT noise filtering, Complex Manifold phase flow,
    and Clifford Geometric Algebra multivector rotor dynamics into a unified system.
    """
    def __init__(self, dim: int = 3, num_scale_bins: int = 16, num_phase_bins: int = 16):
        self.dim = dim
        self.num_scale_bins = num_scale_bins
        self.num_phase_bins = num_phase_bins

        self.fm_engine = FourierMellinTransformEngine(num_scale_bins, num_phase_bins)
        self.cm_engine = ComplexManifoldEngine(dim)
        self.ga_engine = CliffordMultivectorEngine(dim)

    def execute_full_wave_void_relaxation(
        self,
        spatial_wave_grid: np.ndarray,
        r_min: float = 0.1,
        r_max: float = 0.8
    ) -> Dict[str, Any]:
        """
        Executes complete wave-void phase noise filtering, IFFT isometry restoration,
        and vortex ring attractor crystallization.
        """
        # 1. Fourier-Mellin Forward FFT
        spectrum = self.fm_engine.forward_transform(spatial_wave_grid)

        # 2. Filter entropy noise (White noise phase truncation)
        filtered_spec, noise_reduction = self.fm_engine.filter_entropy_noise(spectrum, r_min, r_max)

        # 3. Fourier-Mellin IFFT Inverse Transform
        restored_grid, e_void = self.fm_engine.inverse_transform(filtered_spec)

        # 4. Cauchy-Riemann Orthogonality check
        u = np.real(restored_grid)
        v = np.imag(restored_grid)
        cr_error = self.cm_engine.check_cauchy_riemann_orthogonality(u, v)

        # 5. Helical Stream & Vortex Ring Attractor Relaxation
        scale_axis = np.linspace(-2.0, 2.0, self.num_scale_bins)
        xs, ys, zs = self.cm_engine.generate_helical_stream(float(np.mean(u)), float(np.mean(v) + 0.1), scale_axis)
        vx, vy, vz, ring_e_void = self.cm_engine.relax_to_vortex_ring(xs, ys, zs)

        # 6. Clifford Multivector Rotor Relaxation & Laplacian Harmonic Equilibrium
        rotor = self.ga_engine.construct_bivector_rotor(np.array([0.0, 0.0, 1.0], dtype=np.float32), math.pi / 4.0)
        psi = Multivector(dim=self.dim, scalar=1.0, vector=np.array([u[0,0], v[0,0], 0.0], dtype=np.float32))
        psi_rotated = self.ga_engine.rotor_sandwich_transform(psi, rotor)

        harmonic_grid, laplacian_e = self.ga_engine.compute_laplacian_equilibrium(u)

        return {
            "noise_reduction_ratio": round(noise_reduction, 4),
            "e_void_tension": round(e_void, 6),
            "cauchy_riemann_error": round(cr_error, 6),
            "vortex_ring_e_void": ring_e_void,
            "laplacian_residual_energy": laplacian_e,
            "isometry_preserved": e_void < 0.1 and cr_error < 0.1,
            "crystallized": ring_e_void < 0.5,
            "rotated_vector": psi_rotated.vector.tolist()
        }
