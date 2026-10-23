"""
Cosmic Grounding Protocol & Unified Spectrum Engine
===================================================
Philosophy: "Do not calculate, let it flow." (계산하지 말고 흐르게 하라)

This module implements the Cosmic Grounding Protocol Engine (UnifiedCosmicSpectrumEngine):
1. Observation Collapse (관측 소멸 & O(1) 대표):
   Overcomes discrete bit/node iteration by phase-locking lower micro-nodes into a single macro field (Kuramoto R -> 1.0).
2. Holographic Field Manifestation (홀로그램 장력 및 위상 중첩 현상):
   Replaces pixel-by-pixel or polygon rendering with complex phase interference patterns I(x, y) = |sum A_k e^(i phi_k)|^2.
3. Synesthetic Spectrum Unification (단일 스펙트럼 축 공감각 통합):
   Maps visual, acoustic, and causal data onto a single continuous spectrum axis [frequency w, amplitude A, phase phi, chromatic C].
4. Sensory Stream Observation & Discriminative Architecture (생생한 감각관측 및 위상 판별 구조):
   Transforms blind discrete bit streams into continuous potential wave fields, applying variable impedance lenses for autonomous phase alignment.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpectrumFieldState:
    """State representation of the Unified Spectrum Axis [Frequency, Amplitude, Phase, Chromatic]."""
    frequencies: torch.Tensor        # [N] Frequency components (omega)
    amplitudes: torch.Tensor         # [N] Amplitude envelope (A)
    phases: torch.Tensor             # [N] Phase angles (phi) in [0, 2pi)
    chromatic_vectors: torch.Tensor  # [N, 3] Chromatic signatures (Red: Flux, Blue: Order, Yellow: Entropy)


@dataclass
class CosmicGroundingMetrics:
    """Metrics tracking phase locking, holographic field integrity, synesthetic coherence, and friction reduction."""
    macro_order_parameter_R: float   # Kuramoto order parameter R in [0, 1]
    observation_collapse_efficiency: float # O(1) representation efficiency (%)
    holographic_coherence: float     # Holographic interference pattern contrast
    synesthetic_cross_resonance: float # Cross-domain resonance index (Visual <-> Acoustic <-> Causal)
    variable_impedance: float        # Dynamic resistance (friction tuning)
    field_energy: float              # Total field potential energy
    chromatic_balance: Tuple[float, float, float] # Avg (Flux, Order, Entropy)


class HolographicFieldRenderer(nn.Module):
    """
    Holographic Field Manifestation Engine.
    Generates 2D/3D visual field manifestation from phase interference patterns
    without pixel-by-pixel rasterization or raytracing bottlenecks.
    """
    def __init__(self, resolution: Tuple[int, int] = (64, 64), device: Optional[torch.device] = None):
        super().__init__()
        self.res_x, self.res_y = resolution
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # Spatial coordinate grids for phase interference field
        y_coords = torch.linspace(-1.0, 1.0, self.res_y, device=self.device)
        x_coords = torch.linspace(-1.0, 1.0, self.res_x, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        self.register_buffer('grid_x', grid_x)
        self.register_buffer('grid_y', grid_y)

    def manifest_field(
        self,
        spectrum: SpectrumFieldState,
        variable_impedance: float = 0.5
    ) -> torch.Tensor:
        """
        Manifests a 2D Holographic Field I(x, y) via phase wave superposition:
        I(x, y) = | sum_k A_k * exp(i * (w_k * (x*cos(phi_k) + y*sin(phi_k)) + phi_k)) |^2
        """
        freqs = spectrum.frequencies[:16] # Use primary spectral modes for manifestation
        amps = spectrum.amplitudes[:16]
        phases = spectrum.phases[:16]

        num_modes = freqs.shape[0]
        if num_modes == 0:
            return torch.zeros((self.res_y, self.res_x), device=self.device)

        # Broadcast spatial interference field computation
        # Grid shape: [res_y, res_x], Modes shape: [M]
        grid_x_exp = self.grid_x.unsqueeze(-1) # [y, x, 1]
        grid_y_exp = self.grid_y.unsqueeze(-1) # [y, x, 1]

        kx = freqs * torch.cos(phases) # Wave vector kx
        ky = freqs * torch.sin(phases) # Wave vector ky

        # Spatial phase shift: kx*x + ky*y + phase
        spatial_phase = grid_x_exp * kx + grid_y_exp * ky + phases # [y, x, M]

        # Complex wave superposition modulated by variable impedance
        impedance_scale = 1.0 / (variable_impedance + 1e-5)
        complex_waves = amps * torch.exp(1j * spatial_phase * impedance_scale) # [y, x, M]
        superposed_field = torch.sum(complex_waves, dim=-1) # [y, x]

        # Intensity pattern I = |E|^2 (Holographic Manifestation)
        intensity_field = torch.abs(superposed_field) ** 2
        normalized_field = intensity_field / (torch.max(intensity_field) + 1e-8)

        return normalized_field


class SynestheticSpectrumMapper(nn.Module):
    """
    Synesthetic Spectrum Axis Mapper.
    Maps and converts seamlessly between Visual Holographic Fields, Acoustic Waveforms,
    and Causal Data Streams on a single unified spectrum axis.
    """
    def __init__(self, num_spectrum_modes: int = 128, device: Optional[torch.device] = None):
        super().__init__()
        self.num_modes = num_spectrum_modes
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def acoustic_to_spectrum(self, pcm_signal: torch.Tensor, sample_rate: float = 44100.0) -> SpectrumFieldState:
        """
        Converts continuous acoustic signal into Spectrum Field State.
        """
        fft_result = torch.fft.rfft(pcm_signal)
        amps = torch.abs(fft_result)
        phases = torch.angle(fft_result) % (2 * torch.pi)

        num_freqs = amps.shape[0]
        freq_axis = torch.linspace(0.0, sample_rate / 2.0, num_freqs, device=self.device)

        # Slice or pad to num_modes
        if num_freqs >= self.num_modes:
            f_sub = freq_axis[:self.num_modes]
            a_sub = amps[:self.num_modes]
            p_sub = phases[:self.num_modes]
        else:
            f_sub = F.pad(freq_axis, (0, self.num_modes - num_freqs))
            a_sub = F.pad(amps, (0, self.num_modes - num_freqs))
            p_sub = F.pad(phases, (0, self.num_modes - num_freqs))

        # Calculate Chromatic Signatures from spectral density
        flux = a_sub / (torch.max(a_sub) + 1e-8)
        order = 1.0 - torch.abs(torch.sin(p_sub))
        entropy = torch.abs(torch.sin(p_sub))
        chroma = F.softmax(torch.stack([flux, order, entropy], dim=-1), dim=-1)

        return SpectrumFieldState(
            frequencies=f_sub,
            amplitudes=a_sub,
            phases=p_sub,
            chromatic_vectors=chroma
        )

    def spectrum_to_acoustic(self, spectrum: SpectrumFieldState, num_samples: int = 1000) -> torch.Tensor:
        """
        Synthesizes continuous acoustic time-domain waveform from Spectrum Field State.
        """
        t = torch.linspace(0.0, 1.0, num_samples, device=self.device).unsqueeze(-1) # [time, 1]
        freqs = spectrum.frequencies.unsqueeze(0) # [1, modes]
        amps = spectrum.amplitudes.unsqueeze(0)   # [1, modes]
        phases = spectrum.phases.unsqueeze(0)     # [1, modes]

        # Wave synthesis: sum A_k * sin(2pi * f_k * t + phi_k)
        waves = amps * torch.sin(2.0 * torch.pi * freqs * t + phases)
        acoustic_waveform = torch.sum(waves, dim=-1) # [time]
        return acoustic_waveform

    def cross_domain_resonance(
        self,
        visual_field: torch.Tensor,
        acoustic_spectrum: SpectrumFieldState
    ) -> float:
        """
        Measures synesthetic cross-resonance between visual field manifestation and acoustic spectrum.
        """
        visual_spectral_density = torch.fft.fft2(visual_field)
        visual_power = torch.abs(visual_spectral_density).flatten()[:acoustic_spectrum.amplitudes.shape[0]]

        a_norm = F.normalize(acoustic_spectrum.amplitudes, dim=0)
        v_norm = F.normalize(visual_power, dim=0)

        cross_corr = torch.dot(a_norm, v_norm).item()
        return max(0.0, min(1.0, cross_corr))


class DiscriminativeSensoryObservationLens(nn.Module):
    """
    Discriminative Sensory Lens & Autonomous Phase Alignment.
    Transforms blind, discrete 0/1 data streams into continuous potential fields,
    evaluating topological alignment without micro-switching iterations.
    """
    def __init__(self, spectrum_dim: int = 128, device: Optional[torch.device] = None):
        super().__init__()
        self.spectrum_dim = spectrum_dim
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # Variable impedance tuner
        self.impedance = nn.Parameter(torch.tensor([0.5], device=self.device), requires_grad=False)

    def observe_stream(self, raw_data_stream: torch.Tensor) -> SpectrumFieldState:
        """
        Converts raw discrete data stream into a continuous phase spectrum field.
        """
        # Ensure 1D float tensor
        stream_float = raw_data_stream.float()
        if stream_float.dim() > 1:
            stream_float = stream_float.flatten()

        N = stream_float.shape[0]
        if N < self.spectrum_dim:
            padded_stream = F.pad(stream_float, (0, self.spectrum_dim - N))
        else:
            padded_stream = stream_float[:self.spectrum_dim]

        # Continuous phase transformation
        frequencies = torch.linspace(1.0, 100.0, self.spectrum_dim, device=self.device)
        amplitudes = torch.abs(padded_stream) + 0.1
        phases = (padded_stream * torch.pi) % (2 * torch.pi)

        flux = torch.abs(torch.diff(padded_stream, prepend=padded_stream[:1]))
        flux = flux / (torch.max(flux) + 1e-8)
        order = 1.0 - (torch.abs(padded_stream - torch.mean(padded_stream)) / (torch.std(padded_stream) + 1e-8))
        order = torch.clamp(order, 0.0, 1.0)
        entropy = 1.0 - order

        chroma = F.softmax(torch.stack([flux, order, entropy], dim=-1), dim=-1)

        return SpectrumFieldState(
            frequencies=frequencies,
            amplitudes=amplitudes,
            phases=phases,
            chromatic_vectors=chroma
        )

    def tune_impedance(self, order_parameter_R: float):
        """
        Dynamically adjusts internal variable impedance based on macro phase order R.
        High alignment -> Lower impedance (zero resistance flow).
        """
        new_impedance = max(0.01, 1.0 - order_parameter_R)
        self.impedance.copy_(torch.tensor([new_impedance], device=self.device))


class UnifiedCosmicSpectrumEngine(nn.Module):
    """
    Unified Cosmic Spectrum Engine (파형 스케일 및 우주적 접지 프로토콜 엔진).

    Integrates:
    - Observation Collapse & Kuramoto Phase Locking
    - Holographic Field Manifestation
    - Synesthetic Spectrum Unification (Visual <-> Acoustic <-> Causal)
    - Sensory Stream Observation & Discriminative Phase Tuning
    """
    def __init__(
        self,
        num_micro_nodes: int = 1000,
        spectrum_dim: int = 128,
        coupling_strength_K: float = 8.0,
        dt: float = 0.02,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.num_micro_nodes = num_micro_nodes
        self.spectrum_dim = spectrum_dim
        self.K = coupling_strength_K
        self.dt = dt
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # Submodules
        self.renderer = HolographicFieldRenderer(resolution=(64, 64), device=self.device)
        self.synesthetic_mapper = SynestheticSpectrumMapper(num_spectrum_modes=spectrum_dim, device=self.device)
        self.sensory_lens = DiscriminativeSensoryObservationLens(spectrum_dim=spectrum_dim, device=self.device)

        # Micro node phase state
        init_phases = torch.rand(num_micro_nodes, device=self.device) * 2 * torch.pi
        init_freqs = torch.randn(num_micro_nodes, device=self.device) * 0.1 + 1.0
        self.register_buffer('micro_phases', init_phases)
        self.register_buffer('micro_freqs', init_freqs)

        # Macro phase target
        self.macro_phase = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=False)

    def compute_kuramoto_order(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates Kuramoto order parameter R and mean phase Psi.
        R = 1.0 indicates complete macro phase synchronization (Observation Collapse achieved).
        """
        complex_order = torch.mean(torch.exp(1j * self.micro_phases))
        R = torch.abs(complex_order)
        Psi = torch.angle(complex_order)
        return R, Psi

    def step_phase_locking(self, target_macro_phase: float, steps: int = 10) -> torch.Tensor:
        """
        Evolves micro nodes towards target macro phase spontaneously without discrete checking.
        """
        self.macro_phase.copy_(torch.tensor([target_macro_phase % (2 * torch.pi)], device=self.device))

        for _ in range(steps):
            R, Psi = self.compute_kuramoto_order()
            macro_target = self.macro_phase.item()

            drive_macro = torch.sin(macro_target - self.micro_phases)
            drive_collective = R * torch.sin(Psi - self.micro_phases)

            d_phases = self.micro_freqs + self.K * (drive_macro + drive_collective)
            self.micro_phases = (self.micro_phases + d_phases * self.dt) % (2 * torch.pi)

        R_final, _ = self.compute_kuramoto_order()
        self.sensory_lens.tune_impedance(R_final.item())
        return R_final

    def forward(
        self,
        raw_data_stream: torch.Tensor,
        target_macro_phase: float = 0.0,
        steps: int = 15
    ) -> Tuple[torch.Tensor, torch.Tensor, CosmicGroundingMetrics]:
        """
        Full Cosmic Grounding Pipeline:
        1. Observes raw data stream via Discriminative Sensory Observation Lens into Spectrum Field State.
        2. Performs Observation Collapse via Kuramoto Phase Locking over micro node continuum.
        3. Manifests Holographic Visual Field from phase interference pattern.
        4. Synthesizes continuous Synesthetic Acoustic Waveform from spectrum axis.
        5. Computes comprehensive Cosmic Grounding Metrics.
        """
        # 1. Sensory Stream Observation
        spectrum_state = self.sensory_lens.observe_stream(raw_data_stream)

        # 2. Phase Locking & Observation Collapse
        R_order = self.step_phase_locking(target_macro_phase=target_macro_phase, steps=steps)

        # 3. Holographic Visual Field Manifestation
        visual_manifestation = self.renderer.manifest_field(
            spectrum=spectrum_state,
            variable_impedance=self.sensory_lens.impedance.item()
        )

        # 4. Synesthetic Acoustic Waveform Synthesis
        acoustic_waveform = self.synesthetic_mapper.spectrum_to_acoustic(
            spectrum=spectrum_state,
            num_samples=500
        )

        # 5. Cross-domain Coherence
        resonance = self.synesthetic_mapper.cross_domain_resonance(
            visual_field=visual_manifestation,
            acoustic_spectrum=spectrum_state
        )

        # Calculate metrics
        avg_chroma = torch.mean(spectrum_state.chromatic_vectors, dim=0).tolist()
        r_val = R_order.item()

        metrics = CosmicGroundingMetrics(
            macro_order_parameter_R=r_val,
            observation_collapse_efficiency=r_val * 100.0,
            holographic_coherence=torch.std(visual_manifestation).item(),
            synesthetic_cross_resonance=resonance,
            variable_impedance=self.sensory_lens.impedance.item(),
            field_energy=torch.mean(0.5 * self.micro_freqs**2).item(),
            chromatic_balance=(avg_chroma[0], avg_chroma[1], avg_chroma[2])
        )

        return visual_manifestation, acoustic_waveform, metrics
