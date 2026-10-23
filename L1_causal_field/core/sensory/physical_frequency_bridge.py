import torch
import torch.nn as nn
import math

class PhysicalFrequencySDFBridge(nn.Module):
    """
    Direct tokenless injection of physical audio/visual frequency spectra into continuous latent SDF space.
    """
    def __init__(self, dim=64, num_freq_bands=128):
        super().__init__()
        self.dim = dim
        self.num_freq_bands = num_freq_bands

        # Linear projector mapping physical spectrum bands to 64D wavevector k
        self.freq_to_wavevector = nn.Linear(num_freq_bands, dim, bias=False)

    def forward(self, sdf_field_pos, external_audio_fft, external_visual_spatial_freq):
        """
        sdf_field_pos: [Batch, 64]
        external_audio_fft: [Batch, 128]
        external_visual_spatial_freq: [Batch, 128]
        """
        fused_physical_spectrum = external_audio_fft + external_visual_spatial_freq
        k_vector = self.freq_to_wavevector(fused_physical_spectrum)
        spatial_phase = torch.sum(sdf_field_pos * k_vector, dim=-1, keepdim=True)
        excitation_wave = torch.cos(spatial_phase)
        return excitation_wave


class RealTimeAudioFFTBridge(nn.Module):
    """
    STFT analysis of real-time audio PCM streams converting into spatial wavevector k_ext and angular frequency omega.
    """
    def __init__(self, sample_rate=44100, n_fft=1024, spatial_dim=64):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.spatial_dim = spatial_dim

        self.freq_to_k = nn.Linear(n_fft // 2 + 1, spatial_dim, bias=False)
        self.window = torch.hann_window(n_fft)

    def forward(self, audio_chunk):
        """
        audio_chunk: [Batch, n_fft]
        """
        if self.window.device != audio_chunk.device:
            self.window = self.window.to(audio_chunk.device)

        windowed = audio_chunk * self.window
        fft_complex = torch.fft.rfft(windowed, n=self.n_fft, dim=-1)
        magnitude = torch.abs(fft_complex)
        phase = torch.angle(fft_complex)

        peak_idx = torch.argmax(magnitude, dim=-1)
        peak_freq = peak_idx.float() * (self.sample_rate / self.n_fft)
        omega_ext = 2.0 * math.pi * peak_freq

        k_ext = self.freq_to_k(magnitude)
        amplitude = torch.norm(magnitude, dim=-1, keepdim=True) / math.sqrt(self.n_fft)

        return k_ext, omega_ext, amplitude, phase[:, :self.spatial_dim]
