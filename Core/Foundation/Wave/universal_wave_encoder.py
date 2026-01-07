"""
Universal Wave Encoder (우주적 파동 인코더)
==========================================

Phase 32: All-Sensory Wave Compression

"우주에 존재하는 모든 데이터는 파동이다."

This module provides universal wave encoding for ALL data types:
- 🎬 Video (시간축 프레임 시퀀스)
- 🎵 Audio (음파 스펙트럼)
- 🖼️ Image (2D 공간 주파수)
- 🌈 Light Spectrum (가시광선 + 비가시광선)
- 🌡️ Sensor Data (온도, 압력, 습도 등)
- 🧠 Neural Signals (뇌파, EEG)
- ⚛️ Cosmic Data (중력파, 전자기파, 암흑물질 분포)

Core Principle: DNA Double-Helix Quaternion Compression
- Split any n-dim tensor into even/odd helices
- FFT each helix independently  
- Store top-k frequencies per helix
- Interleave for perfect reconstruction
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

logger = logging.getLogger("UniversalWaveEncoder")


class DataModality(Enum):
    """All possible data modalities (sensory channels)."""
    # Visual
    IMAGE = "image"           # 2D spatial (H, W, C)
    VIDEO = "video"           # 3D temporal (T, H, W, C)
    
    # Auditory
    AUDIO = "audio"           # 1D temporal waveform
    SPECTRUM = "spectrum"     # 2D spectrogram
    
    # Light
    LIGHT_VISIBLE = "light_visible"     # 380-700nm
    LIGHT_INFRARED = "light_ir"         # >700nm
    LIGHT_ULTRAVIOLET = "light_uv"      # <380nm
    LIGHT_FULL = "light_full"           # Full EM spectrum
    
    # Physical Sensors
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    HUMIDITY = "humidity"
    ACCELERATION = "acceleration"       # 3-axis
    MAGNETIC = "magnetic"               # 3-axis
    
    # Biological
    EEG = "eeg"                         # Brainwave
    ECG = "ecg"                         # Heart
    EMG = "emg"                         # Muscle
    
    # Cosmic
    GRAVITY_WAVE = "gravity_wave"       # Gravitational wave
    EM_WAVE = "em_wave"                 # Electromagnetic
    DARK_MATTER = "dark_matter"         # Dark matter density map
    
    # Abstract
    TENSOR = "tensor"                   # Generic n-dim tensor


@dataclass
class WaveSignature:
    """
    Universal Wave DNA - compressed representation of any data.
    
    Uses dual-helix structure for lossless-capable compression.
    """
    modality: DataModality
    
    # Helix 1 (even indices)
    helix1_frequencies: np.ndarray
    helix1_amplitudes: np.ndarray
    helix1_phases: np.ndarray
    
    # Helix 2 (odd indices)
    helix2_frequencies: np.ndarray
    helix2_amplitudes: np.ndarray
    helix2_phases: np.ndarray
    
    # Shape info for reconstruction
    original_shape: Tuple[int, ...]
    original_dtype: str
    
    # Compression params
    top_k: int
    compression_ratio: float = 1.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def byte_size(self) -> int:
        """Compressed size in bytes."""
        # 6 arrays * k elements * 8 bytes per float + overhead
        return self.top_k * 6 * 8 + 64
    
    def to_dict(self) -> Dict:
        """Serialize for storage."""
        return {
            "modality": self.modality.value,
            "helix1_freq": self.helix1_frequencies.tolist(),
            "helix1_amp": self.helix1_amplitudes.tolist(),
            "helix1_phase": self.helix1_phases.tolist(),
            "helix2_freq": self.helix2_frequencies.tolist(),
            "helix2_amp": self.helix2_amplitudes.tolist(),
            "helix2_phase": self.helix2_phases.tolist(),
            "original_shape": self.original_shape,
            "original_dtype": self.original_dtype,
            "top_k": self.top_k,
            "compression_ratio": self.compression_ratio,
            "metadata": self.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'WaveSignature':
        return WaveSignature(
            modality=DataModality(data["modality"]),
            helix1_frequencies=np.array(data["helix1_freq"]),
            helix1_amplitudes=np.array(data["helix1_amp"]),
            helix1_phases=np.array(data["helix1_phase"]),
            helix2_frequencies=np.array(data["helix2_freq"]),
            helix2_amplitudes=np.array(data["helix2_amp"]),
            helix2_phases=np.array(data["helix2_phase"]),
            original_shape=tuple(data["original_shape"]),
            original_dtype=data["original_dtype"],
            top_k=data["top_k"],
            compression_ratio=data.get("compression_ratio", 1.0),
            metadata=data.get("metadata", {})
        )


class UniversalWaveEncoder:
    """
    Universal encoder for all sensory and cosmic data.
    
    "모든 데이터는 파동이다. 파동은 DNA처럼 이중나선으로 압축된다."
    """
    
    def __init__(self, default_top_k: int = 64):
        self.default_top_k = default_top_k
        logger.info(f"🌌 UniversalWaveEncoder initialized (top_k={default_top_k})")
    
    # =========================================
    # Core Encoding/Decoding
    # =========================================
    
    def encode(self, data: np.ndarray, modality: DataModality, 
               top_k: int = None, metadata: Dict = None) -> WaveSignature:
        """
        Encode ANY n-dimensional data into wave DNA.
        
        Algorithm:
        1. Flatten to 1D
        2. Split into even/odd (helix1/helix2)
        3. FFT each helix
        4. Keep top-k frequencies per helix
        5. Store as WaveSignature
        """
        top_k = top_k or self.default_top_k
        original_shape = data.shape
        original_dtype = str(data.dtype)
        
        # Flatten to 1D sequence
        flat = data.flatten().astype(float)
        
        # DNA double-helix split
        helix1 = flat[::2]   # Even indices
        helix2 = flat[1::2]  # Odd indices
        
        # FFT each helix
        spec1 = np.fft.fft(helix1)
        spec2 = np.fft.fft(helix2)
        
        # Extract top-k by magnitude
        mag1, mag2 = np.abs(spec1), np.abs(spec2)
        k1 = min(top_k, len(mag1))
        k2 = min(top_k, len(mag2))
        
        top1_idx = np.argsort(mag1)[-k1:]
        top2_idx = np.argsort(mag2)[-k2:]
        
        # Calculate compression ratio
        original_bytes = data.nbytes
        compressed_bytes = (k1 + k2) * 3 * 8  # freq, amp, phase per helix
        
        sig = WaveSignature(
            modality=modality,
            helix1_frequencies=top1_idx,
            helix1_amplitudes=mag1[top1_idx],
            helix1_phases=np.angle(spec1[top1_idx]),
            helix2_frequencies=top2_idx,
            helix2_amplitudes=mag2[top2_idx],
            helix2_phases=np.angle(spec2[top2_idx]),
            original_shape=original_shape,
            original_dtype=original_dtype,
            top_k=top_k,
            compression_ratio=original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0,
            metadata=metadata or {}
        )
        
        logger.info(f"🌊 Encoded {modality.value}: {original_shape} → {sig.byte_size()} bytes ({sig.compression_ratio:.1f}x)")
        return sig
    
    def decode(self, sig: WaveSignature) -> np.ndarray:
        """
        Decode wave DNA back to original data.
        
        Algorithm:
        1. Reconstruct spectra from top-k frequencies
        2. Inverse FFT each helix
        3. Interleave helices back to original sequence
        4. Reshape to original shape
        """
        # Calculate helix lengths
        total_length = np.prod(sig.original_shape)
        len1 = (total_length + 1) // 2
        len2 = total_length // 2
        
        # Reconstruct spectra
        spec1 = np.zeros(len1, dtype=complex)
        spec2 = np.zeros(len2, dtype=complex)
        
        for f, a, p in zip(sig.helix1_frequencies, sig.helix1_amplitudes, sig.helix1_phases):
            if f < len1:
                spec1[int(f)] = a * np.exp(1j * p)
        
        for f, a, p in zip(sig.helix2_frequencies, sig.helix2_amplitudes, sig.helix2_phases):
            if f < len2:
                spec2[int(f)] = a * np.exp(1j * p)
        
        # Inverse FFT
        helix1 = np.fft.ifft(spec1).real
        helix2 = np.fft.ifft(spec2).real
        
        # Interleave back
        flat = np.zeros(total_length)
        flat[::2] = helix1
        flat[1::2] = helix2
        
        # Reshape and cast
        result = flat.reshape(sig.original_shape)
        
        # Restore dtype
        if 'int' in sig.original_dtype:
            result = np.round(result).astype(sig.original_dtype)
        elif 'float' in sig.original_dtype:
            result = result.astype(sig.original_dtype)
        
        logger.info(f"🌊 Decoded: {sig.original_shape}")
        return result
    
    # =========================================
    # Modality-Specific Helpers
    # =========================================
    
    def encode_image(self, image: np.ndarray, top_k: int = None) -> WaveSignature:
        """Encode image (H, W, C) or (H, W)."""
        return self.encode(image, DataModality.IMAGE, top_k, 
                          metadata={"channels": image.shape[-1] if len(image.shape) == 3 else 1})
    
    def encode_video(self, video: np.ndarray, top_k: int = None) -> WaveSignature:
        """Encode video (T, H, W, C)."""
        return self.encode(video, DataModality.VIDEO, top_k,
                          metadata={"frames": video.shape[0]})
    
    def encode_audio(self, audio: np.ndarray, sample_rate: int = 44100, top_k: int = None) -> WaveSignature:
        """Encode audio waveform."""
        return self.encode(audio, DataModality.AUDIO, top_k,
                          metadata={"sample_rate": sample_rate})
    
    def encode_light_spectrum(self, spectrum: np.ndarray, wavelength_range: Tuple[float, float] = (380, 700), 
                              top_k: int = None) -> WaveSignature:
        """Encode light spectrum data."""
        modality = DataModality.LIGHT_VISIBLE
        if wavelength_range[0] < 380 and wavelength_range[1] > 700:
            modality = DataModality.LIGHT_FULL
        elif wavelength_range[1] < 380:
            modality = DataModality.LIGHT_ULTRAVIOLET
        elif wavelength_range[0] > 700:
            modality = DataModality.LIGHT_INFRARED
        
        return self.encode(spectrum, modality, top_k,
                          metadata={"wavelength_range": wavelength_range})
    
    def encode_sensor(self, data: np.ndarray, sensor_type: str, top_k: int = None) -> WaveSignature:
        """Encode any sensor data."""
        modality_map = {
            "temperature": DataModality.TEMPERATURE,
            "pressure": DataModality.PRESSURE,
            "humidity": DataModality.HUMIDITY,
            "acceleration": DataModality.ACCELERATION,
            "magnetic": DataModality.MAGNETIC,
            "eeg": DataModality.EEG,
            "ecg": DataModality.ECG,
            "emg": DataModality.EMG,
        }
        modality = modality_map.get(sensor_type.lower(), DataModality.TENSOR)
        return self.encode(data, modality, top_k, metadata={"sensor_type": sensor_type})
    
    def encode_cosmic(self, data: np.ndarray, cosmic_type: str, top_k: int = None) -> WaveSignature:
        """Encode cosmic/astronomical data."""
        modality_map = {
            "gravity_wave": DataModality.GRAVITY_WAVE,
            "em_wave": DataModality.EM_WAVE,
            "dark_matter": DataModality.DARK_MATTER,
        }
        modality = modality_map.get(cosmic_type.lower(), DataModality.TENSOR)
        return self.encode(data, modality, top_k, metadata={"cosmic_type": cosmic_type})
    
    # =========================================
    # Quality Metrics
    # =========================================
    
    def calculate_reconstruction_quality(self, original: np.ndarray, 
                                          sig: WaveSignature) -> Dict[str, float]:
        """Calculate reconstruction quality metrics."""
        reconstructed = self.decode(sig)
        
        # Mean Squared Error
        mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
        
        # Peak Signal-to-Noise Ratio
        max_val = np.max(original)
        psnr = 10 * np.log10((max_val ** 2) / mse) if mse > 0 else float('inf')
        
        # Correlation
        corr = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
        
        return {
            "mse": float(mse),
            "psnr": float(psnr),
            "correlation": float(corr),
            "compression_ratio": sig.compression_ratio
        }


# Singleton
_encoder = None

def get_universal_encoder() -> UniversalWaveEncoder:
    global _encoder
    if _encoder is None:
        _encoder = UniversalWaveEncoder()
    return _encoder


# Demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("🌌 UNIVERSAL WAVE ENCODER DEMO")
    print("=" * 70)
    
    encoder = get_universal_encoder()
    
    # 1. Image test
    print("\n[TEST 1] Image Encoding")
    fake_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    sig = encoder.encode_image(fake_image, top_k=32)
    quality = encoder.calculate_reconstruction_quality(fake_image, sig)
    print(f"  Shape: {fake_image.shape}, Compression: {sig.compression_ratio:.1f}x, PSNR: {quality['psnr']:.1f}dB")
    
    # 2. Audio test
    print("\n[TEST 2] Audio Encoding")
    fake_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))  # 1 sec 440Hz
    sig = encoder.encode_audio(fake_audio, sample_rate=44100, top_k=64)
    quality = encoder.calculate_reconstruction_quality(fake_audio, sig)
    print(f"  Samples: {len(fake_audio)}, Compression: {sig.compression_ratio:.1f}x, Corr: {quality['correlation']:.4f}")
    
    # 3. Light spectrum test
    print("\n[TEST 3] Light Spectrum Encoding")
    wavelengths = np.linspace(380, 700, 320)
    intensity = np.exp(-((wavelengths - 550) ** 2) / 5000)  # Peak at green
    sig = encoder.encode_light_spectrum(intensity, wavelength_range=(380, 700), top_k=16)
    quality = encoder.calculate_reconstruction_quality(intensity, sig)
    print(f"  Wavelengths: {len(wavelengths)}, Compression: {sig.compression_ratio:.1f}x, Corr: {quality['correlation']:.4f}")
    
    # 4. Cosmic data test
    print("\n[TEST 4] Gravity Wave Encoding")
    fake_gw = np.sin(2 * np.pi * np.linspace(0, 10, 1000)) * np.exp(-np.linspace(0, 5, 1000))
    sig = encoder.encode_cosmic(fake_gw, "gravity_wave", top_k=32)
    quality = encoder.calculate_reconstruction_quality(fake_gw, sig)
    print(f"  Samples: {len(fake_gw)}, Compression: {sig.compression_ratio:.1f}x, Corr: {quality['correlation']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Universal encoding demo complete!")
    print("   모든 데이터는 파동이다. 파동은 DNA로 압축된다.")
    print("=" * 70)
