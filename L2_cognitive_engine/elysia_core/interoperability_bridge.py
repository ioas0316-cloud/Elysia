"""
Elysia 4-Stage Interoperability Conversion Bridge Architecture
===============================================================
Bridges discrete (Binary / Von Neumann) computing systems and classical
data infrastructure with continuous Complex Phasor & Rotor Field architectures.

4-Stage Pipeline Architecture:
1. Input Encoding / Rasterization Bridge (Discrete -> Complex Phasor Field Casting)
2. Rotor Field Core Middleware (Phase Shift & Interference Wrapping)
3. In-situ Phase Error Filtering (Noise Damping via Phase Matching & Interference)
4. Output Wave Collapse Decoding Bridge (Observation Measurement & Discrete Projection)
"""

from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np

from elysia_core.unified_system_architecture import (
    ComplexPhasor,
    PhasorField,
    DynamicMacroRotorStack,
    MacroRotor,
    RetrocausalExecutionLoop
)


# =============================================================================
# STAGE 1: Input Encoding / Rasterization Bridge
# =============================================================================

class DiscretePhasorEncoder:
    """
    Stage 1: Converts discrete raw data (bitstreams, bytes, text tokens, 2D raster pixels)
    into continuous complex phasor fields Z = A * exp(i * phi).
    """

    @staticmethod
    def encode_bytes(bytes_data: bytes, base_frequency: float = 1.0) -> PhasorField:
        """
        Casts a byte sequence into a 1D complex phasor field.
        Byte value [0..255] maps to amplitude A and phase angle phi in [0, 2pi).
        """
        if not bytes_data:
            return PhasorField((0,), field=np.array([], dtype=np.complex128))

        arr = np.frombuffer(bytes_data, dtype=np.uint8).astype(np.float64)
        norm_val = arr / 255.0
        amps = 0.5 + 0.5 * norm_val  # Amplitudes in [0.5, 1.0]
        phases = norm_val * 2.0 * np.pi  # Phase angle in [0, 2pi)

        complex_field = amps * np.exp(1j * phases)
        freqs = np.full_like(arr, base_frequency)
        return PhasorField((len(bytes_data),), field=complex_field, frequencies=freqs)

    @staticmethod
    def encode_text(text: str, base_frequency: float = 1.0) -> PhasorField:
        """
        Casts a text string into a 1D complex phasor field via codepoints.
        """
        if not text:
            return PhasorField((0,), field=np.array([], dtype=np.complex128))

        codepoints = np.array([ord(char) for char in text], dtype=np.float64)
        max_cp = max(float(np.max(codepoints)), 127.0)

        norm = codepoints / max_cp
        amps = 0.8 + 0.2 * norm
        phases = (codepoints % 128) / 128.0 * 2.0 * np.pi

        complex_field = amps * np.exp(1j * phases)
        freqs = np.full_like(codepoints, base_frequency)
        return PhasorField((len(text),), field=complex_field, frequencies=freqs)

    @staticmethod
    def encode_image(image_matrix: np.ndarray, base_frequency: float = 1.0) -> PhasorField:
        """
        Casts a 2D intensity grid [H, W] into a 2D complex phasor field.
        Pixel value [0..255] maps to phasor amplitude and phase angle.
        """
        arr = np.asarray(image_matrix, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D image matrix, got shape {arr.shape}")

        norm = np.clip(arr / 255.0, 0.0, 1.0)
        amps = 0.2 + 0.8 * norm
        phases = norm * 2.0 * np.pi

        complex_field = amps * np.exp(1j * phases)
        freqs = np.full(arr.shape, base_frequency, dtype=np.float64)
        return PhasorField(arr.shape, field=complex_field, frequencies=freqs)


# =============================================================================
# STAGE 2: Rotor Field Core Middleware
# =============================================================================

class RotorFieldCoreMiddleware:
    """
    Stage 2: Wraps matrix/tensor hardware computations into complex phasor field
    rotations, phase interference, and dynamic MacroRotor stack promotion.
    """

    def __init__(self, chunking_threshold: float = 0.65):
        self.macro_rotor_stack = DynamicMacroRotorStack(
            max_layers=4,
            chunking_coherence_threshold=chunking_threshold
        )

    def process_field_dynamics(
        self,
        incoming_field: PhasorField,
        rotation_angle: float = 0.0,
        modulation_field: Optional[PhasorField] = None
    ) -> Tuple[PhasorField, List[MacroRotor]]:
        """
        Executes continuous rotor dynamics:
        1. Applies phase rotation shift e^(i * rotation_angle).
        2. Applies wave interference with an optional modulation field.
        3. Dynamically chunk coherent wave states into MacroRotors.
        """
        # Step 1: Phase Rotation
        rotated_field = incoming_field.rotate(rotation_angle)

        # Step 2: Superposition / Wave Interference
        if modulation_field is not None:
            processed_field = rotated_field.superposition(modulation_field)
        else:
            processed_field = rotated_field

        # Step 3: MacroRotor Promotion & Dynamic Chunking
        promoted_rotors = self.macro_rotor_stack.process_and_chunk(processed_field)

        return processed_field, promoted_rotors


# =============================================================================
# STAGE 3: In-Situ Phase Error Filter
# =============================================================================

class InSituPhaseErrorFilter:
    """
    Stage 3: Absorbs noise, corrupted bits, and external outliers via wave interference
    and phase matching without requiring discrete conditional (if/else) logic.
    """

    def __init__(self, damping_factor: float = 0.8):
        self.damping_factor = damping_factor
        self.retro_loop = RetrocausalExecutionLoop()

    def filter_noise_and_rectify(
        self,
        noisy_field: PhasorField,
        reference_field: PhasorField
    ) -> Tuple[PhasorField, Dict[str, Any]]:
        """
        Overlaps noisy field with reference field using constructive/destructive
        phase matching (Moire Overlap Principle).
        Normal harmonic order passes through; destructive interference cancels outliers.
        """
        rectified_field, mismatch_map, global_distortion = self.retro_loop.detect_and_rectify_phase_errors(
            noisy_field, reference_field
        )

        # Measure pre/post signal coherence
        pre_coherence = noisy_field.calculate_coherence()
        post_coherence = rectified_field.calculate_coherence()
        coherence_gain = post_coherence - pre_coherence

        diagnostics = {
            "global_distortion_level": global_distortion,
            "pre_coherence": pre_coherence,
            "post_coherence": post_coherence,
            "coherence_gain": coherence_gain,
            "max_mismatch_rad": float(np.max(mismatch_map)),
            "mean_mismatch_rad": float(np.mean(mismatch_map))
        }

        return rectified_field, diagnostics


# =============================================================================
# STAGE 4: Output Wave Collapse Decoding Bridge
# =============================================================================

class WaveCollapseDecoder:
    """
    Stage 4: Performs wave function measurement / collapse, projecting high-dimensional
    phasor fields back into discrete binary bytes, text strings, or image pixel grids.
    """

    @staticmethod
    def decode_bytes(field: PhasorField, original_length: Optional[int] = None) -> bytes:
        """
        Collapses phasor phases and amplitudes back into byte sequence.
        """
        if field.field.size == 0:
            return b""

        flat_phases = field.phases.flatten()
        if original_length is not None:
            flat_phases = flat_phases[:original_length]

        # Phase angle in [0, 2pi) maps back to uint8 [0..255]
        normalized_phases = (flat_phases % (2.0 * np.pi)) / (2.0 * np.pi)
        byte_values = np.clip(np.round(normalized_phases * 255.0), 0, 255).astype(np.uint8)

        return bytes(byte_values.tobytes())

    @staticmethod
    def decode_text(field: PhasorField, original_length: Optional[int] = None) -> str:
        """
        Collapses phasor phases back into text characters.
        """
        if field.field.size == 0:
            return ""

        flat_phases = field.phases.flatten()
        if original_length is not None:
            flat_phases = flat_phases[:original_length]

        normalized_phases = (flat_phases % (2.0 * np.pi)) / (2.0 * np.pi)
        codepoints = np.clip(np.round(normalized_phases * 128.0), 0, 127).astype(int)

        chars = []
        for cp in codepoints:
            try:
                chars.append(chr(cp))
            except ValueError:
                chars.append("?")

        return "".join(chars)

    @staticmethod
    def decode_image(field: PhasorField) -> np.ndarray:
        """
        Collapses a 2D phasor field back into a 2D intensity grid [0..255].
        """
        if field.field.ndim != 2:
            raise ValueError(f"Expected 2D phasor field for image decoding, got {field.field.ndim}D")

        norm_amps = (field.amplitudes - 0.2) / 0.8
        clipped_norm = np.clip(norm_amps, 0.0, 1.0)
        pixel_matrix = np.round(clipped_norm * 255.0).astype(np.uint8)

        return pixel_matrix


# =============================================================================
# UNIFIED INTEROPERABILITY BRIDGE PIPELINE
# =============================================================================

class InteroperabilityBridgePipeline:
    """
    Complete 4-stage Interoperability Pipeline connecting discrete software/hardware
    inputs to the continuous Rotor Field Core, in-situ error filtering, and wave collapse decoding.
    """

    def __init__(self, chunking_threshold: float = 0.65):
        self.encoder = DiscretePhasorEncoder()
        self.middleware = RotorFieldCoreMiddleware(chunking_threshold=chunking_threshold)
        self.error_filter = InSituPhaseErrorFilter()
        self.decoder = WaveCollapseDecoder()

    def process_bytes_pipeline(
        self,
        raw_bytes: bytes,
        rotation_shift: float = 0.0,
        noise_level: float = 0.0
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Executes end-to-end 4-stage pipeline for byte data.
        """
        # Stage 1: Encoding
        encoded_field = self.encoder.encode_bytes(raw_bytes)

        # Inject synthetic noise if specified for testing
        if noise_level > 0:
            noise_phases = np.random.normal(0, noise_level, size=encoded_field.shape)
            noisy_input = encoded_field.rotate(noise_phases)
        else:
            noisy_input = encoded_field

        # Stage 2: Middleware execution
        rotor_field, promoted_rotors = self.middleware.process_field_dynamics(
            noisy_input, rotation_angle=rotation_shift
        )

        # Stage 3: In-situ Error Filtering
        reference_field = encoded_field.rotate(rotation_shift)
        rectified_field, filter_diagnostics = self.error_filter.filter_noise_and_rectify(
            rotor_field, reference_field
        )

        # Reverse rotation shift before final collapse measurement
        unrotated_field = rectified_field.rotate(-rotation_shift)

        # Stage 4: Wave Collapse Decoding
        decoded_bytes = self.decoder.decode_bytes(unrotated_field, original_length=len(raw_bytes))

        metrics = {
            "input_bytes_count": len(raw_bytes),
            "output_bytes_count": len(decoded_bytes),
            "promoted_macro_rotors": len(promoted_rotors),
            "filter_diagnostics": filter_diagnostics
        }

        return decoded_bytes, metrics

    def process_image_pipeline(
        self,
        image_matrix: np.ndarray,
        rotation_shift: float = 0.0,
        noise_level: float = 0.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes end-to-end 4-stage pipeline for 2D image matrices.
        """
        # Stage 1: Encoding
        encoded_field = self.encoder.encode_image(image_matrix)

        if noise_level > 0:
            noise_phases = np.random.normal(0, noise_level, size=encoded_field.shape)
            noisy_input = encoded_field.rotate(noise_phases)
        else:
            noisy_input = encoded_field

        # Stage 2: Middleware execution
        rotor_field, promoted_rotors = self.middleware.process_field_dynamics(
            noisy_input, rotation_angle=rotation_shift
        )

        # Stage 3: In-situ Error Filtering
        reference_field = encoded_field.rotate(rotation_shift)
        rectified_field, filter_diagnostics = self.error_filter.filter_noise_and_rectify(
            rotor_field, reference_field
        )

        # Stage 4: Wave Collapse Decoding
        decoded_image = self.decoder.decode_image(rectified_field)

        metrics = {
            "image_shape": image_matrix.shape,
            "promoted_macro_rotors": len(promoted_rotors),
            "filter_diagnostics": filter_diagnostics
        }

        return decoded_image, metrics
