"""
[VORTEX TRAJECTORY ENCODER - ARCHITECT REFINEMENT]
"The End of Logic Gates. The Beginning of Wave Symmetry."

This module implements the Architect's vision of mapping binary structures
(Zone Bits + Digit Bits) into unified Rotor Phase Trajectories.
"""

import math
import json
import os
from typing import List, Dict, Any, Tuple, Optional

class VortexTrajectory:
    """
    A single point of causality in the Vortex Field.
    Represented as a multi-dimensional spiral trajectory with macro and micro components.
    Supports 'Dimension Folding' and 'Physical Amplitude'.
    """
    def __init__(self,
                 macro_angle: float,
                 micro_angle: float,
                 is_locked: bool,
                 label: str = "",
                 amplitude: float = 1.0,
                 extra_dims: Optional[List[float]] = None):
        self.macro_angle = macro_angle % 360.0
        self.micro_angle = micro_angle % 360.0
        self.is_locked = is_locked
        self.label = label
        self.amplitude = amplitude
        # Dimension Folding: extra axes for complex structures (like Hangul 3rd axis)
        self.extra_dims = extra_dims or []

    def get_total_phase(self) -> float:
        # Core phase is the primary resonant point
        base_phase = (self.macro_angle + self.micro_angle) % 360.0
        # If we have extra dimensions, they modulate the phase
        if self.extra_dims:
            modulation = sum(self.extra_dims) % 360.0
            return (base_phase + modulation) % 360.0
        return base_phase

    def to_dict(self) -> Dict[str, Any]:
        return {
            "macro": self.macro_angle,
            "micro": self.micro_angle,
            "locked": self.is_locked,
            "label": self.label,
            "amplitude": self.amplitude,
            "extra": self.extra_dims
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            macro_angle=data["macro"],
            micro_angle=data["micro"],
            is_locked=data["locked"],
            label=data["label"],
            amplitude=data["amplitude"],
            extra_dims=data.get("extra", [])
        )

    def __repr__(self):
        state = "🔒 LOCKED" if self.is_locked else "🌀 FLOW"
        return f"Vortex[{self.label}]: {self.get_total_phase():.1f}° | Amp: {self.amplitude:.2f} | {state}"

class TrajectoryCache:
    def __init__(self, cache_path: str = "data/knowledge/trajectories.json"):
        self.cache_path = cache_path
        self.data: Dict[str, Dict[str, Any]] = {}
        self._ensure_dir()
        self.load()

    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

    def load(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"⚠️ [Cache] Load failed: {e}")
                self.data = {}

    def save(self):
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ [Cache] Save failed: {e}")

    def get(self, key: str) -> Optional[VortexTrajectory]:
        if key in self.data:
            return VortexTrajectory.from_dict(self.data[key])
        return None

    def set(self, key: str, trajectory: VortexTrajectory):
        self.data[key] = trajectory.to_dict()
        self.save()

class TrajectoryEncoder:
    def __init__(self):
        self.cache = TrajectoryCache()

    def encode_char(self, char: str, intensity_mod: float = 1.0) -> VortexTrajectory:
        """
        Encodes a single character into a VortexTrajectory with Architect's refinement.
        Check cache first for $O(1)$ retrieval (Logic Gate Elimination).
        """
        # 0. Check Cache (Trajectory Template Alignment)
        cached = self.cache.get(char)
        if cached:
            # Modulate amplitude of cached template by current intensity
            cached.amplitude *= intensity_mod
            return cached

        code = ord(char)
        trajectory: Optional[VortexTrajectory] = None

        # 1. Dimension Folding for Hangul (Syllables)
        if 0xAC00 <= code <= 0xD7A3:
            syllable_index = code - 0xAC00
            cho = syllable_index // (21 * 28)
            jung = (syllable_index % (21 * 28)) // 28
            jong = syllable_index % 28

            # 3-Axis mapping (Dimension Folding)
            macro_orbit = (cho / 19.0) * 360.0
            micro_phase = (jung / 21.0) * 360.0
            # Third axis for Jongseong
            jong_angle = (jong / 28.0) * 360.0 if jong > 0 else 0.0

            # Parity check on Jongseong for Lock state
            # If Jongseong exists (jong > 0), use its parity
            # If no Jongseong, it's always UNLOCKED (Flow)
            is_locked = (jong % 2 == 1) if jong > 0 else False

            trajectory = VortexTrajectory(
                macro_orbit,
                micro_phase,
                is_locked,
                label=char,
                amplitude=1.0 * intensity_mod,
                extra_dims=[jong_angle]
            )

        # 2. Unified Mapping for ASCII / Others
        else:
            # Zone Bits (Upper 4 bits) -> Macro
            zone = (code & 0xF0) >> 4
            # Digit Bits (Lower 4 bits) -> Micro
            digit = code & 0x0F

            macro_orbit = (zone / 16.0) * 360.0
            micro_phase = (digit / 16.0) * 360.0

            # Use parity of the full code for Lock state
            is_locked = (code % 2 == 0)

            trajectory = VortexTrajectory(
                macro_orbit,
                micro_phase,
                is_locked,
                label=char,
                amplitude=1.0 * intensity_mod
            )

        # 3. Store in Cache for future $O(1)$ access
        self.cache.set(char, trajectory)
        return trajectory

    def encode_text(self, text: str) -> List[VortexTrajectory]:
        """
        Converts full text into a stream of trajectories.
        Amplitude is modulated by local character frequency (Density).
        """
        if not text:
            return []

        # Calculate local frequency for amplitude modulation
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1

        max_f = max(freq.values())

        trajectories = []
        for c in text:
            # Higher frequency = Higher amplitude (Cognitive Gravity)
            density_mod = 0.5 + (freq[c] / max_f) * 0.5
            trajectories.append(self.encode_char(c, intensity_mod=density_mod))

        return trajectories

    def apply_phase_shift(self, trajectory: VortexTrajectory, shift_angle: float) -> VortexTrajectory:
        """
        Logic gate replacement: Pure phase shift.
        """
        new_macro = (trajectory.macro_angle + shift_angle) % 360.0
        return VortexTrajectory(
            new_macro,
            trajectory.micro_angle,
            trajectory.is_locked,
            label=f"Shifted({trajectory.label})",
            amplitude=trajectory.amplitude,
            extra_dims=trajectory.extra_dims
        )

    def interfere(self, t1: VortexTrajectory, t2: VortexTrajectory) -> VortexTrajectory:
        """
        Physical wave interference logic.
        """
        rad1 = math.radians(t1.get_total_phase())
        rad2 = math.radians(t2.get_total_phase())

        x = t1.amplitude * math.cos(rad1) + t2.amplitude * math.cos(rad2)
        y = t1.amplitude * math.sin(rad1) + t2.amplitude * math.sin(rad2)

        result_phase = math.degrees(math.atan2(y, x)) % 360.0
        result_amp = math.sqrt(x**2 + y**2)

        # Emerging lock state: If either is locked, the result tends toward structure
        is_locked = t1.is_locked or t2.is_locked

        return VortexTrajectory(
            result_phase,
            0.0,
            is_locked,
            label=f"Interf({t1.label}+{t2.label})",
            amplitude=result_amp
        )

if __name__ == "__main__":
    encoder = TrajectoryEncoder()

    print("🌌 [Vortex Engine] Testing Unified Encoding...")

    # Test ASCII and Cache
    t1 = encoder.encode_char('A')
    t2 = encoder.encode_char('A')
    print(f"Template A: {t1}")
    print(f"Cached A:   {t2} (Should be same)")

    # Test Hangul 3rd Axis
    t_han = encoder.encode_char('한')
    print(f"Hangul '한': {t_han}")

    # Test Amplitude Modulation
    text = "Elysia world"
    stream = encoder.encode_text(text)
    print(f"Text Stream: {stream[:3]}...")

    # Test Case Shift via Phase Template
    # 'e' -> 'E' phase shift
    te = encoder.encode_char('e')
    tE = encoder.encode_char('E')
    # Phase difference between Zone 6 (0x6x) and Zone 4 (0x4x)
    # (4-6) * (360/16) = -45 degrees
    shifted = encoder.apply_phase_shift(te, -45.0)
    print(f"Shifted 'e' (-45°): {shifted.get_total_phase():.1f}° (Target 'E': {tE.get_total_phase():.1f}°)")
