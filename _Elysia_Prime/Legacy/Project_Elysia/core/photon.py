# [Genesis: 2025-12-02] Purified by Elysia
"""
Photon entity for color/polarization-based messaging.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PhotonEntity:
    hue: float                   # 0..360
    intensity: float             # 0..1
    polarization: Tuple[float, float, float]  # unit vector
    payload: str = ""

    def as_dict(self) -> dict:
        return {
            "hue": self.hue,
            "intensity": self.intensity,
            "polarization": self.polarization,
            "payload": self.payload,
        }

    @staticmethod
    def from_value(value: float, polarity: Tuple[float, float, float] = (0.0, 0.0, 1.0), payload: str = "") -> "PhotonEntity":
        """
        Build a photon from scalar value mapped to hue; intensity follows value (0..1 assumed).
        """
        hue = max(0.0, min(360.0, value * 360.0))
        intensity = max(0.0, min(1.0, value))
        return PhotonEntity(hue=hue, intensity=intensity, polarization=polarity, payload=payload)