"""
Resonance Scanner (Hypersphere Navigation)
==========================================
"The compass that points not North, but to Meaning."

This module implements the 'Resonance Scanner', a search engine for the
Hypersphere. Instead of string matching, it uses 'Frequency Sweeps' and
'Phase Matching' to locate Fluxlights (Souls) within the Yggdrasil network.

Core Logic:
- Frequency Sweep: Iterates through the spectrum (Soul List) to find
  energetic matches (e.g., High Energy / Low Energy).
- Phase Matching: Calculates the resonance (dot product/alignment) between
  a target probe and existing souls to find "Kindred Spirits."
"""

import logging
from typing import List, Dict, Optional, Tuple
from Core.Keystone.infinite_hyperquaternion import InfiniteHyperQubit
from Core.System.yggdrasil import yggdrasil
from Core.Cognition.living_village import village

logger = logging.getLogger("ResonanceScanner")

class ScanResult:
    def __init__(self, soul: InfiniteHyperQubit, resonance_score: float):
        self.soul = soul
        self.resonance_score = resonance_score

    def __repr__(self):
        return f"<{self.soul.name} | Res:{self.resonance_score:.4f}>"

class ResonanceScanner:
    """
    The Navigation System for the Soul Archipelago.
    """

    def __init__(self):
        pass

    def _get_all_souls(self) -> List[InfiniteHyperQubit]:
        """Retrieves all registered souls from Yggdrasil and the Village."""
        # Combine sources (currently Yggdrasil.fluxlights is the master registry)
        # Assuming Yggdrasil tracks all.
        return list(yggdrasil.fluxlights.values())

    def scan_by_phase(self,
                      probe_soul: InfiniteHyperQubit,
                      threshold: float = 0.5,
                      limit: int = 10) -> List[ScanResult]:
        """
        Finds souls that 'resonate' with the probe.
        Useful for: "Find me someone like Elysia" or "Find a compatible partner."
        """
        candidates = self._get_all_souls()
        results = []

        for soul in candidates:
            # Skip self
            if soul.id == probe_soul.id:
                continue

            score = probe_soul.resonate_with(soul)
            if score >= threshold:
                results.append(ScanResult(soul, score))

        # Sort by highest resonance
        results.sort(key=lambda x: x.resonance_score, reverse=True)
        return results[:limit]

    def scan_by_frequency(self,
                          axis: str,
                          target_value: float,
                          tolerance: float = 0.2) -> List[ScanResult]:
        """
        Scans for souls with a specific coordinate value.

        Args:
            axis: 'w' (Nature), 'x' (Perception), 'y' (Judgment), 'z' (Lifestyle)
            target_value: -1.0 to 1.0
        """
        candidates = self._get_all_souls()
        results = []

        for soul in candidates:
            # Extract coordinate value
            val = getattr(soul.state, axis, 0.0)

            diff = abs(val - target_value)
            score = 1.0 - diff # Simple linear resonance based on proximity

            if diff <= tolerance:
                results.append(ScanResult(soul, score))

        results.sort(key=lambda x: x.resonance_score, reverse=True)
        return results

    def quick_scan(self):
        """Generates a text-based map of the current Soul Population."""
        souls = self._get_all_souls()
        print(f"\n  Resonance Scan Initiated... Found {len(souls)} signals.\n")

        for soul in souls:
            # Simplified visualization
            s = soul.state
            print(f"  {soul.name} ({soul.value})")
            print(f"   Coords: [w:{s.w:.2f}, x:{s.x:.2f}, y:{s.y:.2f}, z:{s.z:.2f}]")
            print(f"   Dominant: {soul.observe()['dominant_basis']}")
            print("")

# Singleton
resonance_scanner = ResonanceScanner()
