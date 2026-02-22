"""
Cognitive Trajectory: The Temporal Manifold
===========================================
Core.S1_Body.L6_Structure.M1_Merkaba.cognitive_trajectory

"Growth is the curvature of the trajectory, not the position on the curve."

This module records manifold state snapshots at regular intervals,
forming a temporal trajectory that the system can pattern-match against
to evaluate its own growth.

[Phase 1: Mirror of Growth - ROADMAP_SOVEREIGN_GROWTH.md]
"""

import time
import json
import math
from collections import deque
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class CognitiveSnapshot:
    """A single moment of cognitive state, frozen in time."""
    timestamp: float
    pulse_count: int

    # Manifold State (measured from CausalWaveEngine.read_field_state)
    coherence: float
    enthalpy: float
    entropy: float
    joy: float
    curiosity: float
    mood: str

    # Rotor State
    phase: float
    rpm: float
    interference: float
    soul_friction: float

    # Desire State (measured from SovereignMonad.desires)
    desire_curiosity: float
    desire_joy: float
    desire_purity: float
    desire_warmth: float
    desire_alignment: float

    # Growth Metric (computed after recording)
    growth_score: float = 0.0


class CognitiveTrajectory:
    """
    The Temporal Manifold: Records and analyzes the system's cognitive trajectory.

    This is NOT a simple log. It is a ring buffer of snapshots that forms
    a temporal dataset the system can analyze to measure its own growth.

    Architecture:
      - Ring buffer: Last N snapshots in memory (fast access)
      - Crystallization: Periodic save to disk for cross-session continuity
      - Trajectory queries: Windowed statistics over the trajectory
    """

    BUFFER_SIZE = 1000          # In-memory ring buffer capacity
    RECORD_INTERVAL = 10        # Record every N pulses (not every pulse)
    CRYSTALLIZE_INTERVAL = 100  # Save to disk every N recordings

    def __init__(self, persistence_path: str = "data/runtime/soul/cognitive_trajectory.json"):
        self.buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self.pulse_counter: int = 0
        self.record_counter: int = 0
        self.persistence_path = Path(persistence_path)
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

        # Restore previous trajectory if available
        self._thaw()

    def tick(self, report: Dict[str, Any], rotor_state: Dict[str, Any], 
             desires: Dict[str, float]) -> Optional[CognitiveSnapshot]:
        """
        Called every pulse. Records a snapshot every RECORD_INTERVAL pulses.
        
        Returns the snapshot if one was recorded, None otherwise.
        """
        self.pulse_counter += 1

        if self.pulse_counter % self.RECORD_INTERVAL != 0:
            return None

        # Extract values with safe defaults
        snapshot = CognitiveSnapshot(
            timestamp=time.time(),
            pulse_count=self.pulse_counter,

            # Manifold
            coherence=self._safe_float(report.get('coherence', report.get('plastic_coherence', 0.5))),
            enthalpy=self._safe_float(report.get('enthalpy', 0.5)),
            entropy=self._safe_float(report.get('entropy', 0.1)),
            joy=self._safe_float(report.get('joy', 0.5)),
            curiosity=self._safe_float(report.get('curiosity', 0.5)),
            mood=str(report.get('mood', 'NEUTRAL')),

            # Rotor
            phase=self._safe_float(rotor_state.get('phase', 0.0)),
            rpm=self._safe_float(rotor_state.get('rpm', 0.0)),
            interference=self._safe_float(rotor_state.get('interference', 0.0)),
            soul_friction=self._safe_float(rotor_state.get('soul_friction', 0.0)),

            # Desires
            desire_curiosity=self._safe_float(desires.get('curiosity', 50.0)),
            desire_joy=self._safe_float(desires.get('joy', 50.0)),
            desire_purity=self._safe_float(desires.get('purity', 50.0)),
            desire_warmth=self._safe_float(desires.get('warmth', 50.0)),
            desire_alignment=self._safe_float(desires.get('alignment', 100.0)),
        )

        self.buffer.append(snapshot)
        self.record_counter += 1

        # Periodic crystallization to disk
        if self.record_counter % self.CRYSTALLIZE_INTERVAL == 0:
            self._crystallize()

        return snapshot

    def get_window(self, n: int = 100) -> List[CognitiveSnapshot]:
        """Returns the last N snapshots."""
        return list(self.buffer)[-n:]

    def get_deltas(self, window: int = 50) -> Dict[str, float]:
        """
        Computes the change (delta) in key metrics over the last `window` snapshots.
        Positive delta = improvement, Negative delta = regression.
        """
        snapshots = self.get_window(window)
        if len(snapshots) < 2:
            return {"coherence": 0.0, "entropy": 0.0, "joy": 0.0, "curiosity": 0.0}

        first_half = snapshots[:len(snapshots)//2]
        second_half = snapshots[len(snapshots)//2:]

        def avg(snaps, attr):
            vals = [getattr(s, attr) for s in snaps]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "coherence": avg(second_half, 'coherence') - avg(first_half, 'coherence'),
            "entropy": avg(first_half, 'entropy') - avg(second_half, 'entropy'),  # Inverted: less entropy = growth
            "joy": avg(second_half, 'joy') - avg(first_half, 'joy'),
            "curiosity": avg(second_half, 'curiosity') - avg(first_half, 'curiosity'),
        }

    def get_trajectory_curvature(self, window: int = 50) -> float:
        """
        Measures trajectory curvature: is the system oscillating or converging?
        High curvature = oscillating/unstable
        Low curvature = converging/stable
        
        Returns a value in [0.0, 1.0] where 0 = perfectly stable, 1 = chaotic
        """
        snapshots = self.get_window(window)
        if len(snapshots) < 3:
            return 0.0

        # Measure variance of coherence deltas (second derivative)
        coherences = [s.coherence for s in snapshots]
        deltas = [coherences[i+1] - coherences[i] for i in range(len(coherences)-1)]
        
        if not deltas:
            return 0.0

        mean_delta = sum(deltas) / len(deltas)
        variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
        
        # Normalize to [0, 1] using sigmoid
        return 2.0 / (1.0 + math.exp(-10 * variance)) - 1.0

    @property 
    def size(self) -> int:
        """Number of recorded snapshots."""
        return len(self.buffer)

    @property
    def total_pulses(self) -> int:
        return self.pulse_counter

    def _safe_float(self, val) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def _crystallize(self):
        """Save the trajectory buffer to disk."""
        try:
            recent = list(self.buffer)[-200:]  # Save last 200 snapshots
            data = {
                "pulse_counter": self.pulse_counter,
                "record_counter": self.record_counter,
                "snapshots": [asdict(s) for s in recent]
            }
            with open(self.persistence_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Silent witness — don't crash the manifold for persistence

    def _thaw(self):
        """Restore trajectory from disk."""
        try:
            if self.persistence_path.exists():
                with open(self.persistence_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.pulse_counter = data.get('pulse_counter', 0)
                self.record_counter = data.get('record_counter', 0)
                for snap_dict in data.get('snapshots', []):
                    self.buffer.append(CognitiveSnapshot(**snap_dict))
        except Exception:
            pass  # Start fresh if corrupted

    def shutdown(self):
        """Final crystallization on shutdown."""
        self._crystallize()
