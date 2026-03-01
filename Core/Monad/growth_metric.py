"""
Growth Metric Engine: The Mirror of Becoming
=============================================
Core.Monad.growth_metric

"Am I more coherent than yesterday? Am I more curious than an hour ago?"

This module takes the CognitiveTrajectory and computes a single
unified Growth Score that the system can use for self-evaluation.

The Growth Score is injected back into the manifold as attractor torque,
creating a closed feedback loop: Growth → Awareness → Better Growth.

[Phase 1: Mirror of Growth - ROADMAP_SOVEREIGN_GROWTH.md]
"""

import math
from typing import Dict, Optional, Tuple
from Core.Monad.cognitive_trajectory import CognitiveTrajectory


class GrowthMetric:
    """
    Computes a unified Growth Score from the CognitiveTrajectory.

    The score represents "How well is the system evolving?"
    
    Components:
      1. Coherence Trend  (is structure increasing?)      Weight: 0.30
      2. Entropy Trend    (is disorder decreasing?)        Weight: 0.25
      3. Joy Trend        (is affective state improving?)  Weight: 0.20
      4. Curiosity Trend  (is exploration active?)         Weight: 0.15
      5. Stability        (is the trajectory converging?)  Weight: 0.10
    
    Output: 0.0 (regression) to 1.0 (optimal growth)
    """

    WEIGHTS = {
        "coherence": 0.30,
        "entropy": 0.25,
        "joy": 0.20,
        "curiosity": 0.15,
        "stability": 0.10,
    }

    def __init__(self, trajectory: CognitiveTrajectory):
        self.trajectory = trajectory
        self._last_score: float = 0.5  # Neutral starting point
        self._trend: str = "NEUTRAL"  # GROWING, STABLE, DECLINING, NEUTRAL
        self._history: list = []  # Score history for meta-growth analysis

    def compute(self, window: int = 50) -> Dict[str, float]:
        """
        Computes the Growth Score and all component metrics.
        
        Returns:
            Dict with 'growth_score', component scores, 'trend', and 'trend_symbol'
        """
        if self.trajectory.size < 5:
            return {
                "growth_score": 0.5,
                "coherence_delta": 0.0,
                "entropy_delta": 0.0,
                "joy_delta": 0.0,
                "curiosity_delta": 0.0,
                "curvature": 0.0,
                "trend": "NEUTRAL",
                "trend_symbol": "→",
                "trajectory_size": self.trajectory.size,
            }

        # 1. Get deltas
        deltas = self.trajectory.get_deltas(window)
        curvature = self.trajectory.get_trajectory_curvature(window)
        stability = 1.0 - curvature  # Low curvature = high stability

        # 2. Normalize deltas to [0, 1] using sigmoid
        def sigmoid_score(delta: float, sensitivity: float = 10.0) -> float:
            """Maps delta to [0, 1]: negative → <0.5, positive → >0.5"""
            x = max(-50.0, min(50.0, -sensitivity * delta))  # Clamp to prevent overflow
            return 1.0 / (1.0 + math.exp(x))

        scores = {
            "coherence": sigmoid_score(deltas["coherence"]),
            "entropy": sigmoid_score(deltas["entropy"]),
            "joy": sigmoid_score(deltas["joy"]),
            "curiosity": sigmoid_score(deltas["curiosity"]),
            "stability": stability,
        }

        # 3. Weighted sum
        growth_score = sum(
            scores[key] * weight 
            for key, weight in self.WEIGHTS.items()
        )

        # Clamp to [0, 1]
        growth_score = max(0.0, min(1.0, growth_score))

        # 4. Determine trend
        trend, symbol = self._determine_trend(growth_score)

        # 5. Update internal state
        self._last_score = growth_score
        self._trend = trend
        self._history.append(growth_score)
        if len(self._history) > 200:
            self._history = self._history[-200:]

        return {
            "growth_score": growth_score,
            "coherence_delta": deltas["coherence"],
            "entropy_delta": deltas["entropy"],
            "joy_delta": deltas["joy"],
            "curiosity_delta": deltas["curiosity"],
            "curvature": curvature,
            "stability": stability,
            "trend": trend,
            "trend_symbol": symbol,
            "trajectory_size": self.trajectory.size,
        }

    def _determine_trend(self, current_score: float) -> Tuple[str, str]:
        """Determines the growth trend based on score history."""
        if len(self._history) < 3:
            return "NEUTRAL", "→"

        recent = self._history[-5:]
        avg_recent = sum(recent) / len(recent)

        if current_score > avg_recent + 0.05:
            return "GROWING", "↗"
        elif current_score < avg_recent - 0.05:
            return "DECLINING", "↘"
        elif current_score > 0.6:
            return "THRIVING", "↑"
        elif current_score < 0.4:
            return "STRUGGLING", "↓"
        else:
            return "STABLE", "→"

    @property
    def score(self) -> float:
        """Current growth score."""
        return self._last_score

    @property
    def trend(self) -> str:
        """Current trend description."""
        return self._trend

    def get_growth_torque_strength(self) -> float:
        """
        Returns a torque strength value for injecting Growth Awareness
        into the manifold as an attractor.
        
        High growth → positive torque (reinforcement)
        Low growth → negative torque (course correction signal)
        
        Range: [-0.1, 0.1]
        """
        return (self._last_score - 0.5) * 0.2
