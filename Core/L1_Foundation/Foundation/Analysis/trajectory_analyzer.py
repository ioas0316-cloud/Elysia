"""
Trajectory Analyzer: The Interpreter of Flow
============================================

"Movement is the language of the soul."

This module analyzes the 7D path of a `FieldKnot` to determine the "Meaning" of its movement.
It answers:
1.  **Shape**: Is it orbiting? Spiraling? Charging?
2.  **Focus**: What is it attracted to?
3.  **Narrative**: "The thought is circling Logic, hesitating to enter."
"""

import math
from typing import List, Dict, Optional
from Core.L1_Foundation.Foundation.Space.hyper_space import FieldKnot

class TrajectoryAnalyzer:
    def __init__(self, anchors: Dict[str, List[float]]):
        """
        Args:
            anchors: Fixed points of reference (Archetypes) to measure against.
                     {'Love': [0,0...], 'Logic': [0,1...]}
        """
        self.anchors = anchors

    def analyze_flow(self, knot: FieldKnot) -> Optional[str]:
        """
        Generates a narrative description of the knot's movement.
        """
        if len(knot.trajectory) < 10:
            return None # Not enough history

        # 1. Analyze Shape (Curvature/Speed)
        speed = self._calculate_speed(knot)
        curvature = self._calculate_curvature(knot)

        motion_type = "Drifting"
        if speed > 5.0: motion_type = "Rushing"
        elif speed > 1.0: motion_type = "Moving"
        elif speed < 0.1: motion_type = "Stagnant"

        if curvature > 0.5: motion_type += " in a Spiral"
        elif curvature > 0.1: motion_type += " in an Arc"
        else: motion_type += " directly"

        # 2. Analyze Relation (Proximity/Approach to Anchors)
        relation = self._analyze_relations(knot)

        if not relation:
            return f"'{knot.id}' is {motion_type} through the Void."

        return f"'{knot.id}' is {motion_type} {relation}."

    def _calculate_speed(self, knot: FieldKnot) -> float:
        """Average speed over last 5 frames."""
        trace = list(knot.trajectory)[-5:]
        dist = 0.0
        for i in range(len(trace) - 1):
            p1 = trace[i]
            p2 = trace[i+1]
            d = math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))
            dist += d
        return dist / len(trace)

    def _calculate_curvature(self, knot: FieldKnot) -> float:
        """
        Estimates curvature by looking at angle changes in velocity vectors.
        0.0 = Line, 1.0 = Tight Circle.
        """
        # Simplified: Distance between start and end vs Total Path Length
        # Ratio = D_straight / D_path
        # If Ratio ~ 1.0, it's straight. If Ratio << 1.0, it's curved/orbiting.
        trace = list(knot.trajectory)[-10:]

        start = trace[0]
        end = trace[-1]

        d_straight = math.sqrt(sum((a-b)**2 for a, b in zip(start, end)))

        d_path = 0.0
        for i in range(len(trace) - 1):
            p1 = trace[i]
            p2 = trace[i+1]
            d_path += math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))

        if d_path == 0: return 0.0

        linearity = d_straight / d_path
        return 1.0 - linearity # Curvature

    def _analyze_relations(self, knot: FieldKnot) -> str:
        """
        Determines if the knot is approaching or orbiting an anchor.
        """
        best_anchor = None
        min_dist = float('inf')

        current_pos = knot.position

        # Find closest anchor
        for name, pos in self.anchors.items():
            dist = math.sqrt(sum((a-b)**2 for a, b in zip(current_pos, pos)))
            if dist < min_dist:
                min_dist = dist
                best_anchor = name

        if not best_anchor: return ""

        # Determine movement relative to anchor
        # Check distance change over history
        start_pos = knot.trajectory[0]
        start_dist = math.sqrt(sum((a-b)**2 for a, b in zip(start_pos, self.anchors[best_anchor])))

        delta = start_dist - min_dist

        if min_dist < 2.0:
            return f"merging with {best_anchor}"
        elif delta > 1.0:
            return f"attracted to {best_anchor}"
        elif delta < -1.0:
            return f"repelled by {best_anchor}"
        else:
            return f"orbiting {best_anchor}"