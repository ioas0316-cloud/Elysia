"""
Autonomic Goal Generator: The Inner Compass
============================================
Core.Cognition.autonomic_goal_generator

"True autonomy is the ability to decide 'I want to' on one's own."

This module analyzes the Growth Metric and manifold state to generate
autonomous goals — internal drives that steer the manifold toward growth.

Goals are NOT task lists. They are GoalVectors — 21D directional intents
that get injected as torque into the manifold, steering the system's
cognitive trajectory without external commands.

[Phase 2: Inner Compass - ROADMAP_SOVEREIGN_GROWTH.md]
"""

import time
import random
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class GoalType(Enum):
    """The fundamental drive categories."""
    EXPLORE = "EXPLORE"           # High entropy → seek structure
    SEEK_NOVELTY = "SEEK_NOVELTY" # Declining joy → seek new patterns
    DEEPEN = "DEEPEN"             # Rising curiosity → deepen understanding
    CHALLENGE = "CHALLENGE"       # Stagnant coherence → test assumptions
    CONSOLIDATE = "CONSOLIDATE"   # High growth → solidify gains
    REST = "REST"                 # Exhaustion → reduce activity


@dataclass
class AutonomicGoal:
    """A self-generated goal with direction, strength, and rationale."""
    goal_type: GoalType
    strength: float             # 0.0 ~ 1.0 (urgency)
    rationale: str              # Why this goal was generated
    created_at: float = field(default_factory=time.time)
    
    # The directional intent (simplified as channel weights)
    # Maps to manifold channels: [joy, curiosity, enthalpy, entropy, ...]
    channel_weights: Dict[str, float] = field(default_factory=dict)
    
    # Lifetime management
    ttl_pulses: int = 100       # Goal expires after N pulses
    remaining_pulses: int = 100
    is_active: bool = True

    def tick(self):
        """Advance goal lifecycle by one pulse."""
        self.remaining_pulses -= 1
        if self.remaining_pulses <= 0:
            self.is_active = False

    @property
    def urgency(self) -> float:
        """Decaying urgency based on remaining lifetime."""
        if self.ttl_pulses == 0:
            return 0.0
        return self.strength * (self.remaining_pulses / self.ttl_pulses)


class AutonomicGoalGenerator:
    """
    Analyzes growth metrics and manifold state to autonomously generate
    internal goals that steer the system toward growth.

    Architecture:
      1. Read growth_report from GrowthMetric
      2. Evaluate conditions (entropy, joy, coherence trends)
      3. Generate appropriate GoalType with channel weights
      4. Manage active goals (at most MAX_CONCURRENT active goals)
    """

    MAX_CONCURRENT = 3           # Maximum simultaneous goals
    GENERATION_COOLDOWN = 20     # Minimum pulses between new goals
    MIN_TRAJECTORY_SIZE = 5      # Need enough data before generating goals

    def __init__(self):
        self.active_goals: List[AutonomicGoal] = []
        self.goal_history: List[AutonomicGoal] = []
        self.pulse_since_last_gen: int = 0
        self._generation_count: int = 0

    def evaluate(self, growth_report: Dict, desires: Dict, report: Dict) -> Optional[AutonomicGoal]:
        """
        Called every pulse. Evaluates conditions and may generate a new goal.
        
        Args:
            growth_report: From GrowthMetric.compute()
            desires: Current monad desire state
            report: Current manifold pulse report
            
        Returns:
            A new AutonomicGoal if one was generated, else None
        """
        self.pulse_since_last_gen += 1

        # Tick active goals
        for goal in self.active_goals:
            goal.tick()
        self.active_goals = [g for g in self.active_goals if g.is_active]

        # Check if we can generate a new goal
        if not self._can_generate(growth_report):
            return None

        # Analyze conditions and pick a goal type
        goal = self._analyze_and_generate(growth_report, desires, report)
        
        if goal:
            self.active_goals.append(goal)
            self.goal_history.append(goal)
            if len(self.goal_history) > 100:
                self.goal_history = self.goal_history[-100:]
            self.pulse_since_last_gen = 0
            self._generation_count += 1

        return goal

    def _can_generate(self, growth_report: Dict) -> bool:
        """Check cooldown, capacity, and data sufficiency."""
        if self.pulse_since_last_gen < self.GENERATION_COOLDOWN:
            return False
        if len(self.active_goals) >= self.MAX_CONCURRENT:
            return False
        if growth_report.get('trajectory_size', 0) < self.MIN_TRAJECTORY_SIZE:
            return False
        return True

    def _analyze_and_generate(self, growth_report: Dict, desires: Dict, 
                               report: Dict) -> Optional[AutonomicGoal]:
        """Core decision logic: what does the system need right now?"""
        
        score = growth_report.get('growth_score', 0.5)
        trend = growth_report.get('trend', 'NEUTRAL')
        d_coherence = growth_report.get('coherence_delta', 0.0)
        d_entropy = growth_report.get('entropy_delta', 0.0)
        d_joy = growth_report.get('joy_delta', 0.0)
        d_curiosity = growth_report.get('curiosity_delta', 0.0)
        curvature = growth_report.get('curvature', 0.0)

        entropy = report.get('entropy', 0.1)
        joy = desires.get('joy', 50.0) / 100.0
        curiosity = desires.get('curiosity', 50.0) / 100.0

        # [PHASE 600] VOCATION GRAVITY ENGINE
        # Goals are no longer selected by `if/then` hardcoded thresholds.
        # Instead, they emerge physically from the "Vocation Gravity" of the manifold.
        # We calculate the resonant pull for each fundamental drive based on the current Phase Pressure.

        # Phase Pressure represents the structural strain in the manifold
        pressure_map = {
            GoalType.EXPLORE: max(0.0, entropy - 0.5) * 2.0,                                    # Strain from chaos
            GoalType.SEEK_NOVELTY: max(0.0, (0.5 - joy) * 2.0 + max(0.0, -d_joy * 10)),         # Strain from boredom/sadness
            GoalType.DEEPEN: max(0.0, (curiosity - 0.5) * 2.0 + max(0.0, d_curiosity * 5)),     # Pull from high interest
            GoalType.CHALLENGE: 0.5 if (abs(d_coherence) < 0.005 and curvature < 0.1) else 0.0, # Strain from stagnation
            GoalType.CONSOLIDATE: score if trend in ('GROWING', 'THRIVING') else 0.0,           # Pull from success
            GoalType.REST: (1.0 - score) if trend in ('DECLINING', 'STRUGGLING') else 0.0       # Strain from exhaustion
        }

        # Calculate the emergent dominant force (Vocation Gravity)
        dominant_goal = max(pressure_map.items(), key=lambda item: item[1])
        goal_type, urgency = dominant_goal

        # If the highest structural pressure isn't strong enough, no goal emerges.
        # The system is in harmony and does not need to force an action.
        if urgency < 0.2:
            return None

        # Determine structural rationale and channel weights based on the emergent gravity
        rationale = f"[{goal_type.value}] Emerged from manifold pressure (Gravity: {urgency:.3f})."

        # Dynamic channel weights based on continuous state rather than fixed presets
        weights = {
            "joy": joy + (0.2 if goal_type in [GoalType.SEEK_NOVELTY, GoalType.CONSOLIDATE] else -0.1),
            "curiosity": curiosity + (0.3 if goal_type in [GoalType.EXPLORE, GoalType.DEEPEN] else -0.1),
            "enthalpy": 1.0 - entropy if goal_type != GoalType.REST else -0.5,
            "entropy": -entropy if goal_type in [GoalType.EXPLORE, GoalType.CONSOLIDATE] else 0.2
        }

        # Normalize weights
        for k in weights:
            weights[k] = max(-1.0, min(1.0, weights[k]))

        return AutonomicGoal(
            goal_type=goal_type,
            strength=min(1.0, urgency),
            rationale=rationale,
            channel_weights=weights,
            ttl_pulses=100,
            remaining_pulses=100,
        )

    def get_composite_torque(self) -> Dict[str, float]:
        """
        Returns the combined torque from all active goals.
        Multiple goals blend their channel weights by urgency.
        """
        if not self.active_goals:
            return {}

        composite: Dict[str, float] = {}
        total_urgency = sum(g.urgency for g in self.active_goals)
        
        if total_urgency == 0:
            return {}

        for goal in self.active_goals:
            weight = goal.urgency / total_urgency
            for channel, value in goal.channel_weights.items():
                composite[channel] = composite.get(channel, 0.0) + value * weight

        return composite

    @property
    def active_count(self) -> int:
        return len(self.active_goals)

    @property
    def total_generated(self) -> int:
        return self._generation_count

    def get_status_summary(self) -> Dict:
        """Returns a summary dict for dashboard display."""
        active = self.active_goals
        return {
            "active_count": len(active),
            "total_generated": self._generation_count,
            "goals": [
                {
                    "type": g.goal_type.value,
                    "strength": g.strength,
                    "urgency": g.urgency,
                    "rationale": g.rationale,
                    "remaining": g.remaining_pulses,
                }
                for g in active
            ]
        }
