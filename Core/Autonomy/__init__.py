"""
Core.Autonomy - Autonomous Goal Setting and Ethical Reasoning

Phase 12 of the Extended Roadmap: Autonomy & Goal Setting

This module provides systems for:
- Autonomous Goal Generation (AutonomousGoalGenerator)
- Ethical Reasoning (EthicalReasoner)
"""

from .goal_generator import AutonomousGoalGenerator
from .ethical_reasoner import EthicalReasoner

__all__ = [
    'AutonomousGoalGenerator',
    'EthicalReasoner',
]
