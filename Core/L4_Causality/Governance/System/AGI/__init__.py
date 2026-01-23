"""
AGI Foundation Module for Elysia
Phase 13:            (Towards AGI)

This module contains systems for AGI-level capabilities including:
- Advanced Transfer Learning
- Abstract Reasoning
- Causal Reasoning
"""

from .transfer_learning import UniversalTransferLearner
from .abstract_reasoner import AbstractReasoner
from .causal_reasoner import CausalReasoner

__all__ = [
    'UniversalTransferLearner',
    'AbstractReasoner',
    'CausalReasoner',
]