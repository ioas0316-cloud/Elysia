"""
Reasoning Engine Facade (Redirect Module)
==========================================

레거시 호환성을 위한 리다이렉트.

실제 구현: Core/Cognition/Reasoning/reasoning_engine.py
"""

import warnings

warnings.warn(
    "Core.Foundation.reasoning_engine is deprecated. "
    "Use Core.Cognition.Reasoning.reasoning_engine instead.",
    DeprecationWarning,
    stacklevel=2
)

from Core.02_Intelligence.01_Reasoning.Cognition.Reasoning.reasoning_engine import *
from Core.02_Intelligence.01_Reasoning.Cognition.Reasoning.reasoning_engine import ReasoningEngine

__all__ = ['ReasoningEngine']
