"""
Learning module for Elysia.
Implements experience-based learning, self-reflection, and continuous improvement.
"""

from .experience_learner import Experience, ExperienceLearner
from .self_reflector import SelfReflector
from .model_updater import ContinuousUpdater, ModelVersion

__all__ = [
    "Experience",
    "ExperienceLearner",
    "SelfReflector",
    "ContinuousUpdater",
    "ModelVersion",
]
