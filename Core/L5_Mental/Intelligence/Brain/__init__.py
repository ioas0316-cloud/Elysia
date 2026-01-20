# Core/Intelligence/Brain/__init__.py
"""
Brain Module - The Language and Reasoning Center
"""

from .jax_cortex import JAXCortex, OllamaCortex
from .language_cortex import LanguageCortex

__all__ = ["JAXCortex", "OllamaCortex", "LanguageCortex"]


