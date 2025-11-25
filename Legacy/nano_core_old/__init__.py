"""Compatibility shim for the relocated nano_core package.

The actual implementation now lives under Project_Sophia.nano_core.
"""

from importlib import import_module
import sys

_pkg = import_module("Project_Sophia.nano_core")
sys.modules[__name__] = _pkg
