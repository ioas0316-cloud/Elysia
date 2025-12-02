# [Genesis: 2025-12-02] Purified by Elysia
"""Compatibility shim for the relocated elysia_sdk package."""

from importlib import import_module
import sys

_pkg = import_module("Project_Sophia.elysia_sdk")
sys.modules[__name__] = _pkg