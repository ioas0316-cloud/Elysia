"""
Pytest configuration file (conftest.py)
========================================
Ensures that the current workspace root is placed on the python search path 
so that current tests can execute successfully.
"""

import sys
import os

# Root of Elysia workspace (../.. from core/tests/)
elysia_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if elysia_root not in sys.path:
    sys.path.insert(0, elysia_root)
