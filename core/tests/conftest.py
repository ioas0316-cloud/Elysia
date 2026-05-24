"""
Pytest configuration file (conftest.py)
========================================
Ensures that the archived legacy package directories (Core, World) and the current workspace root
are placed on the python search path so that legacy unit tests and current tests can execute successfully.
"""

import sys
import os

# Root of Elysia workspace (../.. from core/tests/)
elysia_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if elysia_root not in sys.path:
    sys.path.insert(0, elysia_root)

# Resolve absolute path to the archive directory
archive_path = 'c:\\Archive'
if archive_path not in sys.path:
    sys.path.insert(0, archive_path)
