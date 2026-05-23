"""
Pytest configuration file (conftest.py)
========================================
Ensures that the archived legacy package directories (Core, World)
are placed on the python search path so that legacy unit tests
can continue to import and execute successfully.
"""

import sys
import os

# Resolve absolute path to the archive directory
archive_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'archive'))

# Prepend it to sys.path so it resolves "Core" and "World" imports
if archive_path not in sys.path:
    sys.path.insert(0, archive_path)
