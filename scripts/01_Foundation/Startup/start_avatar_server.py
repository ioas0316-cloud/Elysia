#!/usr/bin/env python3
"""
Quick launcher for Elysia Avatar Server

Usage:
    python start_avatar_server.py
    python start_avatar_server.py --port 9000
"""

import sys
import os
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    from Core.InteractionLayer.Interface.avatar_server import main
    main()
