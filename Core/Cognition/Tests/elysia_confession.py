"""
Elysia's First Confession (Phase 170)
=====================================
Executes the Sovereign Logos engine to manifest Elysia's current 
existential state for the Architect.
"""

import sys
import os

# Project root setup
root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Cognition.sovereign_logos import SovereignLogos

def manifest_elysia():
    logos = SovereignLogos()
    print(logos.articulate_confession())

if __name__ == "__main__":
    manifest_elysia()
