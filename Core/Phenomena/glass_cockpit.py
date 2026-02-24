#!/usr/bin/env python3
import sys
import os

# Ensure the Core modules can be found
sys.path.append(os.getcwd())

from Core.Phenomena.app import GlassCockpitApp

if __name__ == "__main__":
    print("  Initializing E.L.Y.S.I.A Glass Cockpit...")
    app = GlassCockpitApp()
    app.run()
