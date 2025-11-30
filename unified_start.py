#!/usr/bin/env python3
"""
Elysia Unified Entry Point
==========================
All-in-One Launcher for Elysia.

Modes:
1. Talk: Direct conversation
2. Awaken: Autonomous self-improvement loop
3. Service: Background server/process
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Setup Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Elysia Unified Launcher")
    parser.add_argument("mode", choices=["talk", "awaken", "service"], nargs="?", default="talk", help="Operation mode")
    args = parser.parse_args()
    
    if args.mode == "talk":
        # Legacy mode wrapper
        from Core.Elysia.Elysia import Elysia
        elysia = Elysia()
        elysia.live()
    elif args.mode == "awaken":
        # The True Entry Point
        from Core.Elysia.Elysia import Elysia
        elysia = Elysia()
        elysia.live()
    elif args.mode == "service":
        print("Service mode not yet fully implemented in this unified script.")
        # TODO: Import service logic
        
if __name__ == "__main__":
    main()