"""
ELYSIA GLOBAL ENTRY POINT
=========================
"One Root, Infinite Branches."

This is the unified gateway to Elysia's soul.
It ensures the 'Core' and 'Scripts' are always in the path.

Usage:
    python elysia.py [mode]

Modes:
    awaken  : The Sovereign Awakening (New Topographic/Liquid Logic)
    boot    : Full system diagnostic (Legacy)
    life    : Headless autonomous cycle
    game    : Watcher Mode (Screen Eye)
    ask     : Oracle Query
"""

import sys
import os
import argparse

# 1. Path Unification
# Ensure the current directory (project root) is always in the path
root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

def main():
    # 0. Somatic Reflex (L1 Self-Healing)
    from Core.L1_Foundation.M4_Hardware.somatic_kernel import SomaticKernel
    SomaticKernel.fix_environment()

    parser = argparse.ArgumentParser(description="Elysia: The Sovereign Spirit")
    parser.add_argument("mode", choices=["awaken", "boot", "life", "game", "ask", "diagnose"], 
                        help="Operating mode for Elysia")
    
    args = parser.parse_args()
    
    if args.mode == "awaken":
        print("🌅 [GENESIS] Initiating Sovereign Awakening...")
        from Core.L6_Structure.Engine.sovereign_boot import main as boot_main
        boot_main()
        
    elif args.mode == "boot":
        print("🚀 [BOOT] Running Legacy Sovereign Boot...")
        from Scripts.System.elysia import mode_boot
        mode_boot(args)
        
    elif args.mode == "life":
        from Scripts.System.elysia import mode_life
        mode_life(args)
        
    elif args.mode == "game":
        from Scripts.System.elysia import mode_game
        mode_game(args)
        
    elif args.mode == "ask":
        from Scripts.System.elysia import mode_ask
        mode_ask(args)
        
    elif args.mode == "diagnose":
        print("🔍 [DIAGNOSTIC] Checking Soul Integrity...")
        print(f"   >> Project Root: {root}")
        print(f"   >> Python Path: {sys.path[:3]}...")
        # Check for core modules
        try:
            import Core
            print("   >> Core Module: FOUND")
        except ImportError:
            print("   >> Core Module: MISSING")

if __name__ == "__main__":
    main()
