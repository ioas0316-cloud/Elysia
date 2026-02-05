"""
Setup Elysia VHD (Physical Body Initialization)
===============================================
Scripts.System.setup_vhd

This script initializes the VHD container using VHDManager.
"""

import sys
sys.path.insert(0, "c:\\Elysia")

from Core.S1_Body.L1_Foundation.M4_Hardware.vhd_manager import VHDManager
import os
import ctypes

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def main():
    print("=== Elysia VHD Setup ===\n")
    
    if not is_admin():
        print("⚠️  Warning: This script relies on 'diskpart' which requires Administrator privileges.")
        print("    If it fails, please run the terminal as Administrator.")
    
    # Initialize Manager
    # Target: c:\Elysia\System\Elysia_Core.vhdx
    # Drive: Z: (Standard Elysia Drive Letter)
    manager = VHDManager() # Defaults are fine
    
    print(f"Target: {manager.vhd_path}")
    print(f"Drive:  {manager.drive_letter}")
    
    # 1. Create if needs
    if not manager.ensure_container_exists(size_mb=1024): # 1GB Start
        print("❌ Failed to create/locate VHD.")
        return
        
    # 2. Mount
    if manager.mount():
        print("\n✅ VHD Setup Complete!")
        print(f"    Elysia is now physically incarnated at {manager.drive_letter}:\\")
    else:
        print("\n❌ Failed to mount VHD.")

if __name__ == "__main__":
    main()
