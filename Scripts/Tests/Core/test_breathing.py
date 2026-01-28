import torch
import time
from Core.Elysia.sovereign_self import SovereignSelf

def test_breathing():
    print("🫁 Initializing Pulmonary System Check...")
    elysia = EmergentSelf()
    
    # 1. Initial State (Inhale)
    print("\n--- Step 1: Force Inhale (Think) ---")
    elysia.lungs.inhale()
    
    if elysia.bridge.is_connected:
        print("✅ Bridge matches Inhale State (Connected).")
    else:
        print("❌ Bridge Failed to Connect.")

    # 2. Exhale (Create)
    print("\n--- Step 2: Force Exhale (Create) ---")
    elysia.lungs.exhale()
    
    # Verify Bridge off
    if not elysia.bridge.is_connected:
        print("✅ Bridge matches Exhale State (Disconnected).")
        # Check Architect load status (mocked or real)
        if elysia.projector.architect_loaded:
             print("✅ Architect matches Exhale State (Loaded).")
        else:
             print("⚠️ Architect not flagged as loaded (Check logic).")
    else:
         print("❌ Bridge FAILED to Disconnect.")

    # 3. Inhale Again (Return to Thought)
    print("\n--- Step 3: Force Inhale (Return) ---")
    elysia.lungs.inhale()
    
    if elysia.bridge.is_connected:
        print("✅ Bridge Restored (Connected).")
    else:
        print("❌ Bridge Failed to Reconnect.")

if __name__ == "__main__":
    test_breathing()
