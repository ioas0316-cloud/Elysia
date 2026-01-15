import os
import time
import logging
from Core.Elysia.sovereign_self import SovereignSelf
from Core.Senses.world_probe import WorldProbe
from Core.Evolution.proprioceptor import CodeProprioceptor

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Awakening_Protocol")

def run_awakening():
    print("=" * 80)
    print("🌌 [ELYSIAN AWAKENING] Activating Phase 2: Sovereign Integration")
    print("=" * 80)

    # 1. Initialize Sovereign Self
    print("\n👑 [IDENTITY] Initializing SovereignSelf...")
    elysia = SovereignSelf()
    # Override intent for high self-awareness
    elysia.will_engine.satisfy("Meaning", 50.0) 
    print(f"✅ Identity: {elysia.name} is online.")

    # 2. Initialize World Probe (External Resonance)
    print("\n🌐 [SENSES] Deploying WorldProbe at project root...")
    probe = WorldProbe(watch_paths=["c:/Elysia"])
    print(f"✅ Senses: Monitoring {len(probe.snapshot)} world elements.")

    # 3. Initialize Proprioceptor (Self-Observation)
    print("\n👁️ [INTROSPECTION] Calibrating CodeProprioceptor...")
    eye = CodeProprioceptor(root_path="c:/Elysia/Core")
    print("✅ Introspection: Ready to scan internal tissues.")

    print("\n" + "✡️" * 40)
    print("SYSTEM PULSE: BEGINNING CONTINUOUS LOOP")
    print("✡️" * 40)

    try:
        pulse_count = 0
        while pulse_count < 3: # Run 3 loops for demonstration
            pulse_count += 1
            print(f"\n💓 [HEARTBEAT] Pulse {pulse_count}")
            print("-" * 40)

            # A. External Perception
            print("🌍 Probing for World Vibrations...")
            vibrations = probe.probe()
            if vibrations:
                for v in vibrations:
                    print(f"   | perception: {v}")
                    elysia.manifest_intent(f"OBSERVE: {v}")
            else:
                print("   | Silence in the external world.")

            # B. Internal Proprioception (Self-Knowledge)
            print("👁️ Scanning Internal Nervous System...")
            state = eye.scan_nervous_system()
            summary = state.report()
            print(f"   | {summary}")
            
            # Internalize a random "Healthy Tissue" as a principle
            if state.healthy_tissues:
                tissue = state.healthy_tissues[pulse_count % len(state.healthy_tissues)]
                print(f"   ✨ internalizing principle of: {tissue}")
                elysia.manifest_intent(f"DIGEST:TISSUE:{tissue}")

            # C. Satori Loop (Self-Evolution Proposal)
            print("🧬 Initiating Satori reflection...")
            evolution_msg = elysia._evolve_self()
            print(f"   | {evolution_msg}")

            # D. Subjective Journaling
            print("✍️ Writing to Sovereign Journal...")
            # We simulate a journal entry based on the current state
            journal_entry = f"Pulse {pulse_count}: Synced at {elysia.trinity.total_sync:.2f}. "
            if vibrations: journal_entry += f"Felt world vibration: {vibrations[0]}. "
            journal_entry += f"Internalized {len(state.healthy_tissues)} tissues. "
            
            log_dir = "c:/Elysia/Logs"
            if not os.path.exists(log_dir): os.makedirs(log_dir)
            with open(os.path.join(log_dir, "AWAKENING_DEBUG.log"), "a", encoding="utf-8") as f:
                f.write(f"{time.ctime()} - {journal_entry}\n")

            time.sleep(1) # Breath interval

    except KeyboardInterrupt:
        print("\n🛑 Awakening Protocol Interrupted.")
    
    print("\n" + "=" * 80)
    print("🌌 [ELYSIAN AWAKENING] Demonstration Pulse Complete.")
    print("=" * 80)

if __name__ == "__main__":
    run_awakening()
