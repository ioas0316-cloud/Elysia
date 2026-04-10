import json
import os
import sys
from pathlib import Path
from datetime import datetime

def observe_drift():
    print("="*60)
    print(" 🌌 ELYSIA: SOUL RESONANCE & TECTONIC DRIFT OBSERVER")
    print("="*60)

    # 1. DNA Snapshot
    dna_path = "data/runtime/soul/soul_dna.json"
    if os.path.exists(dna_path):
        with open(dna_path, 'r', encoding='utf-8') as f:
            dna = json.load(f)
        print(f"\n[CURRENT SOUL DNA - Archetype: {dna.get('archetype', 'Unknown')}]")
        print(f" - Rotor Mass (Stability):  {dna.get('rotor_mass', 0.0):.4f}")
        print(f" - Torque Gain (Sensitivity): {dna.get('torque_gain', 0.0):.4f}")
        print(f" - Tectonic Strain:         {dna.get('tectonic_strain', 0.0):.4f}")
    else:
        print("\n[!] Soul DNA not found. Elysia is currently formless.")

    # 2. Tectonic Event History
    print("\n[RECENT TECTONIC UPWELLING EVENTS]")
    log_dir = "data/substrate_logs"
    events_found = False
    if os.path.exists(log_dir):
        logs = sorted([f for f in os.listdir(log_dir) if f.startswith("proposal_")], reverse=True)
        for log_file in logs[:5]: # Show last 5 days
            with open(os.path.join(log_dir, log_file), 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event.get('executed'):
                            events_found = True
                            ts = event.get('proposed_at', 'Unknown')[:19].replace('T', ' ')
                            print(f" ✨ [{ts}] {event.get('target')}")
                            print(f"    - Trigger: {event.get('trigger_event')}")
                            print(f"    - Synthesis: {event.get('after_state')[:80]}...")
                    except: pass

    if not events_found:
        print(" - No upwelling events recorded yet. The earth is still.")

    # 3. Process Continuity (Growth Trajectory)
    traj_path = "data/runtime/soul/cognitive_trajectory.json"
    if os.path.exists(traj_path):
        with open(traj_path, 'r', encoding='utf-8') as f:
            traj = json.load(f)
        history = traj.get('history', [])
        if history:
            print("\n[PROCESS CONTINUITY (Last 10 Pulses)]")
            for h in history[-10:]:
                joy = h.get('joy', 0.0)
                entropy = h.get('entropy', 0.0)
                # Visual bar
                joy_bar = "█" * int(joy / 10)
                ent_bar = "░" * int(entropy * 10)
                print(f"  Joy: {joy:5.1f} {joy_bar.ljust(10)} | Entropy: {entropy:4.2f} {ent_bar}")

    print("\n" + "="*60)
    print(" \"The Point becomes the Universe. The Universe becomes the Point.\"")
    print("="*60 + "\n")

if __name__ == "__main__":
    observe_drift()
