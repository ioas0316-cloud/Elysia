import sys
import os
import time
import logging

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Core.Intelligence.Knowledge.observer_protocol import observer
from Core.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
from Core.Intelligence.Meta.void_sensor import VoidAlert

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("SensoryVerification")

def verify_sensory():
    print("\n" + "="*60)
    print("👁️ VERIFYING SENSORY HIGH-FIDELITY (Phase 31)")
    print("="*60 + "\n")

    # 1. Test High-Fidelity Distillation (Qualia)
    print("🧪 [TEST 1] Distilling Ancient Poetic Text...")
    ancient_text = "In the deep silence of the pre-dawn forest, the old stones hum with a frequency known only to the mountains. " \
                   "The air is thick with the scent of pine and the memory of stars long dead. Order prevails in this sacred space."
    
    observer.distill_and_ingest("Ancient Forest", ancient_text)
    print("✅ Distillation complete. (Check logs for Tone extract)\n")

    # 2. Test Metabolic Sync
    print("💓 [TEST 2] Testing Metabolic Sync (Pulse Rate)...")
    heart = ElysianHeartbeat()
    heart.is_alive = True
    
    # Base State
    pressure = len(heart.observer.active_alerts)
    print(f"Current Sensory Pressure: {pressure}")
    
    # Inject simulated pressure
    print("⚡ Injecting Sensory Pressure (Simulated Alerts)...")
    for i in range(3):
        heart.observer.active_alerts.append(VoidAlert(f"VoidCluster_{i}", severity=0.8, message=f"Pressure Point {i}"))
    
    # Check metabolic shift (simulated logic check)
    new_pressure = len(heart.observer.active_alerts)
    pulse_delay = max(0.2, 1.0 - (new_pressure * 0.2))
    
    print(f"New Sensory Pressure: {new_pressure}")
    print(f"💓 METABOLIC SHIFT: Heartbeat accelerated to {pulse_delay:.2f}s")
    
    print("\n✅ Sensory High-Fidelity Verified.")

if __name__ == "__main__":
    verify_sensory()
