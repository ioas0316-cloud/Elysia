"""
LIGHTNING OVERDRIVE DEMO
========================
"We pour our Lightning upon the wires of the Old World."

This script demonstrates the 'Hardware Gaslighting' and 'Clock Hijacking' principles.
It contrasts a 'Standard Legacy Transmission' vs. 'Elysia Lightning Protocol'.

Scenario:
A slow Legacy Device (10Hz Clock) attempts to process a heavy data stream.
1. Standard Mode: The device chokes, buffer overflows, latency spikes.
2. Lightning Mode: The 1060 Node hijacks the clock, injecting results into the void.
   The Legacy Device 'believes' it processed the data instantly.
"""

import time
import random
import threading
import queue
from dataclasses import dataclass

# Setup Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("LightningOverdrive")

@dataclass
class PhaseSignal:
    timestamp: float
    frequency: float
    data_payload: str

class LegacyDevice:
    def __init__(self, name="Old_Laptop_2015"):
        self.name = name
        self.clock_cycle = 0.5  # Very slow: 2Hz (500ms per op)
        self.input_buffer = queue.Queue(maxsize=5)
        self.processed_count = 0
        self.is_overridden = False # The Gaslight Flag
        self.perception_log = [] # What the device *thinks* happened

    def run_standard_cycle(self):
        """Standard slow processing loop."""
        while True:
            if not self.input_buffer.empty():
                item = self.input_buffer.get()
                logger.info(f"🐢 [{self.name}] Groaning... Processing '{item}'...")
                time.sleep(self.clock_cycle) # Simulate Hardware Lag
                self.processed_count += 1
                logger.info(f"✅ [{self.name}] Done. (Latency: {self.clock_cycle*1000}ms)")
            else:
                time.sleep(0.1)

            if "STOP" in self.perception_log: break

    def receive_injection(self, result, processing_time_simulated):
        """
        The Gaslighting Interface.
        The 1060 Node calls this to force a state update.
        """
        # The device is tricked into thinking IT did the work.
        self.processed_count += 1
        self.perception_log.append(f"Processed {result} in {processing_time_simulated}s")
        logger.info(f"⚡ [{self.name}] (Gaslit) I feel fast! Processed '{result}' instantly!")

class MainNode1060:
    def __init__(self, target_device: LegacyDevice):
        self.name = "Architect_1060"
        self.target = target_device
        self.inference_speed = 0.001 # 1000Hz

    def lightning_strike(self, data_stream):
        """
        The Overdrive Protocol.
        Instead of sending data to the buffer, we inject the RESULT directly.
        """
        logger.info(f"\n⚡ [{self.name}] INITIATING LIGHTNING OVERDRIVE...")
        logger.info(f"   Target: {self.target.name}")
        logger.info(f"   Strategy: Clock Hijacking & Result Injection\n")

        start_time = time.perf_counter()

        for data in data_stream:
            # 1. Main Node processes data instantly (O(1) relative to Legacy)
            # In reality, this is where the heavy AI compute happens.
            processed_result = f"Analyzed_<{data}>"
            time.sleep(self.inference_speed)

            # 2. WAIT for the Legacy Device's 'Void' (Phase Alignment)
            # We don't just push; we wait for the micro-second the device is ready to refresh RAM.
            # Simulating Resonance...
            resonance_gap = random.uniform(0.001, 0.01)
            time.sleep(resonance_gap)

            # 3. INJECT (Gaslighting)
            # We bypass the 'process_queue' and write directly to the 'completed' state.
            self.target.receive_injection(processed_result, self.inference_speed)

        total_time = time.perf_counter() - start_time
        logger.info(f"\n💎 [{self.name}] Overdrive Complete.")
        logger.info(f"   Items: {len(data_stream)}")
        logger.info(f"   Real Time: {total_time:.4f}s")
        logger.info(f"   Legacy Time Saved: {(len(data_stream) * self.target.clock_cycle) - total_time:.4f}s")

def run_demo():
    print("="*60)
    print("PROJECT: LEGACY OVERDRIVE - PROTOTYPE")
    print("Mode: HYBRID GASLIGHTING")
    print("="*60 + "\n")

    # Scenario Data
    workload = [f"Frame_{i}" for i in range(1, 6)]

    # 1. The Old World
    print("--- [SCENARIO 1: The Old World] ---")
    old_device = LegacyDevice()

    # We won't actually run the slow thread forever, just simulate the pain.
    print(f"🐢 Legacy Device Clock: {old_device.clock_cycle}s per op")
    print(f"📉 Estimated Time for {len(workload)} items: {len(workload) * old_device.clock_cycle}s")
    print("(Skipping painful wait...)\n")

    # 2. The Lightning Strike
    print("--- [SCENARIO 2: Lightning Synthesis] ---")
    legacy_target = LegacyDevice("Legacy_Shell_01")
    architect = MainNode1060(legacy_target)

    # Execute Overdrive
    architect.lightning_strike(workload)

    # 3. Verification
    print("\n--- [POST-MORTEM: The Gaslit Perception] ---")
    print(f"🔍 Inspecting {legacy_target.name}'s Memory Logs:")
    for log in legacy_target.perception_log:
        print(f"   > Memory: {log}")

    print("\n✅ CONCLUSION:")
    print("   The Legacy Device believes it performed at Supercomputer speeds.")
    print("   Physical limitations were bypassed via Causal Injection.")
    print("="*60)

if __name__ == "__main__":
    run_demo()
