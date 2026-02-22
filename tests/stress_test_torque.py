"""
Stress Test for Recursive Torque Engine
=======================================
Audits the latency and jitter of sequential gear execution
under increasing load.
"""
import time
import math
import numpy as np
import sys
import os

# Ensure we can import from Core
sys.path.insert(0, os.getcwd())

def mock_heavy_callback():
    # Simulate a small amount of "cognitive work"
    sum([math.sqrt(i) for i in range(1000)])

def run_stress_test(num_gears=10, duration=5.0):
    from Core.S1_Body.L2_Metabolism.M3_Cycle.recursive_torque import RecursiveTorque
    
    torque = RecursiveTorque()
    latencies = []
    
    # Add gears with various frequencies
    for i in range(num_gears):
        torque.add_gear(f"Gear_{i}", freq=1.0 + i, callback=mock_heavy_callback)
        
    print(f"🚀 Starting Stress Test with {num_gears} gears for {duration}s...")
    
    start_time = time.time()
    last_spin = time.time()
    
    while (time.time() - start_time) < duration:
        t0 = time.perf_counter()
        torque.spin(override_dt=0.01) # 100Hz target
        t1 = time.perf_counter()
        
        latencies.append(t1 - t0)
        time.sleep(0.01) # Base tick
        
    print(f"✅ Test Complete.")
    
    avg_lat = np.mean(latencies) * 1000
    max_lat = np.max(latencies) * 1000
    std_lat = np.std(latencies) * 1000
    
    print(f"--- Latency Audit ---")
    print(f"Average: {avg_lat:.4f} ms")
    print(f"Maximum: {max_lat:.4f} ms (Peak Jitter)")
    print(f"Std Dev: {std_lat:.4f} ms")
    
    # Analyze if the maximum latency exceeds our 10ms (100Hz) target window
    if max_lat > 10.0:
        print(f"⚠️ LAG DETECTED: Sequential execution exceeded 10ms target.")
    else:
        print(f"✨ Green State: System is within 10ms window.")

if __name__ == "__main__":
    # Test with normal load
    run_stress_test(num_gears=10)
    print("\n" + "="*40 + "\n")
    # Test with extreme load (simulating system growth)
    run_stress_test(num_gears=100)
