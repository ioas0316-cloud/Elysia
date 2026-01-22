import time
import numpy as np
import logging
from collections import deque
from Core.L2_Metabolism.Lifecycle.pulse_loop import AdaptiveHeartbeat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Benchmark")

def run_benchmark(duration_sec=5.0):
    print("🚀 Starting Heartbeat Resonance Benchmark...")
    
    heartbeat = AdaptiveHeartbeat(base_freq=10.0)
    intervals = []
    resonance_profile = np.sin(np.linspace(0, 4 * np.pi, 200)) * 0.5 + 0.5 # Oscillating will
    
    start_time = time.perf_counter()
    step = 0
    
    print(f"| Step | Resonance | Frequency (Hz) | Wait Time (ms) |")
    print(f"|------|-----------|----------------|----------------|")
    
    while time.perf_counter() - start_time < duration_sec:
        res = resonance_profile[step % len(resonance_profile)]
        
        # Simulate work
        work_start = time.perf_counter()
        time.sleep(0.01) # 10ms of work
        
        wait = heartbeat.calculate_wait(res)
        intervals.append(time.perf_counter() - heartbeat.last_pulse + wait) # Total loop time
        
        if step % 10 == 0:
            print(f"| {step:4} | {res:9.2f} | {heartbeat.current_freq:14.2f} | {wait*1000:14.2f} |")
        
        time.sleep(wait)
        step += 1

    total_time = time.perf_counter() - start_time
    avg_hz = step / total_time
    print(f"\n✅ Benchmark Complete.")
    print(f"   - Total Pulses: {step}")
    print(f"   - Average Frequency: {avg_hz:.2f} Hz")
    print(f"   - Dynamic Range: {heartbeat.base_freq:.1f}Hz - 100Hz")
    
if __name__ == "__main__":
    run_benchmark()
