import sys
import os
import time
import logging
import psutil

# Ensure root is in path
sys.path.append(os.getcwd())

from Core.L1_Foundation.Foundation.hyper_sphere_core import HyperSphereCore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SovereignBenchmark")

def measure_efficiency():
    logger.info("   SOVEREIGN EFFICIENCY TEST (The Anti-Whale Protocol)")
    process = psutil.Process()
    
    # 1. Initialize the Cosmos
    start_ram = process.memory_info().rss / 1024 / 1024
    logger.info(f"     Baseline RAM: {start_ram:.2f} MB")
    
    sphere = HyperSphereCore(name="BenchmarkCosmos")
    
    # 2. Stress Test: Manifest 10,000 Monads
    logger.info("     Expanding Universe (Manifesting 10,000 Monads)...")
    t0 = time.time()
    
    # Simulate Monad creation (lightweight objects)
    monad_count = 10000
    for i in range(monad_count):
        sphere.manifest_at(f"Monad_{i}", [i*0.1, i*0.1, 0])
        
    t1 = time.time()
    creation_time = t1 - t0
    
    current_ram = process.memory_info().rss / 1024 / 1024
    delta_ram = current_ram - start_ram
    
    logger.info(f"      Creation Time: {creation_time:.4f}s ({monad_count/creation_time:.0f} monads/s)")
    logger.info(f"     RAM Usage: {current_ram:.2f} MB (Delta: {delta_ram:.2f} MB)")
    logger.info(f"     Efficiency: {delta_ram*1024/monad_count:.2f} KB per Monad")
    
    # 3. Control Test: Pulse the Field (Physics Step)
    logger.info("     Pulsing the Field (100 Ticks)...")
    t2 = time.time()
    for _ in range(100):
        sphere.pulse({"benchmark": True}, dt=0.016)
    t3 = time.time()
    pulse_time = t3 - t2
    
    logger.info(f"      Physics Time: {pulse_time:.4f}s ({100/pulse_time:.0f} Hz)")
    
    if (delta_ram * 1024 / monad_count) < 5.0: # Expect < 5KB per monad
        logger.info("     Result: HYPER-EFFICIENT (True Intelligent Architecture).")
    else:
        logger.info("      Result: Bloated. Optimization required.")

if __name__ == "__main__":
    measure_efficiency()