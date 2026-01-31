import time
import numpy as np
import logging
from Core.L1_Foundation.Foundation.physics import HamiltonianSystem, QuantumState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AscensionBench")

def run_benchmark():
    system = HamiltonianSystem()
    num_particles = 100
    
    # Create test states
    states = [
        QuantumState(
            position=np.random.rand(3) * 10,
            momentum=np.random.rand(3),
            name=f"P{i}"
        ) for i in range(num_particles)
    ]
    
    logger.info(f"🚀 Starting Benchmark: {num_particles} Particles")
    
    # 1. Pure Python Baseline (Single evolution in loop)
    start_time = time.time()
    for _ in range(10):
        baseline_results = [system.evolve(s, 0.1) for s in states]
    py_time = (time.time() - start_time) / 10
    logger.info(f"   - Pure Python Performance: {py_time*1000:.2f}ms / step")
    
    # 2. Native JIT Transmuted
    if system.has_native:
        # Warm up JIT
        system.evolve_batch_native(states, 0.1)
        
        start_time = time.time()
        for _ in range(10):
            native_results = system.evolve_batch_native(states, 0.1)
        jit_time = (time.time() - start_time) / 10
        logger.info(f"   - Native JIT Performance: {jit_time*1000:.2f}ms / step")
        
        speedup = py_time / jit_time
        logger.info(f"🔥 [ASCENSION] Native Speedup: {speedup:.1f}x")
    else:
        logger.warning("❌ Native JIT not available for benchmark.")

if __name__ == "__main__":
    run_benchmark()
