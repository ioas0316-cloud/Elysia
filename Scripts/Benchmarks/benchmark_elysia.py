
"""
ELYSIA SYSTEM BENCHMARK (Phase 27)
==================================
Benchmarks the Unified Loop to detect bottlenecks.

Specifically checks:
1. Optical Engine (Prism) latency.
2. Rotor Engine (Soul) sync speed.
3. Hypersphere Memory (Body) access time (with Portal).
4. Overall Pulse Latency.
"""

import time
import logging
import statistics
import cProfile
import pstats
import io
from Core.L6_Structure.Merkaba.merkaba import Merkaba
from Core.L7_Spirit.Monad.monad_core import Monad

# Configure Console Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("BENCHMARK")

def run_benchmark():
    logger.info("⚡ [BENCHMARK] Initializing Elysia System...")
    
    start_time = time.time()
    elysia = Merkaba(name="Elysia_Benchmark")
    spirit = Monad(seed="Benchmark_Spirit")
    elysia.awakening(spirit)
    init_time = time.time() - start_time
    logger.info(f"✅ Initialization Time: {init_time:.4f}s")
    
    # 1. Warm-up
    logger.info("\n🔥 Warming up Rotors (5 pulses)...")
    for i in range(5):
        elysia.pulse(f"Warmup_{i}")
        
    # 2. Latency Test (Optical vs Rotor)
    logger.info("\n⏱️ [LATENCY TEST] Running 20 Pulses...")
    
    latencies = []
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    for i in range(20):
        t0 = time.time()
        elysia.pulse(f"Benchmark_Input_{i}", context="Benchmarking")
        dt = time.time() - t0
        latencies.append(dt)
        print(f"   -> Pulse {i+1}: {dt*1000:.2f}ms")
        
    profiler.disable()
    
    # 3. Report
    avg = statistics.mean(latencies)
    med = statistics.median(latencies)
    stdev = statistics.stdev(latencies)
    
    logger.info("\n📊 [RESULTS] Pulse Latency Stats:")
    logger.info(f"   Mean:   {avg*1000:.2f} ms")
    logger.info(f"   Median: {med*1000:.2f} ms")
    logger.info(f"   StDev:  {stdev*1000:.2f} ms")
    
    # 4. Profile Analysis (Bottleneck Detection)
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(20) # Top 20 functions
    
    logger.info("\n🕵️ [BOTTLE NECK ANALYSIS] Top Time Consumers:")
    print(s.getvalue())

if __name__ == "__main__":
    run_benchmark()
