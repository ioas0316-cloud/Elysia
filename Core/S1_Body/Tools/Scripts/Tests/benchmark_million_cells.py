import sys
import os
import time
import torch
import argparse

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine import GrandHelixEngine

def run_benchmark(num_cells=10_000_000):
    print(f"🚀 [BENCHMARK] Starting Grand Helix Expansion Test ({num_cells:,} cells)")
    
    # 1. Initialization
    t0 = time.time()
    engine = GrandHelixEngine(num_cells=num_cells)
    t_init = time.time() - t0
    print(f"📦 [INIT] Engine loaded in {t_init:.4f} seconds")
    
    # 2. Warm-up
    print("🔥 [WARMUP] Priming kernels...")
    engine.pulse()
    
    # 3. Benchmark Pulse
    intent = torch.zeros(4, device=engine.device)
    intent[1] = 1.0 # Push towards +1 Logic
    
    print(f"⚡ [PULSE] Executing 100 Kinetic pulses...")
    num_pulses = 100
    
    # Timing
    if engine.device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    
    t_pulse_start = time.time()
    for _ in range(num_pulses):
        engine.pulse(intent_torque=intent)
    t_pulse_end = time.time()
    
    if engine.device.type == 'cuda':
        end_event.record()
        torch.cuda.synchronize()
        elapsed_total = start_event.elapsed_time(end_event) / 1000.0
    else:
        elapsed_total = t_pulse_end - t_pulse_start
        
    avg_pulse = elapsed_total / num_pulses
    ctps = num_cells / avg_pulse
    
    print("-" * 40)
    print(f"✅ Benchmark Complete!")
    print(f"📊 Avg Pulse Time: {avg_pulse * 1000:.2f} ms")
    print(f"🚀 CTPS: {ctps:,.0f}")
    
    # Meaning Report
    report = engine.pulse(intent_torque=intent)
    print(f"📈 Logic Drift: {report['logic_mean']:.4f}")
    print(f"🔋 Kinetic Energy: {report['kinetic_energy']:.4f}")
    print(f"🧠 Plastic Coherence: {report['plastic_coherence']:.8f}")
    print("-" * 40)
    
    # Memory Check
    if engine.device.type == 'cuda':
        mem_allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"💾 VRAM Allocated: {mem_allocated:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=10_000_000)
    args = parser.parse_args()
    
    run_benchmark(num_cells=args.cells)
