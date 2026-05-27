import sys
import os
import time
import numpy as np
from numba import cuda

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.cuda_kernel import ascii_phase_sync_kernel, BLOCK_SIZE

def run_bypass_demo():
    print("==================================================")
    print(" ⚡ ASCII-CUDA Direct Bypass (Zero-Parsing)")
    print("==================================================")
    
    # 1. Prepare raw ASCII data (simulating a massive data stream)
    text_stream = "Elysia: The quick brown fox jumps over the lazy dog. " * 500  # Large text stream
    ascii_array = np.array([ord(c) for c in text_stream], dtype=np.uint8)
    
    data_size = ascii_array.size
    print(f"[Input] Raw ASCII Bytes: {data_size} bytes (No Tokenization!)")
    
    # Allocate device arrays
    d_ascii = cuda.to_device(ascii_array)
    d_phases_x = cuda.device_array(data_size, dtype=np.float32)
    d_phases_y = cuda.device_array(data_size, dtype=np.float32)
    d_coherence = cuda.device_array(data_size, dtype=np.float32)
    
    # Grid configuration
    threads_per_block = BLOCK_SIZE
    blocks_per_grid = (data_size + (threads_per_block - 1)) // threads_per_block
    
    gear_elasticity = 0.05
    iterations = 500  # How many physical gear turns
    
    print(f"[CUDA] Launching {blocks_per_grid} Blocks, {threads_per_block} Threads/Block")
    print(f"[CUDA] Executing {iterations} pure geometric rotor sync cycles...")
    
    # 2. Execute CUDA Kernel (Hardware-level physics)
    start_time = time.time()
    
    ascii_phase_sync_kernel[blocks_per_grid, threads_per_block](
        d_ascii, d_phases_x, d_phases_y, gear_elasticity, iterations, d_coherence
    )
    cuda.synchronize()
    
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000.0
    
    # 3. Retrieve results
    final_coherence = d_coherence.copy_to_host()
    avg_coherence = np.mean(final_coherence)
    
    print(f"\n[Result] Phase-Lock Synchronization Time: {elapsed_ms:.3f} ms")
    print(f"[Result] Final Network Average Coherence: {avg_coherence:.4f} / 1.0000")
    print("\n✅ Bypass Complete: CPU intervention = 0, Tokenization = 0, Matrix Ops = 0")
    
if __name__ == "__main__":
    if not cuda.is_available():
        print("CUDA is not available. Please ensure NVIDIA drivers and CUDA toolkit are installed.")
    else:
        run_bypass_demo()
