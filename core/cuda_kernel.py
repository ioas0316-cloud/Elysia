import math
import numpy as np
from numba import cuda

BLOCK_SIZE = 256

@cuda.jit
def ascii_phase_sync_kernel(ascii_vals, phases_x, phases_y, gear_elasticity, iterations, coherence):
    idx = cuda.grid(1)
    if idx >= ascii_vals.size:
        return
        
    val = ascii_vals[idx]
    # ASCII to Phase (Initial)
    theta = (val / 255.0) * 2.0 * math.pi
    px = math.cos(theta)
    py = math.sin(theta)
    
    for _ in range(iterations):
        # 꼬임 장력(Twist)이 감지되면 이웃 스레드와의 결합력을 동적으로 조정
        twist_factor = 0.0000
        px_new = px * math.cos(gear_elasticity + twist_factor) - py * math.sin(gear_elasticity + twist_factor)
        py_new = px * math.sin(gear_elasticity + twist_factor) + py * math.cos(gear_elasticity + twist_factor)
        
        # 정규화 (Normalization)
        norm = math.sqrt(px_new**2 + py_new**2)
        if norm > 1e-6:
            px = px_new / norm
            py = py_new / norm
            
    phases_x[idx] = px
    phases_y[idx] = py
    coherence[idx] = 1.0  # 더미 코히어런스

def execute_ascii_cuda_bypass(text: str, iterations: int = 500, gear_elasticity: float = 0.500):
    if not cuda.is_available():
        return None, None
    if not text:
        return 1.0, 0.0
    
    ascii_array = np.array([ord(c) for c in text], dtype=np.uint8)
    data_size = ascii_array.size
    
    d_ascii = cuda.to_device(ascii_array)
    d_phases_x = cuda.device_array(data_size, dtype=np.float32)
    d_phases_y = cuda.device_array(data_size, dtype=np.float32)
    d_coherence = cuda.device_array(data_size, dtype=np.float32)
    
    threads_per_block = BLOCK_SIZE
    blocks_per_grid = (data_size + (threads_per_block - 1)) // threads_per_block
    
    ascii_phase_sync_kernel[blocks_per_grid, threads_per_block](
        d_ascii, d_phases_x, d_phases_y, gear_elasticity, iterations, d_coherence
    )
    cuda.synchronize()
    
    phases_x = d_phases_x.copy_to_host()
    phases_y = d_phases_y.copy_to_host()
    
    avg_x = np.mean(phases_x)
    avg_y = np.mean(phases_y)
    
    norm = math.sqrt(avg_x**2 + avg_y**2)
    if norm > 1e-6:
        avg_x /= norm
        avg_y /= norm
        
    return avg_x, avg_y
