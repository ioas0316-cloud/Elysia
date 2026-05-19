
import sys
import time
sys.path.append(r'c:\Elysia')
from Core.FoundationLayer.Foundation.omni_graph import get_omni_graph
from Core.FoundationLayer.Foundation.torch_graph import get_torch_graph

def benchmark_engine():
    print("🏎️  Engine Benchmark: CPU vs GPU (Torch)")
    print("=======================================")
    
    # Setup
    N_NODES = 500 # Start small for safe test on 1060 prototype
    ITERATIONS = 20
    
    # 1. CPU Test
    omni = get_omni_graph()
    omni.nodes = {} # Clear
    print(f"\n[CPU] seeding {N_NODES} nodes...")
    for i in range(N_NODES):
        omni.add_vector(f"Node_{i}", [0.1] * 10)
        
    start_cpu = time.time()
    omni.apply_gravity(iterations=ITERATIONS)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu
    print(f"   ⏱️ CPU Time: {cpu_time:.4f} sec")
    
    # 2. GPU Test
    torch_g = get_torch_graph()
    # Reset (Hack for prototype)
    torch_g.id_to_idx = {}
    import torch
    torch_g.pos_tensor = torch.zeros((0, 4), device=torch_g.device)
    torch_g.vec_tensor = torch.zeros((0, 64), device=torch_g.device)
    
    print(f"\n[GPU] seeding {N_NODES} nodes...")
    # Batch add simulation (add_node is slow one by one, but gravity is the test target)
    for i in range(N_NODES):
        torch_g.add_node(f"Node_{i}", [0.1] * 10)
        
    start_gpu = time.time()
    torch_g.apply_gravity(iterations=ITERATIONS)
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu
    print(f"   ⏱️ GPU Time: {gpu_time:.4f} sec")
    
    # Result
    ratio = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"\n🚀 Speedup: {ratio:.2f}x")
    if torch_g.use_cuda:
        print("   (Using CUDA Acceleration)")
    else:
        print("   (Using CPU Tensors - No CUDA detected)")

if __name__ == "__main__":
    benchmark_engine()
