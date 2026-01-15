import torch
import sys
import os

# Ensure root is in path
sys.path.append(os.getcwd())

def check_acceleration():
    print("✨ [SOVEREIGN SPOTLIGHT] Hardware Acceleration Audit\n")
    
    # 1. System Check
    has_cuda = torch.cuda.is_available()
    print(f"🖥️  CUDA Available: {'✅ YES' if has_cuda else '❌ NO'}")
    
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2
        print(f"🛸 GPU Model: {gpu_name}")
        print(f"🔋 VRAM Status: {vram_free:.1f} MB Free / {vram_total:.1f} MB Total")
    
    print("\n" + "-"*40)
    
    # 2. Organ Audit
    from Core.Foundation.Graph.torch_graph import TorchGraph
    graph = TorchGraph()
    print(f"🧠 TorchGraph Device: {graph.device}")
    
    from Core.Intelligence.LLM.huggingface_bridge import SovereignBridge
    bridge = SovereignBridge()
    print(f"👅 SovereignBridge Device: {bridge.device}")
    
    # 3. Memory Residency Check
    # Let's move a small tensor to GPU and confirm
    if has_cuda:
        test_tensor = torch.ones((100, 100)).cuda()
        print(f"💎 Test Tensor Location: {test_tensor.device}")
        
        # Check current allocation
        print(f"📈 Current GPU Memory Allocation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    print("\n" + "="*40)
    print("Python is the 'Director' (CPU), but Torch is the 'Muscle' (GPU).")
    print("Everything inside the .to('cuda') calls is happening on your 1060.")
    print("="*40)

if __name__ == "__main__":
    check_acceleration()
