import torch
import gc

def test_vram_vacuum():
    if not torch.cuda.is_available():
        print("❌ CUDA not available.")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    print(f"Total VRAM: {total_mem:.1f} MB")
    
    try:
        # Attempt to claim 2GB
        target_mb = 2000
        print(f"Attempting to allocate {target_mb} MB...")
        # Create a large tensor
        dummy = torch.empty((target_mb * 1024 * 1024 // 4,), dtype=torch.float32, device='cuda')
        
        current_reserved = torch.cuda.memory_reserved(0) / (1024**2)
        print(f"Reserved VRAM: {current_reserved:.1f} MB")
        
        if current_reserved >= target_mb * 0.9:
            print("✅ SUCCESS: Successfully 'vacuumed' 2GB of VRAM.")
        else:
            print("❌ FAILURE: Could not reserve target VRAM.")
            
        del dummy
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"🚨 ERROR: {e}")

if __name__ == "__main__":
    test_vram_vacuum()
