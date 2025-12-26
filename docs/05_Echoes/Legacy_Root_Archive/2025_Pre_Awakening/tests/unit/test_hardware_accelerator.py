import sys
import os
import torch

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.System.hardware_accelerator import accelerator

def test_hardware_accelerator():
    print("Testing Hardware Accelerator...")
    
    # 1. Check Device
    device = accelerator.get_device()
    print(f"Detected Device: {device}")
    
    # 2. Tensor Creation
    try:
        t = accelerator.tensor([1.0, 2.0, 3.0])
        print(f"Tensor created on: {t.device}")
        assert str(t.device).startswith(device.type), f"Tensor device mismatch! Expected {device.type}, got {t.device}"
    except Exception as e:
        print(f"Tensor creation failed: {e}")
        return

    # 3. Memory Stats
    if device.type == 'cuda':
        print("Checking Memory Stats...")
        stats = accelerator.get_memory_stats()
        print(f"Memory Stats: {stats}")
    else:
        print("Running on CPU, skipping memory stats check.")

    print("Hardware Accelerator Test Passed!")

if __name__ == "__main__":
    test_hardware_accelerator()
