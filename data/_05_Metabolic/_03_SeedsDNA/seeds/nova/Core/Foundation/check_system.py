
import sys
import os
import time
import psutil

print("--- System Check ---")

# 1. Check Hippocampus Syntax
print("[1] Checking Hippocampus Syntax...")
try:
    from Core._01_Foundation.Foundation.hippocampus import Hippocampus
    print("    ✅ Hippocampus imported successfully.")
except Exception as e:
    print(f"    ❌ Hippocampus Import Failed: {e}")
    import traceback
    traceback.print_exc()

# 2. Check CUDA / Torch
print("\n[2] Checking CUDA / Torch...")
try:
    import torch
    print(f"    ✅ Torch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"    ✅ CUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        print("    ⚠️ CUDA Not Available (Torch installed but no GPU detected by it)")
except ImportError:
    print("    ❌ Torch not installed.")

# 3. Check CPU Usage
print("\n[3] Checking CPU Usage...")
for i in range(3):
    print(f"    CPU: {psutil.cpu_percent(interval=1.0)}%")

print("\n--- End Check ---")
