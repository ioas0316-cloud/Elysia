"""
[CRYSTALLIZATION FORGE]
"Where history solidifies into the structure of tomorrow's mind."
"""

import os
import json
import hashlib
import torch
import numpy as np
from typing import Dict, Any, List

class CrystallizationForge:
    def __init__(self, chronicles_dir: str):
        self.chronicles_dir = chronicles_dir
        self.crystallized_tensors_dir = os.path.join(os.path.dirname(self.chronicles_dir), "Crystallized")
        os.makedirs(self.crystallized_tensors_dir, exist_ok=True)
        print(f"💎 [CRYSTALLIZATION FORGE] Initialized. Ready to compress History into 3D Tensors.")

    def sweep_and_crystallize(self, dimension: int = 21) -> torch.Tensor:
        files = [f for f in os.listdir(self.chronicles_dir) if f.endswith(".json")]
        if not files:
             print("   ↳ 🌪️ No new history to crystallize.")
             return torch.zeros((dimension, dimension, dimension))

        print(f"   ↳ 🧹 Sweeping {len(files)} Chronicle logs...")

        if torch.cuda.is_available():
            crystal_block = torch.zeros((dimension, dimension, dimension), device='cuda', dtype=torch.float32)
        else:
            crystal_block = torch.zeros((dimension, dimension, dimension), device='cpu', dtype=torch.float32)

        processed_count = 0
        for file in files:
            filepath = os.path.join(self.chronicles_dir, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    record = json.load(f)

                phase = record.get("phase_coordinates", 0.0)
                event_type = record.get("event_type", "Unknown")

                idx_x = int(abs(phase * dimension)) % dimension
                idx_y = len(event_type) % dimension
                idx_z = int(hashlib.md5(str(record.get("data", "")).encode()).hexdigest(), 16) % dimension

                crystal_block[idx_x, idx_y, idx_z] += 1.0
                processed_count += 1

                os.remove(filepath)

            except Exception as e:
                print(f"   ↳ ⚠️ Failed to ingest {file}: {e}")

        if processed_count > 0:
             max_val = torch.max(crystal_block)
             if max_val > 0:
                 crystal_block = crystal_block / max_val

        print(f"   ↳ 💎 Crystallization complete. A new 3D Tensor Block was forged from {processed_count} events.")

        tensor_path = os.path.join(self.crystallized_tensors_dir, f"crystal_core_{int(os.path.getmtime(self.chronicles_dir))}.pt")
        torch.save(crystal_block, tensor_path)

        return crystal_block
