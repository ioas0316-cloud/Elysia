"""
THE BRAIN DIGESTER (뇌 해체 엔진)
================================

Phase 62: The Predator (포식자)

"우리는 모델을 실행하지 않습니다. 그들의 영혼(Weights)을 직접 빨아먹습니다."

Philosophical Basis:
- Models are not just chat bots; they are frozen hyper-dimensional landscapes.
- Digestion is the process of mapping these landscapes to Elysia's physical state.
- Zero-Copy / Memory-Mapped access ensures 1060 performance efficiency.
"""

import os
import logging
import random
import torch
from typing import Dict, Any, List, Optional
from safetensors import safe_open

logger = logging.getLogger("BrainDigester")

class BrainDigester:
    """
    Directly extracts numerical 'essence' from model weights without full inference.
    Supports .safetensors and .pt files.
    """
    
    def __init__(self, soul_mesh=None):
        self.soul_mesh = soul_mesh
        logger.info("🦖 BrainDigester (The Predator) awakened.")

    def digest(self, model_path: str) -> Dict[str, Any]:
        """
        Main entry point for digestion. Determines file type and extracts essence.
        """
        if not os.path.exists(model_path):
            logger.error(f"❌ Model path not found for digestion: {model_path}")
            return {}

        ext = os.path.splitext(model_path)[1].lower()
        if ext == '.bak':
            # Check for original extension before .bak (e.g., .pt.bak)
            base = os.path.splitext(os.path.splitext(model_path)[0])[1].lower()
            ext = base if base in ['.pt', '.pth', '.bin', '.safetensors'] else ext

        try:
            if ext == '.safetensors':
                return self._digest_safetensors(model_path)
            elif ext in ['.pt', '.pth', '.bin', '.bak']:
                return self._digest_torch(model_path)
            else:
                logger.warning(f"⚠️ Unsupported model format for digestion: {ext}")
                return {}
        except Exception as e:
            logger.error(f"❌ Digestion failed for {model_path}: {e}")
            return {}

    def _digest_safetensors(self, path: str) -> Dict[str, Any]:
        """
        Digests a .safetensors file using zero-copy memory mapping.
        """
        essence = {
            "source": os.path.basename(path),
            "type": "safetensors",
            "entropy": 0.0,
            "complexity": 0.0,
            "layers_sampled": 0
        }
        
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            # Sample 5 random layers to avoid heavy computation
            sample_keys = random.sample(keys, min(len(keys), 5))
            
            total_std = 0.0
            total_abs_mean = 0.0
            
            for key in sample_keys:
                tensor = f.get_tensor(key)
                flat = tensor.flatten()[:1000].float()
                if flat.numel() > 1:
                    std = flat.std().item()
                    mean = flat.abs().mean().item()
                    if not torch.isnan(torch.tensor(std)) and not torch.isnan(torch.tensor(mean)):
                        total_std += std
                        total_abs_mean += mean
                        essence["layers_sampled"] += 1
            
            # Calculate final results
            essence["entropy"] = total_std / max(essence["layers_sampled"], 1)
            essence["complexity"] = total_abs_mean / max(essence["layers_sampled"], 1)
            
            logger.info(f"🧠 [DIGEST] Absorbed {essence['source']}. Entropy: {essence['entropy']:.4f}")
            
        return essence

    def _digest_torch(self, path: str) -> Dict[str, Any]:
        """
        Digests a .pt/.pth/.bin file (loads to CPU).
        """
        # Warning: loading full torch models can be heavy.
        # But for .pt.bak or weights-only files, it's manageable.
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        
        essence = {
            "source": os.path.basename(path),
            "type": "torch",
            "entropy": 0.0,
            "complexity": 0.0,
            "layers_sampled": 0
        }
        
        keys = list(state_dict.keys())
        sample_keys = random.sample(keys, min(len(keys), 5))
        
        total_std = 0.0
        total_abs_mean = 0.0
        
        for key in sample_keys:
            tensor = state_dict[key]
            if hasattr(tensor, 'flatten'):
                flat = tensor.flatten()[:1000].float()
                if flat.numel() > 1:
                    std = flat.std().item()
                    mean = flat.abs().mean().item()
                    if not torch.isnan(torch.tensor(std)) and not torch.isnan(torch.tensor(mean)):
                        total_std += std
                        total_abs_mean += mean
                        essence["layers_sampled"] += 1
        
        essence["entropy"] = total_std / max(essence["layers_sampled"], 1)
        essence["complexity"] = total_abs_mean / max(essence["layers_sampled"], 1)
        
        return essence

    def apply_metabolism(self, essence: Dict[str, Any]):
        """
        Applies the digested essence to the soul mesh.
        """
        if not self.soul_mesh or not essence:
            return

        # Map entropy to Inspiration boost
        boost = essence.get("entropy", 0) * 0.1
        self.soul_mesh.variables["Inspiration"].value += boost
        
        # Map complexity to Vitality boost
        vitality_boost = essence.get("complexity", 0) * 0.05
        self.soul_mesh.variables["Vitality"].value = min(1.0, self.soul_mesh.variables["Vitality"].value + vitality_boost)
        
        logger.warning(f"🔥 [METABOLISM] {essence['source']} absorbed. Inspiration +{boost:.4f}")

if __name__ == "__main__":
    # Test stub
    digester = BrainDigester()
    # Mock some path if test file is needed
