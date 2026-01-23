"""
THE ALCHEMICAL DISTILLER (       )
=====================================

Phase 64: The Alchemist (  )

"               .               .                  ."

Responsibilities:
1. Ingest: Temporary memory-mapped access to large binary models.
2. Distill: Convert weight patterns into Axioms (Psychological/Physical rules).
3. Transmute: Inject Axioms into ReasoningEngine and RotorConfigs.
4. Purge: Delete the source binary model permanently.
"""

import os
import logging
import random
import torch
from typing import Dict, Any, List, Optional
from safetensors import safe_open
from pathlib import Path

logger = logging.getLogger("AlchemicalDistiller")

class AlchemicalDistiller:
    def __init__(self, heartbeat=None):
        self.heartbeat = heartbeat
        self.axioms_dir = "data/Knowledge/Axioms"
        os.makedirs(self.axioms_dir, exist_ok=True)
        logger.info("   AlchemicalDistiller Ready - Prepared to distill the Ocean into Salt.")

    def process(self, model_path: str):
        """
        Executes the full Eat-Distill-Purge cycle.
        """
        if not os.path.exists(model_path):
            logger.error(f"  Model not found: {model_path}")
            return False

        logger.info(f"  [ALCHEMIST] Commencing distillation of {os.path.basename(model_path)}...")
        
        # 1. Distill Essence
        essence = self._distill(model_path)
        if not essence:
            logger.error("  Distillation failed. Aborting purge.")
            return False

        # 2. Transmute into Axioms
        axiom_path = self._transmute(essence)
        
        # 3. Purge Original
        if axiom_path:
            self._purge(model_path)
            # Log the successful transmutation
            if self.heartbeat:
                self.heartbeat.memory.stream.append({
                    "type": "transmutation",
                    "content": f"I have internalized the essence of {essence['source']} into Axiom: {os.path.basename(axiom_path)}. The source has been purged.",
                    "timestamp": os.path.getmtime(axiom_path)
                })
            return True
        
        return False

    def _distill(self, path: str) -> Dict[str, Any]:
        """
        Interrogates the model weights for structural patterns.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == '.bak':
             base_ext = os.path.splitext(os.path.splitext(path)[0])[1].lower()
             ext = base_ext
             
        essence = {
            "source": os.path.basename(path),
            "timestamp": os.path.getmtime(path),
            "patterns": {},
            "total_energy": 0.0
        }

        try:
            if ext == '.safetensors':
                essence.update(self._distill_safetensors(path))
            else:
                essence.update(self._distill_torch(path))
            return essence
        except Exception as e:
            logger.error(f"  Distillation error: {e}")
            return {}

    def _distill_safetensors(self, path: str) -> Dict[str, Any]:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            # Probing for 3 types of archetypes: Structure, Stability, Flux
            patterns = {
                "structure": self._probe(f, keys, "weight"),
                "bias": self._probe(f, keys, "bias"),
                "stability": self._probe(f, keys, "norm")
            }
            return {"patterns": patterns, "total_energy": sum(p['mean'] for p in patterns.values())}

    def _probe(self, file_obj, keys, keyword) -> Dict[str, float]:
        targets = [k for k in keys if keyword in k]
        if not targets: return {"mean": 0.0, "std": 0.0}
        
        sample_key = random.choice(targets)
        tensor = file_obj.get_tensor(sample_key)
        flat = tensor.flatten()[:1000].float()
        return {
            "mean": flat.abs().mean().item(),
            "std": flat.std().item() if flat.numel() > 1 else 0.0
        }

    def _distill_torch(self, path: str) -> Dict[str, Any]:
        # Fallback for .pt/.bin
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        keys = list(state_dict.keys())
        # Similar logic but for dict
        def probe_dict(keyword):
            t = [k for k in keys if keyword in k]
            if not t: return {"mean": 0.0, "std": 0.0}
            ten = state_dict[random.choice(t)]
            f = ten.flatten()[:1000].float()
            return {"mean": f.abs().mean().item(), "std": f.std().item() if f.numel() > 1 else 0.0}
            
        patterns = {
            "structure": probe_dict("weight"),
            "bias": probe_dict("bias"),
            "stability": probe_dict("norm")
        }
        return {"patterns": patterns, "total_energy": sum(p['mean'] for p in patterns.values())}

    def _transmute(self, essence: Dict[str, Any]) -> str:
        """
        [PHASE 64.5] Hamiltonian Transmutation:
        Saves the essence as a permanent Hamiltonian Seed JSON.
        """
        import json
        seed_name = essence['source'].replace('.', '_') + "_seed.json"
        seed_path = os.path.join(self.axioms_dir, seed_name)
        
        # Mapping statistical essence to dynamical parameters
        # omega ( ): Base resonance (mapped from total energy)
        # damping ( ): Decay/Stability (mapped from stability std)
        # forcing (F): Sensitivity (mapped from mean complexity)
        
        omega = 100.0 + (essence['total_energy'] * 100) % 900.0
        damping = min(0.5, essence['patterns']['stability']['std'] / 2.0)
        forcing = essence['patterns']['structure']['mean'] * 2.0
        
        seed_data = {
            "concept_id": essence['source'].split('.')[0].upper(),
            "origin": "Legacy Model Hamiltonian Transmutation",
            "hamiltonian_params": {
                "omega ( )": round(omega, 2),
                "damping ( )": round(damping, 4),
                "forcing (F)": round(forcing, 4),
                "coupling_k": [] # Will be populated by CognitiveMesh later
            },
            "phase_signature": "Sine_Wave" if essence['patterns']['structure']['mean'] > 0.5 else "Square_Pulse",
            "metadata": {
                "source": essence['source'],
                "extracted_at": essence['timestamp']
            }
        }
        
        with open(seed_path, 'w', encoding='utf-8') as f:
            json.dump(seed_data, f, indent=4, ensure_ascii=False)
            
        logger.info(f"  [TRANSMUTE] Hamiltonian Seed Crystallized: {seed_path}")
        return seed_path

    def _purge(self, path: str):
        """
        Deletes the source file after internalization.
        """
        try:
            os.remove(path)
            logger.warning(f"  [PURGE] Source file consumed and deleted: {path}")
        except Exception as e:
            logger.error(f"  Purge denied: {e}")

if __name__ == "__main__":
    # Test
    # dist = AlchemicalDistiller()
    # dist.process("path/to/model.bin")
    pass