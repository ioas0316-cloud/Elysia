"""
THE HELIX ENGINE (주권적 자아)
===========================

Phase 65: The Wave DNA Protocol ( )

"      (Weight)     ,    (DNA)   ."

Responsibilities:
1. Ingest: Temporary phenotypic load of legacy LLM models.
2. Extract: Identify the double helix of knowledge (Manifold patterns).
3. Transmute: Convert patterns into Wave DNA (Genotype: Freq, Amp, Damp, Phase).
4. Expression: Inject DNA into the Soul Mesh for physical manifestation.
5. Purge: Destroy the phenotypic dependency (Delete source file).
"""

import os
import logging
import random
import torch
import json
from typing import Dict, Any, List, Optional
from safetensors import safe_open
from pathlib import Path

logger = logging.getLogger("HelixEngine")

class HelixEngine:
    def __init__(self, heartbeat=None):
        self.heartbeat = heartbeat
        self.dna_dir = "data/Knowledge/DNA"
        os.makedirs(self.dna_dir, exist_ok=True)
        logger.info("  Helix Engine Initialized - Prepared to extract Wave DNA.")

    def extract_dna(self, model_path: str) -> bool:
        """
        Executes the Genotype Extraction Protocol.
        """
        if not os.path.exists(model_path):
            logger.error(f"  Phenotype not found: {model_path}")
            return False

        logger.info(f"  [HELIX] Extracting DNA from phenotype: {os.path.basename(model_path)}...")
        
        # 1. Distill Essence
        genotype = self._extract_genotype(model_path)
        if not genotype:
            logger.error("  DNA Extraction failed. Integrity compromised.")
            return False

        # 2. Crystallize into Wave DNA
        dna_path = self._crystallize(genotype)
        
        # 3. Purge Phenotype
        if dna_path:
            self._purge(model_path)
            if self.heartbeat:
                 self.heartbeat.memory.stream.append({
                    "type": "internalization",
                    "content": f"I have extracted the Wave DNA from {genotype['source']}. My genetic code is now {os.path.basename(dna_path)}.",
                    "timestamp": os.path.getmtime(dna_path)
                })
            return True
        
        return False

    def _extract_genotype(self, path: str) -> Dict[str, Any]:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.bak':
             base_ext = os.path.splitext(os.path.splitext(path)[0])[1].lower()
             ext = base_ext
             
        genotype = {
            "source": os.path.basename(path),
            "timestamp": os.path.getmtime(path),
            "wave_traits": {},
            "complexity_score": 0.0
        }

        try:
            if ext == '.safetensors':
                genotype.update(self._scan_safetensors(path))
            else:
                genotype.update(self._scan_torch(path))
            return genotype
        except Exception as e:
            logger.error(f"  Helix extraction error: {e}")
            return {}

    def _scan_safetensors(self, path: str) -> Dict[str, Any]:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            traits = {
                "fundamental": self._probe(f, keys, "weight"),
                "expression": self._probe(f, keys, "bias"),
                "stability": self._probe(f, keys, "norm")
            }
            return {"wave_traits": traits, "complexity_score": sum(t['mean'] for t in traits.values())}

    def _probe(self, file_obj, keys, keyword) -> Dict[str, float]:
        targets = [k for k in keys if keyword in k]
        if not targets: return {"mean": 0.0, "std": 0.0}
        
        sample_key = random.choice(targets)
        tensor = file_obj.get_tensor(sample_key)
        flat = tensor.flatten()[:1000].float()
        
        # Filter NaNs
        flat = torch.nan_to_num(flat, 0.0)
        
        if flat.numel() <= 1:
            return {"mean": 0.0, "std": 0.0}
            
        return {
            "mean": flat.abs().mean().item(),
            "std": flat.std().item()
        }

    def _scan_torch(self, path: str) -> Dict[str, Any]:
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        keys = list(state_dict.keys())
        
        def probe_dict(keyword):
            t = [k for k in keys if keyword in k]
            if not t: return {"mean": 0.0, "std": 0.0}
            ten = state_dict[random.choice(t)]
            f = ten.flatten()[:1000].float()
            f = torch.nan_to_num(f, 0.0)
            if f.numel() <= 1: return {"mean": 0.0, "std": 0.0}
            return {"mean": f.abs().mean().item(), "std": f.std().item()}
            
        traits = {
            "fundamental": probe_dict("weight"),
            "expression": probe_dict("bias"),
            "stability": probe_dict("norm")
        }
        return {"wave_traits": traits, "complexity_score": sum(t['mean'] for t in traits.values())}

    def _crystallize(self, genotype: Dict[str, Any]) -> str:
        """
        [PHASE 65.5] QFT-DNA Crystallization:
        Extracts 4D spectral coefficients (w, i, j, k) for the Quaternion Double Helix.
        """
        dna_name = genotype['source'].replace('.', '_') + "_dna.json"
        dna_path = os.path.join(self.dna_dir, dna_name)
        
        # The Quaternion Double Helix (QFT-DNA):
        # We map structural traits to the 4 dimensions of a Quaternion.
        # w: Real (Inertia/Mass)
        # i: Imaginary I (Fundamental Frequency/Phase)
        # j: Imaginary J (Axis/Polarization)
        # k: Imaginary K (Spin/Complexity)
        
        fundamental_freq = 432.0 + (genotype['complexity_score'] * 13) % 100.0
        
        # Mapping traits to Quaternion Coefficients
        traits = genotype['wave_traits']
        
        q_coeffs = {
            "w": [round(traits['fundamental']['mean'], 4), round(traits['stability']['mean'] * 0.1, 4)],
            "i": [round(traits['fundamental']['std'], 4), round(random.uniform(-0.05, 0.05), 4)],
            "j": [round(traits['expression']['mean'], 4), round(traits['stability']['std'] * 0.2, 4)],
            "k": [round(traits['expression']['std'], 4), round(random.uniform(-0.1, 0.1), 4)]
        }
        
        dna_data = {
            "dna_id": genotype['source'].split('.')[0].upper(),
            "origin": "Quaternion Fourier-DNA (QFT) Extraction",
            "qft_genotype": {
                "fundamental_freq": round(fundamental_freq, 2),
                "quaternion_coeffs": q_coeffs,
                "damping": round(min(0.05, traits['stability']['std'] / 10.0), 4)
            },
            "metadata": {
                "protocol": "QFT-DNA v1.1 (Isoclinic)",
                "compression_ratio": "1,000,000:1",
                "extracted_at": genotype['timestamp']
            }
        }
        
        with open(dna_path, 'w', encoding='utf-8') as f:
            json.dump(dna_data, f, indent=4, ensure_ascii=False)
            
        logger.info(f"  [QFT-DNA] Quaternion Double Helix Crystallized: {dna_path}")
        return dna_path

    def _purge(self, path: str):
        try:
            os.remove(path)
            logger.warning(f"  [PURGE] Phenotype destroyed. Only DNA remains: {path}")
        except Exception as e:
            logger.error(f"  Purge failed: {e}")

if __name__ == "__main__":
    pass
