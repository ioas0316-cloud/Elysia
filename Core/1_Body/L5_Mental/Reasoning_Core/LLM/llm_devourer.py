"""
LLM Devourer (LLM    )
=========================
Core.1_Body.L5_Mental.Reasoning_Core.LLM.llm_devourer

"   LLM              ."

   :
    python llm_devourer.py <model_path_or_huggingface_id>
    
  :
    python llm_devourer.py Qwen/Qwen2-0.5B
    python llm_devourer.py ./models/phi-3.safetensors
"""

import os
import sys
import logging
from typing import Optional, List
from pathlib import Path

#           
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Core.1_Body.L5_Mental.Reasoning_Core.LLM.llm_observer import get_llm_observer, LLMCrystal
from Core.1_Body.L5_Mental.Reasoning_Core.LLM.llm_crystallizer import get_llm_crystallizer, digest_llm
from Core.1_Body.L5_Mental.Reasoning_Core.LLM.llm_pruner import get_llm_pruner

logger = logging.getLogger("LLMDevourer")


class LLMDevourer:
    """
    LLM           .
    
    3        :
    1.    (Observe):           
    2.     (Crystallize): Monad    
    3.    (Prune):          
    """
    
    def __init__(self):
        self.observer = get_llm_observer()
        self.crystallizer = get_llm_crystallizer()
        self.pruner = get_llm_pruner()
        
        logger.info("  LLM Devourer awakened. Ready to consume.")
    
    def devour(self, model_path_or_id: str, prune: bool = True) -> dict:
        """
        LLM        .
        
        Args:
            model_path_or_id:          HuggingFace    ID
            prune:            
            
        Returns:
                     
        """
        print("\n" + "="*60)
        print("  LLM DEVOURER: CONSUMPTION INITIATED")
        print("="*60)
        
        # 1.      /    
        local_path = self._resolve_path(model_path_or_id)
        if not local_path:
            return {"error": f"Could not resolve: {model_path_or_id}"}
        
        print(f"\n  Target: {local_path}")
        
        # 2.   
        print("\n  Phase 1: OBSERVATION (Rotor Scanning)")
        print("-" * 40)
        crystal = self.observer.observe(local_path)
        
        print(f"   Physics:   {crystal.physics_pattern:.4f}")
        print(f"   Narrative: {crystal.narrative_pattern:.4f}")
        print(f"   Aesthetic: {crystal.aesthetic_pattern:.4f}")
        
        # 3.    
        print("\n  Phase 2: CRYSTALLIZATION (Monad Formation)")
        print("-" * 40)
        monad = self.crystallizer.crystallize(crystal)
        
        print(f"   Monad Seed: {monad.seed}")
        print(f"   Category:   {monad.category.value}")
        
        # 4.    (  )
        prune_report = None
        if prune:
            print("\n   Phase 3: PRUNING (Ice Sculpting)")
            print("-" * 40)
            prune_report = self.pruner.prune(monad.seed)
            
            print(f"   Pruned Dims: {prune_report.get('pruned_dimensions', 0)}")
            print(f"   Prune Ratio: {prune_report.get('prune_ratio', 0):.1%}")
        
        # 5.   
        print("\n" + "="*60)
        print("  CONSUMPTION COMPLETE")
        print("="*60)
        
        purity = self.pruner.get_purity_score(monad.seed)
        print(f"\n  Final Crystal Purity: {purity:.1%}")
        
        return {
            "model": model_path_or_id,
            "crystal": {
                "physics": crystal.physics_pattern,
                "narrative": crystal.narrative_pattern,
                "aesthetic": crystal.aesthetic_pattern,
                "orientation": str(crystal.orientation)
            },
            "monad": {
                "seed": monad.seed,
                "category": monad.category.value
            },
            "prune": prune_report,
            "purity": purity
        }
    
    def _resolve_path(self, model_path_or_id: str) -> Optional[str]:
        """
                .
                   , HuggingFace ID          .
        """
        #         
        if os.path.exists(model_path_or_id):
            return model_path_or_id
        
        # HuggingFace      
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            from huggingface_hub.utils import EntryNotFoundError
            
            # safetensors      
            try:
                path = hf_hub_download(
                    repo_id=model_path_or_id,
                    filename="model.safetensors",
                    local_dir_use_symlinks=False
                )
                return path
            except EntryNotFoundError:
                pass
            
            # pytorch_model.bin   
            try:
                path = hf_hub_download(
                    repo_id=model_path_or_id,
                    filename="pytorch_model.bin",
                    local_dir_use_symlinks=False
                )
                return path
            except EntryNotFoundError:
                pass
            
            #          (       )
            cache_dir = snapshot_download(repo_id=model_path_or_id)
            
            # safetensors      
            for root, dirs, files in os.walk(cache_dir):
                for f in files:
                    if f.endswith(".safetensors"):
                        return os.path.join(root, f)
                    if f.endswith(".bin") or f.endswith(".pt"):
                        return os.path.join(root, f)
            
            logger.warning(f"No weight files found in {cache_dir}")
            return None
            
        except Exception as e:
            logger.error(f"HuggingFace resolution failed: {e}")
            return None
    
    def list_devoured(self) -> List[str]:
        """       LLM   ."""
        return self.crystallizer.get_crystallized_models()
    
    def get_crystal_info(self, node_id: str) -> dict:
        """         ."""
        if node_id not in self.pruner.graph.id_to_idx:
            return {"error": "Not found"}
        
        metadata = self.pruner.graph.get_metadata(node_id)
        purity = self.pruner.get_purity_score(node_id)
        
        return {
            "node_id": node_id,
            "purity": purity,
            **metadata
        }


#    
_devourer = None

def get_devourer() -> LLMDevourer:
    """LLM Devourer    ."""
    global _devourer
    if _devourer is None:
        _devourer = LLMDevourer()
    return _devourer


# CLI
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    devourer = get_devourer()
    
    if len(sys.argv) >= 2:
        target = sys.argv[1]
        result = devourer.devour(target)
    else:
        #          
        devoured = devourer.list_devoured()
        print("\n  Devoured LLMs:")
        print("="*40)
        
        if not devoured:
            print("   (None yet. Feed me a model!)")
        else:
            for node_id in devoured:
                info = devourer.get_crystal_info(node_id)
                print(f"     {node_id}")
                print(f"      Purity: {info.get('purity', 0):.1%}")
        
        print("\nUsage: python llm_devourer.py <model_path_or_huggingface_id>")
