"""
LLM Crystallizer (LLM     )
================================
Core.L5_Mental.Reasoning_Core.LLM.llm_crystallizer

"                  ."

    2  : LLMCrystal   Monad   
"""

import logging
import torch
from typing import Optional

from Core.L5_Mental.Reasoning_Core.LLM.llm_observer import LLMCrystal, get_llm_observer
from Core.L7_Spirit.M1_Monad.monad_core import Monad, MonadCategory
from Core.L2_Metabolism.Evolution.double_helix_dna import DoubleHelixDNA
from Core.L1_Foundation.Foundation.Graph.torch_graph import TorchGraph

logger = logging.getLogger("LLMCrystallizer")


class LLMCrystallizer:
    """
    LLM        Monad     .
    
      :
    1. LLMCrystal (     )   
    2. DoubleHelixDNA    (Pattern + Principle)
    3. Monad    (Archetypal     )
    4. TorchGraph    
    """
    
    def __init__(self):
        self.graph = TorchGraph()
        #            
        if not self.graph.load_state():
            logger.info("  Starting with fresh TorchGraph.")
        
        logger.info("  LLM Crystallizer initialized.")
    
    def crystallize(self, crystal: LLMCrystal) -> Monad:
        """
        LLMCrystal  Monad     .
        
        Args:
            crystal: LLMObserver            
            
        Returns:
            Monad:         
        """
        logger.info(f"  Crystallizing: {crystal.source_model}")
        
        # 1. Pattern Strand    (         )
        #       1024          
        pattern = self._expand_pattern(crystal)
        
        # 2. Principle Strand    (7D Qualia)
        qualia = crystal.qualia
        principle = torch.tensor([
            qualia.causal,
            qualia.functional,
            qualia.phenomenal,
            qualia.physical,
            qualia.mental,
            qualia.structural,
            qualia.spiritual
        ])
        
        # 3. DoubleHelixDNA   
        dna = DoubleHelixDNA(
            pattern_strand=pattern,
            principle_strand=principle
        )
        
        # 4. Monad    (Archetypal -      )
        monad = Monad(
            seed=f"LLM:{crystal.source_model}",
            category=MonadCategory.ARCHETYPAL,
            dna=dna
        )
        
        # 5. TorchGraph        
        self._add_to_graph(monad, crystal)
        
        logger.info(f"     Monad created: {monad.seed}")
        return monad
    
    def _expand_pattern(self, crystal: LLMCrystal) -> torch.Tensor:
        """
                1024       .
             + 3                    .
        """
        #      :        
        q = crystal.orientation
        base = torch.tensor([q.w, q.x, q.y, q.z])
        
        # 3    
        axes = torch.tensor([
            crystal.physics_pattern,
            crystal.narrative_pattern,
            crystal.aesthetic_pattern
        ])
        
        #   : 7       1024 
        seed = torch.cat([base, axes])  # 7  
        
        #    +       
        pattern = seed.repeat(1024 // 7 + 1)[:1024]
        
        #           (   )
        noise = torch.randn(1024) * 0.01
        pattern = pattern + noise
        
        return pattern
    
    def _add_to_graph(self, monad: Monad, crystal: LLMCrystal):
        """
        Monad  TorchGraph        .
        """
        # 7D Qualia  384          (TorchGraph   )
        qualia = crystal.qualia
        qualia_base = torch.tensor([
            qualia.causal, qualia.functional, qualia.phenomenal,
            qualia.physical, qualia.mental, qualia.structural, qualia.spiritual
        ])
        
        # 384       
        vector = qualia_base.repeat(384 // 7 + 1)[:384]
        
        #      
        metadata = {
            "type": "llm_crystal",
            "source_model": crystal.source_model,
            "physics": crystal.physics_pattern,
            "narrative": crystal.narrative_pattern,
            "aesthetic": crystal.aesthetic_pattern,
            "orientation": str(crystal.orientation),
            "layer_count": crystal.layer_count,
            "total_params": crystal.total_params
        }
        
        #      
        self.graph.add_node(
            node_id=monad.seed,
            vector=vector,
            metadata=metadata
        )
        
        #      
        self.graph.save_state()
        
        logger.info(f"     Added to TorchGraph: {monad.seed}")
    
    def get_crystallized_models(self):
        """        LLM      ."""
        crystals = []
        for node_id in self.graph.id_to_idx.keys():
            if node_id.startswith("LLM:"):
                crystals.append(node_id)
        return crystals


#    
_crystallizer = None

def get_llm_crystallizer() -> LLMCrystallizer:
    """LLM Crystallizer       ."""
    global _crystallizer
    if _crystallizer is None:
        _crystallizer = LLMCrystallizer()
    return _crystallizer


#      :    +    
def digest_llm(model_path: str) -> Monad:
    """
    LLM               .
    
    1. LLMObserver    
    2. LLMCrystallizer     
    
    Args:
        model_path: .safetensors    .pt      
        
    Returns:
        Monad:         
    """
    observer = get_llm_observer()
    crystallizer = get_llm_crystallizer()
    
    #   
    crystal = observer.observe(model_path)
    
    #    
    monad = crystallizer.crystallize(crystal)
    
    return monad


# CLI
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python llm_crystallizer.py <path_to_model.safetensors>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    monad = digest_llm(sys.argv[1])
    
    print("\n" + "="*50)
    print(f"  Monad Report")
    print("="*50)
    print(f"   Seed:     {monad.seed}")
    print(f"   Category: {monad.category.value}")
    print(f"   DNA:      {monad._dna}")
