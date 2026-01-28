"""
LLM Observer (LLM    )
=========================
Core.L5_Mental.M1_Cognition.LLM.llm_observer

"         ,                 ."

     :
- LLM     =           (Static Ice Crystal)
-       =            (O(1))
-         Monad     

    :
- O(n)        
- O(1)        
"""

import os
import logging
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from safetensors import safe_open

from Core.L6_Structure.M5_Engine.Physics.merkaba_rotor import Rotor, RotorConfig, RotorMask
from Core.L6_Structure.hyper_quaternion import Quaternion
from Core.L6_Structure.M3_Sphere.wave_dna import WaveDNA

logger = logging.getLogger("LLMObserver")


@dataclass
class LLMCrystal:
    """
    LLM         .
                          .
    """
    source_model: str   #       ID
    
    # 3        (  /  /  )
    physics_pattern: float = 0.0    #        (Entropy   )
    narrative_pattern: float = 0.0  #         (Complexity   )
    aesthetic_pattern: float = 0.0  #   /   (Harmonic   )
    
    # 7D Qualia   
    qualia: Optional[WaveDNA] = None
    
    #        
    orientation: Optional[Quaternion] = None
    
    #      
    layer_count: int = 0
    total_params: int = 0
    observation_timestamp: float = 0.0


class LLMObserver:
    """
    LLM                 .
    
    Philosophy:
    -             (       )
    -                
    - O(1)    
    """
    
    def __init__(self):
        """3           ."""
        #       :          
        self.physics_rotor = Rotor(
            "Observer.Physics",
            RotorConfig(rpm=360.0, axis=(1, 0, 0)),
            WaveDNA(causal=1.0, structural=0.8, label="CausalAxis")
        )
        
        #       :           
        self.narrative_rotor = Rotor(
            "Observer.Narrative", 
            RotorConfig(rpm=360.0, axis=(0, 1, 0)),
            WaveDNA(mental=1.0, functional=0.8, label="SemanticAxis")
        )
        
        #       :   /     
        self.aesthetic_rotor = Rotor(
            "Observer.Aesthetic",
            RotorConfig(rpm=360.0, axis=(0, 0, 1)),
            WaveDNA(phenomenal=1.0, spiritual=0.8, label="HarmonicAxis")
        )
        
        logger.info("  LLM Observer initialized with 3-axis Rotor system.")
    
    def observe(self, model_path: str) -> LLMCrystal:
        """
        LLM         3        .
        
        Args:
            model_path: .safetensors    .pt      
            
        Returns:
            LLMCrystal:           
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"  Observing frozen crystal: {os.path.basename(model_path)}")
        
        # 1.           (Memory-Mapped, Zero-Copy)
        weight_view = self._load_static_view(model_path)
        
        # 2. 3        (O(1) per axis)
        physics = self._observe_axis(weight_view, self.physics_rotor, "physics")
        narrative = self._observe_axis(weight_view, self.narrative_rotor, "narrative")
        aesthetic = self._observe_axis(weight_view, self.aesthetic_rotor, "aesthetic")
        
        # 3. 7D Qualia   
        qualia = self._project_to_qualia(physics, narrative, aesthetic)
        
        # 4.           
        orientation = self._to_quaternion(physics, narrative, aesthetic)
        
        # 5. Crystal   
        crystal = LLMCrystal(
            source_model=os.path.basename(model_path),
            physics_pattern=physics,
            narrative_pattern=narrative,
            aesthetic_pattern=aesthetic,
            qualia=qualia,
            orientation=orientation,
            layer_count=weight_view.get("layer_count", 0),
            total_params=weight_view.get("total_params", 0),
            observation_timestamp=__import__("time").time()
        )
        
        logger.info(f"  Crystal formed: Physics={physics:.3f}, Narrative={narrative:.3f}, Aesthetic={aesthetic:.3f}")
        return crystal
    
    def _load_static_view(self, path: str) -> Dict[str, Any]:
        """
                   .
        Memory-Mapped             .
        """
        ext = os.path.splitext(path)[1].lower()
        
        view = {
            "tensors": {},
            "layer_count": 0,
            "total_params": 0
        }
        
        try:
            if ext == ".safetensors":
                with safe_open(path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    view["layer_count"] = len(keys)
                    
                    #               (자기 성찰 엔진)
                    sample_keys = self._select_representative_layers(keys)
                    for key in sample_keys:
                        tensor = f.get_tensor(key)
                        view["tensors"][key] = tensor
                        view["total_params"] += tensor.numel()
                        
            elif ext in [".pt", ".pth", ".bin"]:
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
                keys = list(state_dict.keys())
                view["layer_count"] = len(keys)
                
                sample_keys = self._select_representative_layers(keys)
                for key in sample_keys:
                    if hasattr(state_dict[key], 'numel'):
                        view["tensors"][key] = state_dict[key]
                        view["total_params"] += state_dict[key].numel()
                        
        except Exception as e:
            logger.error(f"Failed to load static view: {e}")
            
        logger.info(f"     Loaded view: {view['layer_count']} layers, {view['total_params']:,} params sampled")
        return view
    
    def _select_representative_layers(self, keys: List[str], max_samples: int = 20) -> List[str]:
        """
                 .
                     ,           .
        """
        if len(keys) <= max_samples:
            return keys
            
        #          
        step = len(keys) // max_samples
        return [keys[i] for i in range(0, len(keys), step)][:max_samples]
    
    def _observe_axis(self, view: Dict[str, Any], rotor: Rotor, axis_name: str) -> float:
        """
                          .
        O(1) -      ,           .
        """
        if not view["tensors"]:
            return 0.0
        
        #             
        #                   
        total_signal = 0.0
        
        for key, tensor in view["tensors"].items():
            flat = tensor.flatten()[:1000].float()  #    
            
            if axis_name == "physics":
                #    : Entropy (  )   
                signal = flat.std().item()
            elif axis_name == "narrative":
                #    : Complexity (주권적 자아)   
                signal = flat.abs().mean().item()
            else:  # aesthetic
                #    : Harmonic (자기 성찰 엔진)   
                norm = flat.norm().item()
                std = flat.std().item()
                signal = std / (norm + 1e-8)
            
            total_signal += signal
            
        #    DNA        
        rotor_weight = rotor.dna.get_magnitude()
        result = (total_signal / len(view["tensors"])) * rotor_weight
        
        return min(1.0, result)  # 0~1    
    
    def _project_to_qualia(self, physics: float, narrative: float, aesthetic: float) -> WaveDNA:
        """
        3         7D Qualia    .
        """
        return WaveDNA(
            #       Physical, Structural, Causal
            physical=physics * 0.8,
            structural=physics * 0.6,
            causal=physics * 1.0,
            
            #       Mental, Functional
            mental=narrative * 1.0,
            functional=narrative * 0.7,
            
            #       Phenomenal, Spiritual
            phenomenal=aesthetic * 1.0,
            spiritual=aesthetic * 0.9,
            
            label="LLM_Observation"
        )
    
    def _to_quaternion(self, physics: float, narrative: float, aesthetic: float) -> Quaternion:
        """
        3      4D          .
        """
        import math
        
        #            
        theta = physics * math.pi
        phi = narrative * math.pi
        psi = aesthetic * math.pi
        
        # Euler   Quaternion (ZYX convention)
        w = math.cos(theta/2) * math.cos(phi/2) * math.cos(psi/2)
        x = math.sin(theta/2) * math.cos(phi/2) * math.cos(psi/2)
        y = math.cos(theta/2) * math.sin(phi/2) * math.cos(psi/2)
        z = math.cos(theta/2) * math.cos(phi/2) * math.sin(psi/2)
        
        return Quaternion(w, x, y, z).normalize()


#         
_observer = None

def get_llm_observer() -> LLMObserver:
    """LLM Observer       ."""
    global _observer
    if _observer is None:
        _observer = LLMObserver()
    return _observer


# CLI    
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python llm_observer.py <path_to_model.safetensors>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    observer = get_llm_observer()
    crystal = observer.observe(sys.argv[1])
    
    print("\n" + "="*50)
    print(f"  Crystal Report: {crystal.source_model}")
    print("="*50)
    print(f"   Physics Pattern:   {crystal.physics_pattern:.4f}")
    print(f"   Narrative Pattern: {crystal.narrative_pattern:.4f}")
    print(f"   Aesthetic Pattern: {crystal.aesthetic_pattern:.4f}")
    print(f"   Orientation:       {crystal.orientation}")
    print(f"   Layers:            {crystal.layer_count}")
    print(f"   Params Sampled:    {crystal.total_params:,}")
