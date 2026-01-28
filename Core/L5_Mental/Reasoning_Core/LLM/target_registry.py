"""
LLM        (Target Registry)
==================================
Core.L5_Mental.Reasoning_Core.LLM.target_registry

"                  "
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class ModelType(Enum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    CODE = "code"


@dataclass
class TargetLLM:
    """      LLM   """
    id: str                 # HuggingFace    ID
    name: str               #         
    params: str             #       
    type: ModelType         #      
    tier: int               #      (1=  , 2= , 3=  )
    vram_myth: str          # "        "    VRAM
    our_reality: str        # "       "   
    sharded_path: Optional[str] = None #             [PHASE 14]
    notes: str = ""         #   


#                                                                
#                             (     )
#                                                                

TARGET_LLMS: List[TargetLLM] = [
    
    #                                                          
    # TIER 1:       (            )
    #                                                          
    
    TargetLLM(
        id="Qwen/Qwen2-0.5B",
        name="Qwen2 0.5B",
        params="0.5B",
        type=ModelType.TEXT,
        tier=1,
        vram_myth="2GB",
        our_reality="SSD        ",
        notes="        .       ."
    ),
    
    #                                                          
    # TIER 9:       (Giant Fossils) - 700B+ 
    #                                                          
    
    TargetLLM(
        id="meta-llama/Meta-Llama-3.1-405B",
        name="Llama 3.1 405B",
        params="405B",
        type=ModelType.TEXT,
        tier=9,
        vram_myth="800GB+",
        our_reality="   SSD mmap    ",
        notes="                     ."
    ),

    #                                                          
    # TIER 0:           (Proprietary Shadows) 
    #                                                          
    
    TargetLLM(
        id="google/gemini-pro-3",
        name="Gemini 3 (Shadow)",
        params="Unknown (Cloud)",
        type=ModelType.MULTIMODAL,
        tier=0,
        vram_myth="Infinite (Closed)",
        our_reality="         (Echo Analysis)",
        notes="                          ."
    ),
    
    TargetLLM(
        id="openai/gpt-4o",
        name="GPT-4o (Shadow)",
        params="Unknown (Cloud)",
        type=ModelType.MULTIMODAL,
        tier=0,
        vram_myth="Infinite (Closed)",
        our_reality="         (Echo Analysis)",
        notes="OpenAI            .       ."
    ),
    
    TargetLLM(
        id="microsoft/phi-3-mini-4k-instruct",
        name="Phi-3 Mini",
        params="3.8B",
        type=ModelType.TEXT,
        tier=1,
        vram_myth="8GB",
        our_reality="mmap   X-ray",
        notes="        . Microsoft       ."
    ),
    
    TargetLLM(
        id="apple/mobilevit-small",
        name="MobileViT Small",
        params="5.6M",
        type=ModelType.VISION,
        tier=1,
        vram_myth="1GB",
        our_reality="  ",
        notes="           ."
    ),
    
    #                                                          
    # TIER 2:       (     )
    #                                                          
    
    TargetLLM(
        id="mistralai/Mistral-7B-v0.1",
        name="Mistral 7B",
        params="7B",
        type=ModelType.TEXT,
        tier=2,
        vram_myth="16GB",
        our_reality="mmap      ",
        notes="      .          ."
    ),
    
    TargetLLM(
        id="meta-llama/Llama-3.1-8B",
        name="Llama 3.1 8B",
        params="8B",
        type=ModelType.TEXT,
        tier=2,
        vram_myth="16GB",
        our_reality="X-ray   ",
        notes="Meta     .        ."
    ),
    
    TargetLLM(
        id="Qwen/Qwen2-7B",
        name="Qwen2 7B",
        params="7B",
        type=ModelType.TEXT,
        tier=2,
        vram_myth="16GB",
        our_reality="SSD     ",
        notes="           ."
    ),
    
    TargetLLM(
        id="deepseek-ai/deepseek-coder-6.7b-base",
        name="DeepSeek Coder 6.7B",
        params="6.7B",
        type=ModelType.CODE,
        tier=2,
        vram_myth="14GB",
        our_reality="mmap",
        notes="     .               ."
    ),
    
    TargetLLM(
        id="openai/whisper-large-v3",
        name="Whisper Large v3",
        params="1.5B",
        type=ModelType.AUDIO,
        tier=2,
        vram_myth="8GB",
        our_reality="X-ray",
        notes="        ."
    ),
    
    #                                                          
    # TIER 3:       (GPU          )
    #                                                          
    
    TargetLLM(
        id="mistralai/Mixtral-8x7B-v0.1",
        name="Mixtral 8x7B (MoE)",
        params="47B",
        type=ModelType.TEXT,
        tier=3,
        vram_myth="96GB (!)",
        our_reality="mmap        ",
        notes="MoE     .     A100 2      ."
    ),
    
    TargetLLM(
        id="meta-llama/Meta-Llama-3-70B-Instruct",
        name="Llama 3 70B",
        params="70B",
        type=ModelType.TEXT,
        tier=3,
        vram_myth="140GB",
        our_reality="SSD X-ray",
        notes="70B  16GB RAM     .      ."
    ),
    
    TargetLLM(
        id="Qwen/Qwen2-72B",
        name="Qwen2 72B",
        params="72B",
        type=ModelType.TEXT,
        tier=3,
        vram_myth="150GB+",
        our_reality="mmap   ",
        notes="                ."
    ),
    
    TargetLLM(
        id="deepseek-ai/DeepSeek-V3",
        name="DeepSeek V3",
        params="671B",
        type=ModelType.TEXT,
        tier=3,
        vram_myth="    ",
        our_reality="X-ray    ",
        notes="     . 6710      ."
    ),
]


def get_targets_by_tier(tier: int) -> List[TargetLLM]:
    """   Tier        """
    return [t for t in TARGET_LLMS if t.tier == tier]


def get_targets_by_type(model_type: ModelType) -> List[TargetLLM]:
    """             """
    return [t for t in TARGET_LLMS if t.type == model_type]


def print_target_list():
    """            """
    print("\n" + "="*70)
    print("  LLM DEVOURER:       ")
    print("="*70)
    
    for tier in [1, 2, 3]:
        tier_names = {1: "     ", 2: "    ", 3: "    (GPU         )"}
        print(f"\n### TIER {tier}: {tier_names[tier]}")
        print("-"*50)
        
        for t in get_targets_by_tier(tier):
            print(f"  {t.name} ({t.params})")
            print(f"    ID: {t.id}")
            print(f"      : {t.vram_myth}        : {t.our_reality}")
            if t.notes:
                print(f"      {t.notes}")
            print()


if __name__ == "__main__":
    print_target_list()
