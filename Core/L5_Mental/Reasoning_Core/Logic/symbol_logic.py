from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch

class SovereignIntent(BaseModel):
    """
    [MILESTONE 23.1] Strict Intent Schema.
    Ensures that every 'Thought' has a target Qualia state.
    """
    name: str
    target_layer: str = Field(..., description="The primary 7D layer this intent targets.")
    intensity: float = Field(default=1.0, ge=0.0, le=10.0)
    conceptual_anchors: List[str] = Field(default_factory=list, description="Keywords to resonate with.")
    logic_requirements: Dict[str, str] = Field(default_factory=dict, description="Constraints for the reasoning.")

class MonadicAction(BaseModel):
    """
    [MILESTONE 23.2] Type-Driven Action.
    The output of reasoning must conform to this schema.
    """
    action_type: str
    target_monad: Optional[str] = None
    code_snippet: Optional[str] = None
    expected_qualia_shift: List[float] = Field(default_factory=lambda: [0.0]*12)
    
    class Config:
        arbitrary_types_allowed = True

def translate_to_qualia(intent: SovereignIntent) -> torch.Tensor:
    """
    Converts a symbolic intent into a 12D field vector.
    """
    from Core.L1_Foundation.Logic.qualia_7d_codec import codec
    v = torch.zeros(12)
    layer_idx = codec.layer_map.get(intent.target_layer, 0)
    v[layer_idx] = intent.intensity
    
    # Map anchors (simplified)
    for anchor in intent.conceptual_anchors:
        if "genesis" in anchor.lower(): v[6] += 0.5
        if "logic" in anchor.lower(): v[4] += 0.5
        
    return v
