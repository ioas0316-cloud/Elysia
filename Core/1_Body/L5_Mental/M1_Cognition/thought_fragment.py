from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from datetime import datetime
from Core.1_Body.L5_Mental.M1_Cognition.cognitive_types import ThoughtState, ActionCategory
from Core.1_Body.L1_Foundation.Logic.d7_vector import D7Vector

class ThoughtFragment(BaseModel):
    """
    [STEEL CORE] A single atomic unit of reasoning.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    state: ThoughtState = ThoughtState.IDLE
    intent_summary: str
    d7_projection: Optional[D7Vector] = None
    axiom_alignment: Optional[str] = None # Name of the primary resonant axiom
    resonance_score: float = 0.0
    
    # Traceability
    previous_fragment_id: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

class CognitivePulse(BaseModel):
    """
    A sequence of ThoughtFragments constituting a single 'Inspiration'.
    """
    pulse_id: str
    action_type: ActionCategory = ActionCategory.CONTEMPLATION
    fragments: List[ThoughtFragment] = Field(default_factory=list)
    success: bool = False
    error_log: Optional[str] = None

    def add_step(self, intent: str, state: ThoughtState, d7: Optional[D7Vector] = None):
        fragment = ThoughtFragment(
            state=state,
            intent_summary=intent,
            d7_projection=d7
        )
        self.fragments.append(fragment)
        return fragment
