from dataclasses import dataclass, field
from typing import Any, Dict


from typing import Any, Dict, Optional

# HACK: Use 'Any' to avoid circular import with Thought
# from Project_Sophia.core.thought import Thought

@dataclass
class ConversationContext:
    """Holds the state of a conversation."""
    pending_hypothesis: Any = None
    guiding_intention: Optional['Thought'] = None # Stores the current intention from Logos Engine
    # Future additions:
    # emotional_state: EmotionalState = field(default_factory=EmotionalState)
    # conversation_history: List[str] = field(default_factory=list)
    # custom_data: Dict[str, Any] = field(default_factory=dict)
