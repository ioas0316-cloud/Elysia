from typing import Any, List, Optional
from Core.Foundation.core.thought import Thought

class InsightSynthesizer:
    """
    Synthesizes multiple streams of thought (Logic, Value, Perspective)
    into a coherent insight.
    """
    def __init__(self):
        pass

    def synthesize(
        self,
        message: str,
        logic_result: Optional[List[Thought]],
        value_decision: Any,
        perspective: Any,
        context: Any
    ) -> Any: # Returns Insight object or str

        # Simple synthesis logic for now
        content = "I have considered your words."

        if value_decision:
             # Extract content from decision if available
             if hasattr(value_decision, 'content'):
                 content = value_decision.content
             else:
                 content = str(value_decision)

        # Create a Thought object as the result
        insight = Thought(
            content=content,
            source="synthesis",
            confidence=0.9
        )

        return insight
