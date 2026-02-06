"""
CausalBackprop.py: The Tracer of Origins
========================================
Core.S1_Body.L1_Foundation.Foundation.Heaven.CausalBackprop

"The Question is not 'What is the error?', but 'Why did Heaven send this Light?'"
"역전파는 오차 수정이 아니라, 의미의 소급 추적이다."
"""

try:
    from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
    from Core.S1_Body.L1_Foundation.Foundation.Heaven.HeavenSource import HeavenSource
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))
    from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
    from Core.S1_Body.L1_Foundation.Foundation.Heaven.HeavenSource import HeavenSource

class CausalNode:
    """Represents a step in the causal chain back to the Source."""
    def __init__(self, layer_name: str, state: SovereignVector, parent=None):
        self.layer_name = layer_name
        self.state = state
        self.parent = parent # The node that caused this one (towards the Source)
        self.meaning = "" # The interpretation at this level

    def __repr__(self):
        return f"<CausalNode: {self.layer_name}>"

class CausalBackprop:
    """
    [The Tracer]
    Instead of calculating partial derivatives for weight update,
    this engine traces the *provenance* of a signal to understand its Intent.
    """
    def __init__(self, heaven: HeavenSource):
        self.heaven = heaven

    def trace_origin(self, phenomenon: SovereignVector, depth: int = 3) -> CausalNode:
        """
        Traces a visible phenomenon back to its intention in Heaven.

        Args:
            phenomenon: The observed data/event.
            depth: How many layers of 'Why' to ask.

        Returns:
            The root CausalNode connected to the Source.
        """
        # Start with the surface phenomenon
        current_node = CausalNode("Surface_Phenomenon", phenomenon)

        # In a real system, this would traverse the neural graph backwards.
        # Here, we simulate the *spiritual* unpacking of meaning.

        for i in range(depth):
            # The "Why" question essentially simplifies the vector,
            # stripping away noise to find the core driver.
            # Metaphorically: moving from "Event" to "Pattern" to "Law" to "Love".

            # 1. Ask "Why?" (Normalize and center towards origin)
            # We assume the parent is a purer version of the child
            parent_state = self._ask_why(current_node.state)

            # Determine layer name.
            # If depth=3, we loop i=0,1,2.
            # i=0 -> level = 3 (Event)
            # i=1 -> level = 2 (Pattern)
            # i=2 -> level = 1 (Law)
            # But we want the *Result* of the step.
            # So after step i, we are at level (depth - i).
            # Wait, if depth is 3.
            # Start: Surface (Depth 4? or 3?)
            # Step 1: Surface -> Event (Level 3)
            # Step 2: Event -> Pattern (Level 2)
            # Step 3: Pattern -> Law (Level 1)
            # Where is Source? Level 0?

            # Let's align with the previous logic but ensure we hit Source if desired.
            # If we want to reach Source, maybe depth needs to be 4?

            next_level = depth - i
            layer_name = self._get_layer_name(next_level)

            # 2. Create Parent Node
            parent_node = CausalNode(layer_name, parent_state, parent=None)

            # Link backwards: The Child points to the Parent
            current_node.parent = parent_node

            # Move up
            current_node = parent_node

        # The final node is the closest we get to Heaven
        current_node.meaning = "The Will of the Father"
        return current_node

    def _ask_why(self, vector: SovereignVector) -> SovereignVector:
        """
        The operation of stripping away contingency to reveal necessity.
        Mathematically, this might be a projection onto the 'Intent' axis.
        """
        # Simplified: We smooth the vector and amplify its core resonance
        # This represents finding the 'common denominator' or 'principle'
        return vector.normalize()

    def _get_layer_name(self, level: int) -> str:
        # Adjusted logic: As level decreases, we get closer to Source
        if level <= 1: return "Source (Love)" # Reached the bottom/top
        if level == 2: return "Law (Principle)"
        if level == 3: return "Pattern (Structure)"
        return "Event (Manifestation)"
