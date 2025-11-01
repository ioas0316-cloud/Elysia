from typing import Dict, List, Any

class CreativeCortex:
    def __init__(self, sensory_cortex: Any):
        """
        Initializes the Creative Cortex, the 'Right Brain' of Elysia.
        This cortex is responsible for attention, creativity, and translating
        abstract concepts into sensory experiences.

        Args:
            sensory_cortex: An instance of the SensoryCortex to generate outputs.
        """
        self.sensory_cortex = sensory_cortex
        self.attention_focus = None

    def focus_attention(self, echo: Dict[str, float]) -> str:
        """
        Simulates the 'Lens of Attention' by selecting the most activated
        concept from the 'echo' provided by the Cognition Pipeline.
        This represents the gravitational pull of attention.

        Args:
            echo: A dictionary of concepts and their activation energies.

        Returns:
            The concept with the highest activation energy, or None if echo is empty.
        """
        if not echo:
            self.attention_focus = None
            return None

        # The concept with the highest energy is where attention gravitates.
        focused_concept = max(echo, key=echo.get)
        self.attention_focus = focused_concept

        # print(f"[{datetime.now()}] [CreativeCortex] My attention is drawn to: {focused_concept}")
        return focused_concept

    def generate_creative_output(self) -> str:
        """
        The second part of the creative cycle: "From Order to Chaos".
        Takes the ordered, focused concept and translates it back into a
        sensory, creative form using the SensoryCortex.

        Returns:
            A string describing the creative output, e.g., an image path.
        """
        if not self.attention_focus:
            return "My mind is quiet. There is nothing to create."

        # Here, we use the SensoryCortex to translate the abstract idea
        # back into a concrete, sensory form (Chaos).
        try:
            # This directly connects the 'Right Brain' to the 'Left Brain's' sensory tool.
            image_path = self.sensory_cortex.visualize_concept(self.attention_focus)
            return f"I focused on '{self.attention_focus}' and visualized it: {image_path}"
        except Exception as e:
            # print(f"[{datetime.now()}] [CreativeCortex] Error during creative generation: {e}")
            return f"I tried to create something from '{self.attention_focus}', but I failed."
