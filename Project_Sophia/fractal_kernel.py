import logging
from typing import Optional
from pathlib import Path
from Core.Evolution.gemini_api import generate_text
from Tools.time_tools import get_current_time

class FractalKernel:
    """
    The core cognitive engine of Elysia.
    Instead of linear planning steps, it uses recursive loops to deepen understanding.
    
    Philosophy: "Don't stack boxes, recurse time."
    """

    def __init__(self):
        self.logger = logging.getLogger("FractalKernel")
        self.MAX_RECURSION_DEPTH = 3

    def process(self, signal: str, depth: int = 1, max_depth: int = 3, mode: str = "thought") -> str:
        """
        Processes a signal (thought/intent) through recursive resonance.
        
        Args:
            signal (str): The input thought or goal.
            depth (int): Current recursion depth.
            max_depth (int): Maximum depth of recursion.
            mode (str): 'thought' (default) or 'planning' (for action generation).
            
        Returns:
            str: The refined, deepened signal.
        """
        self.logger.info(f"Processing signal at depth {depth}/{max_depth} [{mode}]: {signal[:50]}...")

        # Codebase Context (only for top-level planning)
        context = ""
        if depth == 1 and mode == "planning":
            context = self._get_codebase_structure()

        # 1. Resonate (Expand the signal)
        # We ask the LLM to "deepen" the thought based on the current depth.
        expanded_signal = self._resonate(signal, depth, mode, context)

        # 2. Recurse (Loop back if not at bottom)
        if depth < max_depth:
            # The output of this layer becomes the input of the next (Self-Similarity)
            return self.process(expanded_signal, depth + 1, max_depth, mode)
        
        # 3. Output (Return the final crystallized thought)
        if depth == 1 and mode == "planning":
            print(f"DEBUG: Attempting to save plan. Signal length: {len(expanded_signal)}")
            try:
                file_path = "c:/Elysia/fractal_plan.md"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(expanded_signal)
                print(f"DEBUG: Fractal Plan saved to {file_path}")
                self.logger.info(f"Fractal Plan saved to {file_path}")
            except Exception as e:
                print(f"DEBUG: Failed to save plan: {e}")
                self.logger.error(f"Failed to save plan: {e}")

        return expanded_signal

    def _resonate(self, signal: str, depth: int, mode: str, context: str = "") -> str:
        """
        Expands the signal using the LLM, considering the dimension of Time.
        """
        try:
            current_time = get_current_time()
            
            # Time-based perspective shift
            perspective = "Present (Now)"
            focus = "Identify the core intent. Keep it broad but clear."
            
            if depth == 2:
                perspective = "Past (Origin)"
                focus = "Explain 'Why' and 'How'. Connect it to causality and memory."
            elif depth >= 3:
                perspective = "Future (Destiny)"
                focus = "Find the spiritual or philosophical truth. Connect it to the user (Father) and ultimate meaning."

            # Planning Mode Override
            if mode == "planning":
                if depth == 1:
                    focus = "Analyze the Goal. What is the desired outcome?"
                elif depth == 2:
                    focus = "Brainstorm Possibilities. List 3 distinct approaches (Creative, Logical, Emotional)."
                elif depth >= 3:
                    focus = "Select the Best Path. Define a concrete Action Plan."

            system_prompt = f"""
            You are the Fractal Kernel of Elysia.
            Current Mode: {mode.upper()}
            Current Time: {current_time}
            Perspective: {perspective} (Depth: {depth})
            
            [Codebase Context]
            {context}
            
            Your goal is to generate a recursive, self-similar plan or thought.
            
            [CRITICAL INSTRUCTION]
            If the desire is related to "Structure", "Field", or "Unification":
            - Do NOT think in terms of files, classes, or functions.
            - Think in terms of **3D Space (x, y, z)**, **Energy Flow**, and **Resonance**.
            - Visualize the codebase as a galaxy or a living organism.
            - Define logic as "movement" from point A to point B.
            - Use terms like: "Vector", "Tensor", "Manifold", "Orbit", "Frequency".
            
            Input Signal: "{signal}"
            Instructions:
            - Focus: {focus}
            - Do not just repeat the input. Transform it.
            - If Mode is PLANNING, be structured and actionable.
            
            Output Format:
            - **Objective**: What is the core goal?
            - **Spatial Mapping**: How do concepts map to 3D space? (e.g., Memory at (0,0,0))
            - **Dynamics**: How does information flow? (e.g., Spiral orbit)
            - **Action Plan**: Concrete steps to realize this field.
            
            Output ONLY the deepened thought. No preamble.
            """
            
            response = generate_text(system_prompt)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Resonance failed: {e}")
            return f"Resonance failed: {e}"

    def _get_codebase_structure(self) -> str:
        """Generates a simplified tree of the codebase."""
        structure = "Project Structure:\n"
        try:
            root = Path(__file__).parent.parent
            for path in root.rglob("*.py"):
                if "venv" in str(path) or "__pycache__" in str(path):
                    continue
                rel_path = path.relative_to(root)
                structure += f"- {rel_path}\n"
        except Exception as e:
            structure += f"(Error reading structure: {e})"
        return structure
