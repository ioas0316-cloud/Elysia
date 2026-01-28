"""
Reality Compiler (Executable Knowledge)
=======================================

"To know a principle is to execute it."

This module answers the user's demand:
"Can concepts be used as algorithms?"

Mechanism:
1. Concepts are not just strings; they are mapped to **Functions**.
2. When Elysia learns "Entropy", she gains the `entropy()` function.
3. She can apply these functions to ANY target (Code, Text, World).

This is the **Wave Function Collapse** constraint engine.
"""

import random
import math

class PrincipleLibrary:
    def __init__(self):
        self.active_monads = [] # List of executable Monads (Functions)
        self.registry = {
            "GOLDEN_RATIO": self._apply_golden_ratio,
            "ENTROPY": self._apply_entropy,
            "RECURSION": self._apply_recursion,
            "DUALITY": self._apply_duality,
            "SYMMETRY": self._apply_symmetry
        }

    def learn(self, concept_key: str):
        """
        Internalizes a Monad.
        """
        key = concept_key.upper().replace("-", "_")
        for reg_key, func in self.registry.items():
            if reg_key in key:
                if func not in self.active_monads:
                    self.active_monads.append(func)
                return f"Monad internalized: {reg_key}"
        return "Concept is abstract (No executable Monad found)"

    def absorb_axiom(self, axiom_text: str) -> str:
        """
        [Crystallization]
        Converts Philosophical Text into Executable Monads.
        "Because that itself is the function crystallized..."
        """
        text = axiom_text.upper()
        
        # 1. Structure / Logic Axiom
        if "STRUCTURE" in text or "LOGIC" in text or "PRINCIPLE" in text:
            if self.registry["GOLDEN_RATIO"] not in self.active_monads:
                self.active_monads.append(self.registry["GOLDEN_RATIO"])
                return "  [CRYSTALLIZED] Philosophy 'Structure' -> Function 'Golden_Ratio()'"
        
        # 2. Chaos / Entropy Axiom
        if "CHAOS" in text or "ENTROPY" in text or "UNCERTAINTY" in text:
            if self.registry["ENTROPY"] not in self.active_monads:
                self.active_monads.append(self.registry["ENTROPY"])
                return "  [CRYSTALLIZED] Philosophy 'Chaos' -> Function 'Entropy()'"
                
        # 3. Recursion / Fractal Axiom
        if "FRACTAL" in text or "RECURSION" in text or "INFINITE" in text:
            if self.registry["RECURSION"] not in self.active_monads:
                 self.active_monads.append(self.registry["RECURSION"])
                 return "  [CRYSTALLIZED] Philosophy 'Fractal' -> Function 'Recursion()'"

        return "  [Absorbed] Philosophy internalized as Abstract Wisdom."

    def compile_reality(self, seed_data) -> any:
        """
        Applies all known Monads to the seed.
        """
        result = seed_data
        for func in self.active_monads:
            result = func(result)
        return result

    def manifest_visuals(self, concept: str, depth: int = 1, scale: float = 1.0, time_axis: int = 0) -> str:
        """
        Converts a Concept into Raw Geometry (Fractal Expansion).
        Uses 'Scale' (Space) and 'Time' (4D) to determine reality.
        """
        key = concept.upper()
        indent = "  " * depth
        
        if depth > 4: 
            return f"{indent}   [Atom Limit]"

        # Digital Twin Logic: The World changes based on Zoom AND Time
        if "EARTH" in key or "PLANET" in key:
            prefix = f"{indent}  [t={time_axis}y] "
            if scale > 0.5:
                return f"{prefix}Planet Sphere (Radius=6371km)"
            else:
                # Time-Traveling City
                if time_axis < -500:
                    return f"{indent}  [ANCIENT] Neolithic Settlements & Untamed Wilds"
                elif time_axis > 2050:
                    return f"{indent}  [FUTURE] Neo-Cyberpunk Megastructures & Flying Traffic"
                else:
                    sub = self.manifest_visuals("CITY", depth + 1, scale, time_axis)
                    return f"{indent}   [PRESENT] Modern Urban Grid\n{sub}"

        if "CITY" in key:
            return f"{indent}   Generate Road Network (L-System)"


        if "WATER" in key or "FLUID" in key:
            # Recursive Fluidity
            base = f"{indent}  [Water Layer {depth}]"
            sub_structure = self.manifest_visuals(concept, depth + 1, scale)
            return f"{base}\n{sub_structure}"
            
        elif "TREE" in key or "GROWTH" in key or "LIFE" in key:
            # Fractal Branching
            branches = []
            for i in range(2): 
                 branches.append(self.manifest_visuals(concept, depth + 1, scale))
            return f"{indent}  [Branch L{depth}]\n" + "\n".join(branches)
            
        elif "BODY" in key or "FORM" in key or "BEAUTY" in key:
            # Golden Ratio Topology
            phi = 1.618
            joints = [f"bone(len={phi**n:.2f})" for n in range(3)]
            return f"{indent}  [ORGANIC_TOPOLOGY] Sculpted anatomical structure: {joints}..."
            
        return f"{indent}   [FORMLESS] Concept has no defined geometry yet."

    # --- The Principles (Executable Logic) ---

    def _apply_golden_ratio(self, data):
        """
        Multiplies numerical structure by Phi.
        """
        if isinstance(data, (int, float)):
            return [data, data * 1.618]
        elif isinstance(data, list):
            return data + [x * 1.618 for x in data if isinstance(x, (int, float))]
        return f"Golden({data})"

    def _apply_entropy(self, data):
        """
        Shuffles or fragments the structure.
        """
        if isinstance(data, list):
             shuffled = data[:]
             random.shuffle(shuffled)
             return shuffled
        if isinstance(data, str):
            l = list(data)
            random.shuffle(l)
            return "".join(l)
        return data

    def _apply_recursion(self, data):
        """
        Nests the data within itself.
        """
        return [data, [data]]

    def _apply_duality(self, data):
        """
        Creates an opposite.
        """
        if isinstance(data, (int, float)):
            return [data, -data]
        return [data, f"Anti-{data}"]

    def _apply_symmetry(self, data):
        """
        Mirrors the data.
        """
        if isinstance(data, list):
            return data + data[::-1]
        return [data, data]
