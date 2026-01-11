"""
Constraint Solver: The Logic of Assembly
========================================

"Logic is not probability. Logic is fit."

This module implements a CAD-style constraint solver for thoughts.
Instead of asking "What is likely?", we ask "Does it fit?".

Concepts:
- **Part**: A semantic unit (e.g., "Apple", "Red").
- **Port**: A connection point on a part (e.g., "Color Slot", "Subject Slot").
- **Constraint**: A rule enforcing how ports connect.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger("ConstraintSolver")

@dataclass
class Port:
    name: str
    type: str  # e.g., "Adjective", "Noun", "Cause"
    value: Optional[str] = None

@dataclass
class Part:
    name: str
    ports: Dict[str, Port] = field(default_factory=dict)

    def add_port(self, name: str, p_type: str, value: str = None):
        self.ports[name] = Port(name, p_type, value)

class ConstraintSolver:
    def __init__(self):
        pass

    def check_fit(self, part_a: Part, port_a_name: str, part_b: Part, port_b_name: str, constraint_type: str) -> bool:
        """
        Checks if Part A and Part B can be assembled via the specified ports.
        """
        if port_a_name not in part_a.ports or port_b_name not in part_b.ports:
            logger.warning(f"⚠️ Ports not found: {port_a_name} / {port_b_name}")
            return False

        port_a = part_a.ports[port_a_name]
        port_b = part_b.ports[port_b_name]

        # 1. Type Check (The Bolt must fit the Nut)
        if constraint_type == "Coincident":
            # Must be same type or compatible
            if port_a.type != port_b.type:
                logger.info(f"❌ Type Mismatch: {port_a.type} != {port_b.type}")
                return False

        # 2. Value Check (Interference)
        if constraint_type == "Concentric":
            # They must share a core value/axis
            if port_a.value and port_b.value and port_a.value != port_b.value:
                logger.info(f"❌ Interference: {port_a.value} clashes with {port_b.value}")
                return False

        logger.info(f"✅ Constraint Satisfied: {part_a.name}.{port_a.name} <-> {part_b.name}.{port_b.name}")
        return True

    def assemble(self, parts: List[Part]) -> bool:
        """
        Mock assembly check.
        """
        # (Placeholder for complex solving)
        return True

if __name__ == "__main__":
    solver = ConstraintSolver()

    # Define Parts
    apple = Part("Apple")
    apple.add_port("ColorProperty", "Color", "Red")

    redness = Part("Redness")
    redness.add_port("ColorValue", "Color", "Red")

    blueness = Part("Blueness")
    blueness.add_port("ColorValue", "Color", "Blue")

    print("--- Testing Assembly ---")

    # Test 1: Apple + Redness (Should Fit)
    print("\nAttempting: Apple + Redness (Concentric)")
    result1 = solver.check_fit(apple, "ColorProperty", redness, "ColorValue", "Concentric")
    print(f"Result: {result1}")

    # Test 2: Apple + Blueness (Should Fail/Interfere)
    print("\nAttempting: Apple + Blueness (Concentric)")
    result2 = solver.check_fit(apple, "ColorProperty", blueness, "ColorValue", "Concentric")
    print(f"Result: {result2}")
