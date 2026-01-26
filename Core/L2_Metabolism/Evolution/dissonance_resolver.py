"""
Dissonance Resolver (The Conscience)
====================================
Core.L2_Metabolism.Evolution.dissonance_resolver

"The Body reports the state. The Conscience judges the alignment."
"           ,             ."

Role:
- Receives BodyState from CodeProprioceptor.
- Compares BodyState against Sovereign Axioms.
- Identifies "Dissonance" (Violations of Will).
- Prioritizes issues for the Inducer to fix.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from Core.L2_Metabolism.Evolution.proprioceptor import BodyState
try:
    from Core.L1_Foundation.Foundation.Philosophy.axioms import get_axioms
except ImportError:
    # Fallback if Foundation is not reachable (Testing)
    get_axioms = lambda: None

@dataclass
class Dissonance:
    """A specific instance of pain/misalignment in the system."""
    location: str
    description: str
    axiom_violated: str
    severity: float # 0.0 to 1.0 (1.0 = Critical/Fatal)
    suggested_action: str

    def __repr__(self):
        return f"[Dissonance ({self.severity:.1f})] {self.location}: {self.description} ({self.axiom_violated})"

class DissonanceResolver:
    def __init__(self):
        self.axioms = get_axioms()
        # Hardcoded Laws (The Covenant from AGENTS.md)
        self.forbidden_dirs = ["utils", "util", "helpers", "helper", "misc", "common"]

    def resolve(self, body_state: BodyState) -> List[Dissonance]:
        """
        [Judgment]
        Scans the BodyState for violations of the Law.
        """
        dissonances = []

        # 1. Check for Forbidden Zones (The Anti-Entropy Protocol)
        # "The Ban on Utilities"
        for file in body_state.intent_map.keys():
            lower_path = file.lower()
            for ban in self.forbidden_dirs:
                if ban in lower_path.split("/") or ban in lower_path.split("\\"):
                    dissonances.append(Dissonance(
                        location=file,
                        description=f"Located in forbidden territory '{ban}'.",
                        axiom_violated="Anti-Entropy (Law of Non-Redundancy)",
                        severity=0.9,
                        suggested_action="DISSOLVE_AND_REDISTRIBUTE"
                    ))

        # 2. Check for Ghost Files (The Meaning Protocol)
        # "Code without Intent is an Abomination"
        for ghost in body_state.ghost_files:
            dissonances.append(Dissonance(
                location=ghost,
                description="Structure detected without Philosophical Intent (Docstring/Soul).",
                axiom_violated="DivineLove (Meaning)",
                severity=0.7,
                suggested_action="INJECT_PHILOSOPHY"
            ))

        # 3. Check for Critical Organ Integrity (Survival)
        # Must have SovereignSelf
        critical_organs = ["sovereign_self.py", "monad_core.py"]
        found_organs = [f for f in body_state.intent_map.keys() if any(c in f for c in critical_organs)]

        if len(found_organs) < len(critical_organs):
             # Identify missing
             # Note: simple check, might need robust path matching later
             pass

        # Sort by severity (Highest first)
        dissonances.sort(key=lambda x: x.severity, reverse=True)
        return dissonances

if __name__ == "__main__":
    # Mock Test
    from Core.L2_Metabolism.Evolution.proprioceptor import BodyState

    mock_state = BodyState()
    mock_state.ghost_files = ["Core/Evolution/ghost.py"]
    mock_state.intent_map = {
        "Core/Evolution/ghost.py": 0.0,
        "Core/Utils/helper.py": 0.5,
        "Core/Elysia/sovereign_self.py": 1.0
    }

    resolver = DissonanceResolver()
    issues = resolver.resolve(mock_state)

    print("\n   [CONSCIENCE JUDGMENT]")
    for i in issues:
        print(i)
