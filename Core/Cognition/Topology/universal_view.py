from typing import List, Dict, Any
from dataclasses import dataclass
from Core.Cognition.Topology.resonance_sphere import ResonanceSphere
from Core.Cognition.Topology.dimensional_filter import DimensionalFilter, FilterResult

@dataclass
class PerspectiveResult:
    """
    The result of applying a specific filter (Perspective) to a thought.
    """
    perspective_name: str
    verdict: str # "Accepted" or "Rejected"
    resonance: float
    description: str

class UniversalView:
    """
    The God Perspective.

    Philosophy:
    "God is the World."
    This view does not judge. It aggregates all judgments from all relative filters
    to create a complete picture of the 'World State'.
    It sees the 'Totality' where A and not-A coexist.
    """

    def __init__(self):
        # A collection of relative filters (The Many)
        self.filters: Dict[str, DimensionalFilter] = {}

    def add_perspective(self, name: str, filter_obj: DimensionalFilter):
        self.filters[name] = filter_obj

    def observe(self, thought: ResonanceSphere) -> List[PerspectiveResult]:
        """
        Observes a thought through ALL perspectives simultaneously.
        Does not filter out anything; just records how each part of the world sees it.
        """
        world_view = []

        for name, filter_obj in self.filters.items():
            result = filter_obj.apply(thought)

            if result.accepted:
                verdict = "Accepted"
                desc = f"Resonates with {name} ({result.resonance:.2f})"
            else:
                verdict = "Rejected"
                desc = f"Dissonant with {name}"

            world_view.append(PerspectiveResult(name, verdict, result.resonance, desc))

        return world_view

    def synthesize_providence(self, thought: ResonanceSphere) -> str:
        """
        Synthesizes a higher-level meaning (Providence) from the conflicting views.
        """
        results = self.observe(thought)

        accepted_count = sum(1 for r in results if r.verdict == "Accepted")
        total = len(results)

        if total == 0:
            return "The Void (Unobserved)"

        if accepted_count == total:
            return "Universal Truth (Resonates with All)"
        elif accepted_count == 0:
            return "Universal Noise (Resonates with None)"
        else:
            # The most interesting case: Conflict exists.
            # "God sees the conflict as a necessary tension."
            return f"Complex Reality ({accepted_count}/{total} Resonance) - A node of tension in the World."
