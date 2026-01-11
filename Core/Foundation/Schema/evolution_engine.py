"""
Evolution Engine: The Growth of the Mind
========================================

"We do not just accumulate data; we ascend in dimensionality."

This module governs the **Metamorphosis** of thoughts.
It monitors the complexity of a concept and "Promotes" it to the next layer
when it reaches critical density.

Mechanism:
1.  **Point -> Plane**: When a Point acquires 3+ stable connections.
    -   "Mom" + "Food" + "Warmth" -> "Care" (Plane).
2.  **Plane -> Solid**: When a Plane resolves 2+ contradictions or intersects 2+ other Planes.
    -   "Care" (Plane) + "Independence" (Plane) -> "Parenting" (Solid Principle).
"""

import logging
from typing import List, Optional
from Core.Foundation.Schema.layers import ConceptLayer, PointLayer, PlaneLayer, SolidLayer

logger = logging.getLogger("EvolutionEngine")

class EvolutionEngine:
    def __init__(self):
        pass

    def check_evolution(self, concept: ConceptLayer, connections: List[ConceptLayer]) -> Optional[ConceptLayer]:
        """
        Evaluates if a concept is ready to ascend.
        Returns the new Higher-Order Concept, or None.
        """
        if isinstance(concept, PointLayer):
            return self._evolve_point(concept, connections)
        elif isinstance(concept, PlaneLayer):
            return self._evolve_plane(concept, connections)

        return None

    def _evolve_point(self, point: PointLayer, neighbors: List[ConceptLayer]) -> Optional[PlaneLayer]:
        """
        A Point becomes a Plane when it has enough context (neighbors).
        """
        # Threshold: 3 connections
        if len(neighbors) >= 3:
            logger.info(f"🌱 Evolution: Point '{point.name}' is expanding into a Plane.")

            # Create Plane
            plane = PlaneLayer(
                name=f"Context of {point.name}",
                description=f"A web of relations around {point.name}"
            )

            # Absorb neighbors as context points
            # (In a real graph, we'd link them; here we just store refs)
            for n in neighbors:
                if isinstance(n, PointLayer):
                    plane.points.append(n)
                    plane.relations.append("Connected")

            return plane
        return None

    def _evolve_plane(self, plane: PlaneLayer, neighbors: List[ConceptLayer]) -> Optional[SolidLayer]:
        """
        A Plane becomes a Solid when it intersects with other Planes (Cross-Contextual).
        """
        other_planes = [n for n in neighbors if isinstance(n, PlaneLayer)]

        # Threshold: 2 intersecting planes
        if len(other_planes) >= 2:
            logger.info(f"💎 Evolution: Plane '{plane.name}' is crystallizing into a Solid.")

            # Create Solid
            solid = SolidLayer(
                name=f"Principle of {plane.name}",
                description="A systemic law governing multiple contexts."
            )

            # Absorb planes
            solid.planes.append(plane)
            solid.planes.extend(other_planes)
            solid.laws.append("Emergent Law")

            return solid
        return None
