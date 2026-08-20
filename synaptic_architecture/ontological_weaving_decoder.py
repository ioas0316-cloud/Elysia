"""
Ontological Weaving Decoder: Multi-contextual Binding and Inverse Causal Extraction.

Rejects numeric reductionism (flattening entities to scalar floating vectors).
Decodes the multi-contextual spectrum from Cognitive Lenses, weaving together
the entity's true ontological reality ("An apple is an apple") across spatial,
organic, relational, and symbolic dimensions without loss of context.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from core.lens.cognitive_lens_engine import ContextualDimension, RefractedObservation


@dataclass
class OntologicalBinding:
    entity_name: str
    primary_ontology: str
    contextual_refractions: Dict[str, Dict[str, Any]]
    woven_causal_invariants: List[str]
    is_reduced_to_scalar_vector: bool = False  # Always False!


class OntologicalWeavingDecoder:
    """Decodes multi-lens refractions into a woven ontological reality."""

    def decode_weaving(self, entity_name: str, spectrum: Dict[ContextualDimension, RefractedObservation]) -> OntologicalBinding:
        contextual_refractions = {}
        all_invariants = set()

        for dim, observation in spectrum.items():
            contextual_refractions[dim.value] = {
                "refraction_angle": observation.refraction_angle,
                "phase_tension": observation.phase_tension,
                "bound_weaving": observation.bound_weaving
            }
            all_invariants.update(observation.causal_invariants)

        # Determine primary ontology based on maximum phase tension or dominant context
        max_tension_dim = max(spectrum.keys(), key=lambda d: spectrum[d].phase_tension)
        primary_ontology = f"{entity_name}_in_{max_tension_dim.value}"

        return OntologicalBinding(
            entity_name=entity_name,
            primary_ontology=primary_ontology,
            contextual_refractions=contextual_refractions,
            woven_causal_invariants=sorted(list(all_invariants)),
            is_reduced_to_scalar_vector=False
        )
