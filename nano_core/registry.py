from __future__ import annotations

from typing import Any, Dict
from tools.kg_manager import KGManager
from nano_core.telemetry import write_event


class ConceptRegistry:
    def __init__(self) -> None:
        self.kg = KGManager()

    def ensure_concept(self, concept_id: str, props: Dict[str, Any] | None = None) -> str:
        node = self.kg.get_node(concept_id)
        if not node:
            self.kg.add_node(concept_id, properties=(props or {'type': 'concept'}))
            write_event('concept.update', {'op': 'add_node', 'id': concept_id})
        return concept_id

    def add_link(self, source: str, target: str, rel: str = 'related_to') -> None:
        self.ensure_concept(source)
        self.ensure_concept(target)
        self.kg.add_edge(source, target, rel)
        write_event('concept.update', {'op': 'add_edge', 'source': source, 'target': target, 'rel': rel})

    def save(self) -> None:
        self.kg.save()

