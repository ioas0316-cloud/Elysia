"""
Layer 3 Knowledge Bus Slot Module
=================================
Implements the 'Mainboard Knowledge Bus Slot' interface in Layer 3 Macro Causal Architecture.
Allows external knowledge sources (e.g., NotebookLM source chunks, physical principles,
human cognitive mappings) to be 'slotted in' directly into ELYSIA's continuous wave manifold.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Any, Optional


@dataclass
class KnowledgeChunk:
    """Represents a chunk of external knowledge slotted into the Knowledge Bus."""
    chunk_id: str
    title: str
    source_type: str  # e.g., 'notebook_lm', 'physical_principle', 'cognitive_map'
    content_text: str
    semantic_vector: np.ndarray  # Vector representation in phase space
    constraint_order: float = 1.0  # Order weight / stiffness of the constraint
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeBusSlot:
    """
    Layer 3 Mainboard Knowledge Bus Slot.
    Acts as the mainboard PCIe/NVMe slot for macro knowledge sources.
    Transforms discrete textual/conceptual knowledge chunks into continuous
    causal phase fields and constraint matrices for ELYSIA layers.
    """

    def __init__(self, slot_id: str = "SLOT_L3_KNOWLEDGE_BUS", vector_dim: int = 5):
        self.slot_id = slot_id
        self.vector_dim = vector_dim
        self.connected_chunks: Dict[str, KnowledgeChunk] = {}
        self.bus_impedance: float = 0.05
        self.aggregate_knowledge_field: np.ndarray = np.zeros(self.vector_dim)

    def slot_in(self, chunk: KnowledgeChunk) -> bool:
        """Slots a new KnowledgeChunk into the knowledge bus."""
        if len(chunk.semantic_vector) != self.vector_dim:
            # Pad or truncate to match vector_dim
            if len(chunk.semantic_vector) < self.vector_dim:
                padded = np.zeros(self.vector_dim)
                padded[:len(chunk.semantic_vector)] = chunk.semantic_vector
                chunk.semantic_vector = padded
            else:
                chunk.semantic_vector = chunk.semantic_vector[:self.vector_dim]

        self.connected_chunks[chunk.chunk_id] = chunk
        self._recalculate_aggregate_field()
        return True

    def slot_out(self, chunk_id: str) -> Optional[KnowledgeChunk]:
        """Unslots a KnowledgeChunk from the knowledge bus."""
        removed = self.connected_chunks.pop(chunk_id, None)
        if removed:
            self._recalculate_aggregate_field()
        return removed

    def _recalculate_aggregate_field(self) -> None:
        """Calculates the weighted aggregate phase field from all slotted chunks."""
        if not self.connected_chunks:
            self.aggregate_knowledge_field = np.zeros(self.vector_dim)
            return

        total_weight = sum(c.constraint_order for c in self.connected_chunks.values())
        if total_weight == 0:
            total_weight = 1.0

        field_sum = np.zeros(self.vector_dim)
        for chunk in self.connected_chunks.values():
            field_sum += chunk.semantic_vector * chunk.constraint_order

        self.aggregate_knowledge_field = field_sum / total_weight

    def generate_phase_constraint_spectrum(self, sample_points: int = 64) -> np.ndarray:
        """
        Generates a continuous phase spectrum from the aggregate knowledge field
        to modulate Layer 2 resonant attractors and Layer 1 prismatic filters.
        """
        t = np.linspace(0, 2 * np.pi, sample_points)
        spectrum = np.zeros(sample_points)

        for i, val in enumerate(self.aggregate_knowledge_field):
            freq = (i + 1) * 2.0
            phase = val * np.pi
            spectrum += np.sin(freq * t + phase) * (1.0 / (i + 1))

        # Normalize
        norm = np.max(np.abs(spectrum))
        if norm > 0:
            spectrum = spectrum / norm
        return spectrum

    def get_bus_status(self) -> Dict[str, Any]:
        """Returns diagnostic status of the Knowledge Bus Slot."""
        return {
            "slot_id": self.slot_id,
            "chunk_count": len(self.connected_chunks),
            "chunks": [
                {
                    "id": c.chunk_id,
                    "title": c.title,
                    "type": c.source_type,
                    "order": c.constraint_order
                }
                for c in self.connected_chunks.values()
            ],
            "aggregate_field": self.aggregate_knowledge_field.tolist()
        }
