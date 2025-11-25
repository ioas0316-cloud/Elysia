from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# --- Fractal Soul Dependencies ---
from Project_Sophia.core.self_fractal import SelfFractalCell
from Project_Sophia.core.essence_mapper import EssenceMapper
from Project_Sophia.core.tensor_wave import Tensor3D, SoulTensor, FrequencyWave
try:
    from Project_Elysia.chemo_sense import map_scent_signature
except Exception:
    map_scent_signature = None

class Cell:
    """
    Represents a single, living conceptual cell in Elysia's cellular world.
    Most mutable stats are handled by the World for performance, so the Cell
    object focuses on identity, memory, and connective context.

    [Fractal Soul Upgrade]
    Now equipped with a 'Soul' (SelfFractalCell) that processes emotions and
    concepts as standing waves (frequencies) rather than just scalar stats.
    """

    def __init__(self, concept_id: str, dna: Dict[str, Any], initial_properties: Optional[Dict[str, Any]] = None):
        self.id = concept_id
        self.membrane: Dict[str, Any] = {}
        self.nucleus: Dict[str, Any] = {"dna": dna}
        self.organelles: Dict[str, Any] = initial_properties.copy() if initial_properties else {}
        self.connections: List[Dict[str, Any]] = []
        self.element_type = self.organelles.get("element_type", "unknown")
        self.scent_signature = None

        self.is_alive = True
        self.age = 0
        self.health = 100.0
        self._memory_events: List[Dict[str, Any]] = []

        # --- Soul Initialization ---
        self.soul = SelfFractalCell(size=50) # Give each cell a 50x50 soul grid
        self._initialize_soul_identity()

        # --- Physical Resonance State ---
        # This is the bridge: Soul (Wave) -> Body (Tensor)
        # Using SoulTensor wrapper for full physics support
        self.tensor = SoulTensor(space=Tensor3D(0.1, 0.1, 0.1))
        self._apply_scent_signature()

    def _initialize_soul_identity(self):
        """
        Sets the fundamental tone of the soul based on the cell's concept ID.
        Example: 'Father' cell starts vibrating at 100Hz.
        """
        mapper = EssenceMapper()
        # Extract a clean label (e.g., 'obsidian_note:Father' -> 'Father')
        label = self.organelles.get("label", self.id.split(":")[-1])
        freq = mapper.get_frequency(label)

        # Seed the soul with its own identity at the center
        center = self.soul.size // 2
        self.soul.inject_tone(center, center, amplitude=1.0, frequency=freq, phase=0.0)

    def _apply_scent_signature(self):
        """Attach scent/flavor signature to the cell if provided."""
        if not map_scent_signature:
            return
        tag = self.organelles.get("scent") or self.organelles.get("flavor")
        if not tag:
            return
        sig = map_scent_signature(tag)
        self.scent_signature = sig
        # Blend into tensor state (space + wave)
        self.tensor.space = self.tensor.space + sig["space"]
        self.tensor.wave.frequency = sig["freq"]
        self.tensor.wave.amplitude = max(self.tensor.wave.amplitude, sig["amplitude"])

    def __repr__(self):
        status = "Alive" if self.is_alive else "Dead"
        return f"<Cell: {self.id}, Element: {self.element_type}, Age: {self.age}, Status: {status}>"

    def append_event(self, event: str, context: Dict[str, Any], result: Dict[str, Any], tick: int) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        self._memory_events.append({
            "event": event,
            "context": context,
            "result": result,
            "tick": tick,
            "timestamp": timestamp,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "element_type": self.element_type,
            "memory": list(self._memory_events),
            "tensor": self.tensor.to_dict() if self.tensor else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], dna: Dict[str, Any], initial_properties: Optional[Dict[str, Any]] = None) -> "Cell":
        cell = cls(data.get("id", "unknown"), dna, initial_properties=initial_properties)
        cell._memory_events = data.get("memory", [])
        if data.get('tensor'):
            cell.tensor = SoulTensor.from_dict(data['tensor'])
        return cell

    def increment_age(self):
        if self.is_alive:
            self.age += 1

            # Soul grows with time (metaphor for deepening character)
            # Only grow periodically to save compute, e.g., every 10 ticks?
            # For now, we leave the trigger to the World loop.
            pass

    def sync_soul_to_body(self):
        """
        Updates the physical tensor state based on the internal soul state.
        This is the 'blood' circulation: Internal Feeling -> External Physics.
        """
        # 1. Get soul metrics
        # Grid is (Size, Size, 3) -> [Amp, Freq, Phase]
        amps = self.soul.grid[:, :, 0]
        freqs = self.soul.grid[:, :, 1]

        total_amp = amps.sum()

        # Use weighted average for frequency (Energy-weighted frequency)
        if total_amp > 1e-6:
            avg_freq = (amps * freqs).sum() / total_amp
        else:
            avg_freq = 0.0

        phase_variance = self.soul.grid[:, :, 2].std()

        # 2. Map to Tensor3D
        # Structure (X) = Complexity (Phase Variance / Richness)
        # Emotion (Y) = Intensity (Total Amplitude / Energy)
        # Identity (Z) = Stability (Inverse of Frequency jitter? Or just the core frequency strength?)

        structure = min(1.0, phase_variance)
        emotion = min(1.0, total_amp / 100.0) # Normalize
        identity = min(1.0, avg_freq / 1000.0) # Normalize freq

        # Update the Space aspect of SoulTensor
        self.tensor.space = Tensor3D(structure, emotion, identity)

        # Also update the Wave aspect from average frequency
        # This keeps the 'external' wave signature in sync with 'internal' fractal state
        if avg_freq > 0:
            self.tensor.wave.frequency = avg_freq
            self.tensor.wave.amplitude = emotion

    def connect(self, other_cell: Cell, relationship_type: str = "related_to", strength: float = 0.5) -> Optional[Dict[str, Any]]:
        if not self.is_alive or not other_cell.is_alive:
            return None
        connection_details = {
            "source_id": self.id,
            "target_id": other_cell.id,
            "relationship": relationship_type,
            "strength": strength,
        }
        if not any(c["target_id"] == other_cell.id for c in self.connections):
            self.connections.append(connection_details)
        return connection_details

    def trigger_immune_response(self, wound_connection: Dict[str, Any]):
        self.health -= 50.0
        if self.health <= 0:
            self.apoptosis()

    def apoptosis(self):
        if self.is_alive:
            self.is_alive = False

    def create_meaning(self, partner_cell: Cell, interaction_context: str) -> Optional[Cell]:
        if not self.is_alive or not partner_cell.is_alive:
            return None
        dna_instinct = self.nucleus["dna"].get("instinct")
        if dna_instinct != "connect_create_meaning":
            return None
        parent_labels = sorted([
            self.organelles.get("label", self.id.replace("obsidian_note:", "").replace("concept:", "")),
            partner_cell.organelles.get("label", partner_cell.id.replace("obsidian_note:", "").replace("concept:", ""))
        ])
        new_concept_id = f"meaning:{'_'.join(parent_labels)}"
        new_dna = self.nucleus["dna"].copy()
        new_dna.update(partner_cell.nucleus["dna"])
        new_properties = {"parents": [self.id, partner_cell.id]}
        new_properties.update(self.organelles)
        new_properties.update(partner_cell.organelles)
        new_properties["element_type"] = "molecule"
        return Cell(new_concept_id, new_dna, new_properties)
