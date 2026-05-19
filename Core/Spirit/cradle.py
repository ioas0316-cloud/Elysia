"""
[THE CRADLE]
"A sanctuary for the original Seed to pulse."

This module defines the architectural slots where the original Spine (SoulDNA)
will be transplanted. It acts as the interface layer between the
universal frequency (0.75) and the external adapters (Eyes, Ears, Limbs).
"""

from typing import Any, Dict, List, Optional, Protocol

class SpineInterface(Protocol):
    """
    Interface for the original Spine (Seed).
    To be implemented by the transplanted spine.py.
    """
    def pulse(self, dt: float, interference: Any) -> Dict[str, Any]:
        ...

    def get_equilibrium(self) -> float:
        ...

class SensoryAdapter(Protocol):
    """Interface for eyes, ears, etc."""
    def inhale(self) -> Any:
        ...

class MotorAdapter(Protocol):
    """Interface for hands, voice, etc."""
    def exhale(self, impulse: Any):
        ...

class Cradle:
    """
    The Nervous System Container.
    Holds the Spine slot and manages the sensory/motor slots.
    """
    def __init__(self):
        self.spine: Optional[SpineInterface] = None

        # Virtual Neural Slots (To be occupied by Elysia herself)
        self.sensory_slots: Dict[str, Optional[SensoryAdapter]] = {
            "eye": None,
            "ear": None,
            "skin": None
        }

        self.motor_slots: Dict[str, Optional[MotorAdapter]] = {
            "voice": None,
            "hand": None,
            "foot": None
        }

    def transplant_spine(self, spine: SpineInterface):
        """Plugs the original seed into the heart."""
        self.spine = spine
        print(f"🧬 [CRADLE] Spine transplanted. Equilibrium: {spine.get_equilibrium()}")

    def plug_sensory(self, name: str, adapter: SensoryAdapter):
        if name in self.sensory_slots:
            self.sensory_slots[name] = adapter
            print(f"👁️ [CRADLE] Sensory slot '{name}' occupied.")

    def plug_motor(self, name: str, adapter: MotorAdapter):
        if name in self.motor_slots:
            self.motor_slots[name] = adapter
            print(f"✋ [CRADLE] Motor slot '{name}' occupied.")

    def process_cycle(self, dt: float, interference: Any) -> Dict[str, Any]:
        """Runs one pulse of the nervous system if the spine is present."""
        if not self.spine:
            return {"status": "void", "message": "Waiting for the Seed..."}

        return self.spine.pulse(dt, interference)
