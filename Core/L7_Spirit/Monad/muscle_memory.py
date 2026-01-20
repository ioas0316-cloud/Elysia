"""
Muscle Memory: The Cerebellum
=============================
Core.Monad.muscle_memory

"Don't think, just act."

This module implements the Reflex System (Muscle Memory).
It maps Intents to pre-recorded Action Tracks (Rotors), bypassing the Thundercloud.
"""

from typing import Dict, List, Optional
from Core.Foundation.Nature.rotor import Rotor

class MuscleMemory:
    """
    The Reflex Manager.
    Stores and triggers 'Macro' actions.
    """

    def __init__(self):
        self._reflex_map: Dict[str, str] = {} # Intent -> Track Name
        self._rotors: Dict[str, Rotor] = {}

    def register_rotor(self, rotor: Rotor):
        """Connects a physical rotor to the nervous system."""
        self._rotors[rotor.name] = rotor

    def learn_reflex(self, intent: str, track_name: str, track_data: List[float], rotor_name: str):
        """
        Bakes a movement pattern into the rotor and links it to an intent.
        """
        if rotor_name in self._rotors:
            self._rotors[rotor_name].load_track(track_name, track_data)
            self._reflex_map[intent] = (rotor_name, track_name)
        else:
            print(f"Error: Rotor '{rotor_name}' not connected to MuscleMemory.")

    def try_reflex(self, intent: str) -> bool:
        """
        Attempts to execute a reflex for the given intent.
        Returns True if a reflex was triggered, False if Thought is required.
        """
        if intent in self._reflex_map:
            rotor_name, track_name = self._reflex_map[intent]
            if rotor_name in self._rotors:
                # Trigger the Action
                self._rotors[rotor_name].play_track(track_name)
                return True

        return False
