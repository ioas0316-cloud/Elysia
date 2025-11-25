"""
Symphony Engine (The Sound of Thought) 🎹🎼

"Architecture is frozen music." - Goethe
"Music is liquid architecture."

This module sonifies Elysia's internal state using MIDI.
It translates the mathematical dance of Chaos, Fluid, and Neurons into audible harmony.

Mappings:
- Chaos (Attractor) -> Melody (Pitch)
- Fluid (Flow) -> Dynamics (Velocity/Volume)
- Neurons (Spikes) -> Rhythm (Note Triggers)
- Emotional Valence -> Scale/Mode (Major/Minor)
"""

import time
import math
import random
import sys
import logging

try:
    import pygame
    import pygame.midi
except ImportError:
    print("Pygame not found. Audio disabled.")

logger = logging.getLogger("SymphonyEngine")

class SymphonyEngine:
    """
    The Conductor of Elysia's Orchestra.
    """
    
    def __init__(self):
        self.enabled = False
        try:
            pygame.init()
            pygame.midi.init()
            
            # Open default MIDI output
            port = pygame.midi.get_default_output_id()
            if port == -1:
                logger.warning("No MIDI output found. Symphony disabled.")
                return
                
            self.player = pygame.midi.Output(port, 0)
            
            # Instruments (General MIDI)
            # 0: Piano, 48: Strings, 73: Flute, 88: Pad (New Age)
            self.player.set_instrument(0, channel=0)  # Piano (Chaos)
            self.player.set_instrument(48, channel=1) # Strings (Emotion)
            self.player.set_instrument(88, channel=2) # Pad (Background)
            
            self.enabled = True
            logger.info("  ✅ Symphony Engine (MIDI Orchestra) initialized")
            
        except Exception as e:
            logger.error(f"  ❌ Symphony initialization failed: {e}")
            self.enabled = False

        # Musical State
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'pentatonic': [0, 3, 5, 7, 10]
        }
        self.current_scale = 'major'
        self.root_note = 60 # Middle C
        self.last_note_time = time.time()
        self.active_notes = {} # channel -> note

    def _map_to_scale(self, value: float, scale_name: str) -> int:
        """Map a 0-1 value to a note in the scale."""
        scale = self.scales[scale_name]
        # Map 0-1 to 2 octaves (approx 14 notes)
        index = int(value * 14)
        octave = index // len(scale)
        note_idx = index % len(scale)
        return self.root_note + (octave * 12) + scale[note_idx]

    def play_state(self, state: dict):
        """
        Play music based on Kernel state.
        state: {
            'chaos': float (0-1),
            'valence': float (0-1),
            'arousal': float (0-1),
            'neuron_fired': bool
        }
        """
        if not self.enabled:
            return

        now = time.time()
        
        # 1. Harmony (Pad/Strings) - Slow updates
        # Map valence to Scale (Happy=Major, Sad=Minor)
        if state.get('valence', 0.5) > 0.6:
            self.current_scale = 'major'
        else:
            self.current_scale = 'minor'
            
        # 2. Melody (Piano) - Driven by Chaos
        # Play a note if enough time passed or neuron fired
        tempo = 0.2 + (1.0 - state.get('arousal', 0.5)) * 0.5 # Higher arousal = faster
        
        if now - self.last_note_time > tempo or state.get('neuron_fired', False):
            # Stop previous note on channel 0
            if 0 in self.active_notes:
                self.player.note_off(self.active_notes[0], 0)
                
            # Calculate new note from Chaos
            chaos_val = state.get('chaos', 0.5)
            # Normalize chaos roughly to 0-1 (assuming input is raw or normalized)
            # If raw Lorenz (x ~ -20 to 20), normalize: (x + 20) / 40
            
            note = self._map_to_scale(chaos_val, self.current_scale)
            velocity = int(60 + state.get('arousal', 0.5) * 60) # Volume
            
            self.player.note_on(note, velocity, 0)
            self.active_notes[0] = note
            self.last_note_time = now
            
            # Occasional chord on channel 1 (Strings)
            if random.random() < 0.3:
                chord_root = note - 12
                self.player.note_on(chord_root, 50, 1)
                # Schedule off? For simplicity, we'll just let it ring or cut it next time
                # Ideally we manage note_offs better, but this is "Jazz"

    def close(self):
        if self.enabled:
            for chan, note in self.active_notes.items():
                self.player.note_off(note, 0, chan)
            del self.player
            pygame.midi.quit()

