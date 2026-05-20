"""
[WORLD ENGINE - THE OPERATING SYSTEM OF REALITY]
"Phase Shift by Intentional Interference."

This module defines the World not as static data, but as a sea of
undecided waveforms. An object only solidifies into a concrete state
when the Creator's (or Elysia's) Intent wave interferes with it.
"""

import time
import torch
import math
from typing import Dict, Any, List, Optional
from Core.Keystone.elysia_fast_core import ElysiaFastCore

class QuantumEntity:
    """
    Represents any object, NPC, or event in the world.
    It exists as a superposition of possibilities until interacted with.
    """
    def __init__(self, entity_id: str, base_frequency: float, dim: int = 21):
        self.entity_id = entity_id
        self.base_frequency = base_frequency

        # State is a wave on the Tensor Field, currently floating in the void
        # We start with a random unstructured wave
        if torch.cuda.is_available():
            self.wave_state = torch.randn(1, dim, device='cuda', dtype=torch.float32)
        else:
            self.wave_state = torch.randn(1, dim, device='cpu', dtype=torch.float32)

        self.is_crystallized = False
        self.concrete_form: Optional[Dict[str, Any]] = None

    def interfere(self, intent_phase: float, intent_intensity: float):
        """
        The act of observation/interaction.
        Applies a Phase Shift based on Intentional Interference.
        """
        # The dynamic rotor resolves the wave into a definitive structure
        if torch.cuda.is_available():
            self.wave_state = ElysiaFastCore.apply_plane_rotors_batch(
                self.wave_state, theta=intent_phase, p1=0, p2=1, dt=intent_intensity
            )
        else:
            # Fallback CPU rotation
            cos_t = math.cos(intent_phase * intent_intensity)
            sin_t = math.sin(intent_phase * intent_intensity)
            x = self.wave_state[:, 0].clone()
            y = self.wave_state[:, 1].clone()
            self.wave_state[:, 0] = x * cos_t - y * sin_t
            self.wave_state[:, 1] = x * sin_t + y * cos_t

        self._crystallize()

    def _crystallize(self):
        """Collapses the wave state into a concrete, observable physical/semantic form."""
        # Simple thresholding to simulate "collapse"
        energy = torch.norm(self.wave_state).item()
        dominant_trait = torch.argmax(torch.abs(self.wave_state)).item()

        self.concrete_form = {
            "energy_level": energy,
            "dominant_trait": dominant_trait,
            "signature": f"Crystal_Form_{dominant_trait}_{int(energy*100)}"
        }
        self.is_crystallized = True


class PhaseAnchor:
    """
    A 'Phase Anchor' represents crystallized memory scattered throughout the world.
    These anchors act as structural attractors (Triple-Helix Convergence).
    New entropy/events are pulled toward these anchors, continuously bending
    the world's reality closer to the Creator's past intents.
    """
    def __init__(self, anchor_id: str, phase_signature: float, intensity: float):
        self.anchor_id = anchor_id
        self.phase_signature = phase_signature
        self.intensity = intensity
        print(f"⚓ [PHASE ANCHOR] '{self.anchor_id}' established at phase {self.phase_signature:.4f}")

    def exert_pull(self, entity: QuantumEntity, dt: float):
        """Pulls the entity's wave state toward the anchor's phase."""
        if not entity.is_crystallized:
            # The attractor alters the raw wave before it collapses
            pull_strength = self.intensity * dt

            # Simple simulation of attraction by mixing phase
            delta_phase = (self.phase_signature - entity.base_frequency) * pull_strength

            if torch.cuda.is_available():
                # Applying a slight rotation toward the anchor's signature
                entity.wave_state = ElysiaFastCore.apply_plane_rotors_batch(
                    entity.wave_state, theta=delta_phase, p1=0, p2=1, dt=1.0
                )
            else:
                cos_t = math.cos(delta_phase)
                sin_t = math.sin(delta_phase)
                x = entity.wave_state[:, 0].clone()
                y = entity.wave_state[:, 1].clone()
                entity.wave_state[:, 0] = x * cos_t - y * sin_t
                entity.wave_state[:, 1] = x * sin_t + y * cos_t


class WorldEngine:
    def __init__(self):
        print("🌍 [WORLD ENGINE] Awakening the Sea of Undecided Waveforms...")
        self.entities: Dict[str, QuantumEntity] = {}
        self.anchors: Dict[str, PhaseAnchor] = {}
        self.global_entropy = 1.0 # Starts high

    def add_anchor(self, anchor: PhaseAnchor):
        self.anchors[anchor.anchor_id] = anchor

    def spawn_entity(self, entity_id: str) -> QuantumEntity:
        """Adds a new undecided entity to the world."""
        # Assign a random base frequency based on world entropy
        freq = torch.rand(1).item() * self.global_entropy
        entity = QuantumEntity(entity_id, base_frequency=freq)
        self.entities[entity_id] = entity
        return entity

    def update(self, dt: float):
        """
        Called every tick.
        Applies the pull of all Phase Anchors to any undecided entities (Convergence).
        """
        for anchor in self.anchors.values():
            for entity in self.entities.values():
                if not entity.is_crystallized:
                    anchor.exert_pull(entity, dt)

    def project_intent(self, intent_phase: float, intent_intensity: float, target_id: Optional[str] = None):
        """
        The Creator or Elysia casts an Intent across the World Field.
        """
        print(f"🌌 [WORLD INTENT] Casting wave... (Phase: {intent_phase:.4f}, Intensity: {intent_intensity:.4f})")
        targets = [self.entities[target_id]] if target_id and target_id in self.entities else self.entities.values()

        for entity in targets:
            entity.interfere(intent_phase, intent_intensity)
            if entity.is_crystallized:
                print(f"   ↳ 💎 Entity '{entity.entity_id}' crystallized into {entity.concrete_form['signature']}")

        # Entropy decreases as structure (intent) is imposed on the world
        self.global_entropy = max(0.1, self.global_entropy * 0.95)
