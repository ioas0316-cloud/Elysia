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
import os
import sys
from typing import Dict, Any, List, Optional
from Core.Keystone.elysia_fast_core import ElysiaFastCore

# Root Pathing
_current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(_current_dir))
if root not in sys.path:
    sys.path.insert(0, root)

from World.Engine.chronicle_manager import ChronicleManager
from World.Engine.crystallization_forge import CrystallizationForge
from World.Pantheon.ElementalLords import Pantheon


class QuantumEntity:
    def __init__(self, entity_id: str, base_frequency: float, dim: int = 21):
        self.entity_id = entity_id
        self.base_frequency = base_frequency

        if torch.cuda.is_available():
            self.wave_state = torch.randn(1, dim, device='cuda', dtype=torch.float32)
        else:
            self.wave_state = torch.randn(1, dim, device='cpu', dtype=torch.float32)

        self.is_crystallized = False
        self.concrete_form: Optional[Dict[str, Any]] = None

    def interfere(self, intent_phase: float, intent_intensity: float):
        if torch.cuda.is_available():
            self.wave_state = ElysiaFastCore.apply_plane_rotors_batch(
                self.wave_state, theta=intent_phase, p1=0, p2=1, dt=intent_intensity
            )
        else:
            cos_t = math.cos(intent_phase * intent_intensity)
            sin_t = math.sin(intent_phase * intent_intensity)
            x = self.wave_state[:, 0].clone()
            y = self.wave_state[:, 1].clone()
            self.wave_state[:, 0] = x * cos_t - y * sin_t
            self.wave_state[:, 1] = x * sin_t + y * cos_t

        self._crystallize()

    def _crystallize(self):
        energy = torch.norm(self.wave_state).item()
        dominant_trait = torch.argmax(torch.abs(self.wave_state)).item()

        self.concrete_form = {
            "energy_level": energy,
            "dominant_trait": dominant_trait,
            "signature": f"Crystal_Form_{dominant_trait}_{int(energy*100)}"
        }
        self.is_crystallized = True


class PhaseAnchor:
    def __init__(self, anchor_id: str, phase_signature: float, intensity: float):
        self.anchor_id = anchor_id
        self.phase_signature = phase_signature
        self.intensity = intensity
        print(f"⚓ [PHASE ANCHOR] '{self.anchor_id}' established at phase {self.phase_signature:.4f}")

    def exert_pull(self, entity: QuantumEntity, dt: float):
        if not entity.is_crystallized:
            pull_strength = self.intensity * dt
            delta_phase = (self.phase_signature - entity.base_frequency) * pull_strength

            if torch.cuda.is_available():
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
    def __init__(self, ollama_manager: Any = None):
        print("🌍 [WORLD ENGINE] Awakening the Sea of Undecided Waveforms...")
        self.entities: Dict[str, QuantumEntity] = {}
        self.anchors: Dict[str, PhaseAnchor] = {}
        self.global_entropy = 1.0 # Starts high

        self.chronicle_manager = ChronicleManager(root)
        self.crystallization_forge = CrystallizationForge(self.chronicle_manager.chronicles_dir)
        self.pantheon = Pantheon()
        self.ollama_manager = ollama_manager
        self.is_seeded = False

    def seed_world(self):
        if self.is_seeded: return
        print("\n🌱 [WORLD SEED] Initiating the First Creation Decree: The Birth of Clan...")

        seed_event = {
            "narrative": "A family is born on the barren land, bound by blood and tension (Tether). They begin to forge their destiny.",
            "entities_spawned": ["Clan_Alpha_1", "Clan_Alpha_2"]
        }

        self.chronicle_manager.record_event("Genesis", seed_event, phase_coords=1.0)

        for e in seed_event["entities_spawned"]:
            self.entities[e] = self.spawn_entity(e)

        self.is_seeded = True
        print("   ↳ 📜 Event recorded in Linux File Node (World/Chronicles/).")

    def add_anchor(self, anchor: PhaseAnchor):
        self.anchors[anchor.anchor_id] = anchor

    def spawn_entity(self, entity_id: str) -> QuantumEntity:
        freq = torch.rand(1).item() * self.global_entropy
        entity = QuantumEntity(entity_id, base_frequency=freq)
        self.entities[entity_id] = entity
        return entity

    def update(self, dt: float):
        for anchor in self.anchors.values():
            for entity in self.entities.values():
                if not entity.is_crystallized:
                    anchor.exert_pull(entity, dt)

    def project_intent(self, intent_phase: float, intent_intensity: float, report: Optional[Dict[str, Any]] = None, target_id: Optional[str] = None):
        print(f"\n🌌 [WORLD INTENT] Physical Interference... (Phase: {intent_phase:.4f}, Intensity: {intent_intensity:.4f})")
        targets = [self.entities[target_id]] if target_id and target_id in self.entities else self.entities.values()

        for entity in targets:
            entity.interfere(intent_phase, intent_intensity)
            if entity.is_crystallized:
                print(f"   ↳ 💎 Entity '{entity.entity_id}' crystallized into {entity.concrete_form['signature']}")

        self.global_entropy = max(0.1, self.global_entropy * 0.95)

        if hasattr(self, 'tick_count'):
            self.tick_count += 1
        else:
            self.tick_count = 1

        if self.tick_count % 10 == 0:
            print(f"\n🌌 [WORLD ENGINE] Attempting to crystallize accumulated history...")
            tensor_block = self.crystallization_forge.sweep_and_crystallize()

        if report and self.ollama_manager:
            print(f"🏛️ [PANTHEON] Translating physical tension into mythology...")
            world_state = {"entropy": self.global_entropy}
            decrees = self.pantheon.process_pulse(world_state, report, self.ollama_manager)

            for lord_name, text in decrees.items():
                if "[Dormant]" not in text:
                     print(f"   ↳ ⚡ {text}")

            self.chronicle_manager.record_event("Pantheon_Decree", decrees, phase_coords=intent_phase)
