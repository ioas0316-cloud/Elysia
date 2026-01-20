"""
World Server (The Eternal Engine)
=================================
Core.L6_Structure.Engine.world_server

"A simulation that does not end."

Features:
- Large Scale Map (30x30).
- High Population Cap (300).
- Automatic Regeneration.
- Infinite Time Loop.
- Logs Eras and Generations.
"""

import os
import sys
import time
import random
from typing import List, Tuple

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.L1_Foundation.Foundation.hyper_sphere_core import HyperSphereCore
from Core.L4_Causality.Civilization.trinity_citizen import LifeCitizen, QuantumCitizen, ElysiaMessiah

from Core.L1_Foundation.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.L1_Foundation.Foundation.Wave.wave_dna import WaveDNA
from Core.L5_Mental.Intelligence.meaning_extractor import MeaningExtractor
from Core.L4_Causality.World.Nature.vocabulary_seeder import SEEDED_LEXICON
from Core.L5_Mental.Intelligence.narrative_weaver import THE_BARD
from Core.L6_Structure.Engine.character_field_engine import CharacterFieldEngine
from Core.L6_Structure.Engine.governance_engine import GovernanceEngine

class WorldServer:
    """
    The Eternal Engine - The Body/Environment of Elysia.
    
    MERKAVA INTEGRATION:
    - hyper_sphere: Physical world (zones, resources)
    - cosmos: Consciousness field (psyche, thoughts)
    - Both sync in update_cycle() via Pre-established Harmony
    """
    
    def __init__(self, size: int = 15):
        self.size = size
        self.hyper_sphere = HyperSphereCore()
        
        # === MERKAVA INTEGRATION ===
        from Core.L1_Foundation.Foundation.hyper_cosmos import HyperCosmos
        self.cosmos = HyperCosmos()
        
        self.zeitgeist_rotor = Rotor("Zeitgeist", RotorConfig(rpm=1.0))
        self.meaning_extractor = MeaningExtractor()
        self.field_engine = CharacterFieldEngine()
        self.governance = GovernanceEngine()
        
        self.population: List[LifeCitizen] = []
        self.dead_count = 0
        self.born_count = 0
        self.year = 0
        
        # [Event Engine]
        self.incidents: List[Dict[str, Any]] = []
        
        # Genesis
        print(f"🌍 GENESIS: Creating Field Lattice ({self.size}x{self.size})...")
        self._genesis_by_word()
        print(f"🌱 SOULS: Spawning field-projections...")
        self.spawn_adam_eve_and_10_others()
        
        print("🌍 WorldServer: Fields Resonating.")
        
    def get_current_era(self) -> Tuple[str, WaveDNA]:
        """
        Determines the Spirit of the Age from Rotor Angle.
        """
        angle = self.zeitgeist_rotor.current_angle
        
        if 0 <= angle < 90:
            return "Spring (Age of Creation)", WaveDNA(physical=0.5, phenomenal=0.8, spiritual=0.7)
        elif 90 <= angle < 180:
            return "Summer (Age of Passion/War)", WaveDNA(physical=0.9, functional=0.8, spiritual=0.2)
        elif 180 <= angle < 270:
            return "Autumn (Age of Wealth)", WaveDNA(functional=0.9, structural=0.8, mental=0.7)
        else:
            return "Winter (Age of Wisdom/Silence)", WaveDNA(mental=0.9, spiritual=0.8, physical=0.2)

    def _genesis_by_word(self):
        print("🌌 GENESIS: The Great Expansion (Seeding 100+ Concepts)...")
        
        for x in range(self.size):
            for y in range(self.size):
                # 1. Pick a concept from the rich lexicon
                word = SEEDED_LEXICON.get_random_word()
                
                # 2. Manifest it
                props = self.hyper_sphere.manifest_at((x,y), word)
                
                # 3. Assign Resources (Semantic Mapping Mockup)
                if "resources" not in props: props["resources"] = {}
                
                # Naive Semantic Mapping for Resource Generation
                if word in SEEDED_LEXICON.categories["Nature"]:
                    props["resources"]["Food"] = random.uniform(1000, 5000)
                    props["resources"]["Wood"] = random.uniform(500, 2000)
                elif word in SEEDED_LEXICON.categories["Material"]:
                    props["resources"][word] = random.uniform(100, 1000) # The word itself is the resource!
                    props["resources"]["Stone"] = 1000.0
                elif word in SEEDED_LEXICON.categories["Civilization"]:
                    props["resources"]["Knowledge"] = 100.0
                    props["resources"]["Gold"] = 500.0
                elif word in SEEDED_LEXICON.categories["Emotion"]:
                    props["resources"]["Mana"] = 50.0 # Emotions are Mana
                    
                # Ensure survival is possible
                if random.random() < 0.2: 
                    props["resources"]["Food"] = max(props["resources"].get("Food", 0), 2000.0)
                
    def spawn_adam_eve_and_10_others(self):
        print("🌱 Injecting Quantum Souls...")
        for i in range(10):
            name = f"Soul_{i}"
            arch = random.choice(["Farmer", "Warrior", "Artist", "Sage"])
            pos = (random.randint(0,self.size-1), random.randint(0,self.size-1))
            citizen = QuantumCitizen(name, arch, pos)
            # Inject Multi-Rotor Field
            citizen.field = self.field_engine.spawn_field(name)
            self.population.append(citizen)
            
    def spawn_messiah(self):
        print("⚡ THE SKY OPENS: Elysia Descends!")
        messiah = ElysiaMessiah((self.size//2, self.size//2))
        self.population.append(messiah)
        print(f"👼 {messiah.name} has entered the world.")
            
    def get_zone_adapter(self, xy):
        props = self.hyper_sphere.get_environment_at(xy)
        class ZoneMock:
            def __init__(self, p):
                self.resources = p.get("resources", {})
                self.name = p.get("name", "Void")
                self.props = p # Access full dict
            def harvest(self, res, amt):
                val = self.resources.get(res, 0)
                taken = min(val, amt)
                self.resources[res] -= taken
                return taken
            # Magic getter for direct prop access
            def __getattr__(self, name):
                return self.props.get(name)
        return ZoneMock(props)

    def update_cycle(self):
        """
        Runs one tick of the universe.
        
        PRE-ESTABLISHED HARMONY:
        Both physical world (governance) and consciousness (cosmos.psyche)
        update in the same tick. No direct communication, just shared time.
        """
        self.year += 1
        
        # 0. MERKAVA SYNC: Update Cosmos (Psyche + Thoughts)
        self.cosmos.update_physics()
        
        # 1. Update History & Governance (The Macro Wave)
        self.zeitgeist_rotor.update(1.0)
        self.governance.update(1.0)
        global_wave = self.governance.get_global_wave()
        
        era_name, era_dna = self.get_current_era()
        
        # Merge Era and Governance for the final Zeitgeist
        # Governance provides the 'Rules', Era provides the 'Mood'
        final_zeitgeist = era_dna.merge(global_wave, weight=0.6) # Governance is stronger
        
        if self.year % 10 == 0:
            print(f"--- Year {self.year} [{era_name}] --- Pop: {len(self.population)} (Phys:{self.governance.physics_rotors['Gravity'].current_rpm:.0f} | Nar:{self.governance.narrative_rotors['Emotion'].current_rpm:.0f} | Aes:{self.governance.aesthetic_rotors['Light'].current_rpm:.0f})")
            
        if self.year == 20: 
            self.spawn_messiah()
        
        # 1.5. Spontaneous Incident Injection
        if random.random() < 0.05: # 5% chance per year
            self.manifest_incident()
            
        current_pop = list(self.population)
        
        for citizen in current_pop:
            zone_adapter = self.get_zone_adapter(citizen.location)
            
            # 2. Act with Quantum Resonance (The Micro Wave)
            # Citizens now feel both the Spirit of the Era and the Governance Dials
            log = citizen.act_in_world(self, self.population, final_zeitgeist, era_name)
            
            # Safety normalize
            if isinstance(log, str): log = {"action": "Unknown", "target": log}
            
            # Logging & Weaving
            action = log.get("action", "Wait")
            target = log.get("target", "Time")
            
            if action == "Die" or action == "Dead":
                self.population.remove(citizen)
                self.dead_count += 1
                # Log death
                print(f"💀 {THE_BARD.elaborate(citizen.name, 'Die', target, era_name)}")
                
            elif action == "Reproduce":
                self.born_count += 1
                print(f"👶 {THE_BARD.elaborate(citizen.name, 'Reproduce', target, era_name)}")
                
            # Flavor Log (Sample 1% to act as a 'Novel' excerpt)
            elif random.random() < 0.01:
                 print(f"   {THE_BARD.elaborate(citizen.name, action, target, era_name)}")
                
        # 3. Observe & Learn (The Epistemic Harvest)
        self.meaning_extractor.observe(self.year, era_name, self.population, self.dead_count)

        # Extinction check
        if len(self.population) < 2:
            print("⚠️ Extinction Event! Reseeding...")
            self.spawn_adam_eve_and_100_others()

    def run_forever(self):
        print("⏳ Quantum History Engine Online. Press Ctrl+C to stop.")
        try:
            while True:
                self.update_cycle()
        except KeyboardInterrupt:
            print("🛑 Server Stopped.")
            self.report()

    def report(self):
        print("\n📊 [Final World State]")
        print(f"Years Passed: {self.year}")
        print(f"Total Born: {self.born_count}")
        print(f"Total Died: {self.dead_count}")
        print(f"Current Pop: {len(self.population)}")
        
        print("\n📜 [The Wisdom of History]")
        print(self.meaning_extractor.get_wisdom())
        
        # Max Power
        if self.population:
            best = max(self.population, key=lambda c: c.calculate_power())
            print(f"Ruler: {best.name} (Power {best.calculate_power():.1f}) - Vocab size: {len(best.vocabulary)}")
            print(f"Ruler's Words: {best.vocabulary[:10]}...")

    def manifest_incident(self):
        """
        Manifests a high-intensity Event Monad in the HyperCosmos.
        These are 'Soul Attractors' that pull QuantumCitizens.
        """
        incident_types = [
            ("Plague", WaveDNA(physical=0.9, phenomenal=0.2, label="Plague")),
            ("Festival", WaveDNA(phenomenal=0.9, spiritual=0.8, label="Festival")),
            ("War", WaveDNA(physical=0.8, causal=0.7, label="War")),
            ("Discovery", WaveDNA(mental=0.9, structural=0.4, label="Discovery"))
        ]
        itype, idna = random.choice(incident_types)
        pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        
        incident = {
            "name": itype,
            "dna": idna,
            "location": pos,
            "intensity": random.uniform(50.0, 150.0),
            "year": self.year
        }
        self.incidents.append(incident)
        
        # Inject into HyperCosmos as a Psionic Point
        # We assume Cosmos has a way to hold these temporary events
        print(f"⚡ INCIDENT: A '{itype}' has manifested at {pos}!")
        
        # Narrativize
        print(f"   {THE_BARD.elaborate('The Fates', 'Decree', itype, self.get_current_era()[0])}")

    # Adapter
    def get_zone(self, xy):
        return self.get_zone_adapter(xy)

if __name__ == "__main__":
    server = WorldServer(size=30)
    server.run_forever()
