"""
World Server (The Eternal Engine)
=================================
Core.Engine.world_server

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

from Core.Foundation.hyper_sphere_core import HyperSphereCore
from Core.Civilization.trinity_citizen import LifeCitizen, QuantumCitizen, ElysiaMessiah

from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Foundation.Wave.wave_dna import WaveDNA
from Core.Intelligence.meaning_extractor import MeaningExtractor
from Core.World.Nature.vocabulary_seeder import SEEDED_LEXICON
from Core.Intelligence.narrative_weaver import THE_BARD

class WorldServer:
    def __init__(self, size: int = 30):
        self.size = size
        self.hyper_sphere = HyperSphereCore()
        
        # [History Engine]
        self.zeitgeist_rotor = Rotor("Zeitgeist", RotorConfig(rpm=1.0))
        self.meaning_extractor = MeaningExtractor()
        
        self.population: List[LifeCitizen] = []
        self.dead_count = 0
        self.born_count = 0
        self.year = 0
        
        # Genesis
        self._genesis_by_word()
        self.spawn_adam_eve_and_100_others()
        
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
                
    def spawn_adam_eve_and_100_others(self):
        print("🌱 Injecting Quantum Souls...")
        for i in range(100):
            name = f"Soul_{i}"
            arch = random.choice(["Farmer", "Warrior", "Artist", "Sage"])
            pos = (random.randint(0,self.size-1), random.randint(0,self.size-1))
            self.population.append(QuantumCitizen(name, arch, pos))
            
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
        Callable by the Sovereign Self.
        """
        self.year += 1
        
        # 1. Update History (The Macro Wave)
        self.zeitgeist_rotor.update(1.0) # Tick 1.0
        era_name, era_dna = self.get_current_era()
        
        if self.year % 10 == 0:
            print(f"--- Year {self.year} [{era_name}] --- Pop: {len(self.population)} (Born: {self.born_count}, Died: {self.dead_count})")
            
        if self.year == 20: 
            self.spawn_messiah()
        
        current_pop = list(self.population)
        
        for citizen in current_pop:
            zone_adapter = self.get_zone_adapter(citizen.location)
            
            # 2. Act with Quantum Resonance (The Micro Wave)
            log = citizen.act_in_world(self, self.population, era_dna)
            
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

    # Adapter
    def get_zone(self, xy):
        return self.get_zone_adapter(xy)

if __name__ == "__main__":
    server = WorldServer(size=30)
    server.run_forever()
