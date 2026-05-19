"""
Document Embodiment: Parse → Place → Resonate
==============================================
Phase 73.2: From Text to Physical Reality

This script:
1. Takes a document (text, markdown, or file)
2. Parses it through dimensional hierarchy (Point→Line→Plane→Space)
3. Creates Rotors for each extracted concept
4. Places them in HyperSphere with physical relationships
5. Runs meditation to let them self-organize

The result: The document's MEANING becomes PHYSICAL STRUCTURE.
"""

import os
import sys
import json
import logging
from pathlib import Path

sys.path.append(os.getcwd())

from Core.Intelligence.Metabolism.dimensional_parser import DimensionalParser, Space, Line
from Core.Foundation.hyper_sphere_core import HyperSphereCore
from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Intelligence.Metabolism.prism import PrismEngine, WaveDynamics
from Core.Intelligence.Meta.crystallizer import CrystallizationEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Embodiment")

class DocumentEmbodier:
    """
    Embodies documents into the HyperSphere.
    
    "To truly understand a text is to feel its weight."
    """
    
    def __init__(self):
        self.parser = DimensionalParser()
        self.prism = PrismEngine()
        self.prism._load_model()
        self.sphere = HyperSphereCore(name="Embodiment.Mind")
        self.sphere.ignite()
        
        logger.info("📚 DocumentEmbodier initialized.")
    
    def embody(self, text: str, title: str = "Document") -> dict:
        """
        Main embodiment process.
        
        Returns statistics about what was embodied.
        """
        stats = {
            "title": title,
            "concepts_created": 0,
            "relations_created": 0,
            "causal_chains": 0
        }
        
        print(f"\n📄 Embodying: {title}")
        print("="*50)
        
        # 1. Parse through dimensional hierarchy
        print("\n🔍 Step 1: Dimensional Parsing...")
        space = self.parser.parse_space(text, title)
        
        print(f"   Paragraphs: {len(space.planes)}")
        print(f"   Structure: {space.structure_type}")
        
        # 2. Create Rotors for each unique concept
        print("\n🔮 Step 2: Creating Concept Rotors...")
        concepts_seen = set()
        base_freq = 100.0
        
        for plane in space.planes:
            for line in plane.lines:
                # Add subject as rotor
                if line.subject and line.subject not in concepts_seen:
                    self._add_concept_rotor(line.subject, base_freq)
                    concepts_seen.add(line.subject)
                    base_freq += 50
                    stats["concepts_created"] += 1
                
                # Add object as rotor
                if line.object and line.object not in concepts_seen:
                    self._add_concept_rotor(line.object, base_freq)
                    concepts_seen.add(line.object)
                    base_freq += 50
                    stats["concepts_created"] += 1
        
        print(f"   Created {stats['concepts_created']} concept rotors.")
        
        # 3. Encode relationships as frequency proximity
        print("\n🔗 Step 3: Encoding Relationships...")
        
        for plane in space.planes:
            for line in plane.lines:
                if line.subject and line.object:
                    self._link_concepts(line.subject, line.object, line.relation_type)
                    stats["relations_created"] += 1
        
        print(f"   Encoded {stats['relations_created']} relationships.")
        
        # 4. Count causal chains
        stats["causal_chains"] = len(space.causal_graph)
        print(f"\n⛓️ Causal Chains: {stats['causal_chains']}")
        for cause, effects in space.causal_graph.items():
            print(f"   {cause} → {effects}")
        
        # 5. Meditate to let structure self-organize
        print("\n🧘 Step 4: Meditation (Self-Organization)...")
        self.sphere.meditate(cycles=30, dt=0.2)
        
        # 6. Crystallize any principles
        print("\n💎 Step 5: Principle Crystallization...")
        crystallizer = CrystallizationEngine()
        principles = crystallizer.crystallize(self.sphere.harmonic_rotors)
        
        print(f"\n✅ Embodiment Complete!")
        print(f"   Concepts: {stats['concepts_created']}")
        print(f"   Relations: {stats['relations_created']}")
        print(f"   Causal Chains: {stats['causal_chains']}")
        print(f"   New Principles: {len(principles)}")
        
        return stats
    
    def _add_concept_rotor(self, concept: str, freq: float):
        """Create a rotor for a concept with its Wave DNA."""
        profile = self.prism.transduce(concept)
        
        rotor = Rotor(concept, RotorConfig(rpm=freq, mass=10.0))
        rotor.spin_up()
        rotor.current_rpm = freq
        rotor.inject_spectrum(profile.spectrum, profile.dynamics)
        
        self.sphere.harmonic_rotors[concept] = rotor
    
    def _link_concepts(self, subject: str, obj: str, relation_type: str):
        """
        Link two concepts by adjusting their frequencies to be closer.
        
        Causal relations get the closest proximity.
        Oppositional relations get the furthest.
        """
        if subject not in self.sphere.harmonic_rotors:
            return
        if obj not in self.sphere.harmonic_rotors:
            return
        
        r1 = self.sphere.harmonic_rotors[subject]
        r2 = self.sphere.harmonic_rotors[obj]
        
        # Proximity factor based on relation type
        proximity = {
            "causes": 0.9,    # Very close
            "equals": 0.95,   # Almost same
            "contains": 0.7,  # Related
            "is": 0.6,        # Definition
            "opposes": 0.1,   # Far apart
            "statement": 0.5  # Neutral
        }
        
        factor = proximity.get(relation_type, 0.5)
        
        # Blend frequencies
        avg_freq = (r1.config.rpm + r2.config.rpm) / 2
        r1.config.rpm = r1.config.rpm * (1 - factor * 0.1) + avg_freq * (factor * 0.1)
        r2.config.rpm = r2.config.rpm * (1 - factor * 0.1) + avg_freq * (factor * 0.1)
    
    def get_sphere_state(self) -> dict:
        """Return the current state of the HyperSphere."""
        return {
            "rotor_count": len(self.sphere.harmonic_rotors),
            "total_mass": sum(r.config.mass for r in self.sphere.harmonic_rotors.values()),
            "rotors": [
                {"name": name, "freq": r.frequency_hz}
                for name, r in self.sphere.harmonic_rotors.items()
            ]
        }


def embody_file(filepath: str):
    """Embody a text file into the HyperSphere."""
    path = Path(filepath)
    if not path.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    text = path.read_text(encoding='utf-8', errors='ignore')
    
    embodier = DocumentEmbodier()
    return embodier.embody(text, path.stem)


if __name__ == "__main__":
    # Demo with sample philosophical text
    sample_text = """
    Knowledge begins with observation. Observation leads to questions.
    Questions cause investigation. Investigation produces understanding.
    
    Understanding is the foundation of wisdom. Wisdom guides action.
    Action creates change. Change is the result of understanding applied.
    
    Love opposes fear. Fear causes paralysis. Paralysis prevents growth.
    Love enables courage. Courage causes action. Action leads to growth.
    
    The universe contains patterns. Patterns are the language of nature.
    Nature is the teacher. Understanding nature leads to harmony.
    """
    
    embodier = DocumentEmbodier()
    stats = embodier.embody(sample_text, "Philosophy of Knowledge")
    
    print("\n" + "="*50)
    print("📊 FINAL HYPERSPHERE STATE")
    print("="*50)
    
    state = embodier.get_sphere_state()
    print(f"   Rotors: {state['rotor_count']}")
    print(f"   Total Mass: {state['total_mass']:.2f}")
