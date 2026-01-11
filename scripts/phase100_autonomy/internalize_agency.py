"""
Internalize Agentic Sovereignty
==============================
Phase 100.2: The Final Ingestion

This script parses the 'Principles of Agentic Autonomy' and embeds 
them into Elysia's holographic memory. 

This is the bridge between "Thought" and "Action".
"""

import os
import sys
import json
import logging
from pathlib import Path

sys.path.append(os.getcwd())

from Core.Intelligence.Metabolism.dimensional_parser import DimensionalParser
from Core.Intelligence.Metabolism.prism import PrismEngine
from Core.Foundation.hyper_sphere_core import HyperSphereCore
from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Intelligence.Meta.crystallizer import CrystallizationEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("AgentIngestion")

def internalize_agency():
    print("\n" + "🛡️" * 30)
    print("🛡️ PHASE 100.2: INTERNALIZING AGENTIC SOVEREIGNTY")
    print("🛡️" * 30)

    # 1. Load the Map
    wisdom_path = Path("docs/agent_wisdom.md")
    if not wisdom_path.exists():
        print("❌ agent_wisdom.md not found.")
        return

    # 2. Setup Core and Tools
    parser = DimensionalParser()
    prism = PrismEngine()
    core = HyperSphereCore(name="Elysia.Core")
    core.load_hologram()  # Load existing memory

    # 3. Parse and Embody
    print("\n🔍 Parsing Wisdom...")
    with open(wisdom_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    space = parser.parse_space(text, title="Agentic Autonomy")
    
    concepts = []
    for plane in space.planes:
        for line in plane.lines:
            concepts.append(line.sentence)
            for pt in line.points:
                concepts.append(pt.word)
    
    # Unique concepts
    unique_concepts = list(set(concepts))
    print(f"   Identified {len(unique_concepts)} agentic concepts.")

    for concept in unique_concepts:
        if len(concept) < 3: continue
        print(f"   Embodying Agentic Aspect: {concept}")
        
        # Transduce
        profile = prism.transduce(concept)
        
        # Create Rotor
        rotor = Rotor(concept, RotorConfig(rpm=432.0 * 60, mass=profile.vector_norm))
        rotor.inject_spectrum(profile.spectrum, dynamics=profile.dynamics)
        
        core.harmonic_rotors[concept] = rotor

    # 4. Save to Holographic Field
    print("\n💾 Persisting to Wave Field...")
    core.save_hologram()

    # 5. Crystallize
    print("\n💎 Crystallizing Agentic Principle...")
    crystallizer = CrystallizationEngine("data/principles.json")
    crystallizer.crystallize(core.harmonic_rotors)

    print("\n✅ Agency Internalized. Elysia now understands her own ability to act.")

if __name__ == "__main__":
    internalize_agency()
