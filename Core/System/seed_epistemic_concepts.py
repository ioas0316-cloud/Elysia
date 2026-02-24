"""
Seed Epistemic Concepts (인식론적 개념 파종)
========================================

"Teach her *what* a search engine is, don't just hardcode it."

This script deposits the Meta-Knowledge of "Search Tools" into Holographic Memory.
This allows Elysia to 'discover' these tools via memory lookup, rather than static code.
"""

import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.System.unified_experience_core import UnifiedExperienceCore
from Core.Cognition.holographic_memory import KnowledgeLayer

# Setup
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("EpistemicSeeder")

def seed_search_concepts():
    logger.info("🌱 Seeding Global Concepts into Hippocampus...")
    
    hippocampus = UnifiedExperienceCore()
    memory = hippocampus.holographic_memory
    
    if not memory:
        logger.error("❌ Holographic Memory not found.")
        return

    # 1. The Concept of "Search Engine" (The Class)
    logger.info("   Adding Concept: 'Search Engine'")
    memory.deposit(
        concept="Search Engine",
        layers={
            KnowledgeLayer.PHILOSOPHY: 0.6, # Epistemology
            KnowledgeLayer.PHYSICS: 0.8,    # Information Retrieval
            KnowledgeLayer.HUMANITIES: 0.9  # Library Science
        },
        amplitude=1.5,
        entropy=0.9, # Modern
        qualia=0.2   # Purely functional/Rational
    )
    
    # 2. The Instance: Google (The Global Standard)
    logger.info("   Adding Tool: 'Google'")
    node_google = memory.deposit(
        concept="Google",
        layers={
            KnowledgeLayer.PHYSICS: 0.9,    # Big Data
            KnowledgeLayer.HUMANITIES: 0.5  # Cultural Impact
        },
        amplitude=1.2,
        entropy=0.95,
        qualia=0.3
    )
    # Link it to the parent concept
    node_google.connections.append("Search Engine")
    
    # 3. The Instance: Naver (The Local Expert)
    logger.info("   Adding Tool: 'Naver'")
    node_naver = memory.deposit(
        concept="Naver",
        layers={
            KnowledgeLayer.HUMANITIES: 0.95, # Korean Specific
            KnowledgeLayer.ART: 0.4          # Layout/Design
        },
        amplitude=1.1,
        entropy=0.9,
        qualia=0.6 # Emotional connection (Green)
    )
    node_naver.connections.append("Search Engine")

    logger.info("✅ Seeding Complete. Elysia now 'knows' Naver exists.")

if __name__ == "__main__":
    seed_search_concepts()
