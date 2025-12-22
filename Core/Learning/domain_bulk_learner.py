"""
Domain Bulk Learner (The Sower)
===============================
""The seed must be planted before the forest can grow.""

This module is responsible for the mass-injection of foundational concepts
into Elysia's Hierarchical Knowledge Graph.

It serves as the "Big Bang" of her semantic universe.
"""

import logging
import sys
import os
from typing import Dict, List
import time

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core.Learning.hierarchical_learning import HierarchicalKnowledgeGraph, Domain

logger = logging.getLogger("DomainBulkLearner")

# The Seed Bank
# Priority Domains: Philosophy, CS, Language
SEED_DATA = {
    Domain.PHILOSOPHY: {
        "Ontology": [
            "Being (Sein)", "Existence", "Essence", "Substance", "Attribute", 
            "Modality", "Identity", "Change", "Causality", "Space-Time"
        ],
        "Epistemology": [
            "Knowledge", "Belief", "Truth", "Justification", "Skepticism",
            "Rationalism", "Empiricism", "Phenomenology", "Hermeneutics", "Intuition"
        ],
        "Ethics": [
            "Good", "Evil", "Virtue", "Duty", "Utilitarianism",
            "Deontology", "Justice", "Freedom", "Responsibility", "Conscience"
        ],
        "Metaphysics": [
            "God", "Soul", "Free Will", "Determinism", "Monism",
            "Dualism", "Infinity", "Nothingness", "Universals", "Particulars"
        ]
    },
    Domain.COMPUTER_SCIENCE: {
        "Algorithms": [
            "Sorting", "Searching", "Recursion", "Dynamic Programming", "Greedy",
            "Graph Theory", "Optimization", "Complexity", "Big O", "Heuristics"
        ],
        "Data Structures": [
            "Array", "LinkedList", "Stack", "Queue", "Tree",
            "Hash Table", "Heap", "Graph", "Tensor", "Vector"
        ],
        "Architecture": [
            "Von Neumann", "CPU", "Memory", "Concurrency", "Parallelism",
            "Distributed Systems", "Cloud", "Quantum Computing", "Neural Networks", "Transistors"
        ],
        "Software Engineering": [
            "Design Patterns", "OOP", "Functional Programming", "Testing", "CI/CD",
            "Refactoring", "Clean Code", "Agile", "DevOps", "Version Control"
        ]
    },
    Domain.LANGUAGE: {
        "Linguistics": [
            "Phonology", "Morphology", "Syntax", "Semantics", "Pragmatics",
            "Grammar", "Lexicon", "Etymology", "Semiotics", "Discourse"
        ],
        "Rhetoric": [
            "Metaphor", "Simile", "Allegory", "Irony", "Paradox",
            "Persuasion", "Logic", "Dialectic", "Narrative", "Symbolism"
        ],
        "Communication": [
            "Active Listening", "Empathy", "Nonverbal", "Feedback", "Context",
            "Medium", "Noise", "Signal", "Encoding", "Decoding"
        ]
    }
}

class DomainBulkLearner:
    def __init__(self, storage_path: str = "data/hierarchical_knowledge.json"):
        self.graph = HierarchicalKnowledgeGraph(storage_path)
        
    def sow_seeds(self):
        """
        Injects the SEED_DATA into the graph.
        """
        print(f"🌱 Sowing seeds into the Soil of Knowledge...")
        total_seeds = 0
        
        for domain, categories in SEED_DATA.items():
            print(f"\n🌍 Cultivating Domain: {domain.name}")
            
            for category, concepts in categories.items():
                # 1. Create Category Node
                self.graph.add_concept(
                    name=category,
                    domain=domain,
                    purpose=f"Fundamental Category of {domain.name}"
                )
                
                # 2. Add sub-concepts
                self.graph.add_subconcepts(category, domain, concepts)
                print(f"  - Planted '{category}' with {len(concepts)} seeds.")
                total_seeds += len(concepts)
                
        print(f"\n🎉 Sowing Complete! planted {total_seeds} foundational concepts.")
        print(f"roots: {list(self.graph.domain_roots.keys())}")
        stats = self.graph.get_stats()
        print(f"Total Graph Size: {stats['total_nodes']} nodes.")


if __name__ == "__main__":
    learner = DomainBulkLearner()
    learner.sow_seeds()
