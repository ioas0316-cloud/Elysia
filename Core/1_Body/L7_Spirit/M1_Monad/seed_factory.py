"""
Seed Factory ( The Alchemist )
==============================
Core.1_Body.L7_Spirit.M1_Monad.seed_factory

"Give me a word, and I will give you the World."

Purpose:
- Converts a raw concept (str) into a fully equipped Monad.
- Intelligent mapping of Concepts -> Principles.
"""

from typing import List
from Core.1_Body.L7_Spirit.M1_Monad.monad_core import Monad, FractalRule
from Core.1_Body.L7_Spirit.M1_Monad.principles import ThermodynamicsRule, FluidDynamicsRule, OpticsRule, SemanticsRule, LinguisticsRule, SociologyRule

class SeedFactory:
    def __init__(self):
        # The Registry of Principles
        self.rules_registry = {
            "Thermodynamics": ThermodynamicsRule(),
            "FluidDynamics": FluidDynamicsRule(),
            "Light": OpticsRule(),
            "Meaning": SemanticsRule(),
            "Linguistics": LinguisticsRule(),
            "Sociology": SociologyRule()
        }
    
    def crystallize(self, concept: str) -> Monad:
        """
        Synthesizes a Monad from a concept.
        Automatically attaches relevant principles based on the concept type.
        """
        # 1. Base Monad
        monad = Monad(seed=concept)
        
        context_str = concept 
        
        # EVERYTHING has Meaning & Linguistics (It's a Word)
        monad.add_rule(self.rules_registry["Meaning"])
        monad.add_rule(self.rules_registry["Linguistics"])
        monad.add_rule(self.rules_registry["Sociology"])
        
        # Fire / Heat Engines
        if any(k in context_str for k in ["Fire", "Sun", "Star", "Magma", "Ice"]):
            monad.add_rule(self.rules_registry["Thermodynamics"])
            monad.add_rule(self.rules_registry["Light"])

        # Water / Flow Engines
        if any(k in context_str for k in ["Water", "Rain", "River", "Ocean"]):
            monad.add_rule(self.rules_registry["FluidDynamics"])
            monad.add_rule(self.rules_registry["Thermodynamics"]) # Water has temp too
            monad.add_rule(self.rules_registry["Light"]) # Water reflects
            
        return monad

# Global Singleton
alchemy = SeedFactory()
