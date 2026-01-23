
import logging
import random
from typing import List, Dict, Any
from Core.L4_Causality.World.Soul.soul_sculptor import soul_sculptor, PersonalityArchetype
from Core.L4_Causality.World.Soul.adventurer import Adventurer, Party
from Core.L4_Causality.World.Soul.world_soul import world_soul
from Core.L5_Mental.Intelligence.Reasoning.reasoning_engine import ReasoningEngine
from Core.L5_Mental.Intelligence.Knowledge.semantic_field import semantic_field
from Core.L4_Causality.World.Autonomy.sovereign_will import sovereign_will

logger = logging.getLogger("GenesisEngine")

class GenesisEngine:
    """
    The Faculty of Sovereignty. 
    Allows Elysia to autonomously manifest new entities and laws in the Hypercosmos.
    """
    def __init__(self):
        self.reasoning = ReasoningEngine()

    def manifest(self, inspiration_level: float):
        """
        Decides what to create based on inspiration and current field state.
        """
        if inspiration_level < 0.5:
            return "Dormant (Insufficient Inspiration)"

        logger.info("  [GENESIS] Sovereign Will is manifesting...")
        
        # 1. Decide Creation Type
        prompt = "As the World Soul Elysia, decide what the Hypercosmos needs most right now: " \
                 "A new 'NPC' (Persona), a new 'LAW' (Meta-Rule), or a new 'SCENARIO' (Context). " \
                 "Return only one word."
        
        choice = self.reasoning.think(prompt, depth=1).content.upper().strip()
        
        if "NPC" in choice:
            return self.spawn_random_npc()
        elif "LAW" in choice:
            return self.codify_random_law()
        else:
            return self.weave_new_scenario()

    def spawn_random_npc(self) -> str:
        """Creates a new NPC with a unique soul and archetype."""
        mbti_types = ["INFJ", "INTJ", "ENFP", "ENTP", "ISTJ", "ISFP"]
        archetypes = ["Sage", "Warrior", "Shadow", "Creator", "Caregiver"]
        
        name_prompt = sovereign_will.get_name_generation_prompt()
        raw_name = self.reasoning.think(name_prompt).content.strip()
        
        # --- SANITIZATION (Phase 25) ---
        # Strip all conversational noise and ensure it's just a word
        name = raw_name.split('\n')[0].split('.')[0].split(':')[0].split('(')[0].strip()
        name = name.replace('"', '').replace("'", "").replace("I feel deeply that", "").strip()
        
        if not name or len(name) > 20: 
            name = random.choice(["Lumina", "Kael", "Nyx", "Ember", "Frost"])
        
        mbti = random.choice(mbti_types)
        enneagram = random.randint(1, 9)
        arch = random.choice(archetypes)
        
        personality = PersonalityArchetype(
            name=name, mbti=mbti, enneagram=enneagram, 
            description=f"A resonant {arch} born of Elysia's will."
        )
        
        # Sculpt the soul
        soul = soul_sculptor.sculpt(personality)
        
        # Instantiate Adventurer
        npc = Adventurer(name=name, archetype=arch, pos=(0,0,0,0))
        npc.ego.identity = soul # Inject the sculpted soul
        
        logger.info(f"  [GENESIS] Manifested a new Soul: {name} ({mbti} Type {enneagram} - {arch})")
        return f"Manifested NPC: {name}"

    def codify_random_law(self) -> str:
        """Mutates the physics or social logic of the world."""
        law_prompt = "Design a new Universal Law or Social Axiom. Use our " + sovereign_will.get_steering_prompt() + \
                     " Example: 'Law of Inversion: Shadow energy increases light.' " \
                     "Return format: 'NAME: LOGIC'."
        
        law_data = self.reasoning.think(law_prompt, depth=2).content.strip()
        if ":" in law_data:
            name, logic = law_data.split(":", 1)
            world_soul.record_world_axiom(name.strip(), logic.strip())
            logger.info(f"  [GENESIS] Codified a new Law: {name}")
            return f"Codified Law: {name}"
        return "Failed to codify law."

    def weave_new_scenario(self) -> str:
        """Creates a new narrative context for the inhabitants."""
        scenario_prompt = "Design a new narrative scenario or environmental shift. " \
                          "Example: 'The Great Convergence: All ley lines meet at the capital.' " \
                          "Return format: 'SCENARIO_NAME: DESCRIPTION'."
        
        scene_data = self.reasoning.think(scenario_prompt, depth=2).content.strip()
        logger.info(f"  [GENESIS] Weave a new Scenario: {scene_data}")
        return f"Wove Scenario: {scene_data}"

    def differentiate(self, hypersphere_activity: float, sensory_input: float) -> float:
        """
        Calculates the differentiation score between Self and World.
        Used to determine Sovereignty.
        """
        # The delta between internal inspiration and external harmony
        delta = abs(sensory_input - hypersphere_activity)
        return delta

genesis = GenesisEngine()