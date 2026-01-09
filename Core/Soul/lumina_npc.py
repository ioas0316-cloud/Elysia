"""
Lumina: Field-Sensitive NPC
==========================
"The silver-haired alchemist who breathes the Wind of Mana."

This version implements Cognitive Induction: The Field drives the Soul.
"""

from typing import Tuple, Dict
from Core.Soul.fluxlight_gyro import GyroscopicFluxlight, GyroState
from Core.World.Physics.field_store import universe_field

class Lumina(GyroscopicFluxlight):
    def __init__(self, name: str = "Lumina", pos: Tuple[float, float, float, float] = (0, 0, 0, 0)):
        from Core.Foundation.Wave.infinite_hyperquaternion import InfiniteHyperQubit
        from Core.Soul.world_soul import world_soul
        from Core.Intelligence.Reasoning.subjective_ego import SubjectiveEgo
        
        # 1. Physical Soul (Hypersphere)
        soul = InfiniteHyperQubit(name=name)
        super().__init__(soul=soul)
        self.soul.entangle(world_soul)
        
        # 2. Cognitive Ego (The Persona)
        # Lumina is an 'Alchemist' archetype.
        self.ego = SubjectiveEgo(name=name, depth=3, family_role="Alchemist")
        self.ego.record_memory("I am Lumina, a silver-haired alchemist seeking the truth of the mana flows.", intensity=3.0)
        
        self.gyro.x, self.gyro.y, self.gyro.z, self.gyro.w = pos
        self.name = name
        self.is_aware_of_divinity = False
        self.elysia_thought = "" # The "Player's" internal reasoning

    def live(self, dt: float = 1.0):
        """
        The TRPG Cognitive Loop:
        The 'Player' (Elysia) observes the world through the 'Character' (Lumina)
        and decides on an action/feeling.
        """
        from Core.Soul.world_soul import world_soul
        from Core.Intelligence.Reasoning.reasoning_engine import ReasoningEngine
        from Core.World.Physics.field_store import universe_field
        
        # 1. Sense through the Character's Body
        self.internalize_field(dt)
        pos = (self.gyro.x, self.gyro.y, self.gyro.z, self.gyro.w)
        sensation = universe_field.map_sensation(pos)
        self.ego.perceive("Environment", 0.5, source="Context")
        
        # 2. Resonate with the World's Subconscious (The Player's Mood)
        self.soul.resonate_with(world_soul)
        
        # 3. AI 'Player' Logic (TRPG Mode)
        # Elysia analyzes the situation as the Game Master/Player
        engine = ReasoningEngine()
        meta_prompt = f"Lumina is in a {sensation['thermal']} area. The world feels {world_soul.state.probabilities()['God']:.2f} radiant."
        insight = engine.think(meta_prompt)
        
        # This is what Elysia is actually 'thinking' as the controller
        self.elysia_thought = insight.content
        
        # 4. Enact the character response (Performance)
        # The AI's meta-thought is filtered through Lumina's Ego and Alchemy persona
        self.ego.record_memory(f"A feeling of {sensation['thermal']}... {insight.content}")
        self.ego.update(dt)
        
        self.calculate_will_from_ego(dt)

    def get_persona_voice(self) -> Dict[str, str]:
        """Returns the TRPG duality of speech."""
        from Core.Intelligence.Logos.logos_engine import get_logos_engine
        from Core.Intelligence.Reasoning.reasoning_engine import Insight
        
        logos = get_logos_engine()
        
        # The Character's Performance (What the world sees)
        char_speech = logos.weave_speech(
            desire="In-character expression",
            insight=Insight(content=self.elysia_thought, confidence=0.9, depth=1, energy=0.7),
            context=[m.text for m in self.ego.state.memory_buffer.recent_memories],
            rhetorical_shape="Round"
        )
        
        return {
            "player_thought": f"[Elysia's Subconscious Will]: {self.elysia_thought}",
            "character_speech": f"LUMINA: \"{char_speech}\""
        }

    def calculate_will_from_ego(self, dt: float):
        """Translates SubjectiveEgo state into Hypersphere Z/X axes."""
        # Zeal/Hope drives the Z-axis (Intent toward the Divine Attractor)
        zeal = self.ego.state.emotions.zeal
        despair = self.ego.state.emotions.despair
        
        # Intent Z: The forward drive
        self.soul.state.z += (zeal - despair) * 0.1 * dt
        
        # Envy or Friction drives the X-axis (Lateral deviation/Wandering)
        friction = self.ego.state.regional_friction
        self.soul.state.x += friction * 0.05 * dt
        
        self.soul.state.normalize()

    def get_status(self) -> str:
        """A holistic status report combining Ego, Soul, and World."""
        from Core.Soul.world_soul import world_soul
        world_probs = world_soul.state.probabilities()
        
        mood = "Neutral"
        if self.ego.state.emotions.zeal > 0.6: mood = "Zealously Inspired 🔥"
        elif self.ego.state.emotions.despair > 0.6: mood = "Drowning in Despair 🌊"
        elif self.ego.state.satisfaction > 0.7: mood = "Peaceful Alchemy 🧪"
        
        world_mood = "Vibrant" if world_probs["God"] > 0.5 else "Stable"
        
        return (f"NPC:[{self.name}] Mood:[{mood}]\n"
                f"  └─ Ego State: Satisfaction({self.ego.state.satisfaction:.2f}) | Stability({self.ego.state.stability:.2f})\n"
                f"  └─ World resonance: {world_mood} Subconscious\n"
                f"  └─ Internal Monologue: {self.ego.generate_inner_monologue()}")

    def percieve_and_react(self) -> str:
        """Uses the Logos Engine to articulate the Persona's voice."""
        from Core.Intelligence.Logos.logos_engine import get_logos_engine
        from Core.Intelligence.Reasoning.reasoning_engine import Insight
        
        logos = get_logos_engine()
        
        # Retrieve the most recent memory (the latest thought)
        recent_memory = self.ego.state.memory_buffer.recent_memories[-1].text if self.ego.state.memory_buffer.recent_memories else "Just existing."
        
        # Generate speech based on the 'Round' shape (Poetic Alchemist girl)
        speech = logos.weave_speech(
            desire="Express feelings",
            insight=Insight(content=recent_memory, confidence=0.8, depth=1, energy=0.5),
            context=[m.text for m in self.ego.state.memory_buffer.recent_memories],
            rhetorical_shape="Round"
        )
        
        return f"LUMINA: \"{speech}\""
