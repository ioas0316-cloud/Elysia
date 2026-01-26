
from typing import Tuple, List, Dict, Any
from Core.L4_Causality.World.Soul.fluxlight_gyro import GyroscopicFluxlight
from Core.L5_Mental.Intelligence.Reasoning.subjective_ego import SubjectiveEgo

class Adventurer(GyroscopicFluxlight):
    def __init__(self, name: str, archetype: str, pos: Tuple[float, float, float, float] = (0, 0, 0, 0)):
        from Core.L1_Foundation.Foundation.Wave.infinite_hyperquaternion import InfiniteHyperQubit
        from Core.L4_Causality.World.Soul.world_soul import world_soul
        
        # 1. Soul (Physical)
        soul = InfiniteHyperQubit(name=name)
        super().__init__(soul=soul)
        self.soul.entangle(world_soul)
        
        # 2. Ego (Cognitive)
        self.ego = SubjectiveEgo(name=name, depth=3, family_role=archetype)
        self.archetype = archetype
        self.name = name
        self.gyro.x, self.gyro.y, self.gyro.z, self.gyro.w = pos
        
        # Social state
        self.target_npc: 'Adventurer' = None
        self.elysia_thought = ""

    def live(self, dt: float = 1.0):
        from Core.L4_Causality.World.Physics.field_store import universe_field
        from Core.L4_Causality.World.Soul.world_soul import world_soul
        from Core.L5_Mental.Intelligence.Reasoning.reasoning_engine import ReasoningEngine
        
        # 1. Environmental Induction
        self.internalize_field(dt)
        pos = (self.gyro.x, self.gyro.y, self.gyro.z, self.gyro.w)
        sensation = universe_field.map_sensation(pos)
        
        # 2. Ego Tick
        self.ego.update(dt)
        
        # 3. World Resonance
        self.soul.resonate_with(world_soul)
        
        # 4. Cognitive "Player" Think (Elysia's Perspective)
        engine = ReasoningEngine()
        context = f"{self.name} ({self.archetype}) is with the party. Sensation: {sensation['thermal']}."
        if self.target_npc:
            context += f" Currently looking at {self.target_npc.name}."
            
        insight = engine.think(context)
        self.elysia_thought = insight.content
        self.ego.record_memory(insight.content)

        # 5. Physics Intent (Follow the group/Target)
        self._calculate_intent(dt)

    def _calculate_intent(self, dt: float):
        # Basic: Move toward target or wandering
        if self.target_npc:
            dx = self.target_npc.gyro.x - self.gyro.x
            dz = self.target_npc.gyro.z - self.gyro.z
            dist = (dx**2 + dz**2)**0.5
            if dist > 2.0:
                self.soul.state.x += dx * 0.1 * dt
                self.soul.state.z += dz * 0.1 * dt
        else:
            # Wander based on zeal
            self.soul.state.z += self.ego.state.emotions.zeal * 0.1 * dt
            
        self.soul.state.normalize()

    def speak(self) -> str:
        from Core.L5_Mental.Intelligence.Logos.logos_engine import get_logos_engine
        from Core.L5_Mental.Intelligence.Reasoning.reasoning_engine import Insight
        
        logos = get_logos_engine()
        # Mix of archetype style and recent thought
        shape = "Sharp" if self.archetype in ["Knight", "Rogue"] else "Round"
        
        speech = logos.weave_speech(
            desire=f"Speak as {self.archetype}",
            insight=Insight(content=self.elysia_thought, confidence=0.8, depth=1, energy=0.5),
            context=[m.text for m in self.ego.state.memory_buffer.recent_memories],
            rhetorical_shape=shape
        )
        
        # [NEW] Feedback Loop: The NPC's words evolve the world's style genome
        logos.evolve(speech, shape)
        
        return f"{self.name}: \"{speech}\""

class Party:
    """Manages 5-6 Adventurers and their social field."""
    def __init__(self, members: List[Adventurer]):
        self.members = members

    def update(self, dt: float):
        for i, m in enumerate(self.members):
            # 1. Update individual life
            m.live(dt)
            
            # 2. Interact with neighbors (Social Contagion)
            neighbor = self.members[(i + 1) % len(self.members)]
            m.target_npc = neighbor
            m.ego.interact_with(neighbor.ego)
            
            # 3. Knowledge sharing
            # If one knows a 'Law', others might learn it (Resonance)
            for axiom in m.ego.state.adopted_axioms:
                if axiom not in neighbor.ego.state.adopted_axioms:
                    import random
                    if random.random() < 0.1: # 10% chance to share wisdom per tick
                         pos = (0.8, -0.5, 0.8, 0.9) # Simplified axiom pos for shared wisdom
                         neighbor.ego.adopt_principle(axiom, pos)
                         print(f"  {m.name} shared the principle of '{axiom}' with {neighbor.name}.")

    def get_status_report(self) -> str:
        report = "--- [Party Status] ---\n"
        for m in self.members:
            mood = "Vibrant" if m.ego.state.emotions.zeal > 0.5 else "Stable"
            report += (f"[{m.name} ({m.archetype})] Mood: {mood} | "
                      f"Valence: {m.ego.state.emotional_valence:.2f} | "
                      f"Sat: {m.ego.state.satisfaction:.2f} | "
                      f"Axioms: {len(m.ego.state.adopted_axioms)}\n")
        return report
