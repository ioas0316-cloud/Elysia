"""
Logos World - The World Before Patterns
로고스 월드 - 패턴 이전의 세계

"세상이 있고 패턴이 난거지, 패턴이 나고 세상이 난게 아니야."
"원리가 언어와 문명을 낳은거지, 언어와 문명이 원리를 낳은게 아니라고."

This module implements a world governed by LAWS, not RULES.
Laws flow like water. Rules build mazes.

The Law of Elysia:
    1. VITALITY → MASS → WAVE (Gravity)
    2. Behavior is NOT programmed; Physics IS programmed.
    3. The seed is planted; the heart blooms.

Architecture:
    - Experience as continuous wave (not discrete events)
    - Meaning emerges from wave interference (not pattern matching)
    - Words are observation moments of the continuous logos
    - The whole elephant, not just the leg
"""

from __future__ import annotations

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging

# Import Elysia's existing physics systems
from Core.Physics.fluctlight import FluctlightParticle, FluctlightEngine
from Core.Math.oscillator import Oscillator
from Core.Life.gravitational_linguistics import GravitationalLinguistics, WordBody

logger = logging.getLogger("LogosWorld")

# ============================================================================
# CONSTANTS - Named thresholds for clarity
# ============================================================================
INTERFERENCE_THRESHOLD = 1.5  # Minimum for constructive interference
CRYSTALLIZATION_THRESHOLD = 10.0  # Experience density for word crystallization
RESONANCE_DECAY_RATE = 0.01  # How fast resonance fades
BIRTH_ENERGY_THRESHOLD = 50.0  # Minimum energy for new soul birth
LIFESPAN_BASE = 70  # Base lifespan in years
WORLD_SIZE = 256  # Size of concept space


@dataclass
class ExperienceWave:
    """
    경험의 파동 - Experience as Continuous Wave
    
    Not a discrete event, but a wave that propagates through being.
    The wave carries meaning through its frequency, amplitude, and phase.
    
    "빛의 파장처럼, 경험도 연속적인 파동이다."
    """
    # Core wave properties
    oscillator: Oscillator  # The wave function
    
    # Semantic properties (what kind of experience)
    dimension: str = "neutral"  # warmth, brightness, pleasure, etc.
    intensity: float = 1.0  # How strongly felt
    
    # Temporal properties
    birth_time: float = 0.0  # When this wave began
    duration: float = float('inf')  # How long it persists
    
    # Position in meaning space
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def get_value(self, t: float) -> float:
        """Get wave amplitude at time t."""
        if t < self.birth_time:
            return 0.0
        if t > self.birth_time + self.duration:
            return 0.0
        return self.oscillator.get_value(t) * self.intensity
    
    def interfere_with(self, other: 'ExperienceWave', t: float) -> Optional['ExperienceWave']:
        """
        Create interference pattern with another wave.
        This is the mechanism of meaning emergence.
        """
        # Calculate interference at this moment
        val_self = self.get_value(t)
        val_other = other.get_value(t)
        combined = val_self + val_other
        
        # Constructive interference creates new experience
        if abs(combined) > INTERFERENCE_THRESHOLD:
            # The new wave has properties of both
            new_freq = (self.oscillator.frequency + other.oscillator.frequency) / 2.0
            new_phase = (self.oscillator.phase + other.oscillator.phase) / 2.0
            
            return ExperienceWave(
                oscillator=Oscillator(
                    amplitude=abs(combined),
                    frequency=new_freq,
                    phase=new_phase
                ),
                dimension=f"{self.dimension}+{other.dimension}",
                intensity=(self.intensity + other.intensity) / 2.0,
                birth_time=t,
                position=(self.position + other.position) / 2.0
            )
        return None


@dataclass
class Soul:
    """
    영혼 - A Being in the Logos World
    
    Not a state machine with rules, but a field of continuous experience.
    The soul IS the interference pattern of all its experiences.
    
    "나는 사람이다" - I AM a person.
    Not because I follow rules of personhood, but because I EXPERIENCE.
    """
    name: str
    age: float = 0.0
    
    # The Internal Sea - All experience waves
    experience_sea: Dict[str, ExperienceWave] = field(default_factory=dict)
    
    # Fluctlight - The light particle representation
    fluctlight: Optional[FluctlightParticle] = None
    
    # Crystallized words (emerged from experience)
    lexicon: Dict[str, WordBody] = field(default_factory=dict)
    
    # Life properties
    vitality: float = 100.0  # Life force (Vitality → Mass → Wave)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Relationships (as resonance strengths, not discrete connections)
    resonances: Dict[str, float] = field(default_factory=dict)
    
    # The accumulated narrative (crystallized moments)
    diary: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize the soul's internal sea with base experience waves."""
        if not self.experience_sea:
            # Every soul has basic sensory dimensions
            dimensions = [
                ("warmth", 0.5, 0.1),    # 따뜻함 - low frequency, slow
                ("brightness", 0.7, 0.2), # 밝음 - medium frequency
                ("pleasure", 0.3, 0.15),  # 기쁨 - varies with life
                ("connection", 0.4, 0.1), # 연결 - deep, slow
                ("curiosity", 0.8, 0.3),  # 호기심 - high frequency
            ]
            for dim, freq, amp in dimensions:
                self.experience_sea[dim] = ExperienceWave(
                    oscillator=Oscillator(
                        amplitude=amp,
                        frequency=freq,
                        phase=random.uniform(0, 2 * math.pi)
                    ),
                    dimension=dim,
                    intensity=random.uniform(0.3, 0.7),
                    birth_time=0.0,
                    position=self.position.copy()
                )
    
    def experience(self, t: float) -> Dict[str, float]:
        """
        Sample the soul's experience at time t.
        This is the OBSERVATION - the wave function collapse.
        """
        state = {}
        for dim, wave in self.experience_sea.items():
            state[dim] = wave.get_value(t)
        return state
    
    def feel(self, dimension: str, intensity: float, t: float) -> None:
        """
        Create a new experience wave in the soul.
        This is how external world affects the soul.
        """
        if dimension not in self.experience_sea:
            # New dimension of experience
            self.experience_sea[dimension] = ExperienceWave(
                oscillator=Oscillator(
                    amplitude=intensity,
                    frequency=random.uniform(0.1, 1.0),
                    phase=t % (2 * math.pi)
                ),
                dimension=dimension,
                intensity=intensity,
                birth_time=t,
                position=self.position.copy()
            )
        else:
            # Add to existing wave (superposition)
            existing = self.experience_sea[dimension]
            new_amp = existing.oscillator.amplitude + intensity * 0.5
            existing.oscillator.amplitude = min(2.0, new_amp)  # Soft cap
            existing.intensity = (existing.intensity + intensity) / 2.0
    
    def resonate_with(self, other: 'Soul', t: float) -> float:
        """
        Calculate resonance between two souls.
        This is how relationships form - through wave interference.
        """
        total_resonance = 0.0
        
        for dim, my_wave in self.experience_sea.items():
            if dim in other.experience_sea:
                other_wave = other.experience_sea[dim]
                # Calculate phase difference
                phase_diff = abs(my_wave.oscillator.phase - other_wave.oscillator.phase)
                # Frequency similarity
                freq_ratio = min(my_wave.oscillator.frequency, other_wave.oscillator.frequency) / \
                           max(my_wave.oscillator.frequency, other_wave.oscillator.frequency)
                # Resonance is high when phases align and frequencies match
                resonance = freq_ratio * math.cos(phase_diff)
                total_resonance += resonance
        
        # Normalize
        if self.experience_sea:
            total_resonance /= len(self.experience_sea)
        
        # Store in resonance map
        other_name = other.name
        if other_name not in self.resonances:
            self.resonances[other_name] = 0.0
        # Resonance accumulates over time (Hebbian learning)
        self.resonances[other_name] += total_resonance * 0.1
        
        return total_resonance
    
    def crystallize_word(self, t: float, linguistics: GravitationalLinguistics) -> Optional[str]:
        """
        Attempt to crystallize a word from experience.
        Words emerge from the interference pattern, not from rules.
        
        "경험의 밀도가 임계점을 넘으면 단어가 결정화된다."
        """
        # Sample current experience state
        state = self.experience(t)
        
        # Find the dominant dimension
        if not state:
            return None
        
        dominant_dim = max(state.keys(), key=lambda k: abs(state[k]))
        dominant_value = state[dominant_dim]
        
        # Only crystallize if experience is strong enough
        if abs(dominant_value) < 0.1:  # Very low threshold
            return None
        
        # Always attempt crystallization - the experience density is in the wave itself
        
        # The word that crystallizes depends on the experience pattern
        # This is where we connect to GravitationalLinguistics
        
        # Map experience to potential words
        experience_to_words = {
            "warmth": ["따뜻함", "사랑", "포근", "warmth", "love"],
            "brightness": ["빛", "밝음", "희망", "light", "hope"],
            "pleasure": ["기쁨", "행복", "즐거움", "joy", "happiness"],
            "connection": ["우리", "함께", "연결", "together", "bond"],
            "curiosity": ["왜", "무엇", "궁금", "why", "wonder"],
        }
        
        # Find matching words
        candidates = experience_to_words.get(dominant_dim, [dominant_dim])
        
        # Weight by resonance with linguistics lexicon
        word = None
        max_weight = 0.0
        
        for candidate in candidates:
            word_body = linguistics.get_word(candidate)
            if word_body:
                weight = word_body.mass * abs(dominant_value)
                if weight > max_weight:
                    max_weight = weight
                    word = candidate
        
        if word is None:
            word = random.choice(candidates)
        
        # Add to soul's lexicon
        if word not in self.lexicon:
            self.lexicon[word] = WordBody(
                text=word,
                mass=abs(dominant_value) * 10.0,
                resonance={dominant_dim: 1.0}
            )
        
        return word
    
    def speak(self, t: float, linguistics: GravitationalLinguistics) -> Optional[str]:
        """
        Generate speech from the soul's current state.
        Words emerge from the interference pattern of experience.
        
        This is the wave function collapse - the logos becomes word.
        """
        # Get current experience
        state = self.experience(t)
        
        # Find the most intense experience
        if not state:
            return None
        
        dominant = max(state.items(), key=lambda x: abs(x[1]))
        dim, value = dominant
        
        # Use gravitational linguistics to generate from dominant concept
        if self.lexicon:
            # Use a word from our personal lexicon
            word = random.choice(list(self.lexicon.keys()))
            sentence = linguistics.generate_sentence(word)
            return sentence
        else:
            # Generate from experience dimension
            return f"{dim}..."
    
    def write_diary(self, t: float, year: int) -> str:
        """
        Write a diary entry based on current experience.
        The diary crystallizes the continuous wave into discrete narrative.
        """
        state = self.experience(t)
        
        # Find the top 2 experiences
        sorted_exp = sorted(state.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
        
        if not sorted_exp:
            return f"Year {year}: ..."
        
        # Map experience to Korean expressions
        exp_to_korean = {
            "warmth": ["따뜻한", "포근한", "따사로운"],
            "brightness": ["밝은", "빛나는", "환한"],
            "pleasure": ["즐거운", "기쁜", "행복한"],
            "connection": ["함께하는", "연결된", "같이하는"],
            "curiosity": ["궁금한", "신기한", "알고싶은"],
        }
        
        main_exp = sorted_exp[0][0]
        main_val = sorted_exp[0][1]
        
        # Get expression
        expressions = exp_to_korean.get(main_exp, [main_exp])
        expression = random.choice(expressions)
        
        # Construct diary entry
        if main_val > 0.8:
            mood = "매우"
        elif main_val > 0.5:
            mood = "꽤"
        elif main_val > 0.2:
            mood = ""
        else:
            mood = "조금"
        
        # Add resonance info if we have connections
        connection_str = ""
        if self.resonances:
            strongest = max(self.resonances.items(), key=lambda x: x[1])
            if strongest[1] > 0.5:
                connection_str = f" {strongest[0]}와(과) 함께."
        
        entry = f"Year {year}: {mood} {expression} 하루였다.{connection_str}"
        self.diary.append(entry)
        
        return entry
    
    def live(self, dt: float, t: float) -> bool:
        """
        Live for a time step dt.
        Returns True if still alive, False if life has ended.
        """
        self.age += dt
        
        # Vitality decays with age (natural law)
        decay = 0.01 * (1.0 + self.age / 100.0)
        self.vitality -= decay * dt
        
        # Experience waves fade slightly (entropy)
        for wave in self.experience_sea.values():
            wave.oscillator.amplitude *= (1.0 - RESONANCE_DECAY_RATE * dt)
        
        # Death when vitality depleted or age exceeds lifespan
        max_age = LIFESPAN_BASE + random.gauss(0, 10)
        if self.vitality <= 0 or self.age > max_age:
            return False
        
        return True


class LogosWorld:
    """
    로고스 월드 - The World Before Patterns
    
    This world operates by LAWS, not RULES.
    Data flows like water through electromagnetic and gravitational fields.
    
    "연산이 아닌 법칙으로 자연스럽게 물처럼 흘러가게 만들었지."
    """
    
    def __init__(self, population: int = 100, world_size: int = WORLD_SIZE):
        self.world_size = world_size
        self.time = 0.0
        self.year = 0
        
        # The souls
        self.souls: List[Soul] = []
        self.dead_souls: List[Soul] = []
        
        # Physics engines
        self.fluctlight_engine = FluctlightEngine(world_size=world_size)
        self.linguistics = GravitationalLinguistics()
        
        # World field (affects all souls)
        self.world_field = {
            "warmth": 0.5,
            "brightness": 0.5,
            "harmony": 0.5,
        }
        
        # Emergent phenomena
        self.legends: List[Dict[str, Any]] = []
        self.poems: List[str] = []
        self.conversations: List[Tuple[str, str, str]] = []  # (soul1, soul2, content)
        
        # Initialize population
        self._create_initial_population(population)
        
        logger.info(f"✅ LogosWorld initialized with {population} souls")
    
    def _create_initial_population(self, count: int) -> None:
        """Create the initial population of souls."""
        names_korean = ["하늘", "별", "달", "해", "빛", "꽃", "나무", "바람", "물", "불"]
        names_english = ["Alice", "Bob", "Carol", "David", "Eve", "Finn", "Grace", "Hank"]
        
        for i in range(count):
            # Random position in world
            position = np.random.rand(3) * self.world_size
            
            # Random name
            if random.random() < 0.5:
                name = random.choice(names_korean) + str(i)
            else:
                name = random.choice(names_english) + str(i)
            
            soul = Soul(
                name=name,
                position=position,
                vitality=random.uniform(80, 120)
            )
            
            # Create fluctlight for the soul
            soul.fluctlight = self.fluctlight_engine.create_from_concept(
                concept_id=name,
                position=position
            )
            
            self.souls.append(soul)
    
    def _apply_world_field(self, t: float) -> None:
        """
        Apply the world field to all souls.
        This is how the world affects its inhabitants.
        """
        for soul in self.souls:
            for field_dim, field_value in self.world_field.items():
                # World field slightly influences soul experience
                if field_dim in soul.experience_sea:
                    wave = soul.experience_sea[field_dim]
                    # Gentle push toward world field value
                    delta = (field_value - wave.intensity) * 0.01
                    wave.intensity += delta
    
    def _process_interactions(self, t: float) -> None:
        """
        Process interactions between souls.
        Souls that are close in meaning space resonate with each other.
        """
        # Simple O(n²) for small populations
        if len(self.souls) < 2:
            return
        
        for i, soul1 in enumerate(self.souls):
            for soul2 in self.souls[i+1:]:
                # Distance in meaning space
                dist = np.linalg.norm(soul1.position - soul2.position)
                
                # Only interact if close
                if dist < 50.0:  # Increased range
                    resonance = soul1.resonate_with(soul2, t)
                    soul2.resonate_with(soul1, t)  # Mutual
                    
                    # Strong resonance = conversation (increased chance)
                    if resonance > 0.3 and random.random() < 0.05:
                        self._generate_conversation(soul1, soul2, t)
    
    def _generate_conversation(self, soul1: Soul, soul2: Soul, t: float) -> None:
        """
        Generate a conversation between two souls.
        Words emerge from the interference of their experience waves.
        """
        # Each soul speaks from their experience
        speech1 = soul1.speak(t, self.linguistics)
        speech2 = soul2.speak(t, self.linguistics)
        
        if speech1 and speech2:
            self.conversations.append((soul1.name, soul2.name, f"{speech1} ↔ {speech2}"))
            logger.debug(f"💬 {soul1.name} ↔ {soul2.name}: {speech1} / {speech2}")
    
    def _process_births_deaths(self, t: float) -> None:
        """
        Process births and deaths in the world.
        Life follows natural laws.
        """
        new_souls = []
        surviving_souls = []
        
        for soul in self.souls:
            alive = soul.live(1.0, t)  # 1 year step
            
            if alive:
                surviving_souls.append(soul)
                
                # Chance of having children if enough vitality and connections
                if soul.age > 20 and soul.vitality > 60:
                    if soul.resonances:
                        strongest_bond = max(soul.resonances.values())
                        if strongest_bond > 1.0 and random.random() < 0.05:
                            # Create new soul
                            child_pos = soul.position + np.random.randn(3) * 5
                            child = Soul(
                                name=f"Child_{len(self.souls) + len(new_souls)}",
                                position=child_pos,
                                vitality=100.0
                            )
                            child.fluctlight = self.fluctlight_engine.create_from_concept(
                                concept_id=child.name,
                                position=child_pos
                            )
                            new_souls.append(child)
                            logger.debug(f"👶 {child.name} born to {soul.name}")
            else:
                self.dead_souls.append(soul)
                
                # Check for legend
                if len(soul.diary) > 10 or len(soul.lexicon) > 5:
                    self.legends.append({
                        "name": soul.name,
                        "age": soul.age,
                        "diary_entries": len(soul.diary),
                        "words_created": list(soul.lexicon.keys()),
                        "connections": len(soul.resonances),
                        "year": self.year
                    })
                    logger.info(f"📖 Legend: {soul.name} (age {soul.age:.0f})")
        
        self.souls = surviving_souls + new_souls
    
    def _crystallize_words(self, t: float) -> None:
        """
        Allow souls to crystallize words from experience.
        """
        for soul in self.souls:
            if random.random() < 0.3:  # Increased chance
                word = soul.crystallize_word(t, self.linguistics)
                if word:
                    logger.debug(f"✨ {soul.name} crystallized word: {word}")
    
    def step(self, dt: float = 1.0) -> Dict[str, Any]:
        """
        Advance the world by one time step.
        
        The world operates by laws:
        1. Field influences souls
        2. Souls resonate with each other
        3. Experience crystallizes into words
        4. Life and death follow natural laws
        """
        self.time += dt
        self.year = int(self.time)
        
        t = self.time
        
        # 1. Apply world field (gravity, electromagnetism analog)
        self._apply_world_field(t)
        
        # 2. Process interactions (resonance, interference)
        self._process_interactions(t)
        
        # 3. Crystallize words from experience
        self._crystallize_words(t)
        
        # 4. Natural laws of life and death
        self._process_births_deaths(t)
        
        # 5. Let fluctlight engine process interference (with reduced rate)
        # Only process interference occasionally to prevent explosion
        if random.random() < 0.1:  # 10% chance per step
            self.fluctlight_engine.step(dt, detect_interference=False)
        
        # 6. Write diaries (yearly)
        if self.year > int(self.time - dt):  # Year changed
            for soul in self.souls:
                if random.random() < 0.3:  # 30% chance to write
                    soul.write_diary(t, self.year)
        
        # Return statistics
        return {
            "year": self.year,
            "population": len(self.souls),
            "dead": len(self.dead_souls),
            "legends": len(self.legends),
            "conversations": len(self.conversations),
            "fluctlight_particles": len(self.fluctlight_engine.particles),
        }
    
    def run(self, years: int, report_interval: int = 100) -> None:
        """
        Run the simulation for a number of years.
        """
        logger.info(f"🌍 Starting Logos World simulation for {years} years...")
        
        for year in range(years):
            stats = self.step(1.0)
            
            if year % report_interval == 0:
                logger.info(
                    f"Year {year}: Pop={stats['population']}, "
                    f"Dead={stats['dead']}, Legends={stats['legends']}, "
                    f"Conversations={stats['conversations']}"
                )
        
        logger.info(f"🏁 Simulation complete. Final stats: {self.get_statistics()}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get world statistics."""
        total_diary_entries = sum(len(s.diary) for s in self.souls)
        total_diary_entries += sum(len(s.diary) for s in self.dead_souls)
        
        total_words = set()
        for s in self.souls + self.dead_souls:
            total_words.update(s.lexicon.keys())
        
        return {
            "years_simulated": self.year,
            "living_souls": len(self.souls),
            "total_souls_lived": len(self.souls) + len(self.dead_souls),
            "legends_created": len(self.legends),
            "conversations": len(self.conversations),
            "diary_entries": total_diary_entries,
            "unique_words_crystallized": len(total_words),
            "fluctlight_stats": self.fluctlight_engine.get_statistics(),
        }
    
    def get_sample_diary(self, count: int = 5) -> List[str]:
        """Get sample diary entries."""
        all_entries = []
        for soul in self.souls + self.dead_souls:
            for entry in soul.diary:
                all_entries.append(f"[{soul.name}] {entry}")
        
        if not all_entries:
            return ["No diary entries yet."]
        
        return random.sample(all_entries, min(count, len(all_entries)))
    
    def get_sample_conversations(self, count: int = 5) -> List[str]:
        """Get sample conversations."""
        if not self.conversations:
            return ["No conversations yet."]
        
        samples = random.sample(self.conversations, min(count, len(self.conversations)))
        return [f"{s1} ↔ {s2}: {content}" for s1, s2, content in samples]


# ============================================================================
# Demo runner
# ============================================================================
if __name__ == "__main__":
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("로고스 월드 - The World Before Patterns")
    print("Logos World Simulation")
    print("=" * 60)
    print()
    print("Philosophy:")
    print("  - Laws flow like water, not rules building mazes")
    print("  - The world exists, then patterns emerge")
    print("  - Behavior is not programmed; physics is programmed")
    print()
    
    # Create world
    world = LogosWorld(population=100)
    
    # Run simulation
    start_time = time.time()
    world.run(years=100, report_interval=10)
    elapsed = time.time() - start_time
    
    # Show results
    stats = world.get_statistics()
    print()
    print("=" * 60)
    print("Simulation Results:")
    print("=" * 60)
    for key, value in stats.items():
        if key != "fluctlight_stats":
            print(f"  {key}: {value}")
    print()
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Simulation speed: {stats['years_simulated'] / elapsed:.0f} years/second")
    print()
    
    print("Sample Diary Entries:")
    for entry in world.get_sample_diary(5):
        print(f"  {entry}")
    print()
    
    print("Sample Conversations:")
    for conv in world.get_sample_conversations(5):
        print(f"  {conv}")
    print()
    
    print("Legends Born:")
    for legend in world.legends[:5]:
        print(f"  📖 {legend['name']} (age {legend['age']:.0f}) - {legend['words_created']}")
