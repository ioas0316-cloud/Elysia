"""
Fractal Soul World (         )
=====================================

"          " -           

     :
1.                  "       "     
2.   (  /  )    (  /  )       
3.              ,       /       
4.       ,        ,       ,       

                     .
"""

from __future__ import annotations

import random
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("FractalSoulWorld")


# =============================================================================
# Configuration Constants
# =============================================================================

# Simulation probabilities
DAILY_INTERACTION_PROB = 0.1      # Probability of two souls meeting each day
DIARY_WRITE_PROB = 0.05           # Probability of writing diary each day
SOCIAL_ACTIVITY_PROB = 0.3        # Probability of social activity when lonely

# Life event parameters
BASE_DEATH_PROB = 0.0001          # Base daily death probability
ELDER_AGE_THRESHOLD = 60          # Age when death probability increases
AGE_DEATH_FACTOR = 0.1            # Death probability increase per year over elder age
ELF_LONGEVITY_FACTOR = 0.3        # Elves have 30% death rate of humans

# Population parameters
MAX_BIRTHS_PER_YEAR = 5           # Maximum new souls born per year


# =============================================================================
# 1.        (Soul Heart) -           
# =============================================================================

@dataclass
class SoulHeart:
    """
           -           
    
          ,           
    8        : [  ,   ,   ,   ,    ,   ,  /  ,   ]
    """
    current_state: List[float] = field(default_factory=lambda: [0.5] * 8)
    
    #      
    hunger: float = 0.0
    thirst: float = 0.0
    fatigue: float = 0.0
    loneliness: float = 0.0
    
    #       
    total_experiences: int = 0
    emotional_memory: List[float] = field(default_factory=list)
    
    def beat(self, world_input: Dict[str, float] = None) -> List[float]:
        """
              -           
        
        Returns:          (8     )
        """
        world_input = world_input or {}
        
        #             
        self.current_state[6] -= self.hunger * 0.1  #        
        self.current_state[6] -= self.fatigue * 0.1  #        
        self.current_state[4] -= self.loneliness * 0.15  #            
        
        #      
        for key, value in world_input.items():
            if key == "warmth":
                self.current_state[0] = value
            elif key == "brightness":
                self.current_state[1] = value
            elif key == "social":
                self.current_state[4] = value
                self.loneliness = max(0, self.loneliness - value * 0.1)
            elif key == "food":
                self.hunger = max(0, self.hunger - value)
                self.current_state[6] += value * 0.2
            elif key == "rest":
                self.fatigue = max(0, self.fatigue - value)
        
        #      
        self.hunger = min(1.0, self.hunger + 0.01)
        self.fatigue = min(1.0, self.fatigue + 0.005)
        self.loneliness = min(1.0, self.loneliness + 0.008)
        
        #       
        self.current_state = [max(-1, min(1, x)) for x in self.current_state]
        
        #      
        self.total_experiences += 1
        if len(self.emotional_memory) < 100:
            self.emotional_memory.append(self.current_state[6])  #  /     
        
        return self.current_state
    
    def get_dominant_feeling(self) -> str:
        """         """
        pleasure = self.current_state[6]
        arousal = self.current_state[7]
        
        if pleasure > 0.5 and arousal > 0.5:
            return "excited"
        elif pleasure > 0.5 and arousal <= 0.5:
            return "peaceful"
        elif pleasure <= 0.5 and arousal > 0.5:
            return "anxious"
        else:
            return "melancholy"


# =============================================================================
# 2.        (Soul Mind) -       
# =============================================================================

class SoulMind:
    """
           -               
    
                 /      
    """
    
    def __init__(self):
        #      
        self.personal_vocabulary: Dict[str, int] = {}  #   :      
        self.favorite_expressions: List[str] = []
        
        #         
        self.language_level = 0  # 0:   , 1:   , 2:   , 3:   
        
        #         
        self.expressions = {
            "hungry": ["   ...", "        ", "         "],
            "tired": ["   ...", "       ", "     "],
            "lonely": ["   ...", "         ", "        "],
            "happy": ["     !", "   ~", "     "],
            "sad": ["  ...", "      ", "     "],
            "peaceful": ["    ", "   ", "      "],
            "excited": ["  !", "    ", "   !"],
            "anxious": ["   ...", "   ", "       "],
        }
        
        #         
        self.activity_expressions = {
            "eating": ["   !", "     ", "    ~"],
            "resting": ["    ", "   ", "     !"],
            "socializing": ["    ", "     !", "        "],
            "working": ["      ", "           ", "        "],
            "music": ["       ", "      ", "       "],
            "nature": ["    ", "      ", "    "],
        }
    
    def express_state(self, heart: SoulHeart) -> str:
        """              """
        expressions = []
        
        #         
        if heart.hunger > 0.7:
            expressions.append(random.choice(self.expressions["hungry"]))
        if heart.fatigue > 0.7:
            expressions.append(random.choice(self.expressions["tired"]))
        if heart.loneliness > 0.7:
            expressions.append(random.choice(self.expressions["lonely"]))
        
        #         
        feeling = heart.get_dominant_feeling()
        if feeling in self.expressions:
            expressions.append(random.choice(self.expressions[feeling]))
        
        if not expressions:
            expressions.append("...")
        
        return " ".join(expressions[:2])  #    2    
    
    def express_activity(self, activity: str) -> str:
        """         """
        if activity in self.activity_expressions:
            return random.choice(self.activity_expressions[activity])
        return f"{activity}  "
    
    def write_diary(self, heart: SoulHeart, events: List[str]) -> str:
        """     """
        feeling = heart.get_dominant_feeling()
        feeling_kr = {
            "excited": "   ",
            "peaceful": "    ", 
            "anxious": "   ",
            "melancholy": "   "
        }.get(feeling, "   ")
        
        diary = f"    {feeling_kr}     . "
        
        if events:
            diary += " ".join(events[:3])
        
        #          
        if heart.loneliness > 0.5:
            diary += "           ."
        elif heart.current_state[6] > 0.6:
            diary += "         ."
        
        return diary
    
    def think(self, heart: SoulHeart) -> str:
        """      """
        thoughts = [
            "       ...",
            "             ",
            "         ",
            "             ",
            "       ",
            "             ",
            "       ",
        ]
        
        #          
        if heart.hunger > 0.8:
            return "   ...            "
        if heart.loneliness > 0.8:
            return "             ..."
        
        feeling = heart.get_dominant_feeling()
        if feeling == "excited":
            return random.choice(["                   !", "      !"])
        elif feeling == "melancholy":
            return random.choice(["          ...", "       "])
        
        return random.choice(thoughts)


# =============================================================================
# 3.        (Fractal Soul) -       
# =============================================================================

@dataclass
class FractalSoul:
    """
           - "       "          
    
                    ,                 .
                               .
    """
    id: int
    name: str
    birth_year: int
    
    #       
    heart: SoulHeart = field(default_factory=SoulHeart)
    mind: SoulMind = field(default_factory=SoulMind)
    
    #      
    race: str = "Human"
    profession: str = "Villager"
    location: str = "village"
    
    #   
    relationships: Dict[int, float] = field(default_factory=dict)  # id:    
    family: List[int] = field(default_factory=list)
    friends: List[int] = field(default_factory=list)
    
    #   
    diary_entries: List[str] = field(default_factory=list)
    memories: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    
    #   
    is_alive: bool = True
    death_year: Optional[int] = None
    
    def get_age(self, current_year: int) -> int:
        return current_year - self.birth_year
    
    def live_day(self, current_year: int, world_context: Dict = None) -> Dict[str, Any]:
        """
               
        
        Returns:          
        """
        world_context = world_context or {}
        daily_record = {
            "name": self.name,
            "year": current_year,
            "events": [],
            "thoughts": [],
            "expressions": [],
        }
        
        # 1.       -            
        self.heart.beat(world_context)
        
        # 2.       
        thought = self.mind.think(self.heart)
        daily_record["thoughts"].append(thought)
        
        # 3.      
        activities = self._daily_activities(current_year)
        daily_record["events"].extend(activities)
        
        # 4.      
        expression = self.mind.express_state(self.heart)
        daily_record["expressions"].append(expression)
        
        # 5.          (5%   )
        if random.random() < 0.05:
            diary = self.mind.write_diary(self.heart, activities)
            self.diary_entries.append(f"Year {current_year}: {diary}")
            daily_record["diary"] = diary
        
        return daily_record
    
    def _daily_activities(self, current_year: int) -> List[str]:
        """     """
        activities = []
        
        #   
        if self.heart.hunger > 0.5:
            self.heart.beat({"food": 0.8})
            activities.append(self.mind.express_activity("eating"))
        
        #   
        if self.heart.fatigue > 0.6:
            self.heart.beat({"rest": 0.7})
            activities.append(self.mind.express_activity("resting"))
        
        #      
        if self.heart.loneliness > 0.4 and random.random() < 0.3:
            self.heart.beat({"social": 0.6})
            activities.append(self.mind.express_activity("socializing"))
        
        #    
        if random.random() < 0.5:
            activities.append(self.mind.express_activity("working"))
        
        #      
        if random.random() < 0.2:
            leisure = random.choice(["music", "nature"])
            activities.append(self.mind.express_activity(leisure))
        
        return activities
    
    def interact_with(self, other: 'FractalSoul') -> Tuple[str, str]:
        """           """
        
        #           
        intimacy = self.relationships.get(other.id, 0.3)
        
        my_feeling = self.heart.get_dominant_feeling()
        other_feeling = other.heart.get_dominant_feeling()
        
        #      
        greetings = {
            "excited": ["  !         !", "   ~"],
            "peaceful": ["  ,     ?", "      "],
            "anxious": [" ...   ...", "       ..."],
            "melancholy": ["...  ", "     ..."],
        }
        
        responses = {
            "excited": ["  !      !", "        !"],
            "peaceful": ["  ,        ", "   "],
            "anxious": ["       ?", "   ?"],
            "melancholy": ["...     ", "  "],
        }
        
        my_line = random.choice(greetings.get(my_feeling, ["  "]))
        other_line = random.choice(responses.get(other_feeling, [" "]))
        
        #       
        if other.id not in self.relationships:
            self.relationships[other.id] = 0.2
        self.relationships[other.id] = min(1.0, self.relationships[other.id] + 0.05)
        
        #       
        self.heart.loneliness = max(0, self.heart.loneliness - 0.1)
        other.heart.loneliness = max(0, other.heart.loneliness - 0.1)
        
        return my_line, other_line
    
    def get_summary(self) -> str:
        """     """
        feeling = self.heart.get_dominant_feeling()
        feeling_kr = {
            "excited": "   ",
            "peaceful": "    ",
            "anxious": "   ",
            "melancholy": "   "
        }.get(feeling, "   ")
        
        return f"   {self.name}. {feeling_kr}     . {self.mind.think(self.heart)}"


# =============================================================================
# 4.        (Fractal World) -        
# =============================================================================

class FractalWorld:
    """
           -             
    
                       ,               
    """
    
    def __init__(self, population: int = 300, seed: int = None):
        if seed:
            random.seed(seed)
        
        self.population = population
        self.souls: Dict[int, FractalSoul] = {}
        self.current_year = 0
        
        #      
        self.locations = ["village", "forest", "mountain", "city", "coast"]
        self.seasons = ["spring", "summer", "autumn", "winter"]
        
        #   
        self.total_conversations = 0
        self.total_diary_entries = 0
        self.legends: List[str] = []
        
        #    
        self._create_initial_souls()
        
        logger.info(f"  Fractal World created with {population} souls")
    
    def _create_initial_souls(self):
        """        """
        names_pool = [
            "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ",
            "Alice", "Luna", "Kai", "Aria", "Finn", "Rose", "Mira", "Thorne",
            "Eugeo", "Asuna", "Kirito", "Leafa", "Sinon", "Yuuki",
        ]
        
        for i in range(self.population):
            name = random.choice(names_pool) + f"_{i}"
            birth_year = -random.randint(0, 50)
            
            soul = FractalSoul(
                id=i,
                name=name,
                birth_year=birth_year,
                race=random.choice(["Human", "Human", "Elf", "Dwarf"]),
                profession=random.choice(["Farmer", "Artisan", "Hunter", "Merchant", "Scholar"]),
                location=random.choice(self.locations),
            )
            
            self.souls[i] = soul
    
    def simulate_day(self) -> List[Dict[str, Any]]:
        """        """
        day_records = []
        season = self.seasons[(self.current_year * 4 // 365) % 4]
        
        #               
        world_context = {
            "spring": {"warmth": 0.5, "brightness": 0.6},
            "summer": {"warmth": 0.8, "brightness": 0.9},
            "autumn": {"warmth": 0.4, "brightness": 0.5},
            "winter": {"warmth": 0.2, "brightness": 0.3},
        }.get(season, {})
        
        alive_souls = [s for s in self.souls.values() if s.is_alive]
        
        #              
        for soul in alive_souls:
            record = soul.live_day(self.current_year, world_context)
            day_records.append(record)
        
        #      (        )
        if len(alive_souls) >= 2 and random.random() < DAILY_INTERACTION_PROB:
            soul1, soul2 = random.sample(alive_souls, 2)
            line1, line2 = soul1.interact_with(soul2)
            
            day_records.append({
                "type": "conversation",
                "participants": [soul1.name, soul2.name],
                "dialogue": [
                    f"[{soul1.name}] {line1}",
                    f"[{soul2.name}] {line2}",
                ]
            })
            self.total_conversations += 1
        
        return day_records
    
    def simulate_year(self) -> Dict[str, Any]:
        """1       """
        year_records = []
        
        for day in range(365):
            records = self.simulate_day()
            #             
            for r in records:
                if r.get("diary") or r.get("type") == "conversation":
                    year_records.append(r)
        
        self.current_year += 1
        
        #          
        self._handle_life_events()
        
        alive = sum(1 for s in self.souls.values() if s.is_alive)
        
        return {
            "year": self.current_year,
            "population": alive,
            "events_count": len(year_records),
            "sample_events": year_records[-5:] if year_records else []
        }
    
    def _handle_life_events(self):
        """      """
        alive_souls = [s for s in self.souls.values() if s.is_alive]
        
        #   
        for soul in alive_souls:
            age = soul.get_age(self.current_year)
            death_prob = BASE_DEATH_PROB * (1 + max(0, age - ELDER_AGE_THRESHOLD) * AGE_DEATH_FACTOR)
            
            if soul.race == "Elf":
                death_prob *= ELF_LONGEVITY_FACTOR
            
            if random.random() < death_prob:
                soul.is_alive = False
                soul.death_year = self.current_year
                
                #              
                if len(soul.achievements) > 3 or len(soul.diary_entries) > 10:
                    self.legends.append(f"The Legend of {soul.name}")
        
        #    (     )
        alive = sum(1 for s in self.souls.values() if s.is_alive)
        if alive < self.population:
            deficit = self.population - alive
            for _ in range(min(deficit, MAX_BIRTHS_PER_YEAR)):
                new_id = max(self.souls.keys()) + 1
                new_soul = FractalSoul(
                    id=new_id,
                    name=f"Soul_{new_id}",
                    birth_year=self.current_year,
                    race=random.choice(["Human", "Elf"]),
                    location=random.choice(self.locations),
                )
                self.souls[new_id] = new_soul
    
    def run_simulation(self, years: int, progress_interval: int = 100) -> Dict[str, Any]:
        """
                   
        """
        logger.info(f"  Starting simulation: {years} years")
        start_time = time.time()
        
        for year in range(years):
            self.simulate_year()
            
            if (year + 1) % progress_interval == 0:
                alive = sum(1 for s in self.souls.values() if s.is_alive)
                diaries = sum(len(s.diary_entries) for s in self.souls.values())
                logger.info(f"  Year {year + 1}: Pop={alive}, Diaries={diaries}, Legends={len(self.legends)}")
        
        elapsed = time.time() - start_time
        
        #      
        results = self._compile_results(elapsed)
        
        logger.info(f"  Simulation complete in {elapsed:.2f}s")
        
        return results
    
    def _compile_results(self, elapsed: float) -> Dict[str, Any]:
        """     """
        alive = [s for s in self.souls.values() if s.is_alive]
        
        total_diaries = sum(len(s.diary_entries) for s in self.souls.values())
        total_memories = sum(len(s.memories) for s in self.souls.values())
        
        #      
        sample_diaries = []
        for soul in self.souls.values():
            if soul.diary_entries:
                sample_diaries.append({
                    "author": soul.name,
                    "entry": soul.diary_entries[-1]
                })
                if len(sample_diaries) >= 5:
                    break
        
        return {
            "simulation": {
                "years": self.current_year,
                "elapsed_seconds": elapsed,
                "years_per_second": self.current_year / elapsed if elapsed > 0 else 0,
            },
            "population": {
                "initial": self.population,
                "final": len(alive),
                "total_souls": len(self.souls),
            },
            "culture": {
                "total_diaries": total_diaries,
                "total_conversations": self.total_conversations,
                "legends_created": len(self.legends),
                "legend_examples": self.legends[:10],
            },
            "sample_diaries": sample_diaries,
            "sample_thoughts": [
                {"soul": s.name, "thought": s.mind.think(s.heart)}
                for s in list(alive)[:5]
            ]
        }


# =============================================================================
#      
# =============================================================================

def run_fractal_world(population: int = 300, years: int = 1000):
    """         """
    print("=" * 70)
    print("  FRACTAL SOUL WORLD")
    print("   '       ' -            ")
    print("=" * 70)
    print(f"\n  Settings:")
    print(f"     Population: {population}")
    print(f"     Duration: {years} years")
    print()
    
    world = FractalWorld(population=population)
    results = world.run_simulation(years, progress_interval=100)
    
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    
    print(f"\n   Performance:")
    print(f"     Real time: {results['simulation']['elapsed_seconds']:.2f}s")
    print(f"     Speed: {results['simulation']['years_per_second']:.0f} years/second")
    
    print(f"\n  Population:")
    print(f"     Final: {results['population']['final']}")
    print(f"     Total souls: {results['population']['total_souls']}")
    
    print(f"\n  Culture:")
    print(f"     Total diaries: {results['culture']['total_diaries']}")
    print(f"     Total conversations: {results['culture']['total_conversations']}")
    print(f"     Legends: {results['culture']['legends_created']}")
    
    print("\n  Sample Diaries:")
    for diary in results['sample_diaries']:
        print(f"   [{diary['author']}] {diary['entry'][:60]}...")
    
    print("\n  Current Thoughts:")
    for thought in results['sample_thoughts']:
        print(f"   [{thought['soul']}] {thought['thought']}")
    
    print("\n" + "=" * 70)
    print("  Fractal World simulation complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_fractal_world(population=300, years=1000)
