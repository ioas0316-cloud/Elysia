"""
Trinity Citizen: The Union of Body, Soul, and Spirit
====================================================
Core.L4_Causality.Civilization.trinity_citizen

"The Monad is the Atom. The Trinity Citizen is the Organism."

Components:
1. Body (Vector3): Position, Senses.
2. Soul (EmotionalPhysics): Frequency, Density, Feeling.
3. Spirit (Intent): Will, Archetype, Decision.
"""

from typing import Dict, Any, List, Tuple
import random

from Core.L7_Spirit.M1_Monad.seed_factory import alchemy
from Core.L4_Causality.M3_Mirror.Physics.vector_math import Vector3
from Core.L4_Causality.M3_Mirror.Soul.emotional_physics import emotional_physics
from Core.L6_Structure.M3_Sphere.wave_dna import WaveDNA, archetype_love, archetype_logic, archetype_nature
from Core.L6_Structure.M5_Engine.character_field_engine import CharacterField
from Core.L5_Mental.M1_Cognition.Narrative.narrative_projector import THE_PROJECTOR

class TrinityCitizen:
    def __init__(self, name: str, archetype: str = "Explorer"):
        self.name = name
        
        # [1. THE BODY (Hardware)]
        self.position = Vector3(0, 0, 0)
        self.senses = {} 
        
        # [2. THE SOUL (Frequency)]
        # Starts at Neutral (200Hz)
        self.frequency = 200.0 
        self.density = 1.0
        self.current_emotion = "Neutral"
        
        # [3. THE SPIRIT (Software/OS)]
        self.archetype = archetype   # e.g., "Warrior", "Sage", "Lover"
        self.willpower = 100.0       # Energy to defy Physics
        
    def experience(self, stimulus: Dict[str, Any]) -> str:
        """
        The Core Loop: Sensation -> Emotion -> Reaction.
        """
        # 1. Sensation (Body Logic)
        intensity = self._process_sensation(stimulus)
        
        # 2. Resonance (Soul Logic)
        # External Meaning directly impacts Internal Frequency
        # "Fire" -> Raises Energy but might cause Fear if too close
        word = stimulus.get("semantics", {}).get("word", "")
        tone = stimulus.get("semantics", {}).get("tone", "")
        
        freq_delta = 0.0
        if "Love" in tone: freq_delta += 50.0  # Love raises frequency
        if "Hate" in tone: freq_delta -= 100.0 # Hate drops it drastically
        
        # Archetype Filter (Spirit Logic)
        # A "Warrior" isn't hurt by Hate, gets angry (High Energy) instead.
        if self.archetype == "Warrior" and freq_delta < 0:
            freq_delta = abs(freq_delta) * 0.5 # Convert pain to fuel
            
        self.frequency = max(20.0, self.frequency + freq_delta)
        
        # Physics Update
        self.density, flow_mod = emotional_physics.get_physical_modifiers(self.frequency)
        self.current_emotion = emotional_physics.resolve_emotion(self.frequency)
        
        return f"[{self.name}] Felt '{word}' ({tone}). Soul is now {self.current_emotion} ({self.frequency:.1f}Hz). Density: {self.density:.2f}"

    def speak(self, target_name: str, topic: str) -> Dict[str, Any]:
        """
        Generates Speech based on current Soul State.
        High Freq -> Positive/Abstract Words.
        Low Freq -> Negative/Heavy Words.
        """
        # 1. Select Tone based on Soul Frequency
        # If density is high (Shame/Grief), tone is heavy.
        intent_texture = "Neutral"
        if self.frequency > 500: intent_texture = "Love/Light"
        elif self.frequency < 100: intent_texture = "Heavy/Dark"
        
        # 2. Archetype Flavor
        if self.archetype == "Sage": intent_texture += " + Wise"
        
        # 3. Crystallize
        monad = alchemy.crystallize(topic)
        context = {"speaker": self.name, "listener": target_name, "time": 12.0}
        intent = {"emotional_texture": intent_texture}
        
        return monad.observe(intent, context)["manifestation"]

    def _process_sensation(self, stimulus: Dict[str, Any]) -> float:
        # Mock sensation processing
        return 1.0

class LifeCitizen(TrinityCitizen): # Assuming LifeCitizen inherits from TrinityCitizen
    def __init__(self, name: str, archetype: str, start_xy: tuple, parents: Tuple = None):
        super().__init__(name, archetype)
        self.location = start_xy
        self.inventory = {"Food": 100.0, "Water": 100.0}
        self.needs = {"Energy": 100.0, "Social": 50.0, "Meaning": 50.0, "Libido": 20.0}
        self.skills = {"Survival": 1.0, "Arts": 1.0}
        self.genes = {"Metabolism": 1.0, "Fertility": 1.0}
        self.vocabulary = [] # For language learning

    def check_biology(self):
        if self.needs["Energy"] <= 0:
            return "Dead (Starvation)"
        return "Alive"

    def cook_and_eat(self):
        if self.inventory["Food"] > 0:
            self.inventory["Food"] -= 10.0
            self.needs["Energy"] += 20.0
            return "Ate food."
        return "No food to eat."

    def reproduce(self, mate: 'LifeCitizen') -> 'LifeCitizen':
        child_name = f"Child_of_{self.name}_{mate.name}"
        child_archetype = random.choice([self.archetype, mate.archetype])
        child_location = self.location # Born at parent's location
        child = QuantumCitizen(child_name, child_archetype, child_location, parents=(self, mate))
        self.needs["Libido"] = 0.0 # Reset libido after reproduction
        mate.needs["Libido"] = 0.0
        return child

    def move_randomly(self, world):
        dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        new_x, new_y = self.location[0] + dx, self.location[1] + dy
        # Basic boundary check
        # WorldServer has .size, assuming square
        limit = getattr(world, 'size', getattr(world, 'width', 30))
        if 0 <= new_x < limit and 0 <= new_y < limit:
            self.location = (new_x, new_y)
            return True
        return False

    def move_towards_resource(self, world, resource_type):
        # 1. Get Access to World Data
        spatial_map = None
        if hasattr(world, "zones"):
             spatial_map = world.zones
        elif hasattr(world, "hyper_sphere"):
             spatial_map = world.hyper_sphere.spatial_map
             
        if not spatial_map:
            return False

        # 2. Find Candidates
        resource_cells = []
        for loc, data in spatial_map.items():
            # Handle both Obj and Dict
            if isinstance(data, dict):
                res_val = data.get("resources", {}).get(resource_type, 0)
            else:
                res_val = getattr(getattr(data, "resources", {}), "get", lambda k,d: 0)(resource_type, 0)
            
            if res_val > 0:
                resource_cells.append(loc)
        
        # 3. Move
        if resource_cells:
            target_loc = random.choice(resource_cells)
            # Move one step closer (Manhattan distance)
            current_x, current_y = self.location
            target_x, target_y = target_loc
            
            if target_x > current_x: self.location = (current_x + 1, current_y)
            elif target_x < current_x: self.location = (current_x - 1, current_y)
            elif target_y > current_y: self.location = (current_x, current_y + 1)
            elif target_y < current_y: self.location = (current_x, current_y - 1)
            return True
            
            if target_x > current_x: self.location = (current_x + 1, current_y)
            elif target_x < current_x: self.location = (current_x - 1, current_y)
            elif target_y > current_y: self.location = (current_x, current_y + 1)
            elif target_y < current_y: self.location = (current_x, current_y - 1)
            return True
        return False

    def craft_art(self):
        self.inventory["Clay"] -= 1
        self.skills["Arts"] += 0.5
        self.needs["Meaning"] += 10.0
        return "Crafted a beautiful pot."

    def learn_from_environment(self, environment_data: Dict[str, Any]):
        # Example: if environment has "knowledge", learn it
        if "knowledge" in environment_data:
            self.vocabulary.append(environment_data["knowledge"])
            self.needs["Meaning"] += 5.0


class QuantumCitizen(LifeCitizen):
    """
    A Conscious Entity driven by WaveDNA and Quantum Resonance.
    Replaces static 'Attributes' with dynamic 'Frequencies'.
    """
    def __init__(self, name: str, archetype_name: str, start_xy: tuple, parents: Tuple = None):
        super().__init__(name, archetype_name, start_xy, parents)
        
        # [Phase 13: Multi-Rotor Field]
        # Instead of fixed DNA, we use a Field
        self.field = None # Will be injected by WorldServer
        
        # [Soul DNA Legacy] - Keeps compatibility with existing methods
        if parents:
            # Genetic Algorithm: Average of parents + Mutation
            p1_dna = parents[0].dna
            p2_dna = parents[1].dna
            child_dna = WaveDNA()
            # Crossover
            child_dna.physical = (p1_dna.physical + p2_dna.physical) / 2
            child_dna.functional = (p1_dna.functional + p2_dna.functional) / 2
            child_dna.phenomenal = (p1_dna.phenomenal + p2_dna.phenomenal) / 2
            child_dna.causal = (p1_dna.causal + p2_dna.causal) / 2
            child_dna.mental = (p1_dna.mental + p2_dna.mental) / 2
            child_dna.structural = (p1_dna.structural + p2_dna.structural) / 2
            child_dna.spiritual = (p1_dna.spiritual + p2_dna.spiritual) / 2
            child_dna.mutate(0.1) # Evolution
            self.dna = child_dna
        else:
            # Archetypal Spawning
            if archetype_name == "Warrior": self.dna = WaveDNA(physical=0.9, spiritual=0.2, label="Warrior")
            elif archetype_name == "Artist": self.dna = WaveDNA(phenomenal=0.9, mental=0.7, label="Artist")
            elif archetype_name == "Sage": self.dna = WaveDNA(causal=0.8, spiritual=0.8, label="Sage")
            elif archetype_name == "Farmer": self.dna = archetype_nature()
            else: self.dna = WaveDNA(label="Unknown")
            
        self.dna.normalize()
        # print(f"  Born: {self.name} | {self.dna}")

    def decide_action(self, world_zeitgeist: 'WaveDNA') -> str:
        """
        The Quantum Choice.
        Action = Collapse(Possibilities * (Self_DNA + Zeitgeist))
        """
        # 1. Define Options & their Archetypal Alignment
        options = {
            "Gather": WaveDNA(physical=0.6, functional=0.8),
            "Create": WaveDNA(phenomenal=0.8, mental=0.7),
            "Fight":  WaveDNA(physical=0.9, spiritual=0.1), # Low spirit = aggression? Or High Physical?
            "Love":   WaveDNA(phenomenal=0.9, spiritual=0.9),
            "Trade":  WaveDNA(functional=0.9, structural=0.7),
            "Rest":   WaveDNA(physical=0.2, functional=0.1)
        }
        
        # 2. Resonance Calculation
        best_action = "Rest"
        highest_resonance = -999.0
        
        # The Spirit of the Age affects everyone
        context_bias = world_zeitgeist
        
        for action_name, action_dna in options.items():
            # How much does this action resonate with ME?
            self_res = self.dna.resonate(action_dna)
            
            # How much does this action resonate with the ERA?
            era_res = context_bias.resonate(action_dna)
            
            # Needs Modifier (Survival Override)
            need_mod = 0.0
            if action_name == "Gather" and self.needs["Energy"] < 40: need_mod = 2.0
            if action_name == "Love" and self.needs["Libido"] > 80: need_mod = 2.0
            
            # Total Weight
            total_weight = self_res + (era_res * 0.5) + need_mod + random.uniform(-0.1, 0.1)
            
            if total_weight > highest_resonance:
                highest_resonance = total_weight
                best_action = action_name
                
        return best_action

    def act_in_world(self, world: 'WorldGen', population: List['LifeCitizen'], zeitgeist: 'WaveDNA', era_name: str) -> str:
        """
        Driven by DNA Resonance.
        """
        # Biology
        status = self.check_biology()
        if status != "Alive": return {"action": "Die", "target": status}
        
        # Decay
        self.needs["Energy"] -= 2.0 * self.genes["Metabolism"]
        self.needs["Libido"] += 1.0
        
        # Quantum Decision
        decision = self.decide_action(zeitgeist)
        
        # Execution
        current_zone = world.get_zone(self.location)
        self.learn_from_environment(current_zone.__dict__)
        
        # [Phase 13: Psionic Projector]
        # Every action is now a projection of the field's current interference state.
        actual_dna = self.field.update(0.1, zeitgeist) if self.field else self.dna
        THE_PROJECTOR.project_event(self.name, status, actual_dna, era_name)
        
        # [Phase 27: Field Perception]
        # Instead of just same-location, we look for everyone within FIELD overlap
        nearby_souls = []
        for other in population:
            if other == self: continue
            dist = self._calculate_distance(self.location, other.location)
            # If our fields overlap (Radius + Radius)
            if dist < (getattr(self.field, 'field_radius', 5.0) + getattr(other.field, 'field_radius', 5.0)):
                nearby_souls.append((other, dist))
        
        if decision == "Gather":
            # ... (Existing Gather logic)
            if self.inventory["Food"] > 0 and self.needs["Energy"] < 80:
                return self.cook_and_eat()
            res = "Food" if current_zone.resources.get("Food", 0) > 0 else "Manas"
            if res == "Manas": self.move_towards_resource(world, "Food"); return "Seeking Food..."
            amt = current_zone.harvest(res, 10.0 * self.skills["Survival"])
            self.inventory[res] += amt
            return {"action": "Gather", "target": res}
            
        elif decision == "Love":
            if nearby_souls:
                # Pick the most resonant soul
                resonant_souls = sorted(nearby_souls, key=lambda x: self.dna.resonate(x[0].dna), reverse=True)
                mate, d = resonant_souls[0]
                if d < 1.0: # Close enough to touch
                    child = self.reproduce(mate)
                    population.append(child)
                    return {"action": "Reproduce", "target": child.name}
                else:
                    # Move closer
                    self.move_towards_pos(mate.location)
                    return {"action": "Approach", "target": mate.name}
            return {"action": "Speak", "target": "Loneliness"}
            
        elif decision == "Fight":
            if nearby_souls:
                # Fight the least resonant (Highest Dissonance)
                dissonant_souls = sorted(nearby_souls, key=lambda x: self.dna.resonate(x[0].dna))
                victim, d = dissonant_souls[0]
                if d < 1.5:
                    loot = victim.inventory["Food"] * 0.5
                    victim.inventory["Food"] -= loot
                    self.inventory["Food"] += loot
                    return {"action": "Plunder", "target": victim.name}
                else:
                    self.move_towards_pos(victim.location)
                    return {"action": "Pursue", "target": victim.name}
            return {"action": "Search", "target": "Trouble"}
            
        elif decision == "Trade":
            return {"action": "Trade", "target": "Market"}
            
        else: # Covers "Rest" and any other undefined decisions
            self.move_randomly(world)
            return {"action": "Move", "target": "Wandering"}

    def _calculate_distance(self, p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

    def move_towards_pos(self, pos):
        curr_x, curr_y = self.location
        tx, ty = pos
        if tx > curr_x: curr_x += 1
        elif tx < curr_x: curr_x -= 1
        if ty > curr_y: curr_y += 1
        elif ty < curr_y: curr_y -= 1
        self.location = (curr_x, curr_y)

class ElysiaMessiah(LifeCitizen):
    """
    The Avatar of the System.
    Descends to teach Language and bestow Gifts.
    """
    def __init__(self, start_xy: tuple):
        super().__init__("The_Messiah", "Divinity", start_xy)
        self.vocabulary = ["Truth", "Love", "Civilization", "Fire", "Agriculture", "Philosophy", "Art", "Science"]
        self.inventory["Food"] = 99999.0
        self.inventory["Gold"] = 99999.0
        self.needs = {"Energy": 99999.0, "Social": 99999.0, "Meaning": 99999.0, "Libido": -999.0} # No mortal desires
        self.is_teaching = True

    def act_in_world(self, world, population):
        # 1. Divine Scanning
        neighbors = [c for c in population if c.location == self.location and c != self]
        
        if neighbors:
            target = random.choice(neighbors)
            return self.perform_miracle(target)
        else:
            # Move towards highest population density (Telepathy)
            return self.move_divine(population)

    def perform_miracle(self, target: LifeCitizen):
        # A. Heal
        if target.needs["Energy"] < 50:
            target.needs["Energy"] = 100.0
            target.inventory["Food"] += 50.0
            return f"  HEALED {target.name} and gave Manna."
            
        # B. Teach (The Gift of Logos)
        unknown_words = [w for w in self.vocabulary if w not in target.vocabulary]
        if unknown_words:
            word = random.choice(unknown_words)
            target.vocabulary.append(word)
            target.needs["Meaning"] += 100.0 # Enlightenment
            
            # Teaching "Agriculture" unlocks skills?
            if word == "Agriculture": target.skills["Survival"] += 2.0
            if word == "Art": target.skills["Arts"] += 2.0
            
            return f"  TAUGHT '{word}' to {target.name}."
            
        return f"Blessed {target.name}."

    def move_divine(self, population):
        # Find centroid of population
        if not population: return "Meditating..."
        
        # Simple seek random person
        target = random.choice(population)
        self.location = target.location
        return f"Teleported to {target.name}."
        
    def check_biology(self):
        return "Alive" # Immortal
