import numpy as np
from typing import List, Dict, Optional
from Core.Language.hangul_physics import HangulPhysicsEngine
from Core.Mind.resonance_engine import ResonanceEngine

class GeneticCell:
    """
    A cell defined by its code (genome) and capable of organic communication.
    """
    def __init__(self, cell_id: str, genome: str, position: np.ndarray, parent_brain: Optional[ResonanceEngine] = None):
        self.id = cell_id
        self.genome = genome
        self.position = np.array(position, dtype=float)
        self.energy = 100.0
        self.age = 0
        self.direction = np.random.rand(3)
        self.direction /= np.linalg.norm(self.direction)
        
        # Communication
        self.inbox: List[str] = []
        self.outbox: List[str] = []
        
        # Language Learning (Sound -> Utility Score)
        self.vocabulary: Dict[str, float] = {} 
        
        # Civilization (Inventory)
        self.inventory: Dict[str, int] = {}
        
        self.physics_engine = HangulPhysicsEngine()
        
        if parent_brain:
            self.brain = parent_brain.clone()
            self.brain.mutate() # Evolution happens here
        else:
            self.brain = ResonanceEngine()
            # Initial Concepts (New ones for Civilization)
            # Note: Basic ones (Eat, Move) are in ResonanceEngine._init_instincts()
            # HyperQubit nodes are auto-created when needed
            self.brain.add_node("Gather")  # ✅ Protocol-50 compliant
            self.brain.add_node("Experiment")  # ✅ Protocol-50 compliant
            
            # Connect instincts for new actions
            self.brain.connect("Energy", "Gather", 0.3) 
            self.brain.connect("Hunger", "Gather", 0.1) 
            
            # Experiment is curiosity-driven (Entropy?)
            self.brain.connect("SELF", "Experiment", 0.1)
        
        # Compile genome (Legacy support, though not used in resonance mode)
        self.executable = self._compile_genome(genome)

    def _compile_genome(self, source: str):
        try:
            scope = {}
            exec(source, scope)
            if "update" not in scope:
                return None
            return scope["update"]
        except Exception:
            return None

    def run(self, world):
        """Execute behavior based on Resonance."""
        # 1. Gather Inputs (The Sensorium)
        inputs = {}
        
        # Internal State
        if self.energy < 30:
            inputs["Hunger"] = np.array([1.0, 1.0, 1.0])
        if self.energy > 80:
            inputs["Energy"] = np.array([1.0, 1.0, 1.0])
            
        # Sight (Visual Field)
        visual_vector = self.sense_sight(world)
        if np.linalg.norm(visual_vector) > 0:
            inputs["Sight"] = visual_vector
            
        # Smell (Pheromones)
        scent_vector = self.sense_smell(world)
        if np.linalg.norm(scent_vector) > 0:
            inputs["Smell"] = scent_vector
            
        # Touch (Collision)
        touch_vector = self.sense_touch(world)
        if np.linalg.norm(touch_vector) > 0:
            inputs["Touch"] = touch_vector
            
        # Taste (Energy Density)
        taste_vector = self.sense_taste(world)
        if np.linalg.norm(taste_vector) > 0:
            inputs["Taste"] = taste_vector

        # Hearing (Inbox)
        if self.inbox:
            combined_signal = np.mean([v for v in self.inbox if isinstance(v, np.ndarray)], axis=0)
            if combined_signal is not None:
                inputs["Sound"] = combined_signal

        # Spiritual (Pantheon)
        divine_vector = self.sense_divine(world)
        if np.linalg.norm(divine_vector) > 0:
            inputs["Divine"] = divine_vector

        # Reading (Nearby Artifacts)
        artifacts = world.get_nearby_artifacts(self.position, radius=10.0)
        for artifact in artifacts:
            self.read(artifact.get_read_signal())

        # 2. Resonance Engine Update (Imagination Mode)
        action = self.brain.imagination_step(inputs)
        
        # 3. Execute Action
        if action == "Move":
            self.move_forward()

    # --- Senses ---
    def sense_sight(self, world) -> np.ndarray:
        """Visual scan: Returns vector sum of nearby entities."""
        neighbors = world.get_neighbors(self.position, radius=50.0)
        visual_sum = np.zeros(3)
        for n in neighbors:
            if n is self: continue
            # Vector to neighbor
            vec = n.position - self.position
            dist = np.linalg.norm(vec)
            if dist > 0:
                # Closer = Stronger signal
                visual_sum += (vec / dist) * (10.0 / dist)
        return visual_sum

    def sense_smell(self, world) -> np.ndarray:
        """Olfactory: Returns gradient towards food/energy."""
        return world.get_scent(self.position)

    def sense_touch(self, world) -> np.ndarray:
        """Tactile: Detects immediate collisions."""
        neighbors = world.get_neighbors(self.position, radius=2.0) # Very close
        if len(neighbors) > 1: # Self + at least one other
            return np.array([1.0, 0.0, 0.0]) # "Obstacle" sensation (Roughness)
        return np.zeros(3)

    def sense_taste(self, world) -> np.ndarray:
        """Gustatory: Samples local energy density."""
        env = world.get_environment(self.position)
        density = env.get("energy_density", 0.0)
        return np.array([density, density, density])

    def sense_divine(self, world) -> np.ndarray:
        """Spiritual: Senses the balance of Order and Chaos."""
        if hasattr(world, "pantheon"):
            laws = world.pantheon.get_active_laws()
            # Vector: [Order, Chaos, 0]
            return np.array([laws.get("Order", 0.0), laws.get("Chaos", 0.0), 0.0])
        return np.zeros(3)

    def read(self, artifact_concepts: List[str]):
        """Absorbs a sequence of concepts from an artifact."""
        for concept in artifact_concepts:
            self.brain.process_input_sequence(concept)
            self.energy -= 0.05

    def gather(self, world):
        """Collects resources from nearby nodes."""
        resources = world.get_nearby_resources(self.position, radius=5.0)
        if resources:
            target = resources[0] # Just take the first one
            if target.amount > 0:
                target.amount -= 1
                self.inventory[target.type] = self.inventory.get(target.type, 0) + 1
                self.energy -= 2.0
                # Learn: Resource -> Good
                self.brain.hebbian_update("Gather", "Energy", 0.1)

    def experiment(self, world):
        """Attempts to combine materials to discover new items."""
        if hasattr(world, "material_system") and len(self.inventory) >= 2:
            # Pick 2 random ingredients
            ingredients = []
            keys = list(self.inventory.keys())
            import random
            for _ in range(2):
                key = random.choice(keys)
                if self.inventory[key] > 0:
                    ingredients.append(key)
                    self.inventory[key] -= 1
            
            if len(ingredients) == 2:
                item = world.material_system.combine(ingredients)
                if item:
                    self.inventory[item.name] = self.inventory.get(item.name, 0) + 1
                    self.energy -= 10.0
                    
                    # Learn from the properties of the created item
                    dominant_prop = max(item.properties, key=item.properties.get)
                    # Hebbian: Experiment -> Dominant Property (e.g., Cutting)
                    # We assume the brain has a concept for the property (or creates it)
                    self.brain.add_node(dominant_prop, np.random.rand(3)) # Learn the concept
                    self.brain.hebbian_update("Experiment", dominant_prop, 0.5)
                    self.brain.hebbian_update(dominant_prop, "Energy", 0.2) # Assume it's useful

    # --- Actions ---
    def move_forward(self):
        self.position += self.direction * 1.0
        self.energy -= 0.2

    def turn_left(self):
        theta = np.radians(15)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        self.direction = R.dot(self.direction)

    def turn_right(self):
        theta = np.radians(-15)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        self.direction = R.dot(self.direction)
        
    def eat(self):
        import random
        if random.random() < 0.05:
            self.energy += 20.0 
            return True
        return False

    def split(self):
        self.energy /= 2.0
        return True

    def speak(self, content: str):
        """Emits a sound wave (Communication)."""
        # Try to form a sentence if content is simple
        if "_" not in content and hasattr(self.brain, "form_sentence"):
            # We need inputs for form_sentence. 
            # Ideally we pass current inputs, but for now we'll just use what's in the brain (cached?)
            # Actually, let's just use the content as the 'Verb' and let the brain fill the rest?
            # Or, more simply:
            sentence = self.brain.form_sentence({}) # Inputs are needed!
            # Wait, form_sentence needs inputs.
            # Let's just emit the content for now, but if it's "Speak", we generate a sentence.
            pass
            
        if content == "ㄱ": # Default babble
             # Generate sentence
             # We need to access the inputs from the last run step.
             # This is hard without refactoring run().
             # Let's just emit "SELF_Speak_Void" for now to test SVO.
             self.outbox.append("SELF_Speak_Void")
        elif "_" in content:
            # Already a sentence or sequence
            sequence = content.split("_")
            for token in sequence:
                self.outbox.append(token)
        else:
            # Simple concept
            self.outbox.append(content)
        self.energy -= 0.5

    def listen(self) -> List[str]:
        """Check inbox and convert vectors back to sound."""
        messages = []
        for vector in self.inbox:
            if isinstance(vector, np.ndarray):
                char = self.physics_engine.vector_to_jamo(vector)
                messages.append(char)
            else:
                messages.append(str(vector))
        self.inbox = [] 
        return messages
        
    def learn_utility(self, sound: str, reward: float):
        """Update utility score for a sound."""
        current = self.vocabulary.get(sound, 0.0)
        self.vocabulary[sound] = current + (reward * 0.1) 
