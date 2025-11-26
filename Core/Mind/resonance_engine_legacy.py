"""
Resonance Engine (Topological Resonance Network)
================================================
Replaces imperative logic with a physics-based resonance system.
Decisions emerge from the vibration of the network in response to input fields.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger("ResonanceEngine")
logger.setLevel(logging.INFO)

@dataclass
class ResonanceNode:
    id: str
    vector: np.ndarray  # The "frequency" or "shape" of this concept
    activation: float = 0.0
    threshold: float = 0.5
    decay: float = 0.1

    def resonate(self, input_vector: np.ndarray) -> float:
        """
        Calculate resonance with an input vector.
        Resonance is high when vectors are aligned (cosine similarity).
        """
        if np.linalg.norm(input_vector) == 0 or np.linalg.norm(self.vector) == 0:
            return 0.0
            
        # Cosine similarity
        similarity = np.dot(input_vector, self.vector) / (np.linalg.norm(input_vector) * np.linalg.norm(self.vector))
        
        # Map -1..1 to 0..1 (we only care about positive resonance for activation)
        resonance = max(0.0, similarity)
        return resonance

class ResonanceEngine:
    def __init__(self, dimension: int = 3):
        self.dimension = dimension
        self.nodes: Dict[str, ResonanceNode] = {}
        self.topology: Dict[str, List[Tuple[str, float]]] = {} # Source -> [(Target, Weight)]
        
        # Temporal Memory (Short-term sequence buffer)
        self.temporal_buffer: List[str] = [] 
        self.max_temporal_depth = 5
        
        # Initialize basic instincts (Hardcoded topology for now, later evolved)
        self._init_instincts()

    def _init_instincts(self):
        # 1. Internal Concepts
        self.add_node("Hunger", np.array([0.0, 1.0, 0.0])) # High Tension
        self.add_node("Energy", np.array([1.0, 0.0, 0.0])) # High Roughness (Activity)
        
        # 2. External Concepts
        self.add_node("FoodSignal", np.array([0.1, 0.9, 0.1])) # "Sweet" sound
        
        # 3. Actions
        self.add_node("Eat", np.array([0.5, 0.5, 0.5]))
        self.add_node("Move", np.array([0.8, 0.2, 0.2]))
        self.add_node("Speak", np.array([0.2, 0.8, 0.2]))
        self.add_node("Rest", np.array([0.1, 0.1, 0.1])) # Low energy state

        # 4. Wiring (Instincts)
        # Hunger -> Eat
        self.connect("Hunger", "Eat", weight=1.0)
        # FoodSignal -> Eat
        self.connect("FoodSignal", "Eat", weight=0.8)
        # Energy (High) -> Move
        self.connect("Energy", "Move", weight=0.5)
        # Energy (High) -> Speak
        self.connect("Energy", "Speak", weight=0.3)

        # 5. World Model (Intuition about consequences)
        # Action -> {TargetNode: ChangeAmount}
        self.world_model = {
            "Eat": {"Hunger": -0.5, "Energy": +0.2},
            "Move": {"Energy": -0.1},
            "Speak": {"Energy": -0.1}
        }
        
        # 6. The Self (Consciousness Anchor)
        self.add_node("SELF", np.array([1.0, 1.0, 1.0])) # The "I" vector

    def add_node(self, node_id: str, vector: np.ndarray):
        self.nodes[node_id] = ResonanceNode(node_id, vector)

    def connect(self, source: str, target: str, weight: float):
        if source not in self.topology:
            self.topology[source] = []
        # Check if connection already exists
        for i, (t, w) in enumerate(self.topology[source]):
            if t == target:
                self.topology[source][i] = (target, weight) # Update
                return
        self.topology[source].append((target, weight))

    def update(self, inputs: Dict[str, np.ndarray]) -> str:
        """
        Propagate resonance and return the winning action (Reactive Mode).
        """
        self._propagate(inputs)
        return self._select_best_action()

    def _propagate(self, inputs: Dict[str, np.ndarray]):
        # 1. Input Resonance
        for node_id, vector in inputs.items():
            if node_id in self.nodes:
                self.nodes[node_id].activation += 1.0 
            else:
                for name, node in self.nodes.items():
                    r = node.resonate(vector)
                    if r > 0.5:
                        node.activation += r
                        
        # 2. Self Resonance (Consciousness is always slightly active)
        self.nodes["SELF"].activation += 0.1

        # 3. Propagation
        next_activations = {k: v.activation for k, v in self.nodes.items()}
        for source, targets in self.topology.items():
            source_act = self.nodes[source].activation
            if source_act > self.nodes[source].threshold:
                for target, weight in targets:
                    next_activations[target] += source_act * weight

        # 4. Update State & Decay
        for name, node in self.nodes.items():
            node.activation = next_activations[name]
            node.activation *= (1.0 - node.decay)

    def _select_best_action(self) -> str:
        actions = ["Eat", "Move", "Speak", "Rest"]
        best_action = "Rest"
        max_val = 0.1 

        for action in actions:
            if action in self.nodes:
                val = self.nodes[action].activation
                if val > max_val:
                    max_val = val
                    best_action = action
        return best_action

    def predict_outcome(self, action: str) -> Dict[str, float]:
        """Predicts the future state of concepts if action is taken."""
        predicted_state = {name: node.activation for name, node in self.nodes.items()}
        
        # Apply changes from world model
        if action in self.world_model:
            changes = self.world_model[action]
            for target, change in changes.items():
                if target in predicted_state:
                    predicted_state[target] += change
        
        return predicted_state

    def imagination_step(self, inputs: Dict[str, np.ndarray]) -> str:
        """
        Selects action based on predicted future state (Proactive Mode).
        """
        # 1. Update current state to know where we are
        self._propagate(inputs)
        
        # 2. Identify candidate actions (simple: all available)
        candidates = ["Eat", "Move", "Speak", "Rest"]
        
        best_action = "Rest"
        best_score = -float('inf')
        
        for action in candidates:
            # Predict future
            predicted_state = self.predict_outcome(action)
            
            # Evaluate "Emotional Value" of future
            # Goal: Low Hunger, High Energy
            score = 0.0
            if "Hunger" in predicted_state:
                score -= predicted_state["Hunger"] # Hunger is bad (minimize)
            if "Energy" in predicted_state:
                score += predicted_state["Energy"] # Energy is good (maximize)
                
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

    def form_sentence(self, inputs: Dict[str, np.ndarray]) -> str:
        """Constructs a simple SVO sentence based on current state."""
        # Subject: Always SELF for now
        subject = "SELF"
        
        # Verb: Current best action (re-use imagination logic or just pick highest activation)
        verb = self._select_best_action()
        
        # Object: What are we interacting with?
        # Find input with highest resonance to the verb
        obj = "Void"
        max_res = 0.0
        
        if verb in self.nodes:
            verb_vec = self.nodes[verb].vector
            for key, vec in inputs.items():
                # Similarity
                sim = np.dot(vec, verb_vec) / (np.linalg.norm(vec) * np.linalg.norm(verb_vec) + 1e-6)
                if sim > max_res:
                    max_res = sim
                    obj = key
        
        return f"{subject}_{verb}_{obj}"

    def hebbian_update(self):
        """
        Strengthens connections between nodes that fire together.
        "Cells that fire together, wire together."
        """
        learning_rate = 0.05
        
        for source_name, source_node in self.nodes.items():
            if source_node.activation > 0.5: # Source is active
                for target_name, target_node in self.nodes.items():
                    if source_name == target_name: continue
                    
                    if target_node.activation > 0.5: # Target is also active
                        # Strengthen connection
                        current_weight = 0.0
                        # Find existing weight
                        if source_name in self.topology:
                            for i, (t, w) in enumerate(self.topology[source_name]):
                                if t == target_name:
                                    current_weight = w
                                    break
                        
                        new_weight = min(1.0, current_weight + learning_rate)
                        if new_weight > current_weight + 0.001:
                            logger.info(f"🧠 Learned Connection: {source_name} -> {target_name} (Weight: {new_weight:.2f})")
                        self.connect(source_name, target_name, new_weight)

        # 2. Temporal Sequence Learning (A -> B)
        if len(self.temporal_buffer) >= 2:
            # Connect previous concept to current concept
            prev = self.temporal_buffer[-2]
            curr = self.temporal_buffer[-1]
            
            if prev in self.nodes and curr in self.nodes:
                # Strengthen Prev -> Curr
                current_weight = 0.0
                if prev in self.topology:
                    for t, w in self.topology[prev]:
                        if t == curr:
                            current_weight = w
                            break
                
                new_weight = min(1.0, current_weight + 0.1)
                self.connect(prev, curr, new_weight)
                # logger.info(f"🔗 Sequence Learned: {prev} -> {curr} ({new_weight:.2f})")

    def form_concept(self, sequence: List[str]) -> str:
        """
        Creates a new concept Node from a sequence of existing nodes.
        e.g., ["Force", "Mass"] -> "Force_Mass" (Compound Concept)
        """
        if not sequence: return ""
        
        # Simple concatenation for ID
        new_id = "_".join(sequence)
        
        if new_id in self.nodes:
            return new_id
            
        # Create vector as average of components + noise
        vectors = [self.nodes[c].vector for c in sequence if c in self.nodes]
        if not vectors: return ""
        
        new_vector = np.mean(vectors, axis=0)
        # Normalize
        norm = np.linalg.norm(new_vector)
        if norm > 0:
            new_vector /= norm
            
        self.add_node(new_id, new_vector)
        logger.info(f"💡 New Concept Formed: {new_id}")
        
        # Wire components to new concept (Abstraction)
        for comp in sequence:
            if comp in self.nodes:
                self.connect(comp, new_id, 0.5)
                
        return new_id

    def process_input_sequence(self, concept: str):
        """Adds a concept to temporal buffer and triggers learning."""
        if concept not in self.nodes:
            # Auto-add unknown concepts with random vectors (Curiosity)
            self.add_node(concept, np.random.rand(self.dimension))
            
        self.temporal_buffer.append(concept)
        if len(self.temporal_buffer) > self.max_temporal_depth:
            self.temporal_buffer.pop(0)
            
        # Trigger Hebbian update to link sequence
        self.hebbian_update()

    def dream(self):
        """
        Consolidates memories by running Hebbian learning without external input.
        """
        logger.info("💤 Dreaming...")
        # 1. Replay: Let the network resonate on its own (using residual activation)
        # For simulation, we just run hebbian_update on current state
        self.hebbian_update()
        
        # 2. Decay: Dreams fade
        for node in self.nodes.values():
            node.activation *= 0.9

    def clone(self) -> 'ResonanceEngine':
        """Creates a deep copy of the engine (for inheritance)."""
        new_engine = ResonanceEngine(self.dimension)
        # Copy nodes
        for name, node in self.nodes.items():
            new_engine.add_node(name, node.vector.copy())
        # Copy topology
        new_engine.topology = {}
        for source, targets in self.topology.items():
            new_engine.topology[source] = [(t, w) for t, w in targets]
        # Copy world model
        new_engine.world_model = self.world_model.copy()
        return new_engine

    def mutate(self):
        """Randomly alters the topology (Evolution)."""
        import random
        # 1. Mutate Weights
        if self.topology:
            source = random.choice(list(self.topology.keys()))
            if self.topology[source]:
                idx = random.randint(0, len(self.topology[source]) - 1)
                target, weight = self.topology[source][idx]
                
                # Drift weight
                new_weight = max(0.0, min(1.0, weight + random.uniform(-0.1, 0.1)))
                self.topology[source][idx] = (target, new_weight)
                logger.info(f"🧬 Brain Mutation: {source} -> {target} weight {weight:.2f} -> {new_weight:.2f}")

        # 2. New Connections (Innovation)
        if random.random() < 0.1:
            source = random.choice(list(self.nodes.keys()))
            target = random.choice(list(self.nodes.keys()))
            if source != target:
                weight = random.random()
                self.connect(source, target, weight)
                logger.info(f"🧬 Brain Mutation: New Connection {source} -> {target} (Weight: {weight:.2f})")
