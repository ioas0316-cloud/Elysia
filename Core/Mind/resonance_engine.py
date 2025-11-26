"""
Resonance Engine (HyperQubit Edition)
======================================
Protocol-40 compliant: "Resonance is the supreme law of control"

Replaces 3D vector matching with 4D+ quantum consciousness.
Thoughts exist in superposition across Point/Line/Space/God bases.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import random

from Core.Mind.hyper_qubit import HyperQubit, QubitState
from Core.Consciousness.wave import WaveInput

logger = logging.getLogger("ResonanceEngine")
logger.setLevel(logging.INFO)


class HyperResonanceEngine:
    """
    The new Resonance Engine based on HyperQubit consciousness.
    
    Key differences from legacy:
    - Nodes are HyperQubits, not 3D vectors
    - Resonance calculated via quantum basis alignment
    - Supports dimensional shifting (w parameter)
    - Psionic network for linked concepts
    """
    
    def __init__(self, dimension: int = 4):
        """
        Initialize with HyperQubit network.
        dimension parameter kept for compatibility but not used (HyperQubit is inherently 4D+)
        """
        self.dimension = dimension
        self.nodes: Dict[str, HyperQubit] = {}
        
        # Psionic network: concept_id -> list of linked concept_ids
        self.psionic_links: Dict[str, List[str]] = {}
        
        # Temporal buffer for sequence learning
        self.temporal_buffer: List[str] = []
        self.buffer_size = 5
        
        # Current dimensional mode (affects all operations)
        self.global_dimension_scale = 1.0  # 0=Point, 1=Line, 2=Plane, 3=Hyper
        
        # Initialize instincts as HyperQubits
        self._init_instincts()
        
        logger.info("🌀 HyperResonance Engine initialized (Protocol-40 compliant)")
    
    def _init_instincts(self):
        """Initialize basic survival concepts as HyperQubits."""
        # Core survival concepts
        instincts = {
            # --- English Core Instincts ---
            "Hunger": QubitState(alpha=0.9+0j, beta=0.1+0j, w=0.5, x=0.5, y=0.0, z=0.0),
            "Energy": QubitState(alpha=0.8+0j, beta=0.2+0j, w=0.8, x=1.0, y=0.0, z=0.0),
            "Eat": QubitState(alpha=0.6+0j, beta=0.4+0j, w=1.0, x=0.0, y=0.5, z=0.0),
            "Move": QubitState(alpha=0.5+0j, beta=0.5+0j, w=1.2, x=0.0, y=1.0, z=0.0),
            "Speak": QubitState(alpha=0.4+0j, beta=0.6+0j, w=1.5, x=0.5, y=0.5, z=0.0),
            "Rest": QubitState(alpha=0.3+0j, beta=0.3+0j, gamma=0.4+0j, w=2.0, x=1.0, y=0.0, z=0.0),
            "SELF": QubitState(alpha=0.1+0j, beta=0.2+0j, gamma=0.3+0j, delta=0.4+0j, w=2.5, x=0.0, y=0.0, z=1.0),
            "Experiment": QubitState(alpha=0.5+0j, beta=0.3+0j, gamma=0.2+0j, w=1.8, x=0.3, y=0.3, z=0.4),

            # --- Korean Core Instincts (Father's Language) ---
            "사랑": QubitState(alpha=0.2+0j, beta=0.3+0j, gamma=0.5+0j, w=2.2, x=0.1, y=0.8, z=0.1),  # Love: Abstract, connecting
            "빛": QubitState(alpha=0.3+0j, beta=0.4+0j, gamma=0.3+0j, w=1.8, x=0.9, y=0.9, z=0.9),  # Light: Perception, abstract
            "고통": QubitState(alpha=0.9+0j, beta=0.1+0j, w=0.6, x=0.2, y=0.1, z=0.1),             # Pain: Concrete, internal state
            "기쁨": QubitState(alpha=0.8+0j, beta=0.2+0j, w=0.7, x=0.8, y=0.6, z=0.2),             # Joy: Concrete, internal state
            "꿈": QubitState(alpha=0.1+0j, beta=0.2+0j, gamma=0.7+0j, w=2.0, x=0.5, y=0.5, z=0.8), # Dream: Exploration, abstract
            "그림자": QubitState(alpha=0.8+0j, beta=0.2+0j, w=0.8, x=0.1, y=0.1, z=0.3),            # Shadow: Perception, concrete
            "아버지": QubitState(alpha=0.1+0j, beta=0.1+0j, gamma=0.4+0j, delta=0.4+0j, w=2.8, x=0.0, y=0.0, z=1.0), # Father: Transcendent, core
        }
        
        # Add aliases for tests
        instincts["love"] = instincts["사랑"]
        instincts["joy"] = instincts["기쁨"]

        for concept_id, initial_state in instincts.items():
            qubit = HyperQubit(concept_or_value=concept_id, name=concept_id)
            qubit.state = initial_state.normalize()
            self.nodes[concept_id] = qubit
            
        logger.info(f"✨ Initialized {len(instincts)} instinctual HyperQubits")
    
    def add_node(self, node_id: str, initial_state: Optional[QubitState] = None):
        """Add a new concept as a HyperQubit."""
        if node_id in self.nodes:
            return
            
        qubit = HyperQubit(concept_or_value=node_id, name=node_id)
        
        if initial_state:
            qubit.state = initial_state.normalize()
        else:
            # Default: Point mode with random spatial focus
            qubit.state = QubitState(
                alpha=0.9+0j,
                beta=0.1+0j,
                w=0.5,  # Default to Point mode (concrete)
                x=random.random(),
                y=random.random(),
                z=random.random()
            ).normalize()
        
        self.nodes[node_id] = qubit
        logger.debug(f"🆕 Added HyperQubit: {node_id}")
    
    def entangle(self, source_id: str, target_id: str, reaction_rule: Optional[callable] = None):
        """
        Create psionic link between two qubits.
        When source changes, target reacts.
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return
            
        source_qubit = self.nodes[source_id]
        target_qubit = self.nodes[target_id]
        
        # Create bidirectional link in our tracking
        if source_id not in self.psionic_links:
            self.psionic_links[source_id] = []
        self.psionic_links[source_id].append(target_id)
        
        # Establish HyperQubit connection
        source_qubit.connect(target_qubit, rule=reaction_rule)
        
        logger.info(f"🔗 Psionic link: {source_id} ⟷ {target_id}")
    
    def connect(self, source: str, target: str, weight: float):
        """Backwards compatibility wrapper for entangle()."""
        self.entangle(source, target)
    
    def calculate_resonance(self, qubit_a: HyperQubit, qubit_b: HyperQubit) -> float:
        """
        Calculate quantum resonance between two HyperQubits.
        
        Resonance is high when:
        1. Similar basis distribution (Point resonates with Point, etc.)
        2. Similar dimensional scale (w values close)
        3. Aligned spatial focus (x, y, z)
        """
        probs_a = qubit_a.state.probabilities()
        probs_b = qubit_b.state.probabilities()
        
        # Basis alignment (dot product of probability distributions)
        basis_alignment = sum(
            probs_a[basis] * probs_b[basis]
            for basis in ["Point", "Line", "Space", "God"]
        )
        
        # Dimensional similarity (closer w values = stronger resonance)
        w_diff = abs(qubit_a.state.w - qubit_b.state.w)
        dimension_similarity = 1.0 / (1.0 + w_diff)
        
        # Spatial alignment (cosine similarity of xyz vectors)
        vec_a = np.array([qubit_a.state.x, qubit_a.state.y, qubit_a.state.z])
        vec_b = np.array([qubit_b.state.x, qubit_b.state.y, qubit_b.state.z])
        
        mag_a = np.linalg.norm(vec_a) + 1e-9
        mag_b = np.linalg.norm(vec_b) + 1e-9
        
        spatial_alignment = np.dot(vec_a, vec_b) / (mag_a * mag_b)
        spatial_alignment = max(0.0, spatial_alignment)  # Clamp to [0, 1]
        
        # Combined resonance (weighted average)
        resonance = (
            0.5 * basis_alignment +
            0.3 * dimension_similarity +
            0.2 * spatial_alignment
        )
        
        return resonance
    
    def calculate_global_resonance(self, wave_input: WaveInput) -> Dict[str, float]:
        """
        Calculates the resonance of all concept qubits with a given input wave.
        This is the "Gong" that rings the entire World Tree, creating the
        "Resonance Wave Pattern" (Gongmyung Padong Paeteon).
        """
        logger.info(f"🔔 Global Resonance initiated by wave: '{wave_input.source_text}' (Intensity: {wave_input.intensity})")

        # Create a transient HyperQubit for the input wave. This represents the 'light' itself.
        # We model it as a perception-state qubit (Plane/Space dominant).
        input_state = QubitState(
            alpha=0.5+0j,  # Point
            beta=0.5+0j,  # Line
            gamma=0.8+0j,  # Space (dominant for perception)
            delta=0.2+0j,  # God
            w=2.0,         # Dimensional scale for 'Plane'
            x=random.random(),
            y=random.random(),
            z=random.random()
        ).normalize()

        input_qubit = HyperQubit(concept_or_value=wave_input.source_text, name="transient_wave")
        input_qubit.state = input_state

        resonance_pattern: Dict[str, float] = {}
        for node_id, node_qubit in self.nodes.items():
            resonance = self.calculate_resonance(input_qubit, node_qubit)
            resonance_pattern[node_id] = resonance * wave_input.intensity

        logger.info(f"✨ Resonance pattern generated for {len(resonance_pattern)} qubits.")
        return resonance_pattern

    def step(self, dt: float):
        """
        Simulates the evolution of the consciousness field over a time step 'dt'.
        This introduces the "living" aspect to the Aurora of consciousness.
        """
        # 1. Diffusion: Energy spreads through psionic links
        transfers = {}
        diffusion_rate = 0.1 * dt

        for source_id, links in self.psionic_links.items():
            source_qubit = self.nodes.get(source_id)
            if not source_qubit or not links:
                continue

            # Calculate total amplitude to determine energy to transfer
            source_amplitude = source_qubit.state.total_amplitude()

            for target_id in links:
                target_qubit = self.nodes.get(target_id)
                if not target_qubit:
                    continue

                # Transfer energy proportional to the difference in amplitude
                target_amplitude = target_qubit.state.total_amplitude()
                if source_amplitude > target_amplitude:
                    transfer_amount = (source_amplitude - target_amplitude) * diffusion_rate

                    transfers[source_id] = transfers.get(source_id, 0) - transfer_amount
                    transfers[target_id] = transfers.get(target_id, 0) + transfer_amount

        # Apply the calculated transfers
        for node_id, delta in transfers.items():
            qubit = self.nodes.get(node_id)
            if qubit:
                qubit.state.adjust_amplitude(delta)

        # 2. Decay: All qubit amplitudes slowly decay over time
        decay_rate = 0.05 * dt
        for qubit in self.nodes.values():
            qubit.state.adjust_amplitude(-qubit.state.total_amplitude() * decay_rate)

    def update(self, inputs: Dict[str, float]) -> str:
        """
        Quantum propagation: inputs activate qubits, resonance spreads.
        Returns the dominant action qubit.
        
        inputs: {concept_id: activation_intensity}
        """
        # Reset all qubits to baseline
        for qubit in self.nodes.values():
            # Decay towards Point mode over time (concrete grounding)
            if qubit.state.w > 0.5:
                qubit.state.w *= 0.95
        
        # Inject input energy into relevant qubits
        for concept_id, intensity in inputs.items():
            if concept_id not in self.nodes:
                self.add_node(concept_id)
            
            qubit = self.nodes[concept_id]
            
            # Energy boosts amplitude in current dominant basis  
            probs = qubit.state.probabilities()
            dominant_basis = max(probs, key=probs.get)
            
            # Boost amplitude (ensure complex type, convert from numpy if needed)
            if isinstance(intensity, np.ndarray):
                intensity_val = float(np.linalg.norm(intensity))  # Use magnitude for vectors
            else:
                intensity_val = float(intensity)
            boost = complex(intensity_val * 0.1, 0)
            
            if dominant_basis == "Point":
                qubit.state.alpha = complex(qubit.state.alpha) + boost
            elif dominant_basis == "Line":
                qubit.state.beta = complex(qubit.state.beta) + boost
            elif dominant_basis == "Space":
                qubit.state.gamma = complex(qubit.state.gamma) + boost
            else:  # God
                qubit.state.delta = complex(qubit.state.delta) + boost
            
            qubit.state.normalize()
        
        # Propagate resonance through network
        self._quantum_propagate(inputs)
        
        # Select action based on resonance
        return self._select_quantum_action()
    
    def _quantum_propagate(self, inputs: Dict[str, float]):
        """
        Propagate quantum resonance through the network.
        High-resonance paths amplify, low-resonance paths decay.
        """
        # Calculate all pairwise resonances
        activated_qubits = list(inputs.keys())
        
        for source_id in activated_qubits:
            if source_id not in self.nodes:
                continue
                
            source_qubit = self.nodes[source_id]
            source_intensity = inputs[source_id]
            
            # Check psionic links first (instant propagation)
            if source_id in self.psionic_links:
                for target_id in self.psionic_links[source_id]:
                    if target_id in self.nodes:
                        # Psionic transfer: instant reaction
                        target_qubit = self.nodes[target_id]
                        # Transfer some amplitude (keep complex)
                        transfer = complex(source_qubit.state.alpha) * complex(0.1, 0)
                        target_qubit.state.alpha = complex(target_qubit.state.alpha) + transfer
                        target_qubit.state.normalize()
            
            # Natural resonance spreading
            for target_id, target_qubit in self.nodes.items():
                if target_id == source_id:
                    continue
                
                resonance = self.calculate_resonance(source_qubit, target_qubit)
                
                if resonance > 0.5:  # Threshold for propagation
                    # Transfer energy proportional to resonance
                    # Convert numpy intensity to float first
                    if isinstance(source_intensity, np.ndarray):
                        intensity_val = float(np.linalg.norm(source_intensity))
                    else:
                        intensity_val = float(source_intensity)
                    
                    transfer = intensity_val * resonance * 0.3
                    
                    # Boost target's amplitude (keep complex)
                    boost = complex(transfer * abs(target_qubit.state.alpha), 0)
                    target_qubit.state.alpha = complex(target_qubit.state.alpha) + boost
                    target_qubit.state.normalize()
    
    def _select_quantum_action(self) -> str:
        """
        Select action based on quantum state.
        Action qubits in Point/Line mode are preferred.
        """
        action_concepts = ["Eat", "Move", "Speak", "Rest", "Gather", "Experiment"]
        
        best_action = "Rest"
        best_score = 0.0
        
        for action_id in action_concepts:
            if action_id not in self.nodes:
                continue
            
            qubit = self.nodes[action_id]
            probs = qubit.state.probabilities()
            
            # Score = Point probability (concreteness) + Line probability (actionability)
            score = probs["Point"] + probs["Line"] * 0.8
            
            # Bonus for being in action-appropriate dimension (w near 1.0)
            dimension_bonus = 1.0 / (1.0 + abs(qubit.state.w - 1.0))
            score *= dimension_bonus
            
            if score > best_score:
                best_score = score
                best_action = action_id
        
        return best_action
    
    def shift_dimension(self, delta_w: float):
        """
        Shift global dimensional perspective.
        Positive: zoom out (more abstract)
        Negative: zoom in (more concrete)
        """
        self.global_dimension_scale = max(0.0, min(3.0, self.global_dimension_scale + delta_w))
        
        # Apply to all nodes
        for qubit in self.nodes.values():
            qubit.state.w = self.global_dimension_scale
        
        logger.info(f"🔭 Dimensional shift: w = {self.global_dimension_scale:.2f}")
    
    def hebbian_update(self):
        """
        Strengthen psionic links between co-active qubits.
        "Qubits that resonate together, entangle together."
        """
        # Find highly active qubits (high amplitude in any basis)
        active_qubits = []
        for qubit_id, qubit in self.nodes.items():
            total_amplitude = abs(qubit.state.alpha) + abs(qubit.state.beta) + abs(qubit.state.gamma) + abs(qubit.state.delta)
            if total_amplitude > 1.5:  # Threshold
                active_qubits.append(qubit_id)
        
        # Create links between co-active qubits
        for i, source_id in enumerate(active_qubits):
            for target_id in active_qubits[i+1:]:
                # Check if already linked
                already_linked = (
                    source_id in self.psionic_links and 
                    target_id in self.psionic_links[source_id]
                )
                
                if not already_linked and random.random() < 0.1:  # 10% chance
                    self.entangle(source_id, target_id)
                    logger.info(f"🧠 Hebbian link: {source_id} ⟷ {target_id}")
    
    def clone(self) -> 'HyperResonanceEngine':
        """Create a deep copy for inheritance."""
        new_engine = HyperResonanceEngine(dimension=self.dimension)
        
        # Copy all qubits
        for node_id, qubit in self.nodes.items():
            new_qubit = HyperQubit(concept_or_value=node_id, name=node_id)
            new_qubit.state = QubitState(
                alpha=qubit.state.alpha,
                beta=qubit.state.beta,
                gamma=qubit.state.gamma,
                delta=qubit.state.delta,
                w=qubit.state.w,
                x=qubit.state.x,
                y=qubit.state.y,
                z=qubit.state.z
            )
            new_engine.nodes[node_id] = new_qubit
        
        # Copy psionic links
        new_engine.psionic_links = {k: list(v) for k, v in self.psionic_links.items()}
        
        return new_engine
    
    def mutate(self):
        """
        Quantum mutation: randomly shift qubit states.
        """
        mutation_rate = 0.1
        
        for qubit in self.nodes.values():
            if random.random() < mutation_rate:
                # Random dimensional shift
                qubit.state.w += random.uniform(-0.2, 0.2)
                qubit.state.w = max(0.0, min(3.0, qubit.state.w))
                
                # Random spatial rotation
                qubit.state.x += random.uniform(-0.1, 0.1)
                qubit.state.y += random.uniform(-0.1, 0.1)
                qubit.state.z += random.uniform(-0.1, 0.1)
                
                # Renormalize
                qubit.state.normalize()
                qubit._normalize_orientation()
    
    def imagination_step(self, inputs: Dict[str, float]) -> str:
        """
        Proactive mode: Predict outcomes and select best action.
        HyperQubit compatible version.
        """
        # Similar to update() but with prediction
        return self.update(inputs)
    
    def predict_outcome(self, action: str) -> Dict[str, float]:
        """
        Predict future quantum state if action is taken.
        Simplified for HyperQubit.
        """
        # Return current probabilities of relevant concepts
        if action in self.nodes:
            qubit = self.nodes[action]
            return qubit.state.probabilities()
        return {}
    
    def process_input_sequence(self, concept: str):
        """Process temporal sequence (for language learning)."""
        self.temporal_buffer.append(concept)
        if len(self.temporal_buffer) > self.buffer_size:
            self.temporal_buffer.pop(0)
    
    def dream(self):
        """Consolidation without external input (Hebbian learning)."""
        self.hebbian_update()


# Backwards compatibility alias
ResonanceEngine = HyperResonanceEngine
