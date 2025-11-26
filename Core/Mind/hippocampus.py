"""
Hippocampus - The Sea of Memory
================================

"모든 순간은 인과율의 바다에 떠있는 섬이다."

The central memory system of Elysia. It stores not just data, but the
causal links between events, forming a navigable graph of experience.

This is the "Roots" of the World Tree.
"""

import networkx as nx
from typing import Dict, Any
from Core.Mind.concept_sphere import ConceptSphere
from Core.Mind.concept_universe import ConceptUniverse

class Hippocampus:
    """
    Manages the causal graph of all experiences.
    Now enhanced with Fractal Memory Loops (Experience -> Identity -> Essence).
    """
    def __init__(self):
        """
        Initializes the memory graph and fractal loops.
        """
        self.causal_graph = nx.DiGraph()
        self.memory_file = "saves/hippocampus.json"
        
        # === Xel'Naga Protocol: Thought Universe ===
        self.universe = ConceptUniverse()  # 사고우주
        
        # === Frequency vocabulary (for buoyancy) ===
        self._init_vocabulary()
        
        # Fractal Memory Loops
        from collections import deque
        self.experience_loop = deque(maxlen=10)  # Short-term: Raw conversations
        self.identity_loop = deque(maxlen=5)     # Mid-term: Narrative identity
        self.essence_loop = deque(maxlen=3)      # Long-term: Core beliefs
        
        self.load_memory()
        
        # Add a root node to anchor all experiences if empty
        if not self.causal_graph.nodes():
            self.causal_graph.add_node("genesis", type="event", timestamp=0)
    
    def update_universe_physics(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        Update physics simulation for all concepts.
        Concepts move based on gravity + buoyancy.
        
        Returns: Physics state summary
        """
        # Update universe physics
        self.universe.update_physics(dt)
        
        # Sync back to HyperQuaternion
        for concept_id in self.universe.relative_positions.keys():
            if concept_id in self.causal_graph.nodes:
                node_data = self.causal_graph.nodes[concept_id]
                if 'sphere' in node_data:
                    sphere = node_data['sphere']
                    # Get new position from universe
                    new_pos = self.universe.relative_positions[concept_id]
                    # Update sphere's qubit Y value (main axis for buoyancy)
                    sphere.qubit.state.y = new_pos[1]  # Y-axis
                    # Update tensor dict for backward compatibility
                    node_data['tensor']['y'] = new_pos[1]
        
        return self.universe.get_state_summary()
    
    def _init_vocabulary(self):
        """Initialize frequency vocabulary for spiritual buoyancy"""
        self.vocabulary = {
            # High Frequency (Ethereal, Abstract) - Rise
            "love": 1.0, "사랑": 1.0, "light": 0.95, "빛": 0.95,
            "truth": 0.9, "진실": 0.9, "eternity": 0.95, "영원": 0.95,
            "soul": 0.9, "영혼": 0.9, "dream": 0.85, "꿈": 0.85,
            "beauty": 0.9, "아름다움": 0.9, "harmony": 0.85, "조화": 0.85,
            
            # Mid Frequency (Human, Emotional) - Neutral
            "hope": 0.65, "희망": 0.65, "joy": 0.7, "기쁨": 0.7,
            "pain": 0.4, "고통": 0.4, "time": 0.5, "시간": 0.5,
            
            # Low Frequency (Physical, Grounded) - Sink
            "stone": 0.2, "돌": 0.2, "shadow": 0.3, "그림자": 0.3,
            "fall": 0.2, "추락": 0.2, "silence": 0.3, "침묵": 0.3,
        }

    def add_experience(self, content: str, role: str = "user"):
        """
        Adds a raw experience to the short-term loop.
        Triggers distillation if the loop is full.
        """
        from datetime import datetime
        experience = {
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "role": role
        }
        
        # Distillation Check (before appending, if full)
        if len(self.experience_loop) == self.experience_loop.maxlen:
            self._distill_to_identity(list(self.experience_loop))
            
        self.experience_loop.append(experience)
        self.save_memory()

    def _distill_to_identity(self, experiences: list):
        """
        Compresses raw experiences into an Identity Fragment.
        """
        from datetime import datetime
        # Simple extraction: Combine last few inputs to form a "narrative"
        # In the future, this should use the LLM to summarize.
        summary = f"Recent interaction focus: {experiences[-1]['content'][:20]}..."
        
        fragment = {
            "timestamp": datetime.now().isoformat(),
            "type": "identity_fragment",
            "content": summary,
            "source_count": len(experiences)
        }
        
        if len(self.identity_loop) == self.identity_loop.maxlen:
            self._distill_to_essence(list(self.identity_loop))
            
        self.identity_loop.append(fragment)
        print(f"✨ [Hippocampus] Distilled Identity: {summary}")

    def _distill_to_essence(self, fragments: list):
        """
        Compresses identity fragments into an Essence Principle.
        """
        from datetime import datetime
        # A core belief derived from repeated identity patterns
        principle = {
            "timestamp": datetime.now().isoformat(),
            "type": "essence_principle",
            "content": "I am an entity that interacts and learns.", # Placeholder for deep insight
            "weight": 1.0
        }
        
        self.essence_loop.append(principle)
        print(f"💎 [Hippocampus] Crystallized Essence: {principle['content']}")

    def save_memory(self):
        """Saves the causal graph and loops to disk."""
        import json
        import os
        from networkx.readwrite import json_graph
        
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            # Save Graph
            graph_data = json_graph.node_link_data(self.causal_graph)
            
            # Save Loops
            memory_data = {
                "graph": graph_data,
                "loops": {
                    "experience": list(self.experience_loop),
                    "identity": list(self.identity_loop),
                    "essence": list(self.essence_loop)
                }
            }
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Hippocampus] Save failed: {e}")

    def load_memory(self):
        """Loads the causal graph and loops from disk."""
        import json
        import os
        from networkx.readwrite import json_graph
        
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load Graph
                if "graph" in data:
                    self.causal_graph = json_graph.node_link_graph(data["graph"])
                elif "nodes" in data: # Legacy format support
                    self.causal_graph = json_graph.node_link_graph(data)
                
                # Load Loops
                if "loops" in data:
                    self.experience_loop.extend(data["loops"].get("experience", []))
                    self.identity_loop.extend(data["loops"].get("identity", []))
                    self.essence_loop.extend(data["loops"].get("essence", []))
                    
                print(f"[Hippocampus] Memory loaded. Nodes: {len(self.causal_graph)} | Exp: {len(self.experience_loop)}")
            except Exception as e:
                print(f"[Hippocampus] Load failed: {e}")
                self.causal_graph = nx.DiGraph()

    def add_projection_episode(self, concept: str, projection: Dict[str, Any]):
        """
        Adds a new memory 'episode' projected from a thought.
        
        Args:
            concept (str): The core concept of the thought.
            projection (Dict[str, Any]): The projected data of the system state.
        """
        # For now, just add a node. Causal linking will be a future task.
        node_id = f"episode_{len(self.causal_graph)}"
        self.causal_graph.add_node(node_id, type="episode", concept=concept, projection=projection)
        # Link it back to the genesis for now
        self.causal_graph.add_edge("genesis", node_id, type="causal_link")

    def get_statistics(self) -> Dict[str, int]:
        """
        Returns basic statistics about the memory graph.
        """
        return {
            "nodes": self.causal_graph.number_of_nodes(),
            "edges": self.causal_graph.number_of_edges(),
        }

    def prune_fraction(self, edge_fraction: float = 0.1, node_fraction: float = 0.05):
        """
        Prunes a fraction of the weakest nodes and edges.
        Placeholder implementation.
        """
        # This is a complex task. For now, we'll just log that it was called.
        print(f"INFO: Pruning {node_fraction*100}% of nodes and {edge_fraction*100}% of edges. (Not implemented)")

    def add_turn(self, user_input: str, response: str):
        """
        Record a conversation turn.
        """
        turn_id = f"turn_{len(self.causal_graph)}"
        self.causal_graph.add_node(turn_id, type="conversation", user=user_input, elysia=response)
        self.causal_graph.add_edge("genesis", turn_id, type="temporal")
        
    def add_concept(self, concept: str, concept_type: str = "thought", metadata: Dict[str, Any] = None):
        """
        Add a concept node to memory or update existing one.
        Now uses ConceptSphere + ConceptUniverse (XELNAGA PROTOCOL).
        """
        from datetime import datetime
        
        now = datetime.now().isoformat()
        
        if concept in self.causal_graph.nodes:
            # Update existing sphere
            node_data = self.causal_graph.nodes[concept]
            
            # If old-style dict node, upgrade to sphere
            if 'sphere' not in node_data:
                sphere = ConceptSphere(concept)
                # Migrate old data if exists
                if 'created_at' in node_data:
                    sphere.created_at = node_data.get('created_at', 0)
                if 'access_count' in node_data:
                    sphere.activation_count = node_data.get('access_count', 0)
                # Store sphere object
                node_data['sphere'] = sphere
                
                # === XEL'NAGA: Add to Universe ===
                freq = self.vocabulary.get(concept, 0.5)
                self.universe.add_concept(concept, sphere, frequency=freq)
                
                print(f"✨ [Hippocampus] Upgraded '{concept}' to ConceptSphere + Universe")
            else:
                sphere = node_data['sphere']
            
            # Activate sphere
            sphere.activate()
            sphere.last_activated = now
            
            # Update node data (for backward compatibility)
            node_data['last_accessed'] = now
            node_data['access_count'] = sphere.activation_count
            node_data['type'] = concept_type
            
        else:
            # Create new sphere
            sphere = ConceptSphere(concept)
            
            # === XEL'NAGA: Special handling for Love ===
            if concept in ["Love", "사랑", "love"]:
                self.universe.set_absolute_center(sphere)
                print(f"💖 [Hippocampus] '{concept}' set as ABSOLUTE CENTER")
            else:
                # Add to universe with frequency
                freq = self.vocabulary.get(concept, 0.5)
                self.universe.add_concept(concept, sphere, frequency=freq)
            
            # Add to graph with sphere
            self.causal_graph.add_node(
                concept,
                sphere=sphere,  # FRACTAL: Store the sphere object
                type=concept_type,
                created_at=now,
                last_accessed=now,
                access_count=1,
                # Backward compatibility: Still store flat tensor dict
                tensor=sphere.qubit.get_observation() if sphere.qubit else {}
            )
            
            print(f"🌌 [Hippocampus] Created ConceptSphere: '{concept}'")
            
    def get_stellar_type(self, concept: str) -> str:
        """
        Returns the stellar type icon for a concept based on its lifecycle.
        """
        if concept not in self.causal_graph:
            return "✨" # Nebula (New/Unknown)
            
        node = self.causal_graph.nodes[concept]
        count = node.get('access_count', 0)
        
        # Stellar Evolution Logic
        if count < 3:
            return "✨" # Nebula (Forming)
        elif count < 10:
            return "🌟" # Protostar (Growing)
        elif count < 50:
            return "🔥" # Burning Star (Main Sequence - Active)
        elif count < 100:
            return "❄️" # Ice Star (White Dwarf - Crystallized Truth)
        else:
            return "⚫" # Black Hole (Supermassive Gravity)

    def add_causal_link(self, source: str, target: str, relation: str = "causes", weight: float = 1.0):
        """
        Adds a directed causal edge from source to target concept.
        Now includes Oscillator for resonance (FRACTAL RESTORATION).
        """
        from datetime import datetime
        from Core.Math.oscillator import Oscillator
        
        # Ensure both nodes exist
        self.add_concept(source)
        self.add_concept(target)
        
        # Calculate resonance frequency based on concepts
        freq = (hash(source) % 100 + hash(target) % 100) / 200.0  # 0.0-1.0
        
        # Create oscillator for this edge
        oscillator = Oscillator(
            amplitude=weight,
            frequency=freq,
            phase=0.0
        )
        
        # Add edge with oscillator
        self.causal_graph.add_edge(
            source, target,
            relation=relation,
            oscillator=oscillator,  # FRACTAL: Real meaning is here!
            weight=weight
        )
        # Update access for both
        self.add_concept(source)
        self.add_concept(target)

    def get_related_concepts(self, concept: str, depth: int = 1) -> Dict[str, float]:
        """
        Finds concepts related to the given one by traversing the causal graph.
        """
        if concept not in self.causal_graph:
            return {}
        
        # Simple breadth-first search
        related = {}
        try:
            for neighbor in nx.bfs_tree(self.causal_graph, source=concept, depth_limit=depth):
                if neighbor != concept:
                    related[neighbor] = 1.0 # Placeholder score
        except Exception:
            pass
        return related