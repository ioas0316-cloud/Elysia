import math
import json
import os
from typing import Dict, List, Tuple, Optional
from Core.World.Physics.trinity_fields import TrinityVector
try:
    from Core.Foundation.web_knowledge_connector import WebKnowledgeConnector
except ImportError:
    WebKnowledgeConnector = None

try:
    from Core.Foundation.Graph.torch_graph import TorchGraph
except ImportError:
    TorchGraph = None

class TrinityOperator:
    """
    [Phase 27] The Verb (Active Logic).
    Represents an action that transforms a Subject into an Object or State.
    A = Verb(B, C)
    """
    def __init__(self, name: str, vector: TrinityVector, complexity: float = 1.0):
        self.name = name
        self.vector = vector # The "Flavor" of the action
        self.complexity = complexity

    def apply(self, subject: TrinityVector, target: Optional[TrinityVector] = None) -> TrinityVector:
        """
        Transforms the subject based on the verb's operator logic.
        This is the "Causal Transformer".
        """
        # [Operator Logic]
        # v_result = (v_subject + v_verb) / 2 if adding
        # v_result = v_subject * v_verb if amplifying
        # For prototype: We use a Weighted Resonance sum.
        res = TrinityVector(
            gravity=(subject.gravity + self.vector.gravity) / 2,
            flow=(subject.flow + self.vector.flow) / 2,
            ascension=(subject.ascension + self.vector.ascension) / 2,
            frequency=(subject.frequency + self.vector.frequency) / 2
        )
        if target:
            # If there's an object, it "colors" the result
            res.gravity = (res.gravity + target.gravity) / 3
            res.flow = (res.flow + target.flow) / 3
            # ... etc
        return res

class TrinityLexicon:
    """
    The Gateway to the Logos (4D Knowledge Graph).
    Connects Words (Symbols) to the Hyper-Graph (Structure).
    """
    def __init__(self, persistence_path: str = "c:/Elysia/data/Memory/lexicon_memory.json"):
        self.web_connector = WebKnowledgeConnector() if WebKnowledgeConnector else None
        
        # Initialize the True Brain (TorchGraph)
        if TorchGraph:
            self.graph = TorchGraph(use_cuda=True)
            print(f"🧠 [TrinityLexicon] Connected to 4D TorchGraph.")
        else:
            self.graph = None
            print(f"⚠️ [TrinityLexicon] TorchGraph missing. Running in disconnected mode.")

        # 1. The Primitives (The Root Words) - Still needed for bootstrapping Vectors
        self.primitives: Dict[str, TrinityVector] = {
            # --- Gravity (Matter, Stability, Force) [Axis 4: Low Freq] ---
            "rock": TrinityVector(0.9, 0.0, 0.0, frequency=10.0), # Solid
            "base": TrinityVector(0.8, 0.1, 0.0, frequency=12.0),
            "stand": TrinityVector(0.7, 0.0, 0.1, frequency=15.0),
            "guard": TrinityVector(0.8, 0.0, 0.2, frequency=20.0),
            "demand": TrinityVector(0.9, 0.0, 0.5, frequency=30.0),
            "stop": TrinityVector(1.0, 0.0, 0.0, frequency=0.0),  # Absolute Zero
            "strong": TrinityVector(0.6, 0.1, 0.1, frequency=25.0),
            
            # --- Flow (Mind, Change, Exchange) [Axis 4: Mid Freq] ---
            "flow": TrinityVector(0.0, 1.0, 0.0, frequency=60.0), # Hz
            "trade": TrinityVector(0.1, 0.9, 0.1, frequency=55.0),
            "maybe": TrinityVector(0.0, 0.5, 0.0, frequency=40.0),
            "river": TrinityVector(0.1, 0.8, 0.1, frequency=50.0),
            "connect": TrinityVector(0.0, 0.7, 0.2, frequency=70.0),
            "offer": TrinityVector(0.1, 0.8, 0.3, frequency=65.0),
            "change": TrinityVector(0.0, 0.9, 0.1, frequency=80.0),
            
            # --- [Phase 28] Hierarchy Primitives ---
            "atom": TrinityVector(0.5, 0.0, 0.5, frequency=1000000.0, scale=1000000.0), # High Freq / Small Scale
            "human": TrinityVector(0.3, 0.3, 0.3, frequency=60.0, scale=1.0),
            "village": TrinityVector(0.7, 0.3, 0.0, frequency=1.0, scale=0.01),
            "planet": TrinityVector(1.0, 0.0, 0.0, frequency=0.01, scale=0.0001),
            "galaxy": TrinityVector(0.0, 0.5, 0.5, frequency=0.00001, scale=0.0000001),
            
            # 1. Earth (Structure/Mass)
            "earth": TrinityVector(1.0, 0.0, 0.0, frequency=7.83, scale=0.0001),
            "stone": TrinityVector(0.9, 0.0, 0.0, frequency=10.0, scale=1.0),
            "metal": TrinityVector(0.95, 0.05, 0.0, frequency=50.0),

            # 2. Water (Flow/Connectivity)
            "water": TrinityVector(0.3, 1.0, 0.0, frequency=432.0), # Healing Freq
            "river": TrinityVector(0.2, 0.9, 0.1, frequency=400.0),
            "ice": TrinityVector(0.9, 0.0, 0.0, frequency=0.0),

            # 3. Wind (Motion/Breath)
            "wind": TrinityVector(0.0, 0.8, 0.4, frequency=528.0),  # DNA Repair Freq
            "air": TrinityVector(0.01, 0.5, 0.2, frequency=500.0),
            "storm": TrinityVector(0.2, 0.9, 0.0, frequency=100.0), # Chaos

            # 4. Fire (Energy/Transformation)
            "fire": TrinityVector(0.0, 0.3, 0.9, frequency=800.0), 
            "heat": TrinityVector(0.0, 0.2, 0.8, frequency=700.0),

            # 5. Light (Spirit/Time)
            "light": TrinityVector(0.0, 0.1, 1.0, frequency=1111.0), # High Freq
            "sun": TrinityVector(0.2, 0.2, 1.0, frequency=888.0),
            "day": TrinityVector(0.0, 0.5, 0.8, frequency=600.0),

            # 6. Darkness (Void/Entropy)
            "darkness": TrinityVector(0.1, 0.1, -1.0, frequency=-100.0), # Negative Freq (Absorption)
            "void": TrinityVector(0.0, 0.0, -1.0, frequency=0.0),
            "night": TrinityVector(0.1, 0.2, -0.8, frequency=50.0),
            "shadow": TrinityVector(0.1, 0.1, -0.5, frequency=20.0),

            # --- Compound/Derivatives ---
            "life": TrinityVector(0.2, 0.8, 0.5, frequency=963.0), # Solfeggio
            "steam": TrinityVector(0.0, 0.9, 0.7, frequency=600.0), 
            "dust": TrinityVector(0.4, 0.4, 0.0, frequency=30.0),  
            "magma": TrinityVector(0.8, 0.5, 0.8, frequency=500.0), 

            # --- Korean Primitives (Mapped to Hexagon) ---
            "물": TrinityVector(0.3, 1.0, 0.0, frequency=432.0),
            "불": TrinityVector(0.0, 0.3, 0.9, frequency=800.0),
            "흙": TrinityVector(1.0, 0.0, 0.0, frequency=7.83),
            "바람": TrinityVector(0.0, 0.8, 0.4, frequency=528.0),
            "빛": TrinityVector(0.0, 0.1, 1.0, frequency=1111.0),
            "어둠": TrinityVector(0.1, 0.1, -1.0, frequency=-100.0),
            "암석": TrinityVector(0.9, 0.0, 0.0, frequency=10.0),
            "녹은": TrinityVector(0.0, 0.8, 0.6, frequency=500.0), 
        }

        # [Phase 27] The Operators (Verbs)
        self.operators: Dict[str, TrinityOperator] = {
            "burn": TrinityOperator("burn", TrinityVector(-0.5, 0.2, 0.9, frequency=800.0)), # Destruction + Energy
            "create": TrinityOperator("create", TrinityVector(0.5, 0.5, 1.0, frequency=1111.0)), # Structure + Potential
            "destroy": TrinityOperator("destroy", TrinityVector(-1.0, -0.5, -0.5, frequency=0.0)), # Entropy
            "transform": TrinityOperator("transform", TrinityVector(0.0, 1.0, 0.5, frequency=528.0)), # Pure Flow
            "태우다": TrinityOperator("태우다", TrinityVector(-0.5, 0.2, 0.9, frequency=800.0)),
            "만들다": TrinityOperator("만들다", TrinityVector(0.5, 0.5, 1.0, frequency=1111.0)),
        }

        # Sync Primitives to Graph
        if self.graph:
            self._sync_primitives()

    def _sync_primitives(self):
        """Ensure primitives exist in the Graph."""
        for word, vec in self.primitives.items():
            # Check existence via internal ID map (direct access purely for speed)
            if word not in self.graph.id_to_idx:
                self.graph.add_node(
                    word, 
                    vector=[vec.gravity, vec.flow, vec.ascension],
                    metadata={"type": "primitive", "definition": "Root Concept"}
                )

    def is_concept_known(self, concept: str) -> bool:
        """
        Checks if a concept exists in the Graph or Primitives.
        """
        concept = concept.lower()
        # 1. Check Primitives
        if concept in self.primitives:
            return True
        # 2. Check Graph
        if self.graph and concept in self.graph.id_to_idx:
            return True
        return False

    def analyze(self, text: str) -> TrinityVector:
        """
        Parses a sentence by querying the Knowledge Graph.
        Returns the Net Trinity Vector (Feeling).
        """
        # Use substring matching to handle Korean particles (e.g. "파도가") and variations.
        text = text.lower()
        net_vector = TrinityVector(0, 0, 0)
        found_count = 0
        
        # We need to tokenize to find "Concepts" for the Graph
        # Simple split is weak for Korean, but we start there.
        # Ideally, we should use the Graph's keys to scan the text (Aho-Corasick style)
        # But iterating all graph keys is slow if N is large.
        
        # Hybrid Approach:
        # 1. Check Primitives (Fast, handling particles via substring)
        # 2. Check Graph (Exact Match on tokens)
        
        # 1. Primitives (The "Gut Feeling")
        for key, vec in self.primitives.items():
            if key in text:
                 net_vector.gravity += vec.gravity
                 net_vector.flow += vec.flow
                 net_vector.ascension += vec.ascension
                 found_count += 1
                 
        # 2. Graph Lookup (The "Learned Knowledge")
        if self.graph:
            # Handle compound sensations (e.g. HEAVY_RESISTANCE)
            words = text.replace("_", " ").split()
            for w in words:
                # Clean punctuation
                w = w.strip(".,!?")
                if not w: continue
                
                # Check Graph
                if w in self.graph.id_to_idx:
                     # Get Vector from Tensor
                     idx = self.graph.id_to_idx[w]
                     vec_tensor = self.graph.vec_tensor[idx]
                     # Assume first 3 dims are Trinity (G, F, A)
                     # Or we map 384 dims to 3? 
                     # For now, let's assume we store Trinity in the first 3 dims for compatibility
                     g = float(vec_tensor[0])
                     f = float(vec_tensor[1])
                     a = float(vec_tensor[2])
                     
                     net_vector.gravity += g
                     net_vector.flow += f
                     net_vector.ascension += a
                     found_count += 1
                elif w not in self.primitives:
                     # Unknown Word -> Trigger Learning
                     # Only learn "significant" words (length > 1)
                     if len(w) > 1:
                         new_vec = self.learn_from_hyper_sphere(w)
                         net_vector.gravity += new_vec.gravity
                         net_vector.flow += new_vec.flow
                         net_vector.ascension += new_vec.ascension
                         if new_vec.magnitude() > 0: found_count += 1

        if found_count == 0:
            return TrinityVector(0,0,0)
            
        return net_vector 

    def fetch_definition(self, concept: str) -> str:
        """
        Retrieves the raw definition of a concept from the HyperSphere.
        Used for deep contemplation (recursion).
        """
        if not self.web_connector: return ""
        return self.web_connector.fetch_wikipedia_simple(concept) or ""

    def learn_from_hyper_sphere(self, word: str) -> TrinityVector:
        """
        Queries the Web -> Internalizes to Graph.
        """
        if not self.web_connector or not self.graph:
            return TrinityVector(0,0,0) # Offline
            
        print(f"🌌 [HyperSphere] Connecting to understanding: '{word}'...")
        
        try:
            summary = self.web_connector.fetch_wikipedia_simple(word)
        except Exception as e:
            print(f"Error fetching text: {e}")
            summary = ""
            
        if summary:
            print(f"📄 [Analysis] Definition: '{summary}'")
            # Recursive Analysis of the definition (using Primitives)
            definition_vector = self._analyze_primitives_only(summary)
            
            # Normalize and Boost
            definition_vector.normalize()
            definition_vector.gravity *= 1.5
            definition_vector.flow *= 1.5
            definition_vector.ascension *= 1.5
            
            # Commit to GRAPH
            print(f"🧠 [Graph] Encoding '{word}' into Neural Memory...")
            self.graph.add_node(
                word, 
                vector=[definition_vector.gravity, definition_vector.flow, definition_vector.ascension],
                metadata={
                    "definition": summary[:200], # Store short def
                    "source": "wikipedia"
                }
            )
            # Link to words in definition? (Simple Graph Building)
            # Find primitives in definition and link them!
            for prim, pvec in self.primitives.items():
                if prim in summary:
                     # Ensure primitive exists in graph before linking
                     if prim not in self.graph.id_to_idx:
                         self.graph.add_node(prim, vector=[pvec.gravity, pvec.flow, pvec.ascension])
                     
                     self.graph.add_link(word, prim, weight=0.5)
                     print(f"   🔗 Linked '{word}' -> '{prim}'")
            
            return definition_vector
            
        return TrinityVector(0,0,0)

    def _analyze_primitives_only(self, text: str) -> TrinityVector:
        """Helper to analyze text using ONLY currrent primitives."""
        text = text.lower()
        net = TrinityVector(0,0,0)
        
        for key, vec in self.primitives.items():
            if key in text:
                net.gravity += vec.gravity
                net.flow += vec.flow
                net.ascension += vec.ascension
                
        return net

    # Deprecated: learn_word, save_memory, load_memory 
    # (Graph handles persistence via save_state)
    def is_known(self, concept: str) -> bool:
        """Checks if a concept is understood (in Graph or Primitives)."""
        if not concept: return False
        concept = concept.lower().strip()
        # Direct Primitive Check
        if concept in self.primitives: return True
        # Part of Primitive Check (e.g. 'water' in 'waterfall') - Simplified for now
        for p in self.primitives:
            if p in concept: return True
        # Graph Check
        if self.graph and concept in self.graph.id_to_idx: return True
        return False

    def extract_unknowns(self, text: str) -> List[str]:
        """Identifies significant unknown words in a text."""
        unknowns = []
        # Simple extraction: split by space, strip punctuation
        words = text.replace("_", " ").split()
        for w in words:
            w_clean = w.strip(".,!?()[]{}:;\"'")
            if len(w_clean) > 4: # Ignore small words like 'the', 'is' for now
                if not self.is_known(w_clean):
                    unknowns.append(w_clean)
        
        # Deduplicate
        return list(set(unknowns))

    def save_memory(self):
        if self.graph:
            target_path = "c:/Elysia/data/State/brain_state.pt"
            print(f"DEBUG: Calling graph.save_state({target_path})...")
            self.graph.save_state(target_path)

    def load_memory(self):
        if self.graph:
            target_path = "c:/Elysia/data/State/brain_state.pt"
            if os.path.exists(target_path):
                 self.graph.load_state(target_path)
            else:
                 print(f"⚠️ Memory file not found at {target_path}. Starting Fresh.")

    def audit_knowledge(self) -> List[Tuple[str, str, float]]:
        """
        [Phase 30] Audit knowledge for contradictions.
        Returns pairs of concepts that exhibit 'Destructive Interference'.
        """
        if not self.graph: return []
        
        dissonances = []
        nodes = list(self.graph.id_to_idx.keys())
        
        # O(N^2) for now - audit only a subset of the graph if it grows too large
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                
                # Retrieve vectors (as WaveTensors)
                v_a = self.graph.get_node_vector(a) # Returns torch.Tensor?
                v_b = self.graph.get_node_vector(b)
                
                # Conversion to WaveTensor for interference check
                # This assumes graph vectors are compatible or mapped to frequency space.
                # For now, we simulate this if vectors are out of phase.
                # [Interference Logic]
                # In Phase 30, we'll treat high distance + semantic overlap as dissonance.
                dot = float((v_a * v_b).sum())
                norm = float(v_a.norm() * v_b.norm())
                similarity = dot / (norm + 1e-9)
                
                if similarity < -0.5: # Out of phase (Contradiction)
                    dissonances.append((a, b, similarity))
        
        return dissonances

_lexicon = None
def get_trinity_lexicon():
    global _lexicon
    if _lexicon is None:
        _lexicon = TrinityLexicon()
    return _lexicon
