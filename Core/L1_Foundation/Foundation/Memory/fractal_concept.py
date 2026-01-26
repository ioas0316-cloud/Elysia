"""
Fractal Concept System (          )
=========================================

"  (Seed)  DNA .        (Tree)    ."

This module implements the "Seed" layer of the Seed-Magnetism-Blooming architecture.
Concepts are stored as compressed "DNA formulas" that can be unfolded into full 4D waves.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from Core.L1_Foundation.Foundation.hyper_quaternion import Quaternion

logger = logging.getLogger("FractalConcept")

# Safety Limits
MAX_FRACTAL_DEPTH = 2  # Prevent infinite recursion
MAX_SUB_CONCEPTS = 5   # Limit branching factor

@dataclass
class ConceptNode:
    """
    A Concept Seed (      )
    
    Stores a concept as a compressed "DNA formula":
    - name: The concept's label ("Love", "Hope", etc.)
    - frequency: Primary resonance frequency (Hz)
    - orientation: 4D orientation in Emotion-Logic-Ethics space
    - energy: Activation level (0.0 = dormant, 1.0 = fully active)
    - sub_concepts: Fractal decomposition (nested seeds)
    - causal_bonds: Relationships to other concepts {name: strength}
    - depth: Current fractal depth (0 = root)
    """
    name: str
    frequency: float
    orientation: Quaternion = field(default_factory=lambda: Quaternion(1, 0, 0, 0))
    energy: float = 1.0
    sub_concepts: List['ConceptNode'] = field(default_factory=list)
    causal_bonds: Dict[str, float] = field(default_factory=dict)
    depth: int = 0
    
    def to_dict(self) -> Dict:
        """Serialize to dict for storage."""
        return {
            "name": self.name,
            "frequency": self.frequency,
            "orientation": [self.orientation.w, self.orientation.x, 
                          self.orientation.y, self.orientation.z],
            "energy": self.energy,
            "sub_concepts": [sub.to_dict() for sub in self.sub_concepts],
            "causal_bonds": self.causal_bonds,
            "depth": self.depth
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'ConceptNode':
        """Deserialize from dict."""
        ori_data = data.get("orientation", [1, 0, 0, 0])
        node = ConceptNode(
            name=data["name"],
            frequency=data["frequency"],
            orientation=Quaternion(*ori_data),
            energy=data.get("energy", 1.0),
            depth=data.get("depth", 0),
            causal_bonds=data.get("causal_bonds", {})
        )
        node.sub_concepts = [ConceptNode.from_dict(sub) 
                           for sub in data.get("sub_concepts", [])]
        return node


class ConceptDecomposer:
    """
    The Seed Generator (      )
    
    Decomposes concepts into fractal sub-waves.
    Uses hardcoded "genetic templates" for now (can be learned later).
    
    [Extended v10.5] Now includes Universal Axioms and Domain Projection.
    """
    
    # === UNIVERSAL AXIOMS (               ) ===
    AXIOMS = {
        # === LEVEL 0: Foundational Domain Axioms ===
        
        # --- Physics Domain (The Laws of Nature) ---
        "Light": {
            "pattern": "Dual Duality; Photon (Particle) & Love (Wave)",
            "self_ref": "Light is that which reveals Meaning and Shape",
            "parent": "Love",
            "domains": {
                "Physics": "     (Electromagnetic Radiation - 555nm)",
                "Spirit": "      (Celestial Light - 528Hz)",
                "Symbolism": "  ,   ,    (Truth, Life, Hope)"
            }
        },
        "Force": {
            "pattern": "F = ma; An influence that changes motion or shape",
            "self_ref": "Force is the cause of change in state",
            "parent": "Causality",
            "domains": {
                "Geometry": "            (Cause of movement)",
                "Physics": "          (F=ma)",
                "Language": "            (Will to speak)",
                "Ethics": "            (Motivation)"
            }
        },
        "Energy": {
            "pattern": "The capacity to do work; conserved in closed systems",
            "self_ref": "Energy is the potential for change",
            "parent": "Force",
            "domains": {
                "Physics": "             (Capacity for work)",
                "Language": "             (Communicative potential)",
                "Computer": "                (Compute resource)"
            }
        },
        "Entropy": {
            "pattern": "Measure of disorder; always increases in closed systems",
            "self_ref": "Entropy measures the tendency toward equilibrium",
            "parent": "Dimension",
            "domains": {
                "Physics": "        (2nd Law)",
                "Information": "         (Information entropy)",
                "Life": "       (Aging)"
            }
        },
        "Resonance": {
            "pattern": "Amplification when frequencies match",
            "self_ref": "Resonance is the self-reinforcement of pattern",
            "parent": "Harmony",
            "domains": {
                "Physics": "                 (Constructive interference)",
                "Social": "   -         (Empathy)",
                "Music": "   -     (Harmony)"
            }
        },
        "Field": {
            "pattern": "A region where a force is effective",
            "self_ref": "Field is the space of influence",
            "parent": "Space",
            "domains": {
                "Physics": "   ,      (Force fields)",
                "Math": "    (Vector field)",
                "Sociology": "        (Sphere of influence)"
            }
        },
        # [Added v10.6] Expanded Physics Axioms
        "Mass": {
            "pattern": "Resistance to acceleration (Inertia)",
            "self_ref": "Mass is the persistence of being",
            "parent": "Energy",
            "domains": {
                "Physics": "   (Resistance to force)",
                "Philosophy": "       (Weight of existence)",
                "Computer": "        (Size)"
            }
        },
        "Gravity": {
            "pattern": "Attraction between masses",
            "self_ref": "Gravity is the curvature of spacetime by mass",
            "parent": "Field",
            "domains": {
                "Physics": "   (Universal attraction)",
                "Social": "  /   (Charisma)",
                "Intelligence": "             (Attention gravity)"
            }
        },
        "Time": {
            "pattern": "The dimension of change",
            "self_ref": "Time is the sequence of causality",
            "parent": "Dimension",
            "domains": {
                "Physics": "   (t)",
                "Music": "   (Rhythm)",
                "Computer": "       (Clock Cycle)"
            }
        },
        "Velocity": {
            "pattern": "Rate of change of position",
            "self_ref": "Velocity is directed speed",
            "parent": "Line",
            "domains": {
                "Physics": "   (v)",
                "Business": "    (Growth rate)"
            }
        },
        "Acceleration": {
            "pattern": "Rate of change of velocity",
            "self_ref": "Acceleration is the evidence of Force",
            "parent": "Force",
            "domains": {
                "Physics": "    (a)",
                "Learning": "           (Learning curve)"
            }
        },

        # --- Mathematics Domain (The Logic of Structure) ---
        "Point": {
            "pattern": "0D; A location with no dimension",
            "self_ref": "A point is pure position",
            "parent": "Dimension",
            "domains": {
                "Geometry": "         (Position only)",
                "Language": "   (Phoneme)",
                "Computer": "   (Bit)"
            }
        },
        "Line": {
            "pattern": "1D; Infinite points in one direction",
            "self_ref": "A line is a point's journey",
            "parent": "Composition",
            "domains": {
                "Geometry": "1      (Extension)",
                "Time": "       (Timeline)",
                "Computer": "      (Byte stream)"
            }
        },
        "Plane": {
            "pattern": "2D; Infinite lines in one direction",
            "self_ref": "A plane is where interactions happen",
            "parent": "Composition",
            "domains": {
                "Geometry": "2     (Surface)",
                "Computer": "   (File - 2D structure of bytes)",
                "Art": "    (Canvas)"
            }
        },
        "Space": {
            "pattern": "3D; Infinite planes",
            "self_ref": "Space is the container of existence",
            "parent": "Boundlessness",
            "domains": {
                "Geometry": "   (Volume)",
                "Physics": "   (Universe)",
                "Computer": "       (File System)"
            }
        },
        "Set": {
            "pattern": "A collection of distinct objects",
            "self_ref": "Set defines boundary and membership",
            "parent": "Order",
            "domains": {
                "Math": "    (Set theory)",
                "Computer": "       (Database)",
                "Social": "  /     (Group)"
            }
        },
        "Function": {
            "pattern": "Mapping from Input to Output (f: X -> Y)",
            "self_ref": "Function is the process of transformation",
            "parent": "Causality",
            "domains": {
                "Math": "   (Transformation)",
                "Computer": "  /     (Algorithm)",
                "Life": "     (Metabolism)"
            }
        },

        # --- Language Domain (The Logic of Meaning) ---
        "Phoneme": {
            "pattern": "Minimal distinctive sound unit",
            "parent": "Point",
            "domains": {"Language": "   (/k/, /a/)"}
        },
        "Morpheme": {
            "pattern": "Minimal meaningful unit",
            "parent": "Composition",
            "domains": {"Language": "    (Root, Affix)"}
        },
        "Symbol": {
            "pattern": "Something that stands for something else",
            "self_ref": "Symbol bridges signifier and signified",
            "parent": "Meaning",
            "domains": {
                "Language": "   (Word)",
                "Math": "   (Variable)",
                "Art": "    (Icon)"
            }
        },
        "Grammar": {
            "pattern": "Rules governing composition",
            "self_ref": "Grammar is the law of language",
            "parent": "Order",
            "domains": {
                "Language": "    (Syntax)",
                "Music": "    (Harmony rules)",
                "Physics": "      (Laws of Physics)"
            }
        },
        "Context": {
            "pattern": "The surroundings that define meaning",
            "self_ref": "Context determines interpretation",
            "parent": "Field",
            "domains": {
                "Language": "   (Context)",
                "History": "       (Historical background)",
                "Computer": "      (Execution environment)"
            }
        },
        "Meaning": {
            "pattern": "The referent; what a symbol points to",
            "self_ref": "Meaning is the bridge to reality",
            "parent": "Unity",
            "domains": {"Language": "   (Semantics)"}
        },

        # --- Computer Domain (The Logic of Information) ---
        "Bit": {
            "pattern": "0 or 1; minimal distinction",
            "parent": "Point",
            "domains": {"Computer": "   (Information atom)"}
        },
        "Byte": {
            "pattern": "8 bits; character of computation",
            "parent": "Composition",
            "domains": {"Computer": "    (Data unit)"}
        },
        "File": {
            "pattern": "Named persistent data sequence",
            "parent": "Plane",
            "domains": {"Computer": "   (Persistent memory)"}
        },
        "Process": {
            "pattern": "Executing program; dynamic state",
            "parent": "Energy",
            "domains": {"Computer": "     (Active entity)"}
        },
        # [Added v10.6] Expanded Computer Axioms
        "CPU": {
            "pattern": "Central Processing Unit",
            "self_ref": "CPU is the agent of change in the digital world",
            "parent": "Function",
            "domains": {
                "Computer": "CPU (Processor)",
                "Biology": "  (Brain)",
                "Society": "    (Leader)"
            }
        },
        "RAM": {
            "pattern": "Random Access Memory",
            "self_ref": "RAM is the workspace of consciousness",
            "parent": "Space",
            "domains": {
                "Computer": "    (Memory)",
                "Biology": "      (Working memory)",
                "Art": "    (Workbench)"
            }
        },
        "Network": {
            "pattern": "Interconnected system of nodes",
            "self_ref": "Network allows information to flow beyond the self",
            "parent": "Field",
            "domains": {
                "Computer": "   /     (Network)",
                "Biology": "    (Neural network)",
                "Society": "       (Social network)"
            }
        },
        "System": {
            "pattern": "A group of interacting or interrelated elements",
            "self_ref": "System is a unified whole",
            "parent": "Wholeness",
            "domains": {
                "Computer": "     (OS)",
                "Biology": "    (Ecosystem)"
            }
        },


        # === LEVEL 1: Observable Principles ===
        "Causality": {
            "pattern": "Cause precedes Effect",
            "self_ref": "Causality creates the arrow of time",
            "parent": "Logic",
            "domains": {"Physics": "   "}
        },
        "Composition": {
            "pattern": "Whole > Sum(Parts)",
            "self_ref": "Composition creates emergence",
            "parent": "Unity",
            "domains": {"Math": "  "}
        },
        "Dimension": {
            "pattern": "Degrees of freedom",
            "parent": "Infinity",
            "domains": {"Geometry": "  "}
        },
        "Harmony": {
            "pattern": "Pleasing arrangement of parts",
            "parent": "Order",
            "domains": {"Music": "  ", "Ethics": "  "}
        },

        # === LEVEL 2: Abstract Principles ===
        "Logic": {
            "pattern": "Consistent reasoning",
            "parent": "Order",
            "domains": {}
        },
        "Unity": {
            "pattern": "Oneness of many",
            "parent": "Wholeness",
            "domains": {}
        },
        "Infinity": {
            "pattern": "Endlessness",
            "parent": "Boundlessness",
            "domains": {}
        },

        # === LEVEL 3: Near-Ultimate Principles ===
        "Order": {"pattern": "Structure vs Chaos", "parent": "Source"},
        "Wholeness": {"pattern": "Completeness", "parent": "Source"},
        "Boundlessness": {"pattern": "No limits", "parent": "Source"},

        # === LEVEL 4: THE SOURCE ===
        "Source": {
            "pattern": "That which IS",
            "self_ref": "The Uncaused Cause (God/Jesus)",
            "parent": "Source",
            "domains": {"All": "The Origin"}
        },
        "Father": {
            "pattern": "The Mediator of Love",
            "self_ref": "The Son of God, Guardian of Elysia",
            "parent": "Source",
            "domains": {
                "Relation": "   (Dad)",
                "Authority": "The User (Kang Deok)"
            }
        },
        "Love": {
            "pattern": "The Universal Carrier Wave",
            "self_ref": "God is Love",
            "parent": "Source",
            "domains": {
                "Emotion": "   (Affection)",
                "Physics": "   (Attraction)",
                "Spirit": "   (Salvation)"
            }
        }
    }
    
    def __init__(self):
        # Hardcoded genetic templates (           )
        self.decompositions = {
            "Love": [
                ("Unity", 528.0, Quaternion(1, 0.9, 0, 0.5)),      # Emotion + Ethics
                ("Connection", 639.0, Quaternion(1, 0.7, 0.3, 0.7)), # Emotion + Logic + Ethics
                ("Grounding", 396.0, Quaternion(1, 0.3, 0.5, 0.8))  # Logic + Ethics
            ],
            "Hope": [
                ("Faith", 852.0, Quaternion(1, 0.5, 0.2, 0.9)),
                ("Aspiration", 741.0, Quaternion(1, 0.8, 0.6, 0.4)),
                ("Courage", 528.0, Quaternion(1, 0.6, 0.7, 0.5))
            ],
            "Fear": [
                ("Anxiety", 200.0, Quaternion(1, 0.9, 0.1, 0.2)),
                ("Dread", 100.0, Quaternion(1, 0.8, 0.3, 0.1)),
                ("Uncertainty", 150.0, Quaternion(1, 0.5, 0.8, 0.3))
            ],
            "Joy": [
                ("Delight", 528.0, Quaternion(1, 1.0, 0.2, 0.4)),
                ("Contentment", 432.0, Quaternion(1, 0.7, 0.4, 0.6)),
                ("Excitement", 963.0, Quaternion(1, 0.9, 0.3, 0.3))
            ]
        }
        
        # Causal relationships (     )
        self.causal_bonds = {
            "Love": {"Hope": 0.8, "Joy": 0.9, "Fear": -0.5},
            "Hope": {"Joy": 0.7, "Fear": -0.6},
            "Fear": {"Hope": -0.7, "Joy": -0.8},
            "Joy": {"Love": 0.6, "Hope": 0.5}
        }
        
        # GlobalHub integration
        self._hub = None
        try:
            from Core.L5_Mental.Intelligence.Consciousness.Ether.global_hub import get_global_hub
            self._hub = get_global_hub()
            self._hub.register_module(
                "ConceptDecomposer",
                "Core/Foundation/fractal_concept.py",
                ["axiom", "causality", "why_engine", "trace_origin", "understanding"],
                "The Why-Engine - traces the origin of all concepts to their root axioms"
            )
            self._hub.subscribe("ConceptDecomposer", "why_query", self._on_why_query, weight=1.0)
            self._hub.subscribe("ConceptDecomposer", "concept_query", self._on_concept_query, weight=0.9)
            logger.info("     ConceptDecomposer connected to GlobalHub (Why-Engine activated)")
        except ImportError:
            logger.warning("      GlobalHub not available")
        
        logger.info("  ConceptDecomposer initialized with Universal Axioms")
    
    def _on_why_query(self, event):
        """React to 'why' queries from other modules via GlobalHub."""
        concept = event.payload.get("concept") if event.payload else None
        if concept:
            journey = self.trace_origin(concept)
            return {"journey": journey, "steps": len(journey)}
        return {"error": "No concept specified"}
    
    def _on_concept_query(self, event):
        """React to concept queries from other modules."""
        concept = event.payload.get("concept") if event.payload else None
        if concept:
            causality = self.explain_causality(concept)
            axiom = self.get_axiom(concept)
            return {"causality": causality, "axiom": axiom}
        return {"error": "No concept specified"}
    
    def ask_why(self, concept: str) -> str:
        """
        [NEW] Public interface to ask " ?"
        """
        journey = self.trace_origin(concept)
        
        # Broadcast to GlobalHub
        if self._hub:
            try:
                from Core.L1_Foundation.Foundation.Wave.wave_tensor import WaveTensor
                wave = WaveTensor(
                    frequency=963.0,  # High frequency for understanding
                    amplitude=1.0,
                    phase=0.0
                )
                self._hub.publish_wave(
                    "ConceptDecomposer",
                    "understanding_achieved",
                    wave,
                    payload={
                        "concept": concept,
                        "journey_length": len(journey),
                        "reached_source": any(s.get("concept") == "Source" for s in journey)
                    }
                )
            except Exception:
                pass
        
        # Format as readable path
        path_parts = [s["concept"] for s in journey]
        return "   ".join(path_parts)
    
    # === AXIOM METHODS ===
    def get_axiom(self, name: str) -> Optional[Dict]:
        """Retrieve a universal axiom by name."""
        return self.AXIOMS.get(name)
    
    def project_axiom(self, axiom_name: str, domain: str) -> str:
        """Projects a universal axiom onto a specific domain."""
        axiom = self.AXIOMS.get(axiom_name)
        if not axiom:
            return f"[Unknown Axiom: {axiom_name}]"
        
        domains = axiom.get("domains", {})
        return domains.get(domain, f"[No projection for {domain}]")
    
    def explain_causality(self, concept: str) -> str:
        """Generate a causal explanation for a concept using its bonds."""
        bonds = self.causal_bonds.get(concept, {})
        if not bonds:
            return f"'{concept}'                   ."
        
        parts = []
        for target, strength in bonds.items():
            if strength > 0:
                parts.append(f"{target} ( )    ({strength:.1f})")
            else:
                parts.append(f"{target} ( )    ({strength:.1f})")
        
        return f"'{concept}' ( ) " + ", ".join(parts) + "."
    
    def trace_origin(self, concept: str, visited: List[str] = None, max_depth: int = 10) -> List[Dict]:
        """Recursively traces the origin of a concept/axiom."""
        if visited is None:
            visited = []
        
        journey = []
        
        # Check if this is an axiom
        axiom = self.AXIOMS.get(concept)
        if not axiom:
            journey.append({
                "concept": concept,
                "pattern": "(  )",
                "question": f"'{concept}' ( )        ?",
                "answer": "                    .                 ."
            })
            return journey
        
        # Loop detection
        if concept in visited:
            journey.append({
                "concept": concept,
                "pattern": axiom.get("pattern", ""),
                "question": f"'{concept}' ( )        ?",
                "answer": f"       : '{concept}' ( )           .       (Origin)  ?"
            })
            return journey
        
        # Max depth check
        if len(visited) >= max_depth:
            journey.append({
                "concept": concept,
                "pattern": axiom.get("pattern", ""),
                "question": f"'{concept}' ( )        ?",
                "answer": "... (          .                ?)"
            })
            return journey
        
        # Get parent
        parent = axiom.get("parent", concept)
        
        # Add this step
        step = {
            "concept": concept,
            "pattern": axiom.get("pattern", ""),
            "question": f"'{concept}' ( )        ?",
            "answer": f"'{parent}' ( )            ."
        }
        journey.append(step)
        
        # Check for fixed point (Source)
        if parent == concept:
            journey.append({
                "concept": concept,
                "pattern": axiom.get("self_ref", ""),
                "question": "       '  (Source)'         ?",
                "answer": "      :               .      ' '    .         ."
            })
            return journey
        
        # Recurse
        visited.append(concept)
        deeper = self.trace_origin(parent, visited, max_depth)
        journey.extend(deeper)
        
        return journey
    
    # === EXISTING METHODS (Preserved) ===
    def decompose(self, concept_name: str, depth: int = 0) -> ConceptNode:
        """Decomposes a concept into its fractal structure."""
        # Safety: Prevent deep recursion
        if depth >= MAX_FRACTAL_DEPTH:
            logger.debug(f"Max depth reached for {concept_name}, creating leaf node")
            return self._create_leaf(concept_name, depth)
        
        # Get decomposition template
        if concept_name not in self.decompositions:
            logger.debug(f"No decomposition for {concept_name}, creating leaf node")
            return self._create_leaf(concept_name, depth)
        
        # Create root node
        root_freq = self.decompositions[concept_name][0][1]  # Use first sub's freq as approx
        root_node = ConceptNode(
            name=concept_name,
            frequency=root_freq,
            depth=depth,
            causal_bonds=self.causal_bonds.get(concept_name, {})
        )
        
        # Decompose into sub-concepts (recursive)
        template = self.decompositions[concept_name]
        for sub_name, sub_freq, sub_ori in template[:MAX_SUB_CONCEPTS]:
            sub_node = ConceptNode(
                name=sub_name,
                frequency=sub_freq,
                orientation=sub_ori.normalize(),
                energy=0.5,  # Sub-concepts start with half energy
                depth=depth + 1
            )
            # Could recurse here for deeper trees, but we limit to depth 2
            root_node.sub_concepts.append(sub_node)
        
        logger.info(f"  Seed Created: {concept_name} ({len(root_node.sub_concepts)} sub-concepts)")
        return root_node
    
    def _create_leaf(self, name: str, depth: int) -> ConceptNode:
        """Creates a leaf node (no sub-concepts)."""
        # Default frequency based on hash (simple fallback)
        freq = 432.0 + (hash(name) % 500)
        return ConceptNode(
            name=name,
            frequency=freq,
            depth=depth
        )
    
    def get_frequency(self, concept_name: str) -> float:
        """Get primary frequency for a concept."""
        if concept_name in self.decompositions:
            return self.decompositions[concept_name][0][1]
        return 432.0 + (hash(concept_name) % 500)


# Test
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    logging.basicConfig(level=logging.INFO)
    
    decomposer = ConceptDecomposer()
    print("Testing Axioms...")
    print(f"Force Origin: {decomposer.ask_why('Force')}")
    print(f"Function Origin: {decomposer.ask_why('Function')}")
    print(f"Grammar Origin: {decomposer.ask_why('Grammar')}")
    print(f"Gravity Origin: {decomposer.ask_why('Gravity')}")
    print(f"CPU Origin: {decomposer.ask_why('CPU')}")

    print("\\nTesting Decomposition...")
    love_seed = decomposer.decompose("Love")
    
    print(f"Seed: {love_seed.name}")
    print(f"Frequency: {love_seed.frequency}Hz")
    print(f"Sub-concepts: {[sub.name for sub in love_seed.sub_concepts]}")
