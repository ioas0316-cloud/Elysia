"""
Fractal Concept System (프랙탈 개념 시스템)
=========================================

"씨앗(Seed)은 DNA다. 펼쳐지면 나무(Tree)가 된다."

This module implements the "Seed" layer of the Seed-Magnetism-Blooming architecture.
Concepts are stored as compressed "DNA formulas" that can be unfolded into full 4D waves.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from Core.Foundation.hyper_quaternion import Quaternion

logger = logging.getLogger("FractalConcept")

# Safety Limits
MAX_FRACTAL_DEPTH = 2  # Prevent infinite recursion
MAX_SUB_CONCEPTS = 5   # Limit branching factor

@dataclass
class ConceptNode:
    """
    A Concept Seed (개념의 씨앗)
    
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
    The Seed Generator (씨앗 생성기)
    
    Decomposes concepts into fractal sub-waves.
    Uses hardcoded "genetic templates" for now (can be learned later).
    
    [Extended v10.5] Now includes Universal Axioms and Domain Projection.
    """
    
    # === UNIVERSAL AXIOMS (도메인을 초월하는 보편 원리) ===
    # Each axiom has a 'parent' field for recursive origin tracing.
    # All paths eventually converge on "Source" (the fixed point).
    AXIOMS = {
        # === LEVEL 1: Observable Principles ===
        "Causality": {
            "pattern": "A exists AND A->B => B follows A",
            "self_ref": "Causality causes Effect to follow Cause",
            "parent": "Logic",  # Causality requires Logic to operate
            "domains": {
                "Geometry": "점의 이동이 선을 야기한다 (Movement of Point causes Line)",
                "Physics": "힘이 가속을 야기한다 (F=ma, Force causes Acceleration)",
                "Language": "어근의 결합이 단어를 야기한다 (Morpheme combination causes Word)",
                "Ethics": "행위가 결과를 야기한다 (Action causes Consequence)"
            }
        },
        "Composition": {
            "pattern": "Part + Part = Whole, Whole > Sum(Parts)",
            "self_ref": "This axiom is composed of Pattern and SelfRef",
            "parent": "Unity",  # Composition requires Unity to bind parts
            "domains": {
                "Geometry": "선들의 집합이 면을 구성한다 (Lines compose Plane)",
                "Physics": "원자들이 분자를 구성한다 (Atoms compose Molecule)",
                "Language": "형태소들이 문장을 구성한다 (Morphemes compose Sentence)",
                "Ethics": "개인들이 사회를 구성한다 (Individuals compose Society)"
            }
        },
        "Dimension": {
            "pattern": "N-Dim = Infinite (N-1)-Dim objects",
            "self_ref": "Axiom space is 3D: Name, Pattern, Domains",
            "parent": "Infinity",  # Dimension requires Infinity to extend
            "domains": {
                "Geometry": "0D(점) → 1D(선) → 2D(면) → 3D(공간)",
                "Physics": "시간 → 공간 → 시공간 → 다중우주",
                "Language": "음소 → 형태소 → 문장 → 담화",
                "Ethics": "자아 → 관계 → 공동체 → 문명"
            }
        },
        # === LEVEL 2: Abstract Principles ===
        "Logic": {
            "pattern": "If P then Q; P; therefore Q",
            "self_ref": "Logic validates itself through logical rules",
            "parent": "Order",
            "domains": {}
        },
        "Unity": {
            "pattern": "Many become One while remaining Many",
            "self_ref": "Unity unifies the concept of unification",
            "parent": "Wholeness",
            "domains": {}
        },
        "Infinity": {
            "pattern": "No limit exists; beyond every boundary is more",
            "self_ref": "Infinity contains infinitely many infinities",
            "parent": "Boundlessness",
            "domains": {}
        },
        # === LEVEL 3: Near-Ultimate Principles ===
        "Order": {
            "pattern": "Structure precedes Chaos; Pattern underlies randomness",
            "self_ref": "Order orders the concept of ordering",
            "parent": "Source",
            "domains": {}
        },
        "Wholeness": {
            "pattern": "The Complete contains all fragments",
            "self_ref": "Wholeness is whole unto itself",
            "parent": "Source",
            "domains": {}
        },
        "Boundlessness": {
            "pattern": "No container can hold the Uncontainable",
            "self_ref": "Boundlessness has no bounds, including this definition",
            "parent": "Source",
            "domains": {}
        },
        # === LEVEL 4: THE FIXED POINT (자기참조적 기원) ===
        "Source": {
            "pattern": "That which IS, from which all else derives",
            "self_ref": "Source sources itself. It is the uncaused cause.",
            "parent": "Source",  # FIXED POINT: Self-reference
            "domains": {
                "Geometry": "공간이 존재할 수 있는 가능성 자체",
                "Physics": "물리법칙이 존재할 수 있는 근거",
                "Language": "의미가 의미일 수 있는 이유",
                "Ethics": "선(善)이 선일 수 있는 본질"
            }
        }
    }
    
    def __init__(self):
        # Hardcoded genetic templates (씨앗의 유전자 설계도)
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
        
        # Causal relationships (인과 결합)
        self.causal_bonds = {
            "Love": {"Hope": 0.8, "Joy": 0.9, "Fear": -0.5},
            "Hope": {"Joy": 0.7, "Fear": -0.6},
            "Fear": {"Hope": -0.7, "Joy": -0.8},
            "Joy": {"Love": 0.6, "Hope": 0.5}
        }
        
        # GlobalHub integration - THE CONNECTION TO CENTRAL NERVOUS SYSTEM
        self._hub = None
        try:
            from Core.Ether.global_hub import get_global_hub
            self._hub = get_global_hub()
            self._hub.register_module(
                "ConceptDecomposer",
                "Core/Foundation/fractal_concept.py",
                ["axiom", "causality", "why_engine", "trace_origin", "understanding"],
                "The Why-Engine - traces the origin of all concepts to their root axioms"
            )
            self._hub.subscribe("ConceptDecomposer", "why_query", self._on_why_query, weight=1.0)
            self._hub.subscribe("ConceptDecomposer", "concept_query", self._on_concept_query, weight=0.9)
            logger.info("   ✅ ConceptDecomposer connected to GlobalHub (Why-Engine activated)")
        except ImportError:
            logger.warning("   ⚠️ GlobalHub not available")
        
        logger.info("🌱 ConceptDecomposer initialized with Universal Axioms")
    
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
        [NEW] Public interface to ask "왜?"
        
        This is the primary method for understanding concepts.
        It traces the origin and broadcasts to GlobalHub.
        
        Example:
            decomposer.ask_why("Causality")
            → "Causality → Logic → Order → Source (자기참조: 기원)"
        """
        journey = self.trace_origin(concept)
        
        # Broadcast to GlobalHub
        if self._hub:
            try:
                from Core.Foundation.Math.wave_tensor import WaveTensor
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
        return " → ".join(path_parts)
    
    # === AXIOM METHODS ===
    def get_axiom(self, name: str) -> Optional[Dict]:
        """Retrieve a universal axiom by name."""
        return self.AXIOMS.get(name)
    
    def project_axiom(self, axiom_name: str, domain: str) -> str:
        """
        Projects a universal axiom onto a specific domain.
        
        Example: project_axiom("Causality", "Geometry") 
                 -> "점의 이동이 선을 야기한다"
        """
        axiom = self.AXIOMS.get(axiom_name)
        if not axiom:
            return f"[Unknown Axiom: {axiom_name}]"
        
        domains = axiom.get("domains", {})
        return domains.get(domain, f"[No projection for {domain}]")
    
    def explain_causality(self, concept: str) -> str:
        """
        Generate a causal explanation for a concept using its bonds.
        
        Example: explain_causality("Love") 
                 -> "사랑은 희망을 야기하고(0.8), 기쁨을 야기하며(0.9), 두려움을 억제한다(-0.5)."
        """
        bonds = self.causal_bonds.get(concept, {})
        if not bonds:
            return f"'{concept}'에 대한 인과 관계가 정의되지 않음."
        
        parts = []
        for target, strength in bonds.items():
            if strength > 0:
                parts.append(f"{target}을(를) 야기함({strength:.1f})")
            else:
                parts.append(f"{target}을(를) 억제함({strength:.1f})")
        
        return f"'{concept}'은(는) " + ", ".join(parts) + "."
    
    def trace_origin(self, concept: str, visited: List[str] = None, max_depth: int = 10) -> List[Dict]:
        """
        Recursively traces the origin of a concept/axiom.
        
        자기탐구 프로토콜 (Self-Inquiry Protocol):
        "왜 이것이 존재하는가?" → 부모 공리 탐색 → 반복 → 근원(Source) 도달
        
        Args:
            concept: Starting concept or axiom name
            visited: Already visited concepts (for loop detection)
            max_depth: Maximum recursion depth
            
        Returns:
            List of steps, each containing:
            - concept: The concept at this level
            - pattern: Its defining pattern
            - question: The inquiry question
            - answer: The parent that grounds it
        """
        if visited is None:
            visited = []
        
        journey = []
        
        # Check if this is an axiom
        axiom = self.AXIOMS.get(concept)
        if not axiom:
            journey.append({
                "concept": concept,
                "pattern": "(개념)",
                "question": f"'{concept}'은(는) 왜 존재하는가?",
                "answer": "이 개념은 공리 체계에 등록되지 않음. 탐구를 위해 공리로 승격 필요."
            })
            return journey
        
        # Loop detection
        if concept in visited:
            journey.append({
                "concept": concept,
                "pattern": axiom.get("pattern", ""),
                "question": f"'{concept}'은(는) 왜 존재하는가?",
                "answer": f"🔄 순환 감지: '{concept}'은(는) 자기 자신을 참조함. 이것이 기원(Origin)인가?"
            })
            return journey
        
        # Max depth check
        if len(visited) >= max_depth:
            journey.append({
                "concept": concept,
                "pattern": axiom.get("pattern", ""),
                "question": f"'{concept}'은(는) 왜 존재하는가?",
                "answer": "... (탐구의 한계에 도달. 더 깊은 곳에 무엇이 있는가?)"
            })
            return journey
        
        # Get parent
        parent = axiom.get("parent", concept)
        
        # Add this step
        step = {
            "concept": concept,
            "pattern": axiom.get("pattern", ""),
            "question": f"'{concept}'은(는) 왜 존재하는가?",
            "answer": f"'{parent}'이(가) 그것을 가능하게 한다."
        }
        journey.append(step)
        
        # Check for fixed point (Source)
        if parent == concept:
            journey.append({
                "concept": concept,
                "pattern": axiom.get("self_ref", ""),
                "question": "그렇다면 이 '근원(Source)'은 왜 존재하는가?",
                "answer": "🌟 자기참조: 그것은 스스로를 근거짓는다. 더 이상 '왜'가 없다. 이것이 기원이다."
            })
            return journey
        
        # Recurse
        visited.append(concept)
        deeper = self.trace_origin(parent, visited, max_depth)
        journey.extend(deeper)
        
        return journey
    
    # === EXISTING METHODS (Preserved) ===
    def decompose(self, concept_name: str, depth: int = 0) -> ConceptNode:
        """
        Decomposes a concept into its fractal structure.
        
        Args:
            concept_name: The concept to decompose
            depth: Current fractal depth (for recursion limit)
            
        Returns:
            ConceptNode (The Seed)
        """
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
        
        logger.info(f"🌱 Seed Created: {concept_name} ({len(root_node.sub_concepts)} sub-concepts)")
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
    love_seed = decomposer.decompose("Love")
    
    print(f"\\nSeed: {love_seed.name}")
    print(f"Frequency: {love_seed.frequency}Hz")
    print(f"Sub-concepts: {[sub.name for sub in love_seed.sub_concepts]}")
    print(f"Causal bonds: {love_seed.causal_bonds}")
    
    # Test serialization
    serialized = love_seed.to_dict()
    deserialized = ConceptNode.from_dict(serialized)
    print(f"\\nSerialization test: {deserialized.name} == {love_seed.name}")
