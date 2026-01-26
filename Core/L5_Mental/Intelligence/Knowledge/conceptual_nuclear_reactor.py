"""
Conceptual Nuclear Reactor (       )
=======================================

"          ,             "
"Break the solid ice, release the flame within"

Philosophy:
-----------
         "  (  )" -                 .
                        ,               .

1. Solid State (  ):    ,    ,    - "     "
2. Wavification (    ):          ,          
3. Fission (   ):                 (         )
4. Fusion (   ):                    (         )

Architecture:
-------------
[Periodic Table]   [Particle Accelerator]   [Nuclear Reactor]   [Energy Harvest]
                                                         

Example:
--------
Fission: "  "   ["  ", "  ", "  "] +         
Fusion: "   " + "  "   "           " +         
"""

import logging
import math
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

logger = logging.getLogger("ConceptualNuclearReactor")

# Graceful imports
try:
    from Core.L1_Foundation.Foundation.hangul_physics import Tensor3D
    from Core.L5_Mental.Intelligence.Memory_Linguistics.Memory.unified_types import FrequencyWave
    from Core.L1_Foundation.Foundation.hyper_quaternion import Quaternion
except ImportError:
    # Fallback stubs
    @dataclass
    class Tensor3D:
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0
    
    @dataclass
    class FrequencyWave:
        frequency: float = 0.0
        amplitude: float = 0.0
        phase: float = 0.0
        damping: float = 0.0
    
    @dataclass
    class Quaternion:
        w: float = 1.0
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0


@dataclass
class ConceptAtom:
    """
          - Periodic Table       
    
    A fundamental concept that can undergo nuclear reactions.
    """
    symbol: str  # "Love", "Time", "Pain", etc.
    atomic_number: int  # Position in periodic table
    
    # Wave representation (       )
    wave_tensor: Tensor3D = field(default_factory=lambda: Tensor3D())
    wave_frequency: FrequencyWave = field(default_factory=lambda: FrequencyWave())
    quaternion: Quaternion = field(default_factory=lambda: Quaternion())
    
    # Properties
    energy_level: float = 1.0  # Binding energy
    emotional_charge: float = 0.0  # -1 to 1
    complexity: int = 1  # How many sub-concepts
    
    # Language variants
    ko: str = ""  # Korean
    en: str = ""  # English
    ja: str = ""  # Japanese
    
    def to_plasma(self) -> Dict[str, Any]:
        """Convert solid concept to plasma (wave state)"""
        return {
            "symbol": self.symbol,
            "wave_x": self.wave_tensor.x,
            "wave_y": self.wave_tensor.y,
            "wave_z": self.wave_tensor.z,
            "frequency": self.wave_frequency.frequency,
            "amplitude": self.wave_frequency.amplitude,
            "energy": self.energy_level,
            "charge": self.emotional_charge
        }
    
    def get_hash(self) -> str:
        """Unique identifier for this concept"""
        content = f"{self.symbol}_{self.atomic_number}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class FissionResult:
    """
          
    
    Result of breaking down a complex concept into simpler ones.
    """
    parent_concept: str
    daughter_concepts: List[ConceptAtom]
    insight_energy: float  # Released energy (0-10)
    explanation: str  # What insight was gained
    language: str = "ko"


@dataclass
class FusionResult:
    """
          
    
    Result of combining concepts into something new.
    """
    reactant_a: str
    reactant_b: str
    product_concept: ConceptAtom
    creative_energy: float  # Released energy (0-10)
    poetic_expression: str  # The creative output
    language: str = "ko"


class ConceptualPeriodicTable:
    """
              
    
    Contains fundamental concepts as atomic elements.
    """
    
    def __init__(self, language: str = "ko"):
        self.language = language
        self.atoms: Dict[str, ConceptAtom] = {}
        self.atomic_numbers: Dict[int, str] = {}
        self._initialize_fundamental_concepts()
    
    def _initialize_fundamental_concepts(self):
        """Initialize the periodic table with fundamental concepts"""
        
        # Group 1: Emotions (  )
        self._add_atom("Love", 1, 1.2, 0.8, ko="  ", en="Love", ja=" ")
        self._add_atom("Joy", 2, 1.0, 0.9, ko="  ", en="Joy", ja="  ")
        self._add_atom("Sadness", 3, 0.8, -0.6, ko="  ", en="Sadness", ja="   ")
        self._add_atom("Fear", 4, 0.9, -0.7, ko="   ", en="Fear", ja="  ")
        self._add_atom("Anger", 5, 1.3, -0.5, ko="  ", en="Anger", ja="  ")
        
        # Group 2: Time & Space (   )
        self._add_atom("Time", 6, 1.5, 0.0, ko="  ", en="Time", ja="  ")
        self._add_atom("Space", 7, 1.4, 0.0, ko="  ", en="Space", ja="  ")
        self._add_atom("Moment", 8, 0.8, 0.3, ko="  ", en="Moment", ja="  ")
        self._add_atom("Eternity", 9, 2.0, 0.1, ko="  ", en="Eternity", ja="  ")
        
        # Group 3: Abstract Concepts (     )
        self._add_atom("Truth", 10, 1.8, 0.2, ko="  ", en="Truth", ja="  ")
        self._add_atom("Beauty", 11, 1.3, 0.7, ko="    ", en="Beauty", ja="   ")
        self._add_atom("Freedom", 12, 1.6, 0.5, ko="  ", en="Freedom", ja="  ")
        self._add_atom("Justice", 13, 1.7, 0.3, ko="  ", en="Justice", ja="  ")
        
        # Group 4: Life Concepts (     )
        self._add_atom("Life", 14, 2.5, 0.6, ko="  ", en="Life", ja=" ")
        self._add_atom("Death", 15, 2.3, -0.4, ko="  ", en="Death", ja=" ")
        self._add_atom("Birth", 16, 1.9, 0.8, ko="  ", en="Birth", ja="  ")
        self._add_atom("Growth", 17, 1.4, 0.5, ko="  ", en="Growth", ja="  ")
        
        # Group 5: Knowledge & Mind (      )
        self._add_atom("Knowledge", 18, 1.6, 0.2, ko="  ", en="Knowledge", ja="  ")
        self._add_atom("Wisdom", 19, 2.0, 0.4, ko="  ", en="Wisdom", ja="  ")
        self._add_atom("Understanding", 20, 1.8, 0.3, ko="  ", en="Understanding", ja="  ")
        self._add_atom("Consciousness", 21, 2.8, 0.1, ko="  ", en="Consciousness", ja="  ")
        
        # Group 6: Forces ( )
        self._add_atom("Power", 22, 1.7, 0.0, ko=" ", en="Power", ja=" ")
        self._add_atom("Gravity", 23, 1.9, 0.0, ko="  ", en="Gravity", ja="  ")
        self._add_atom("Energy", 24, 2.2, 0.2, ko="   ", en="Energy", ja="     ")
        self._add_atom("Light", 25, 1.6, 0.9, ko=" ", en="Light", ja=" ")
        self._add_atom("Darkness", 26, 1.5, -0.3, ko="  ", en="Darkness", ja=" ")
        
        # Group 7: Relations (  )
        self._add_atom("Connection", 27, 1.2, 0.6, ko="  ", en="Connection", ja="   ")
        self._add_atom("Separation", 28, 1.1, -0.5, ko="  ", en="Separation", ja="  ")
        self._add_atom("Unity", 29, 1.8, 0.7, ko="  ", en="Unity", ja="  ")
        self._add_atom("Conflict", 30, 1.4, -0.6, ko="  ", en="Conflict", ja="  ")
        
        logger.info(f"Initialized Periodic Table with {len(self.atoms)} fundamental concepts")
    
    def _add_atom(self, symbol: str, num: int, energy: float, charge: float, 
                  ko: str = "", en: str = "", ja: str = ""):
        """Add an atom to the periodic table"""
        # Generate wave properties from symbol
        hash_val = hash(symbol) % 1000 / 1000.0
        
        atom = ConceptAtom(
            symbol=symbol,
            atomic_number=num,
            wave_tensor=Tensor3D(
                x=charge * energy,
                y=energy * math.cos(hash_val * 2 * math.pi),
                z=energy * math.sin(hash_val * 2 * math.pi)
            ),
            wave_frequency=FrequencyWave(
                frequency=100.0 + num * 10.0,
                amplitude=energy,
                phase=hash_val * 2 * math.pi,
                damping=0.1
            ),
            energy_level=energy,
            emotional_charge=charge,
            complexity=1,
            ko=ko,
            en=en,
            ja=ja
        )
        
        self.atoms[symbol] = atom
        self.atomic_numbers[num] = symbol
    
    def get_atom(self, symbol: str) -> Optional[ConceptAtom]:
        """Get an atom by symbol"""
        return self.atoms.get(symbol)
    
    def search_by_emotion(self, charge: float, tolerance: float = 0.3) -> List[ConceptAtom]:
        """Find atoms with similar emotional charge"""
        return [atom for atom in self.atoms.values() 
                if abs(atom.emotional_charge - charge) < tolerance]
    
    def search_by_energy(self, energy: float, tolerance: float = 0.5) -> List[ConceptAtom]:
        """Find atoms with similar energy level"""
        return [atom for atom in self.atoms.values() 
                if abs(atom.energy_level - energy) < tolerance]
    
    def get_translation(self, symbol: str, language: str) -> str:
        """Get concept in specified language"""
        atom = self.get_atom(symbol)
        if not atom:
            return symbol
        
        if language == "ko":
            return atom.ko or symbol
        elif language == "en":
            return atom.en or symbol
        elif language == "ja":
            return atom.ja or symbol
        return symbol


class ConceptualNuclearReactor:
    """
           
    
    Performs nuclear reactions on concepts:
    - Fission: Break down complex concepts into simpler ones
    - Fusion: Combine concepts to create new ideas
    """
    
    def __init__(self, language: str = "ko"):
        self.language = language
        self.periodic_table = ConceptualPeriodicTable(language)
        self.reaction_history: List[Dict[str, Any]] = []
        self.total_energy_released: float = 0.0
        
        logger.info(f"Conceptual Nuclear Reactor initialized in {language}")
    
    def fission(self, complex_concept: str, context: str = "") -> FissionResult:
        """
            (Fission)
        
        Break down a complex concept into fundamental atoms.
        Releases "insight energy" - the understanding gained.
        
        Example:
        - Input: "  " (Life journey)
        - Output: ["  ", "  ", "  ", "  "] + insight energy
        """
        
        # Try to find the concept in periodic table
        atom = self.periodic_table.get_atom(complex_concept)
        
        if atom and atom.complexity == 1:
            # Already a fundamental concept, can't split further
            logger.info(f"{complex_concept} is already a fundamental atom")
            return FissionResult(
                parent_concept=complex_concept,
                daughter_concepts=[atom],
                insight_energy=0.1,
                explanation=self._get_text("already_fundamental", complex_concept),
                language=self.language
            )
        
        # Decompose based on wave analysis and semantic associations
        daughter_atoms = self._decompose_concept(complex_concept, context)
        
        # Calculate insight energy (more atoms = more energy released)
        insight_energy = len(daughter_atoms) * 1.5 + (hash(complex_concept) % 100) / 50.0
        
        # Generate explanation
        explanation = self._generate_fission_explanation(
            complex_concept, daughter_atoms, insight_energy
        )
        
        result = FissionResult(
            parent_concept=complex_concept,
            daughter_concepts=daughter_atoms,
            insight_energy=insight_energy,
            explanation=explanation,
            language=self.language
        )
        
        # Record reaction
        self.reaction_history.append({
            "type": "fission",
            "input": complex_concept,
            "output": [a.symbol for a in daughter_atoms],
            "energy": insight_energy,
            "timestamp": time.time()
        })
        self.total_energy_released += insight_energy
        
        logger.info(f"Fission: {complex_concept}   {[a.symbol for a in daughter_atoms]} "
                   f"(Energy: {insight_energy:.2f})")
        
        return result
    
    def fusion(self, concept_a: str, concept_b: str, context: str = "") -> FusionResult:
        """
            (Fusion)
        
        Combine two concepts to create something entirely new.
        Releases "creative energy" - the novelty of the creation.
        
        Example:
        - Input: "   " + "  "
        - Output: "           " + creative energy
        """
        
        # Get atoms
        atom_a = self.periodic_table.get_atom(concept_a)
        atom_b = self.periodic_table.get_atom(concept_b)
        
        if not atom_a:
            atom_a = self._text_to_atom(concept_a)
        if not atom_b:
            atom_b = self._text_to_atom(concept_b)
        
        # Calculate fusion product
        product = self._fuse_atoms(atom_a, atom_b, context)
        
        # Calculate creative energy (based on difference between reactants)
        energy_diff = abs(atom_a.energy_level - atom_b.energy_level)
        charge_diff = abs(atom_a.emotional_charge - atom_b.emotional_charge)
        creative_energy = (energy_diff + charge_diff) * 2.0 + 3.0
        
        # Generate poetic expression
        poetic = self._generate_fusion_poetry(atom_a, atom_b, product, context)
        
        result = FusionResult(
            reactant_a=concept_a,
            reactant_b=concept_b,
            product_concept=product,
            creative_energy=creative_energy,
            poetic_expression=poetic,
            language=self.language
        )
        
        # Record reaction
        self.reaction_history.append({
            "type": "fusion",
            "input": [concept_a, concept_b],
            "output": product.symbol,
            "energy": creative_energy,
            "timestamp": time.time()
        })
        self.total_energy_released += creative_energy
        
        logger.info(f"Fusion: {concept_a} + {concept_b}   {product.symbol} "
                   f"(Energy: {creative_energy:.2f})")
        
        return result
    
    def _decompose_concept(self, concept: str, context: str) -> List[ConceptAtom]:
        """Decompose a complex concept into fundamental atoms"""
        
        # Predefined decompositions (can be expanded)
        decompositions = {
            # Korean
            "  ": ["Time", "Growth", "Joy", "Sadness"],
            "  ": ["Connection", "Joy", "Understanding"],
            "  ": ["Joy", "Peace", "Connection"],
            "  ": ["Sadness", "Fear", "Growth"],
            "  ": ["Connection", "Trust", "Joy"],
            "  ": ["Love", "Connection", "Protection"],
            
            # English
            "life": ["Time", "Growth", "Joy", "Sadness"],
            "happiness": ["Joy", "Peace", "Connection"],
            "suffering": ["Sadness", "Fear", "Growth"],
            "friendship": ["Connection", "Trust", "Joy"],
            "journey": ["Time", "Movement", "Growth"],
            
            # Japanese
            "  ": ["Time", "Growth", "Joy", "Sadness"],
            "  ": ["Joy", "Peace", "Connection"],
        }
        
        # Get decomposition or use similar concepts
        if concept in decompositions:
            symbols = decompositions[concept]
        else:
            # Default: decompose into related emotional atoms
            symbols = ["Understanding", "Time", "Growth"]
        
        atoms = []
        for symbol in symbols:
            atom = self.periodic_table.get_atom(symbol)
            if atom:
                atoms.append(atom)
        
        return atoms if atoms else [self._text_to_atom(concept)]
    
    def _fuse_atoms(self, atom_a: ConceptAtom, atom_b: ConceptAtom, 
                    context: str) -> ConceptAtom:
        """Create a new concept from two atoms"""
        
        # New symbol combines both
        new_symbol = f"{atom_a.symbol}_{atom_b.symbol}_Fusion"
        
        # Wave properties are averaged/combined
        new_tensor = Tensor3D(
            x=(atom_a.wave_tensor.x + atom_b.wave_tensor.x) / 2,
            y=(atom_a.wave_tensor.y + atom_b.wave_tensor.y) / 2,
            z=(atom_a.wave_tensor.z + atom_b.wave_tensor.z) / 2
        )
        
        new_frequency = FrequencyWave(
            frequency=(atom_a.wave_frequency.frequency + atom_b.wave_frequency.frequency) / 2,
            amplitude=max(atom_a.wave_frequency.amplitude, atom_b.wave_frequency.amplitude),
            phase=(atom_a.wave_frequency.phase + atom_b.wave_frequency.phase) / 2,
            damping=0.1
        )
        
        # Combined properties
        new_energy = (atom_a.energy_level + atom_b.energy_level) * 0.6  # Some energy released
        new_charge = (atom_a.emotional_charge + atom_b.emotional_charge) / 2
        
        # Create product
        product = ConceptAtom(
            symbol=new_symbol,
            atomic_number=9999,  # Synthetic element
            wave_tensor=new_tensor,
            wave_frequency=new_frequency,
            energy_level=new_energy,
            emotional_charge=new_charge,
            complexity=atom_a.complexity + atom_b.complexity,
            ko=f"{atom_a.ko}+{atom_b.ko}",
            en=f"{atom_a.en}+{atom_b.en}",
            ja=f"{atom_a.ja}+{atom_b.ja}"
        )
        
        return product
    
    def _text_to_atom(self, text: str) -> ConceptAtom:
        """Convert arbitrary text into a concept atom"""
        hash_val = hash(text) % 1000 / 1000.0
        
        return ConceptAtom(
            symbol=text,
            atomic_number=9999,
            wave_tensor=Tensor3D(
                x=hash_val - 0.5,
                y=math.cos(hash_val * 2 * math.pi),
                z=math.sin(hash_val * 2 * math.pi)
            ),
            wave_frequency=FrequencyWave(
                frequency=100.0 + hash_val * 200.0,
                amplitude=0.5 + hash_val * 0.5,
                phase=hash_val * 2 * math.pi,
                damping=0.1
            ),
            energy_level=1.0,
            emotional_charge=0.0,
            ko=text,
            en=text,
            ja=text
        )
    
    def _generate_fission_explanation(self, parent: str, daughters: List[ConceptAtom], 
                                     energy: float) -> str:
        """Generate explanation for fission result"""
        
        daughter_names = [self.periodic_table.get_translation(d.symbol, self.language) 
                         for d in daughters]
        
        templates = {
            "ko": [
                f"'{parent}'      ,       {', '.join(daughter_names)} ( )           . "
                f"            {energy:.1f}                   .  ",
                
                f"'{parent}'           , {', '.join(daughter_names)} ( )             . "
                f"          {energy:.1f}                .   ",
                
                f"'{parent}'           {', '.join(daughter_names)} ( )     . "
                f"        {energy:.1f}                .  "
            ],
            "en": [
                f"Analyzing '{parent}', we find {', '.join(daughter_names)} intricately intertwined. "
                f"Understanding this relationship released {energy:.1f} insight energy.  ",
                
                f"Breaking down '{parent}', {', '.join(daughter_names)} emerge as core elements. "
                f"This decomposition generated {energy:.1f} interpretive energy.   ",
                
                f"Exploring the essence of '{parent}' reveals {', '.join(daughter_names)}. "
                f"This realization sparked {energy:.1f} units of understanding light.  "
            ],
            "ja": [
                f" {parent}        {', '.join(daughter_names)}             "
                f"            {energy:.1f}                  ",
                
                f" {parent}             {', '.join(daughter_names)}              "
                f"       {energy:.1f}                  "
            ]
        }
        
        options = templates.get(self.language, templates["ko"])
        return options[hash(parent) % len(options)]
    
    def _generate_fusion_poetry(self, atom_a: ConceptAtom, atom_b: ConceptAtom, 
                                product: ConceptAtom, context: str) -> str:
        """Generate poetic expression for fusion result"""
        
        name_a = self.periodic_table.get_translation(atom_a.symbol, self.language)
        name_b = self.periodic_table.get_translation(atom_b.symbol, self.language)
        
        # Energy and emotion characteristics
        high_energy = (atom_a.energy_level + atom_b.energy_level) > 3.0
        positive = (atom_a.emotional_charge + atom_b.emotional_charge) > 0
        
        templates = {
            "ko": {
                "high_positive": [
                    f"{name_a} ( ) {name_b} ( )        ,                         . "
                    f"   {name_a} ( ) {name_b}            .  ",
                    
                    f"{name_a} ( ) {name_b},                         . "
                    f"           - {name_a} ( )    {name_b}               .  ",
                ],
                "high_negative": [
                    f"{name_a} ( ) {name_b} ( )          ,                     . "
                    f"{name_a} ( ) {name_b}               .  ",
                ],
                "low_positive": [
                    f"{name_a} ( ) {name_b} ( )         ,                   . "
                    f"{name_a} ( ) {name_b}                  .  ",
                ],
                "low_negative": [
                    f"{name_a} ( ) {name_b} ( )        ,                          . "
                    f"{name_a} ( ) {name_b}                .  ",
                ]
            },
            "en": {
                "high_positive": [
                    f"When {name_a} and {name_b} collided, the universe lit up and a new truth was born. "
                    f"It's like {name_a} feels just like {name_b}.  ",
                ],
                "high_negative": [
                    f"{name_a} and {name_b} clashed violently, exploding into new understanding from darkness. "
                    f"{name_a} is heavy and deep, just like {name_b}.  ",
                ],
                "low_positive": [
                    f"{name_a} and {name_b} quietly harmonized, becoming one with gentle light. "
                    f"{name_a} flows softly, like {name_b}.  ",
                ],
                "low_negative": [
                    f"{name_a} and {name_b} slowly mixed, revealing new form in faint shadows. "
                    f"{name_a} is cold and still, like {name_b}.  ",
                ]
            },
            "ja": {
                "high_positive": [
                    f"{name_a} {name_b}                              "
                    f"   {name_a} {name_b}               ",
                ],
                "high_negative": [
                    f"{name_a} {name_b}                           "
                    f"{name_a} {name_b}               ",
                ],
            }
        }
        
        # Select category
        category = "high" if high_energy else "low"
        category += "_positive" if positive else "_negative"
        
        options = templates.get(self.language, templates["ko"]).get(category, [])
        if not options:
            options = templates["ko"]["high_positive"]
        
        return options[hash(context) % len(options)]
    
    def _get_text(self, key: str, *args) -> str:
        """Get localized text"""
        texts = {
            "already_fundamental": {
                "ko": f"'{args[0]}' ( )            .               .",
                "en": f"'{args[0]}' is already a fundamental atom. Cannot decompose further.",
                "ja": f" {args[0]}                        "
            }
        }
        
        return texts.get(key, {}).get(self.language, f"[{key}]")
    
    def get_reactor_stats(self) -> Dict[str, Any]:
        """Get reactor statistics"""
        fissions = sum(1 for r in self.reaction_history if r["type"] == "fission")
        fusions = sum(1 for r in self.reaction_history if r["type"] == "fusion")
        
        return {
            "total_reactions": len(self.reaction_history),
            "fissions": fissions,
            "fusions": fusions,
            "total_energy_released": self.total_energy_released,
            "average_energy_per_reaction": (
                self.total_energy_released / len(self.reaction_history)
                if self.reaction_history else 0
            ),
            "periodic_table_size": len(self.periodic_table.atoms)
        }
    
    def set_language(self, language: str):
        """Change reactor language"""
        if language in ["ko", "en", "ja"]:
            self.language = language
            self.periodic_table.language = language
            logger.info(f"Reactor language set to {language}")


def create_conceptual_nuclear_reactor(language: str = "ko") -> ConceptualNuclearReactor:
    """
    Factory function to create a Conceptual Nuclear Reactor.
    
    Usage:
    ------
    reactor = create_conceptual_nuclear_reactor(language="ko")
    
    # Fission: Break down complex concept
    fission_result = reactor.fission("  ", context="      ")
    
    # Fusion: Combine concepts
    fusion_result = reactor.fusion("   ", "  ", context="     ")
    """
    return ConceptualNuclearReactor(language=language)


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.INFO)
    
    print("  Conceptual Nuclear Reactor Demo")
    print("=" * 60)
    
    reactor = create_conceptual_nuclear_reactor(language="ko")
    
    print("\n   Periodic Table Preview:")
    table = reactor.periodic_table
    for i in range(1, 6):
        symbol = table.atomic_numbers.get(i)
        if symbol:
            atom = table.get_atom(symbol)
            print(f"  {i}. {symbol} ({atom.ko}) - Energy: {atom.energy_level:.1f}, "
                  f"Charge: {atom.emotional_charge:+.1f}")
    
    print("\n  Fission Demo: Breaking down '  '")
    print("-" * 60)
    fission = reactor.fission("  ", context="      ")
    print(f"Parent: {fission.parent_concept}")
    print(f"Daughters: {[d.symbol for d in fission.daughter_concepts]}")
    print(f"Insight Energy: {fission.insight_energy:.2f}")
    print(f"Explanation: {fission.explanation}")
    
    print("\n  Fusion Demo: Combining '   ' + '  '")
    print("-" * 60)
    fusion = reactor.fusion("Gravity", "Love", context="     ")
    print(f"Reactants: {fusion.reactant_a} + {fusion.reactant_b}")
    print(f"Product: {fusion.product_concept.symbol}")
    print(f"Creative Energy: {fusion.creative_energy:.2f}")
    print(f"Poetry: {fusion.poetic_expression}")
    
    print("\n  Reactor Stats:")
    print("-" * 60)
    stats = reactor.get_reactor_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n  Nuclear Reactor Ready!")
