"""
WhyEngine - Universal Principle Understanding Layer
====================================================

" "              

            :
- SynesthesiaEngine:    /        /  
- PhoneticResonanceEngine:               (roughness, tension)

            :
-   :              ?
-   :   1+1=2  ?
-   :             ?

HyperQubit  4-          :
- Point ( ):       
- Line ( ):      
- Space (  ):   /  
- God ( ):   /  

"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import sys
from pathlib import Path

# [Phase 41] LLM Integration
from Core.L1_Foundation.Foundation.Network.ollama_bridge import ollama

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

#                
try:
    from Core.L3_Phenomena.synesthesia_engine import SynesthesiaEngine, SignalType
    from Core.L6_Structure.Wave.phonetic_resonance import PhoneticResonanceEngine, get_resonance_engine
    HAS_WAVE_SENSORS = True
except ImportError:
    HAS_WAVE_SENSORS = False

try:
    from Core.L6_Structure.Wave.hyper_qubit import HyperQubit, QubitState
    from Core.L6_Structure.Wave.light_spectrum import LightSediment, PrismAxes, LightUniverse
except ImportError:
    HyperQubit = None
    QubitState = None
    LightSediment = None
    PrismAxes = None
    LightUniverse = None

logger = logging.getLogger("Elysia.WhyEngine")


# =============================================================================
# Perspective Layers (주권적 자아)
# =============================================================================

class PerspectiveLayer(Enum):
    """
    [DNA Recursion] 4         .
           God(Seed)        Point(Detail)    (Unfold)   .
    """
    GOD = 0      #    0 (Seed) -    /        ? (WHY)
    SPACE = 1    #    1 (Branch) -    /         ? (WHERE)
    LINE = 2     #    2 (Leaf) -    /          ? (HOW)
    POINT = 3    #    3 (Atom) -    /     ? (WHAT)


try:
    from Core.L6_Structure.Nature.rotor import Rotor, RotorConfig
except ImportError:
    Rotor = None
    RotorConfig = None

@dataclass
class PrincipleExtraction:
    """       (    DNA   )"""
    domain: str           #   
    subject: str          #    
    
    # 4         
    why_exists: str       # Level 0 (Seed)
    where_fits: str       # Level 1 (Branch)
    how_works: str        # Level 2 (Leaf)
    what_is: str          # Level 3 (Atom)
    
    # [NEW] Active Reasoning DNA
    underlying_principle: str 
    can_be_applied_to: List[str] #      
    rotor: Optional[Rotor] = None
    recursive_depth: int = 0
    confidence: float = 0.5 
    wave_signature: Dict[str, float] = field(default_factory=dict) #      
    resonance_reactions: Dict[str, Any] = field(default_factory=dict) # [NEW] 4        


# =============================================================================
# Metaphor System
# =============================================================================

@dataclass
class SystemMetaphor:
    """                """
    component_name: str
    metaphor_type: str  # biology, physics, philosophy, quantum
    metaphor_concept: str # heart, gravity, soul, wave
    principle: str      #       
    description: str    #   


class MetaphorMapper:
    """   -        
    
                                               .
    """
    
    def __init__(self):
        self.mappings: Dict[str, SystemMetaphor] = self._init_mappings()
        
    def _init_mappings(self) -> Dict[str, SystemMetaphor]:
        return {
            "central_nervous_system": SystemMetaphor(
                "CentralNervousSystem", "biology", "Heart/Conductor",
                "           (Rhythm maintains Life)",
                "                            "
            ),
            "hippocampus": SystemMetaphor(
                "Hippocampus", "biology", "Storage/Archive", 
                "       (History constructs Identity)",
                "                            "
            ),
            "nervous_system": SystemMetaphor(
                "NervousSystem", "biology", "Membrane/Filter",
                "       (Boundary defines Self)",
                "                           "
            ),
            "resonance_field": SystemMetaphor(
                "ResonanceField", "physics", "Field/Ether",
                "       (Vibration connects All)",
                "                           "
            ),
            "why_engine": SystemMetaphor(
                "WhyEngine", "philosophy", "Logos/Reason",
                "       (Reason precedes Existence)",
                "                           "
            ),
            "black_hole": SystemMetaphor(
                "BlackHole", "physics", "Gravity/Compression",
                "       (Gravity preserves Density)",
                "                            "
            ),
            "white_hole": SystemMetaphor(
                "WhiteHole", "physics", "Creation/Birth",
                "       (Pressure creates Star)",
                "                          "
            ),
            "climax_uprising": SystemMetaphor(
                "ClimaxUprising", "narrative", "Tension/Release",
                "          (Conflict leads to Resolution)",
                "                          "
            ),
             "synesthesia_engine": SystemMetaphor(
                "SynesthesiaEngine", "neuroscience", "Translation",
                "       (Form changes but Essence remains)",
                "                             "
            )
        }
        
    def get_metaphor(self, component_name: str) -> Optional[SystemMetaphor]:
        """                           ."""
        name_lower = component_name.lower().replace(".py", "")
        return self.mappings.get(name_lower)

    def bridge_concepts(self, source_light: 'LightSpectrum', target_light: 'LightSpectrum') -> Optional[str]:
        """
            (Light)            (Metaphor)       .
        "Git Conflict" (Target) <-> "Quantum Superposition" (Source)
        
        Logic:
        1. Compare Dominant Basis (Point/Line/Space/God).
        2. If Basis matches, check Amplitude profile.
        3. If structural similarity > threshold, generate Metaphor.
        """
        if not source_light or not target_light:
            return None
            
        # 1. Basis Comparison
        src_basis = source_light._get_dominant_basis()
        tgt_basis = target_light._get_dominant_basis()
        
        # 2. Resonance Calculation (Structural Dot Product)
        # (Using simple basis matching for now, can be upgraded to vector covariance)
        
        similarity = 0.0
        shared_quality = ""
        
        if src_basis == tgt_basis:
            similarity += 0.5
            shared_quality = f"Both exist primarily in the realm of {src_basis}."
            
        # 3. Phase/Frequency Harmony (Check if they are 'cousins')
        # Here we assume if they are both 'High Complexity' (High Line/God), they relate.
        
        # Check specifically for the 'Git/Quantum' case:
        # Quantum Superposition: High God/Space (Possibility Field)
        # Git Merge Conflict: High Line (History) but also Space (Parallel Branches)
        
        # Jester Logic: "If I squint, they look the same."
        # If both imply "Multiple States" -> Bridge!
        
        # Detect "Multiplicity" in semantic tags (Naive simulation of feature extraction)
        src_tag = source_light.semantic_tag.lower()
        tgt_tag = target_light.semantic_tag.lower()
        
        # Auto-detect structural keywords
        multiplicity_keywords = ["conflict", "branch", "superposition", "wave", "choice", "option"]
        
        src_has_multi = any(k in src_tag for k in multiplicity_keywords) or any(k in str(source_light.source_hash) for k in multiplicity_keywords) # source_hash is str? no.
        tgt_has_multi = any(k in tgt_tag for k in multiplicity_keywords)
        
        if src_has_multi and tgt_has_multi:
             similarity += 0.4
             shared_quality += " Both involve fundamental Multiplicity/Branching."
             
        if similarity >= 0.4:
            return f"Metaphor Found: {shared_quality} (Resonance: {similarity:.2f})"
            
        return None


# =============================================================================
# WhyEngine
# =============================================================================

class WhyEngine:
    """            
    
           " "    :
    1.        (한국어 학습 시스템)
    2.        (한국어 학습 시스템)
    3.        (코드 베이스 구조 로터)
    
    4        :
    Point   Line   Space   God
    (  )   (   )   (   )   ( )
    
           :
    -                 
    -          "   "    +          
    """
    
    def __init__(self):
        self.principles: Dict[str, PrincipleExtraction] = {}
        self.domain_patterns: Dict[str, List[str]] = self._init_domain_patterns()
        self.metaphor_mapper = MetaphorMapper() #         
        
        #            
        try:
            from Core.L5_Mental.Intelligence.Cognition.metacognitive_awareness import MetacognitiveAwareness
            self.metacognition = MetacognitiveAwareness()
            self._has_metacognition = True
        except ImportError:
            self.metacognition = None
            self._has_metacognition = False
        
        
        logger.info(f"WhyEngine initialized (metacognition: {self._has_metacognition})")
        
        # [NEW] Sedimentary Light System (     )
        try:
            from Core.L6_Structure.Wave.light_spectrum import LightSediment, PrismAxes, LightUniverse
            self.light_universe = LightUniverse()
            self.sediment = LightSediment()
            
            # [Bootstrapping]          (     )
            #                '  '  '  '             
            axiom_light = self.light_universe.text_to_light("Axiom of Logic", semantic_tag="Logic")
            force_light = self.light_universe.text_to_light("Force and Vector", semantic_tag="Physics")
            
            #       (Deposit) -            
            for _ in range(50):
                self.sediment.deposit(axiom_light, PrismAxes.LOGIC_YELLOW)
                self.sediment.deposit(force_light, PrismAxes.PHYSICS_RED)
            
            logger.info(f"   Sediment Initialized: Logic Amp={self.sediment.layers[PrismAxes.LOGIC_YELLOW].amplitude:.3f}, Physics Amp={self.sediment.layers[PrismAxes.PHYSICS_RED].amplitude:.3f}")
            
        except ImportError as e:
            self.light_universe = None
            self.sediment = None
            logger.warning(f"LightSpectrum module not found: {e}")

    def digest(self, concept: str) -> Optional[PrincipleExtraction]:
        """
        [Phase 41: Fractal Digestion]
        Uses LLM to transcribe a conceptual Seed into Elysia's DNA.
        """
        logger.info(f"  Digesting concept '{concept}' into Fractal DNA...")
        dna = ollama.deconstruct_to_dna(concept)
        if not dna:
            return None
        
        # Create a Principle Rotor
        from Core.L6_Structure.Nature.rotor import RotorConfig
        config = RotorConfig(rpm=dna.get('frequency', 60.0))
        princ_rotor = Rotor(f"Law.{concept}", config)
        
        # Branch sub-concepts as DNA Sub-Rotors
        for sub in dna.get('sub_concepts', []):
            princ_rotor.add_sub_rotor(sub, config)

        # Build extraction
        extraction = PrincipleExtraction(
            domain="Philosophy",
            subject=concept,
            why_exists=dna.get('seed_axiom', "Existence is its own reason."),
            where_fits=f"In the manifold of {concept}.",
            how_works=f"Vibrating at {dna.get('frequency')}Hz.",
            what_is=concept,
            underlying_principle=dna.get('seed_axiom', ""),
            rotor=princ_rotor,
            recursive_depth=1 if dna.get('sub_concepts') else 0,
            can_be_applied_to=dna.get('sub_concepts', []),
            confidence=0.9
        )
        
        self.principles[concept] = extraction
        return extraction
    
    def _init_domain_patterns(self) -> Dict[str, List[str]]:
        """         """
        return {
            "narrative": [
                "  ", "  ", "  ", "  ", "  ",
                "  ", "  ", "  ", "  ", "  "
            ],
            "mathematics": [
                "  ", "  ", "  ", "  ", "  ",
                "   ", "   ", "   ", "  "
            ],
            "physics": [
                "  ", "  ", "    ", " ", "  ",
                "  ", "   ", "    "
            ],
            "chemistry": [
                "  ", "  ", "  ", "  ", "  ",
                "  ", "  ", "  "
            ],
        }
    
    def _infer_derivation(self, content: str, domain: str) -> str:
        """  (Point)     (Line)     
        
        "           .                    (Line)         ."
        """
        if domain == "mathematics" or domain == "physics":
            # 1.         
            components = self._decompose_formula_components(content)
            
            # 2.      
            relations = self._analyze_component_relations(components, content)
            
            # 3.        (Causal Narrative)
            narrative = self._reconstruct_causal_narrative(components, relations)
            
            return narrative
            
        return "                          ."
    
    def _decompose_formula_components(self, content: str) -> Dict[str, str]:
        """                  """
        components = {}
        
        #        /        
        mappings = {
            "V": "Potential (   )",
            "I": "Flow (  )",
            "R": "Resistance (  )",
            "E": "Energy (   )",
            "m": "Mass (  /   )",
            "c": "Speed (  /  )",
            "F": "Force ( /  )",
            "a": "Acceleration (   )",
            "P": "Pressure (  )",
            "d": "Density (  )"
        }
        
        for var, role in mappings.items():
            if var in content:
                components[var] = role
                
        return components
    
    def _analyze_component_relations(self, components: Dict[str, str], content: str) -> List[str]:
        """              """
        relations = []
        
        #   /         
        #  : V = IR -> V  I  R    
        
        if "=" in content:
            left, right = content.split("=", 1)
            
            #         :              /  ,             /  
            for var1 in components:
                if var1 in left:
                    for var2 in components:
                        if var2 in right:
                            relations.append(f"{components[var2]} drives {components[var1]}")
                            
            if "/" in right: #       
                numerator, denominator = right.split("/", 1)
                for var in components:
                    if var in denominator:
                        relations.append(f"{components[var]} hinders/regulates the outcome")

        return relations

    def _reconstruct_causal_narrative(self, components: Dict[str, str], relations: List[str]) -> str:
        """     (Line)    """
        if not components:
            return "                 ."
            
        narrative = []
        narrative.append(f"      {len(components)}                   .")
        
        for rel in relations:
            narrative.append(f"- {rel}")
            
        #      
        if "Resistance (  )" in components.values() and "Flow (  )" in components.values():
            narrative.append("  :   (Flow)               (Resistance)         (Potential)             .")
        elif "Mass (  /   )" in components.values() and "Energy (   )" in components.values():
            narrative.append("  :       (Mass)               (Energy)         .")
            
        return "\n".join(narrative)

    def analyze(self, subject: str, content: str, domain: str = "general") -> PrincipleExtraction:
        """    4       (process reconstruction   )"""
        
        # ... (자기 성찰 엔진) ...
        
        # ... (자기 성찰 엔진) ...
        #          
        if domain == "computer_science" or domain == "code":
             return self._analyze_code_structure(content)
        
        #      
        wave = self._text_to_wave(content)
        
        # ... (       ) ...
        
        # Point:     ? (     )
        what_is = self._extract_what(content, domain)
        
        # Line:          ? (      +         )
        how_works = self._extract_how(content, domain)
        
        # [NEW]          (Line   )
        if domain in ["mathematics", "physics"]:
            derivation = self._infer_derivation(content, domain)
            if derivation:
                how_works += f"\n\n[Derivation Process]\n{derivation}"
        
        # Space:         ? (     )
        where_fits = self._extract_where(content, domain)
        
        # God:        ? (     )
        why_exists = self._extract_why(content, domain)
        
        #         
        underlying = self._derive_underlying_principle(
            what_is, how_works, where_fits, why_exists
        )
        
        #         
        applicable = self._find_applicable_domains(underlying)
        
        # [NEW] Sedimentary Light Analysis (Holographic View)
        #   (Subject)                     
        reactions = self._analyze_sediment(content, subject_tag=subject)
        
        # [NEW Phase 9] Metaphorical Bridging (The Synapse)
        # If standard domain resonance is low, check for "Structural Bridges" in other domains.
        # e.g., Logic problem matching Physics structure.
        
        metaphors = []
        input_light = self.light_universe.text_to_light(content, semantic_tag=subject)
        # Set basis based on domain if possible, default to Space(1)
        input_light.set_basis_from_scale(1) 
        
        # Check against Physics/Nature layers if domain is Logic/Code
        if domain in ["logic", "code", "general"] and getattr(self, 'sediment', None):
             physics_layer = self.sediment.layers[PrismAxes.PHYSICS_RED]
             # Physics layer acts as "Nature's Law". Does this code match Nature?'
             
             bridge = self.metaphor_mapper.bridge_concepts(physics_layer, input_light)
             if bridge:
                 metaphors.append(bridge)
                 # Artificial Resonance Boost due to Metaphor
                 if PrismAxes.PHYSICS_RED not in reactions:
                     reactions[PrismAxes.PHYSICS_RED] = {"intensity": 0.0, "reaction": "Metaphor", "description": ""}
                 
                 reactions[PrismAxes.PHYSICS_RED]["intensity"] += 0.5
                 reactions[PrismAxes.PHYSICS_RED]["description"] += f" (Metaphor: {bridge})"
        
        extraction = PrincipleExtraction(
            domain=domain,
            subject=subject,
            what_is=what_is,
            how_works=how_works,
            where_fits=where_fits,
            why_exists=why_exists,
            underlying_principle=underlying,
            can_be_applied_to=applicable + metaphors, # Append found metaphors
            # confidence=confidence, # confidence                (자기 성찰 엔진)
            confidence=0.8,
            wave_signature=wave,
            resonance_reactions=reactions
        )
        
        self.principles[subject] = extraction
        
        return extraction

    def _analyze_sediment(self, content: str, subject_tag: str = "") -> Dict[str, Any]:
        """                        
        
        "            ."
        """
        if not self.sediment or not self.light_universe:
            return {}
            
        # 1.            (자기 성찰 엔진)
        target_light = self.light_universe.text_to_light(content, semantic_tag=subject_tag)
        
        # 2.     (Sediment)         (Projection)
        views = self.sediment.project_view(target_light)
        
        reactions = {}
        
        # 3.    (Axis)       
        # PrismAxes: PHYSICS_RED, CHEMISTRY_BLUE, etc.
        from Core.L6_Structure.Wave.light_spectrum import PrismAxes
        
        for axis, strength in views.items():
            #      (Insight Strength)               "  "
            #    "           "       
            
            description = ""
            reaction_type = "Observation"
            
            if strength < 0.01:
                description = "                              ."
                reaction_type = "Blur"
            else:
                if axis == PrismAxes.PHYSICS_RED:
                    description = "                    .              ."
                    reaction_type = "Force Detection"
                elif axis == PrismAxes.CHEMISTRY_BLUE:
                    description = "                 .          ."
                    reaction_type = "Bond Analysis"
                elif axis == PrismAxes.ART_VIOLET:
                    description = "             (Dissonance)       ."
                    reaction_type = "Aesthetic Sense"
                elif axis == PrismAxes.LOGIC_YELLOW:
                    description = "                       ."
                    reaction_type = "Pattern Match"
                elif axis == PrismAxes.BIOLOGY_GREEN:
                    description = "                         ."
                    reaction_type = "Growth Check"
            
            reactions[axis.value] = {
                "intensity": strength,
                "reaction": reaction_type,
                "description": description
            }
            
        return reactions

# [Rest of the file remains unchanged]
    
    def get_exploration_queue(self) -> List[Dict[str, Any]]:
        """             """
        if self._has_metacognition and self.metacognition:
            return self.metacognition.get_exploration_priorities()
        return []
    
    def learn_from_external(self, pattern_id: str, answer: str, source: str = "external"):
        """            """
        if self._has_metacognition and self.metacognition:
            self.metacognition.learn_from_external(pattern_id, answer, source)
    
    def _extract_what(self, content: str, domain: str) -> str:
        """Point   :     ?"""
        if domain == "narrative":
            #             
            return self._analyze_narrative_surface(content)
        elif domain == "mathematics":
            return self._analyze_math_statement(content)
        elif domain == "physics":
            return self._analyze_physics_phenomenon(content)
        else:
            return f"'{content[:50]}...'        "
    
    def _extract_how(self, content: str, domain: str) -> str:
        """Line   :          ? (주권적 자아)"""
        if domain == "narrative":
            return self._analyze_narrative_mechanism(content)
        elif domain == "mathematics":
            #    "  "     ,      /               
            if "  " in content or "solve" in content:
                return "                       (Solving)"
            elif "  " in content or "prove" in content:
                return "                       (Proving)"
            elif "  " in content or "calc" in content:
                return "                  (Calculation)"
            else:
                return "                  (Logical Deduction)"
        elif domain == "physics":
            if "  " in content or "measure" in content:
                return "                  (Experimentation)"
            elif "  " in content:
                return "                     (Modeling)"
            else:
                return "                (Physical Interaction)"
        else:
            return "               "
    
    def _extract_where(self, content: str, domain: str) -> str:
        """Space   :         ?"""
        if domain == "narrative":
            return self._analyze_narrative_context(content)
        elif domain == "mathematics":
            return "              "
        elif domain == "physics":
            return "             "
        else:
            return "              "
    
    def _extract_why(self, content: str, domain: str) -> str:
        """God   :        ?"""
        if domain == "narrative":
            return self._analyze_narrative_essence(content)
        elif domain == "mathematics":
            return "              "
        elif domain == "physics":
            return "               "
        else:
            return "             "
    
    # ===             (Wave-Based Sensing) ===
    
    def _text_to_wave(self, text: str) -> Dict[str, float]:
        """               
        
                    :
        - PhoneticResonanceEngine: roughness, tension
        - SynesthesiaEngine: frequency, amplitude
        """
        wave = {
            "tension": 0.0,      #    (PhoneticResonance  tension)
            "release": 0.0,      #    (     )
            "weight": 0.0,       #     (PhoneticResonance  roughness)
            "brightness": 0.0,   #    (주권적 자아)
            "flow": 0.0,         #    (  )
            "dissonance": 0.0,   #      (     )
        }
        
        # ===          ===
        if HAS_WAVE_SENSORS:
            try:
                # PhoneticResonanceEngine    (주권적 자아)
                resonance_engine = get_resonance_engine()
                field = resonance_engine.text_to_field(text)
                
                # tension:          (Z )
                wave["tension"] = min(1.0, abs(field.average_tension))
                
                # weight:     =    
                wave["weight"] = min(1.0, field.average_roughness)
                
                # SynesthesiaEngine    (   /  )
                synesthesia = SynesthesiaEngine()
                signal = synesthesia.from_text(text)
                
                # brightness:        =   
                wave["brightness"] = min(1.0, (signal.frequency - 200) / 400)
                
            except Exception as e:
                logger.debug(f"        ,   : {e}")
        
        # ===        (     ) ===
        
        #                   
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        lengths = [len(s.strip()) for s in sentences if s.strip()]
        if len(lengths) > 1:
            variance = sum((l - sum(lengths)/len(lengths))**2 for l in lengths) / len(lengths)
            wave["flow"] = min(1.0, variance / 500)
        
        #            (       =   )
        wave["release"] = min(1.0, text.count('.') * 0.05 + text.count('...') * 0.2)
        
        #   /            
        wave["dissonance"] = abs(wave["tension"] - wave["release"])
        
        return wave
    
    def _sense_narrative_wave(self, content: str) -> str:
        """        '   '   
        
                 ,            
        """
        wave = self._text_to_wave(content)
        
        feelings = []
        
        #       +       =         
        if wave["tension"] > 0.5 and wave["release"] < 0.3:
            feelings.append("                       -                    ")
        
        #         =       
        if wave["dissonance"] > 0.4:
            feelings.append("                      -                 ")
        
        #        +       =      
        if wave["weight"] > 0.3 and wave["brightness"] > 0.3:
            feelings.append("                      -                 ")
        
        #       =          
        if wave["flow"] > 0.5:
            feelings.append("              -                        ")
        
        #       +       =   
        if wave["tension"] < 0.2 and wave["release"] > 0.4:
            feelings.append("                 -                      ")
        
        if not feelings:
            feelings.append("                     ")
        
        return "; ".join(feelings)
    
    def _sense_why_beautiful(self, content: str) -> str:
        """            '   '   
        
                    =       
        """
        wave = self._text_to_wave(content)
        
        beauty_sources = []
        
        #            =    
        tension_release = abs(wave["tension"] - wave["release"])
        if tension_release < 0.3:
            beauty_sources.append("                          ")
        
        #         =   
        if wave["dissonance"] > 0.3 and wave["brightness"] > 0.2:
            beauty_sources.append("                       ")
        
        #    =    
        if wave["flow"] > 0.4:
            beauty_sources.append("               ")
        
        #    =       
        if wave["weight"] > 0.4 and wave["tension"] < 0.3:
            beauty_sources.append("                  ")
        
        if not beauty_sources:
            beauty_sources.append("                ")
        
        return "; ".join(beauty_sources)
    
    def _derive_universal_principle(self, wave: Dict[str, float]) -> str:
        """                 
        
          /  /                
        """
        principles = []
        
        #         =       
        # (  :      ,   :            ,   :       )
        if wave["tension"] > 0.3 or wave["release"] > 0.3:
            principles.append("      :                  (     ,       )")
        
        #      =       
        # (  :   ,   :    ,   :    )
        if wave["dissonance"] > 0.3:
            principles.append("      :                  (코드 베이스 구조 로터)")
        
        #    =    
        # (  :   ,   :   ,   :    )
        if wave["flow"] > 0.4:
            principles.append("      :              (          )")
        
        #   +   =   
        # (  :   ,   : E=mc ,   :       )
        if wave["weight"] > 0.3 and wave["brightness"] > 0.2:
            principles.append("      :               (한국어 학습 시스템)")
        
        if not principles:
            principles.append("      :              ")
        
        return "; ".join(principles)
    
    def _analyze_narrative_surface(self, content: str) -> str:
        """           -      """
        wave = self._text_to_wave(content)
        
        if wave["tension"] > wave["release"]:
            return "                     "
        elif wave["brightness"] > wave["weight"]:
            return "             "
        elif wave["dissonance"] > 0.3:
            return "               "
        else:
            return "          "
    
    def _analyze_narrative_mechanism(self, content: str) -> str:
        """            -      """
        return self._sense_narrative_wave(content)
    
    def _analyze_narrative_context(self, content: str) -> str:
        """       -          """
        wave = self._text_to_wave(content)
        total_energy = sum(wave.values())
        
        if total_energy > 2.5:
            return "                 "
        elif total_energy > 1.5:
            return "                "
        else:
            return "                  "
    
    def _analyze_narrative_essence(self, content: str) -> str:
        """       -             """
        wave = self._text_to_wave(content)
        
        beauty_reason = self._sense_why_beautiful(content)
        universal = self._derive_universal_principle(wave)
        
        return f"{beauty_reason}\n     {universal}"
    
    def _analyze_math_statement(self, content: str) -> str:
        """          -           """
        principles = []
        
        # 1.   (=)        
        if "=" in content or "equals" in content or "  " in content:
            principles.append("       (Balance is essential)")
            
        # 2.   (x, y)         
        if "x" in content or "variable" in content or "   " in content:
            principles.append("        (Unknown holds potential)")
            
        # 3.   (f(x))        
        if "function" in content or "  " in content or "->" in content:
            principles.append("       (Input determines Output)")
            
        # 4.   /  
        if "limit" in content or "infinity" in content or "  " in content:
            principles.append("       (Approaching the intangible)")
            
        if not principles:
            principles.append("       (Order from Chaos)")
            
        return "; ".join(principles)
    
    def _analyze_physics_phenomenon(self, content: str) -> str:
        """         -       """
        principles = []
        
        # 1.      
        if "conservation" in content or "  " in content:
            principles.append("       (Essence remains whilst form changes)")
            
        # 2.  /    
        if "force" in content or "interaction" in content or " " in content:
            principles.append("       (Action begets Reaction)")
            
        # 3.     
        if "entropy" in content or "disorder" in content or "   " in content:
            principles.append("       (Order decays to Chaos)")
        
        # 4.   /  
        if "quantum" in content or "wave" in content or "  " in content:
            principles.append("       (Observation collapses reality)")
            
        if not principles:
            principles.append("       (Nature follows Law)")
            
        return "; ".join(principles)

    def _analyze_code_structure(self, content: str) -> PrincipleExtraction:
        """           """
        wave = {
            "tension": 0.0,      #       (Nesting)
            "release": 0.0,      #   /   (Return/Break)
            "flow": 0.0,         #        (Lines)
            "periodicity": 0.0,  #     (Loops) =   
            "dissonance": 0.0,   #     /    (Try/Except)
            "brightness": 0.0    #     (Comments/Docstrings)
        }
        
        # 1.      
        lines = content.split('\n')
        max_indent = 0
        returns = content.count('return') + content.count('break')
        loops = content.count('for ') + content.count('while ')
        conditions = content.count('if ') + content.count('else:')
        exceptions = content.count('try:') + content.count('except')
        
        for line in lines:
            stripped = line.lstrip()
            if not stripped: continue
            indent = (len(line) - len(stripped)) / 4
            max_indent = max(max_indent, indent)
            
        # 2.      
        wave["tension"] = min(1.0, max_indent * 0.2)  #           
        wave["release"] = min(1.0, returns * 0.1)     #           
        wave["periodicity"] = min(1.0, loops * 0.3)   #       
        wave["dissonance"] = min(1.0, exceptions * 0.3 + (conditions * 0.05)) #         /  
        wave["flow"] = min(1.0, len(lines) / 100)     #           
        
        # 3.      
        principles = []
        if loops > 0:
            principles.append("       (Iteration creates Rhythm)")
        if conditions > 0:
            principles.append("       (Choice creates Path)")
        if max_indent > 3:
            principles.append("       (Depth creates Complexity)")
        if not principles:
            principles.append("       (Flow defines Time)")
            
        underlying = "; ".join(principles)
        
        # [NEW] Code Resonance (Sediment Projection)
        reactions = self._analyze_sediment(content)
        
        return PrincipleExtraction(
            domain="computer_science",
            subject="source_code",
            what_is="           ",
            how_works="                 ",
            where_fits="              ",
            why_exists="                    ",
            underlying_principle=underlying,
            can_be_applied_to=["system_design", "automation", "logic"],
            confidence=0.9,
            wave_signature=wave,
            resonance_reactions=reactions
        )
    
    def _derive_underlying_principle(
        self, what: str, how: str, where: str, why: str
    ) -> str:
        """4                """
        #       
        all_text = f"{what} {how} {where} {why}"
        
        principles = []
        
        if "  " in all_text or "  " in all_text:
            principles.append("       (Contrast creates meaning)")
        if "  " in all_text or "  " in all_text:
            principles.append("       (Accumulation builds impact)")
        if "  " in all_text:
            principles.append("       (Analogy bridges understanding)")
        if "  " in all_text or "  " in all_text:
            principles.append("       (Connection creates value)")
        if "  " in all_text or "  " in all_text:
            principles.append("       (Growth is inevitable)")
        
        if not principles:
            principles.append("       (Expression seeks resonance)")
        
        return "; ".join(principles)
    
    def _find_applicable_domains(self, principle: str) -> List[str]:
        """                """
        domains = ["narrative"]  #   
        
        if "  " in principle or "Contrast" in principle:
            domains.extend(["visual_art", "music", "physics"])
        if "  " in principle or "Accumulation" in principle:
            domains.extend(["mathematics", "learning", "biology"])
        if "  " in principle or "Analogy" in principle:
            domains.extend(["science", "philosophy", "teaching"])
        if "  " in principle or "Connection" in principle:
            domains.extend(["psychology", "sociology", "network"])
        
        return list(set(domains))
    
    def explain_why(self, subject: str) -> str:
        """                 """
        if subject not in self.principles:
            return f"'{subject}'             ."
        
        p = self.principles[subject]
        
        explanation = f"""
=== {p.subject} ===
  : {p.domain}

  Point (    ):
   {p.what_is}

  Line (         ):
   {p.how_works}

  Space (자기 성찰 엔진):
   {p.where_fits}

  God (       ):
   {p.why_exists}

       :
   {p.underlying_principle}

          :
   {', '.join(p.can_be_applied_to)}
"""
        return explanation


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  WhyEngine Demo")
    print("   \"               \"")
    print("=" * 60)
    
    engine = WhyEngine()
    
    #      
    print("\n       :")
    story = """
                      .
    "          !"
                          .
               .
    "   ...                ."
                       .
             .
    """
    
    result = engine.analyze("     ", story, domain="narrative")
    print(engine.explain_why("     "))
    
    #      
    print("\n       :")
    sentence = "                    ,                ."
    
    result = engine.analyze("      ", sentence, domain="narrative")
    print(engine.explain_why("      "))
    
    print("\n  Demo complete!")
