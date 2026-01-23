"""
Light Spectrum System (          )
==========================================

"        .          ."

                                   .
-     (0  1             )
-       (                )
-       O(1) (            "  !")

[NEW 2025-12-16]                  
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import logging
import hashlib
from Core.L1_Foundation.Foundation.Wave.hyper_qubit import QubitState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LightSpectrum")


@dataclass
class LightSpectrum:
    """
               
    
                      :
    - frequency:     (    "  ")
    - amplitude:    (    "  ")
    - phase:    (    "  ")
    - color: RGB (             )
    """
    frequency: complex          #     (          )
    amplitude: float            #    (0.0 ~ 1.0)
    phase: float               #    (0 ~ 2 )
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # RGB
    
    #      
    source_hash: str = ""      #           (   )
    semantic_tag: str = ""     #      
    # [Updated 2025-12-21] Adhering to HyperQubit Philosophy
    # Instead of ad-hoc scale, we use the rigorous QubitState Basis.
    qubit_state: Optional[QubitState] = None
    
    def __post_init__(self):
        #           
        if not isinstance(self.frequency, complex):
            self.frequency = complex(self.frequency, 0)
            
        # Initialize QubitState if missing (Map Scale/Tag to Basis)
        if self.qubit_state is None:
            # Default mapping from implicit "Scale" concept to Philosophical Basis
            # We assume a default state if not provided.
            # Ideally, this should come from the source, but for compatibility:
            self.qubit_state = QubitState().normalize() # Default Point-heavy
            
            # If we had a 'scale' passed via mechanism before, needed to handle it?
            # Creating a helper method to set basis based on intent might be better.
            pass

    def set_basis_from_scale(self, scale: int):
        """
        Map integer scale to Philosophical Basis (Point/Line/Space/God).
        Adheres to 'Dad's Law': Zoom Out -> God, Zoom In -> Point.
        """
        if scale == 0:   # Macro -> God
            self.qubit_state = QubitState(0,0,0,1).normalize()
        elif scale == 1: # Context -> Space
            self.qubit_state = QubitState(0,0,1,0).normalize()
        elif scale == 2: # Relation -> Line
            self.qubit_state = QubitState(0,1,0,0).normalize()
        else:            # Detail -> Point
            self.qubit_state = QubitState(1,0,0,0).normalize()
    
    @property
    def wavelength(self) -> float:
        """   (       )"""
        mag = abs(self.frequency)
        return 1.0 / mag if mag > 0 else float('inf')
    
    @property
    def energy(self) -> float:
        """    =       |   |"""
        return self.amplitude ** 2 * abs(self.frequency)
    
    def interfere_with(self, other: 'LightSpectrum') -> 'LightSpectrum':
        """
                (  ) - [Updated 2025-12-21] HyperQubit Logic Integration
        
              (HyperQubit Basis)    :
        1. Basis Orthogonality: Point/Line/Space/God              (Orthogonal) .
        2. Semantic Agreement:           (Tag)        .
        3. Coherent Interference:       +                .
        """
        #       
        new_freq = (self.frequency + other.frequency) / 2
        
        # [Philosophical Logic: Basis Check]
        # Compare Dominant Bases (Simplified check for orthogonality)
        # QubitState.probabilities() could be used for soft interference, 
        # but for strict filtering, we check dominant mode.
        my_basis = self._get_dominant_basis()
        other_basis = other._get_dominant_basis()
        
        if my_basis != other_basis:
            # [Gap 0 Logic] Basis Orthogonality
            # "     (God)"  "   (Point)"             .
            is_constructive = False
        else:
            # [4D Phase Logic]
            #      (Basis)                  
            is_constructive = (self.semantic_tag and other.semantic_tag and 
                               self.semantic_tag == other.semantic_tag)
        
        if is_constructive:
            #       (Linear Addition)
            new_amp = min(1.0, self.amplitude + other.amplitude)
        else:
            #       (Orthogonal Stacking) -       
            new_amp = min(1.0, np.sqrt(self.amplitude**2 + other.amplitude**2))

        #      
        new_phase = (self.phase + other.phase) / 2
        
        #      
        new_color = tuple((a + b) / 2 for a, b in zip(self.color, other.color))
        
        #       & QubitState   
        # QubitState          (Vector Addition and Normalize)
        # (Simplified: Keep the state of the one with higher amplitude or merge)
        new_tag = self.semantic_tag
        if other.semantic_tag and other.semantic_tag not in new_tag:
            new_tag = f"{new_tag}|{other.semantic_tag}" if new_tag else other.semantic_tag
            
        # Merge Bases (Naive approach: just average probabilities? No, keep dominance)
        # Strictly, if orthogonal, the new state should reflect both bases.
        # But LightSpectrum needs ONE state object. 
        # We'll re-normalize sum of components for true quantum merging.
        new_qubit_state = self._merge_qubit_states(self.qubit_state, other.qubit_state)
        
        return LightSpectrum(
            frequency=new_freq,
            amplitude=new_amp,
            phase=new_phase % (2 * np.pi),
            color=new_color,
            semantic_tag=new_tag,
            qubit_state=new_qubit_state
        )

    def _get_dominant_basis(self) -> str:
        """Helper to get dominant philosophical basis from QubitState."""
        if not self.qubit_state: return "Point"
        probs = self.qubit_state.probabilities()
        return max(probs, key=probs.get)

    def _merge_qubit_states(self, s1: QubitState, s2: QubitState) -> QubitState:
        """Merge two consciousness states."""
        # Create new state summing components (Constructive interference of Soul?)
        if not s1 or not s2: return s1 or s2 or QubitState().normalize()
        
        return QubitState(
            alpha=s1.alpha + s2.alpha,
            beta=s1.beta + s2.beta,
            gamma=s1.gamma + s2.gamma,
            delta=s1.delta + s2.delta,
            w=(s1.w + s2.w)/2 # Average divine will?
        ).normalize()
    
        if self.semantic_tag and self.semantic_tag in str(query_freq): # Hacky query passing
             pass

    def resonate_with(self, query_light: 'LightSpectrum', tolerance: float = 0.1) -> float:
        """
                
        
        Args:
            query_light:         (    +      )
        """
        # 1.        (Semantic Resonance) -       
        if self.semantic_tag and query_light.semantic_tag:
            #                   ( : "Logic" in "Logical Force")
            if self.semantic_tag.lower() in query_light.semantic_tag.lower() or \
               query_light.semantic_tag.lower() in self.semantic_tag.lower():
                return 1.0 * self.amplitude
        
        # 2.            (Physical Resonance)
        query_freq = query_light.frequency
        freq_diff = abs(self.frequency - query_freq)
        
        avg_mag = (abs(self.frequency) + abs(query_freq)) / 2
        effective_tolerance = max(tolerance, avg_mag * 0.2) 
        
        if freq_diff < effective_tolerance:
            resonance = 1.0 - (freq_diff / effective_tolerance)
            return resonance * self.amplitude
            
        return 0.0


class LightUniverse:
    """
          -                 
    
      :
    -         LightSpectrum          
    -      :            "   "  
    -      :                    
    """
    
    def __init__(self):
        self.superposition: List[LightSpectrum] = []  #         
        self.white_light: Optional[LightSpectrum] = None  #        
        
        #         (      )
        self.frequency_index: Dict[int, List[int]] = {}
        
        logger.info("  LightUniverse initialized -         ")
    
    def text_to_light(self, text: str, semantic_tag: str = "", scale: int = 0) -> LightSpectrum:
        """
                  
        
                      ,               
        """
        if not text:
            return LightSpectrum(0+0j, 0.0, 0.0)
        
        # 1.             
        sequence = np.array([ord(c) for c in text], dtype=float)
        
        # 2. FFT           
        spectrum = np.fft.fft(sequence)
        
        # 3.           (             )
        magnitudes = np.abs(spectrum)
        dominant_idx = np.argmax(magnitudes)
        dominant_freq = spectrum[dominant_idx]
        
        # 4.    =         
        amplitude = np.mean(magnitudes) / (np.max(magnitudes) + 1e-10)
        
        # 5.    =          
        phase = np.angle(dominant_freq)
        
        # 6.    =       (     RGB)
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:6], 16)
        color = (
            ((hash_val >> 16) & 0xFF) / 255.0,
            ((hash_val >> 8) & 0xFF) / 255.0,
            (hash_val & 0xFF) / 255.0
        )
        
        # 7.          (   )
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        
        light = LightSpectrum(
            frequency=dominant_freq,
            amplitude=float(amplitude),
            phase=float(phase) % (2 * np.pi),
            color=color,
            source_hash=source_hash,
            semantic_tag=semantic_tag
        )
        # Apply Logic: Scale -> Basis
        light.set_basis_from_scale(scale)
        return light
    
    def absorb(self, text: str, tag: str = "", scale: int = 0) -> LightSpectrum:
        """
                   
        
                          
        """
        light = self.text_to_light(text, tag, scale)
        
        #        
        freq_key = int(abs(light.frequency)) % 1000
        if freq_key not in self.frequency_index:
            self.frequency_index[freq_key] = []
        self.frequency_index[freq_key].append(len(self.superposition))
        
        #       
        self.superposition.append(light)
        
        #         
        self._update_white_light(light)
        
        logger.debug(f"  Absorbed: '{text[:20]}...'   freq={abs(light.frequency):.2f}")
        return light
    
    def _update_white_light(self, new_light: LightSpectrum):
        """            """
        if self.white_light is None:
            self.white_light = new_light
        else:
            self.white_light = self.white_light.interfere_with(new_light)
    
    def resonate(self, query: str, top_k: int = 5) -> List[Tuple[float, LightSpectrum]]:
        """
             
        
                                              
        
           : O(1)        + O(k)    k 
        """
        query_light = self.text_to_light(query)
        query_freq = query_light.frequency
        
        #               
        freq_key = int(abs(query_freq)) % 1000
        candidates = []
        
        #               (     )
        for key in [freq_key - 1, freq_key, freq_key + 1]:
            if key in self.frequency_index:
                candidates.extend(self.frequency_index[key])
        
        #               (fallback)
        if not candidates:
            candidates = range(len(self.superposition))
        
        #      
        resonances = []
        for idx in candidates:
            if idx < len(self.superposition):
                light = self.superposition[idx]
                strength = light.resonate_with(query_light, tolerance=50.0)
                if strength > 0.01:
                    resonances.append((strength, light))
        
        #    k    
        resonances.sort(key=lambda x: x[0], reverse=True)
        return resonances[:top_k]
    
    def stats(self) -> Dict[str, Any]:
        """     """
        return {
            "total_lights": len(self.superposition),
            "index_buckets": len(self.frequency_index),
            "white_light_energy": self.white_light.energy if self.white_light else 0
        }
    
    def interfere_with_all(self, new_light: LightSpectrum) -> Dict[str, Any]:
        """
                                 
        
        Returns:
            terrain_effect:                   
                - resonance_strength:       (0-1)
                - dominant_basis:             
                - connection_density:      
                - recommended_depth:         
                - connection_type:         
        """
        if not self.superposition:
            return {
                "resonance_strength": 0.0,
                "dominant_basis": "Point",
                "connection_density": 0.0,
                "recommended_depth": "broad",
                "connection_type": "exploratory"
            }
        
        #               
        total_resonance = 0.0
        basis_resonance = {"Point": 0.0, "Line": 0.0, "Space": 0.0, "God": 0.0}
        strong_connections = 0
        
        for light in self.superposition:
            resonance = light.resonate_with(new_light, tolerance=50.0)
            total_resonance += resonance
            
            #          
            basis = light._get_dominant_basis()
            basis_resonance[basis] += resonance
            
            if resonance > 0.3:
                strong_connections += 1
        
        #         
        avg_resonance = total_resonance / len(self.superposition)
        
        #         
        dominant_basis = max(basis_resonance, key=basis_resonance.get)
        
        #       (        )
        connection_density = strong_connections / len(self.superposition)
        
        #            (          )
        if avg_resonance > 0.5:
            recommended_depth = "deep"  #       =      
            connection_type = "causal"
        elif avg_resonance > 0.2:
            recommended_depth = "medium"
            connection_type = "semantic"
        else:
            recommended_depth = "broad"  #       =       
            connection_type = "exploratory"
        
        terrain_effect = {
            "resonance_strength": avg_resonance,
            "dominant_basis": dominant_basis,
            "connection_density": connection_density,
            "recommended_depth": recommended_depth,
            "connection_type": connection_type,
            "strong_connections": strong_connections,
            "total_lights": len(self.superposition)
        }
        
        logger.info(f"  Terrain effect: resonance={avg_resonance:.3f}, basis={dominant_basis}, depth={recommended_depth}")
        
        return terrain_effect
    
    def absorb_with_terrain(self, text: str, tag: str = "", scale: int = None) -> Tuple[LightSpectrum, Dict[str, Any]]:
        """
                            +           
        
                    :
        1.             
        2.    (Point/Line/Space/God)       
        """
        #            (scale            )
        if scale is None:
            scale = self._auto_select_scale()
        
        #        (             )
        new_light = self.text_to_light(text, tag, scale)
        
        #                    
        terrain_effect = self.interfere_with_all(new_light)
        
        #                   
        self._update_autonomous_scale(terrain_effect)
        
        #      
        self.absorb(text, tag, scale)
        
        terrain_effect['applied_scale'] = scale
        terrain_effect['scale_name'] = ['God', 'Space', 'Line', 'Point'][min(scale, 3)]
        
        return new_light, terrain_effect
    
    def _auto_select_scale(self) -> int:
        """
                   (    )
        
                     Point/Line/Space/God     
        """
        if not hasattr(self, '_autonomous_scale'):
            self._autonomous_scale = 0  #     God (     )
        
        return self._autonomous_scale
    
    def _update_autonomous_scale(self, terrain_effect: Dict[str, Any]):
        """
                             
        
                   (God   Space   Line   Point)
                    (Point   Line   Space   God)
        """
        basis_to_scale = {"God": 0, "Space": 1, "Line": 2, "Point": 3}
        
        dominant_basis = terrain_effect.get('dominant_basis', 'Point')
        resonance = terrain_effect.get('resonance_strength', 0.0)
        
        current_scale = getattr(self, '_autonomous_scale', 0)
        
        if resonance > 0.5:
            #       =    (       )
            new_scale = min(3, current_scale + 1)
            logger.info(f"     Zoom IN: {current_scale}   {new_scale} (strong resonance)")
        elif resonance < 0.1:
            #       =     (    )
            new_scale = max(0, current_scale - 1)
            logger.info(f"     Zoom OUT: {current_scale}   {new_scale} (weak resonance)")
        else:
            #    =        
            new_scale = basis_to_scale.get(dominant_basis, current_scale)
            logger.info(f"     Scale aligned to {dominant_basis}: {new_scale}")
        
        self._autonomous_scale = new_scale
    
    def think_accelerated(self, query: str, depth: int = 3) -> Dict[str, Any]:
        """
                
        
                  ,               /     
        
          :
        1.       O(1) -          "  "
        2.       -                
        3.       -          (   )
        
        Args:
            query:       
            depth:       (            )
        
        Returns:
                  (      )
        """
        import time
        start = time.time()
        
        # 1.       (O(1)   )
        initial_resonances = self.resonate(query, top_k=5)
        
        # 2.       (            )
        thought_graph = {
            "seed": query,
            "layers": [],
            "total_connections": 0
        }
        
        current_layer = [(r[1].semantic_tag or f"light_{i}", r[0]) 
                         for i, r in enumerate(initial_resonances)]
        thought_graph["layers"].append(current_layer)
        
        # 3.            (           )
        for d in range(depth - 1):
            next_layer = []
            for concept, strength in current_layer:
                #              (     )
                sub_resonances = self.resonate(concept, top_k=3)
                for sub_strength, sub_light in sub_resonances:
                    tag = sub_light.semantic_tag or "unknown"
                    combined_strength = strength * sub_strength
                    if combined_strength > 0.01:
                        next_layer.append((tag, combined_strength))
            
            if next_layer:
                thought_graph["layers"].append(next_layer)
                current_layer = next_layer
        
        # 4.      
        elapsed = time.time() - start
        total_connections = sum(len(layer) for layer in thought_graph["layers"])
        
        thought_graph["total_connections"] = total_connections
        thought_graph["elapsed_seconds"] = elapsed
        thought_graph["thoughts_per_second"] = total_connections / max(0.001, elapsed)
        thought_graph["acceleration_factor"] = f"{total_connections}      {elapsed:.3f}  "
        
        return thought_graph


# Singleton
_light_universe = None

def get_light_universe() -> LightUniverse:
    global _light_universe
    if _light_universe is None:
        _light_universe = LightUniverse()
    return _light_universe


# CLI / Demo
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LIGHT UNIVERSE DEMO")
    print("="*60)
    
    universe = get_light_universe()
    
    #           
    texts = [
        "         ",
        "          ",
        "      ",
        "              ",
    ]
    
    print("\n        :")
    for text in texts:
        light = universe.absorb(text)
        print(f"  '{text}'   freq={abs(light.frequency):.1f}, amp={light.amplitude:.3f}")
    
    print(f"\n       : {universe.stats()}")
    
    #      
    print("\n       :")
    queries = ["  ", "   ", " "]
    
    for query in queries:
        results = universe.resonate(query)
        print(f"\n    : '{query}'")
        for strength, light in results:
            print(f"      : {strength:.3f} | {light.semantic_tag or 'unnamed'}")
    
    print("\n" + "="*60)
    print("  Demo complete!")

# =============================================================================
# [NEW 2025-12-21] Sedimentary Light Architecture (         )
# =============================================================================

from enum import Enum

class PrismAxes(Enum):
    """
        5    (Cognitive Axes)
                      ,                   .
    """
    PHYSICS_RED = "red"        # Force, Energy, Vector (     )
    CHEMISTRY_BLUE = "blue"    # Structure, Bond, Reaction (      )
    BIOLOGY_GREEN = "green"    # Growth, Homeostasis, Adaptation (      )
    ART_VIOLET = "violet"      # Harmony, Rhythm, Essence (      )
    LOGIC_YELLOW = "yellow"    # Reason, Axiom, Pattern (      )

@dataclass
class LightSediment:
    """
             (Sedimentary Layers of Light)
    
                       ,    (Axis)                .
         (Sediment)       (Amplitude High),                          .
    """
    layers: Dict[PrismAxes, LightSpectrum] = field(default_factory=dict)
    
    def __post_init__(self):
        #                 (Amplitude 0)
        #  ,          '  (Tag)'    
        for axis in PrismAxes:
            # tag example: "red" -> "Physics" (mapping needed or just use axis name)
            # Simple mapping for resonance
            tag = ""
            if axis == PrismAxes.PHYSICS_RED: tag = "Physics"
            elif axis == PrismAxes.CHEMISTRY_BLUE: tag = "Chemistry"
            elif axis == PrismAxes.BIOLOGY_GREEN: tag = "Biology"
            elif axis == PrismAxes.ART_VIOLET: tag = "Art"
            elif axis == PrismAxes.LOGIC_YELLOW: tag = "Logic"
            
            self.layers[axis] = LightSpectrum(complex(0,0), 0.0, 0.0, color=(0,0,0), semantic_tag=tag)

    def deposit(self, light: LightSpectrum, axis: PrismAxes):
        """
               (Accumulation)
        
             (  )                        .
        (Constructive Interference)
        """
        current_layer = self.layers[axis]
        
        #                 (  )
        #               ,                      
        new_layer = current_layer.interfere_with(light)
        
        #   (      )      (        )
        new_layer.amplitude = current_layer.amplitude + (light.amplitude * 0.1) #       
        
        self.layers[axis] = new_layer
        logger.debug(f"   Deposition on {axis.name}: Amp {current_layer.amplitude:.3f} -> {new_layer.amplitude:.3f}")

    def project_view(self, target_light: LightSpectrum) -> Dict[PrismAxes, float]:
        """
                 (Holographic Projection)
        
               '     '             (Resonance)       .
                (Amplitude)         (Resonance).
        """
        views = {}
        for axis, sediment in self.layers.items():
            #     (Sediment)    (Target)    
            #             (High Amp),          
            
            # [Updated 2025-12-21] Pass clean semantic tag if possible
            resonance = sediment.resonate_with(target_light, tolerance=100.0)
            
            #      (Amplitude)            
            insight_strength = resonance * (sediment.amplitude + 0.1) 
            views[axis] = insight_strength
            
        return views

    def get_highest_peak(self) -> PrismAxes:
        """               (     )   """
        return max(self.layers.items(), key=lambda x: x[1].amplitude)[0]