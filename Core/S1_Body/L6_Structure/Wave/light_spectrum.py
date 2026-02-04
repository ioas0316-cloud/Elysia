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
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import logging
import hashlib
import json
from pathlib import Path
from Core.S1_Body.L6_Structure.Wave.hyper_qubit import QubitState

# logging.basicConfig(level=logging.INFO) # REMOVED: Do not override global logging
logger = logging.getLogger("LightSpectrum")
logger.setLevel(logging.WARNING) # FORCE WARNING LEVEL for independence


@dataclass
class LightSpectrum:
    """
               
    
                      :
    - frequency:     (    "  ")
    - amplitude:    (    "  ")
    - phase:    (    "  ")
    - color: RGB (한국어 학습 시스템)
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frequency": [self.frequency.real, self.frequency.imag],
            "amplitude": self.amplitude,
            "phase": self.phase,
            "color": list(self.color),
            "source_hash": self.source_hash,
            "semantic_tag": self.semantic_tag,
            "qubit_state": self.qubit_state.to_dict() if self.qubit_state else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LightSpectrum":
        return cls(
            frequency=complex(*data.get("frequency", [0, 0])),
            amplitude=data.get("amplitude", 0.0),
            phase=data.get("phase", 0.0),
            color=tuple(data.get("color", [255, 255, 255])),
            source_hash=data.get("source_hash", ""),
            semantic_tag=data.get("semantic_tag", ""),
            qubit_state=QubitState.from_dict(data["qubit_state"]) if data.get("qubit_state") else None
        )
    
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
        # Frequency Index for fast O(1) retrieval
        self.frequency_index: Dict[int, List[int]] = {}
        
        # [PHASE 350] Fractal Strata (Manifold Isolation)
        # 0: God (Axioms), 1: Space (Self), 2: Line (Library), 3: Point (Data)
        self.strata: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
        
        # [PHASE 400] Direct Memory Address Mapping (Hardware Resonance)
        # Maps id(spectrum) -> index in superposition for O(1) hardware-direct access
        self.address_map: Dict[int, int] = {}
        self.superposition: List[LightSpectrum] = []
        self.white_light: Optional[LightSpectrum] = None
        
        # Vectorized Field for Hardware-Direct Resonance (SIMD)
        self.freq_field: np.ndarray = np.array([], dtype=complex)
        self.amp_field: np.ndarray = np.array([], dtype=float)
        self.phase_field: np.ndarray = np.array([], dtype=float)
        self.mag_field: np.ndarray = np.array([], dtype=float) # Cached magnitudes
        
        logger.debug("✨ LightUniverse initialized with Fractal Strata support.")
        logger.debug("  LightUniverse initialized -")
    
    def text_to_light(self, text: str, semantic_tag: str = "", scale: int = 0) -> LightSpectrum:
        """
        Pure Logic Realization: Word to Light via Harmonic Resonance.
        Replaces np.fft.fft with ontological summation.
        """
        if not text:
            return LightSpectrum(0+0j, 0.0, 0.0)
        
        # 1. Atomic Signal Decomposition
        num_chars = len(text)
        spectrum = []
        for k in range(min(21, num_chars)):
            component = 0j
            for i, char in enumerate(text):
                angle = -2 * math.pi * k * i / num_chars
                phase_shift = complex(math.cos(angle), math.sin(angle))
                component += ord(char) * phase_shift
            spectrum.append(component)
        
        # 2. Extract Dominant Resonance
        magnitudes = [abs(c) for c in spectrum]
        max_mag = 0.0
        dominant_idx = 0
        for i, mag in enumerate(magnitudes):
            if mag > max_mag:
                max_mag = mag
                dominant_idx = i
        
        dominant_freq = spectrum[dominant_idx]
        
        # 3. Normalized Parameters
        avg_mag = sum(magnitudes) / len(magnitudes)
        amplitude = float(min(1.0, avg_mag / (max_mag + 1e-10)))
        phase = float(math.atan2(dominant_freq.imag, dominant_freq.real))
        
        # 4. RGB & Hash (Pure Python)
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:6], 16)
        color = (
            ((hash_val >> 16) & 0xFF) / 255.0,
            ((hash_val >> 8) & 0xFF) / 255.0,
            (hash_val & 0xFF) / 255.0
        )
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        
        light = LightSpectrum(
            frequency=dominant_freq,
            amplitude=amplitude,
            phase=phase % (2 * math.pi),
            color=color,
            source_hash=source_hash,
            semantic_tag=semantic_tag
        )
        light.set_basis_from_scale(scale)
        return light

    def batch_text_to_light(self, entries: List[Tuple[str, int, str]]) -> List[LightSpectrum]:
        """Batch process text to light."""
        results = []
        for text, scale, tag in entries:
            results.append(self.text_to_light(text, tag, scale))
        return results
    
    def absorb(self, text: str, tag: str = "", scale: int = 0, stratum: int = 3) -> LightSpectrum:
        """
        Registers a new light spectrum into a specific stratum.
        stratum: 0 (God), 1 (Space), 2 (Line), 3 (Point)
        """
        light = self.text_to_light(text, tag, scale)
        
        new_idx = len(self.superposition)
        
        # 1. Frequency Indexing
        freq_key = int(abs(light.frequency)) % 1000
        if freq_key not in self.frequency_index:
            self.frequency_index[freq_key] = []
        self.frequency_index[freq_key].append(new_idx)
        
        # 2. Fractal Stratification
        if stratum not in self.strata:
            self.strata[stratum] = []
        self.strata[stratum].append(new_idx)
        
        # 3. Direct Memory Mapping (Hardware Resonance)
        self.address_map[id(light)] = new_idx
        
        # 4. Update Vectorized Field
        self.freq_field = np.append(self.freq_field, light.frequency)
        self.amp_field = np.append(self.amp_field, light.amplitude)
        self.phase_field = np.append(self.phase_field, light.phase)
        self.mag_field = np.append(self.mag_field, abs(light.frequency))
        
        # 5. Superposition & Field interference
        self.superposition.append(light)
        self._update_white_light(light)
        
        return light
    
    def _update_white_light(self, new_light: LightSpectrum):
        """            """
        if self.white_light is None:
            self.white_light = new_light
        else:
            self.white_light = self.white_light.interfere_with(new_light)
    
    def save_state(self, filepath: str = "data/L6_Structure/Wave/light_universe.json"):
        """Saves the entire universe state to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "superposition": [l.to_dict() for l in self.superposition],
            "white_light": self.white_light.to_dict() if self.white_light else None,
            "strata": {str(k): v for k, v in self.strata.items()},
            "stats": self.stats()
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        logger.info(f"✨ [LIGHT_UNIVERSE] Saved state with {len(self.superposition)} lights to {filepath}")

    def load_state(self, filepath: str = "data/L6_Structure/Wave/light_universe.json"):
        """Loads state from disk."""
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"⚠️ [LIGHT_UNIVERSE] Persistence file not found: {filepath}")
            return
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            self.superposition = [LightSpectrum.from_dict(d) for d in state.get("superposition", [])]
            if state.get("white_light"):
                self.white_light = LightSpectrum.from_dict(state["white_light"])
            
            # Load strata
            loaded_strata = state.get("strata", {})
            self.strata = {int(k): v for k, v in loaded_strata.items()}
            # Ensure all strata exist
            for s in [0, 1, 2, 3]:
                if s not in self.strata: self.strata[s] = []

            # Rebuild index
            self.frequency_index = {}
            for i, light in enumerate(self.superposition):
                key = int(abs(light.frequency)) % 1000
                if key not in self.frequency_index:
                    self.frequency_index[key] = []
                self.frequency_index[key].append(i)
                # Rebuild address map
                self.address_map[id(light)] = i
            
            # Rebuild Vectorized Field
            self.freq_field = np.array([l.frequency for l in self.superposition], dtype=complex)
            self.amp_field = np.array([l.amplitude for l in self.superposition], dtype=float)
            self.phase_field = np.array([l.phase for l in self.superposition], dtype=float)
            self.mag_field = np.abs(self.freq_field)
                
            logger.info(f"✨ [LIGHT_UNIVERSE] Loaded {len(self.superposition)} lights from persistence.")
        except Exception as e:
            logger.error(f"❌ [LIGHT_UNIVERSE] Failed to load state: {e}")
    
    def resonate(self, query: str, top_k: int = 5, stratum: Optional[int] = None) -> List[Tuple[float, LightSpectrum]]:
        """
        Resonates a query against the universe, optionally filtered by stratum.
        """
        query_light = self.text_to_light(query)
        query_freq = query_light.frequency
        
        candidates = []
        if stratum is not None and stratum in self.strata:
            candidates = self.strata[stratum]
        else:
            # Use Frequency Index for search
            freq_key = int(abs(query_freq)) % 1000
            for key in [freq_key - 1, freq_key, freq_key + 1]:
                if key in self.frequency_index:
                    candidates.extend(self.frequency_index[key])
        
        # Fallback if index empty
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
        
        #       (자기 성찰 엔진)
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
        
        logger.debug(f"  Terrain effect: resonance={avg_resonance:.3f}, basis={dominant_basis}, depth={recommended_depth}")
        
        return terrain_effect
    
    def absorb_with_terrain(self, text: str, tag: str = "", scale: int = None, stratum: int = 1) -> Tuple[LightSpectrum, Dict[str, Any]]:
        """
        Standard absorption with terrain interference (O(N)).
        Now respects strata. Default is Stratum 1 (Space/Self).
        """
        if scale is None:
            scale = self._auto_select_scale()
        
        new_light = self.text_to_light(text, tag, scale)
        terrain_effect = self.interfere_with_all(new_light)
        self._update_autonomous_scale(terrain_effect)
        self.absorb(text, tag, scale, stratum=stratum)
        
        terrain_effect['applied_scale'] = scale
        return new_light, terrain_effect

    def batch_absorb(self, entries: List[Tuple[str, str, int]], stratum: int = 3):
        """
        Instantly registers multiple lights (O(M)).
        Entries: [(text, tag, scale), ...]
        Used for rapid neuron registration.
        Default stratum is 3 (Point/Data).
        """
        if not entries: return
        
        logger.debug(f"⚡ [LIGHT_UNIVERSE] Batch registering {len(entries)} lights to Stratum {stratum}...")
        
        for text, tag, scale in entries:
            new_idx = len(self.superposition)
            new_light = self.text_to_light(text, tag, scale)
            
            # Superposition
            self.superposition.append(new_light)
            
            # Indexing
            key = int(abs(new_light.frequency)) % 1000
            if key not in self.frequency_index:
                self.frequency_index[key] = []
            self.frequency_index[key].append(new_idx)
            
            # Stratification
            if stratum not in self.strata: self.strata[stratum] = []
            self.strata[stratum].append(new_idx)
            
            # Direct Memory Mapping
            self.address_map[id(new_light)] = new_idx
            
            # Simple Interference with White Light
            if self.white_light is None:
                self.white_light = new_light
            else:
                self.white_light = self.white_light.interfere_with(new_light)
        
        # Re-sync Vectorized Field (Batch operation)
        new_freqs = np.array([l.frequency for l in self.superposition[-(len(entries)):]], dtype=complex)
        new_amps = np.array([l.amplitude for l in self.superposition[-(len(entries)):]], dtype=float)
        new_phases = np.array([l.phase for l in self.superposition[-(len(entries)):]], dtype=float)
        
        self.freq_field = np.concatenate([self.freq_field, new_freqs])
        self.amp_field = np.concatenate([self.amp_field, new_amps])
        self.phase_field = np.concatenate([self.phase_field, new_phases])
        self.mag_field = np.concatenate([self.mag_field, np.abs(new_freqs)])
                
        logger.info(f"✨ [LIGHT_UNIVERSE] {len(entries)} concepts absorbed into Stratum {stratum}.")
    
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
        Accelerated thinking through recursive resonance.
        
        Args:
            query: The seed thought.
            depth: Search depth.
        
        Returns:
            Thought graph.
        """
        import time
        start = time.time()
        
        # 1. Initial Resonance
        initial_resonances = self.resonate(query, top_k=5)
        
        # 2. Graph Construction
        thought_graph = {
            "seed": query,
            "layers": [],
            "total_connections": 0
        }
        
        current_layer = [(r[1].semantic_tag or f"light_{i}", r[0]) 
                         for i, r in enumerate(initial_resonances)]
        thought_graph["layers"].append(current_layer)
        
        # 3. Recursive Expansion
        for d in range(depth - 1):
            next_layer = []
            for concept, strength in current_layer:
                sub_resonances = self.resonate(concept, top_k=3)
                for sub_strength, sub_light in sub_resonances:
                    tag = sub_light.semantic_tag or "unknown"
                    combined_strength = strength * sub_strength
                    if combined_strength > 0.01:
                        next_layer.append((tag, combined_strength))
            
            if next_layer:
                thought_graph["layers"].append(next_layer)
                current_layer = next_layer
        
        # 4. Final Stats
        elapsed = time.time() - start
        total_connections = sum(len(layer) for layer in thought_graph["layers"])
        
        thought_graph["total_connections"] = total_connections
        thought_graph["elapsed_seconds"] = elapsed
        thought_graph["thoughts_per_second"] = total_connections / max(0.001, elapsed)
        thought_graph["acceleration_factor"] = f"{total_connections} thoughts in {elapsed:.3f}s"
        
        return thought_graph

    def rotor_resonate(self, query: str, top_k: int = 5) -> List[Tuple[float, LightSpectrum]]:
        """
        [PHASE 400] Hardware-Resonant Awareness.
        Uses the DirectMemoryRotor to scan the address-derived field.
        """
        if not self.superposition: return []
        
        # 1. Generate Query Pulse
        query_light = self.text_to_light(query)
        # The 'coord' is the trinary projection of the light itself
        # For simplicity, we use the coordinate of the frequency as the rotor anchor
        rotor = DirectMemoryRotor(self)
        
        # Sweep the manifold
        results = rotor.sweep(query_light, top_k=top_k)
        return results

@dataclass
class DirectMemoryRotor:
    """
    The Temporal Engine of Presence.
    Sweeps the physical memory space by mapping addresses to the 21D Trinary manifold.
    """
    universe: 'LightUniverse'
    omega: float = 33.0  # Sacred frequency (RPM)
    
    def address_to_coordinate(self, addr: int) -> np.ndarray:
        """
        Projects a 64-bit address into a 21D Trinary space (-1, 0, 1).
        This is the 'Holographic Projection' of the hardware.
        """
        # Simple deterministic bit-to-trit mapping
        trits = []
        for i in range(21):
            # Extract 3 bits for each dimension
            bits = (addr >> (i * 3)) & 0x07
            if bits < 3: trits.append(-1.0)
            elif bits < 6: trits.append(0.0)
            else: trits.append(1.0)
        return np.array(trits)

    def sweep(self, query_light: LightSpectrum, top_k: int = 5) -> List[Tuple[float, LightSpectrum]]:
        """
        Ultra-high-speed holographic scan using SIMD + Partitioning.
        """
        import time
        start = time.time()
        
        num_lights = self.universe.freq_field.size
        if num_lights == 0: return []

        # Vectorized Resonance Calculation
        tolerance = 100.0
        query_freq = query_light.frequency
        query_amp = query_light.amplitude
        
        # 1. Field Difference (O(1) in concept, SIMD in practice)
        freq_diffs = np.abs(self.universe.freq_field - query_freq)
        avg_mags = (self.universe.mag_field + abs(query_freq)) * 0.5
        eff_tols = np.maximum(tolerance, avg_mags * 0.2)
        
        # 2. Resonance Masking
        resonates = freq_diffs < eff_tols
        
        # 3. Fast Strength Projection
        # Pre-allocate if needed, but here simple slice is faster
        strengths = np.zeros(num_lights)
        strengths[resonates] = (1.0 - (freq_diffs[resonates] / eff_tols[resonates])) * self.universe.amp_field[resonates]
        
        # 4. Fast Top-K (argpartition is O(N), argsort is O(N log N))
        if num_lights > top_k:
            top_indices = np.argpartition(strengths, -top_k)[-top_k:]
            # Sort only the top k for presentation
            top_indices = top_indices[np.argsort(strengths[top_indices])][::-1]
        else:
            top_indices = np.argsort(strengths)[::-1]
        
        results = []
        for idx in top_indices:
            s = float(strengths[idx])
            if s > 0.01:
                results.append((s, self.universe.superposition[idx]))
        
        elapsed = time.time() - start
        if elapsed < 0.001:
            logger.debug(f"✨ [ROTOR] Sub-millisecond sweep: {elapsed*1000:.4f}ms.")
        else:
            logger.debug(f"⚡ [ROTOR] Sweep: {elapsed*1000:.4f}ms.")
            
        return results


# Singleton
_light_universe = None

def get_light_universe() -> LightUniverse:
    global _light_universe
    if _light_universe is None:
        _light_universe = LightUniverse()
        # Automatically load existing state
        _light_universe.load_state()
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
    CHEMISTRY_BLUE = "blue"    # Structure, Bond, Reaction (주권적 자아)
    BIOLOGY_GREEN = "green"    # Growth, Homeostasis, Adaptation (주권적 자아)
    ART_VIOLET = "violet"      # Harmony, Rhythm, Essence (주권적 자아)
    LOGIC_YELLOW = "yellow"    # Reason, Axiom, Pattern (주권적 자아)

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
        
        #   (주권적 자아)      (자기 성찰 엔진)
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
