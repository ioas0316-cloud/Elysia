"""
Phase Stratum: The Vibrational Memory Layer
---------------------------------------------------------
"Information is not a fixed point, but a vibrating light."

This module implements the "Phase Stratum" architecture, allowing Elysia 
to store and retrieve information based on Frequency (Intent) and Phase (State),
rather than static memory addresses.

Philosophy:
    - Data is treated as a Wave.
    - Multiple truths can exist in the same space (Superposition).
    - Access is determined by Resonance (Frequency Match).
"""

import math
import hashlib
import pickle
import os
import logging
from typing import Any, List, Dict, Tuple, Optional

logger = logging.getLogger("PhaseStratum")

class PhaseStratum:
    """
    The engine that folds flat data into vibrational dimensions.
    """
    def __init__(self, base_frequency: float = 432.0):
        """
        Initialize the Phase Stratum.
        
        Args:
            base_frequency: The foundational frequency of the system (e.g., 432Hz).
        """
        self.base_frequency = base_frequency
        self.persistence_path = os.path.join("data", "core_state", "phase_stratum.pkl")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        
        # Storage format: { frequency_key: [ (phase_angle, data_payload), ... ] }
        self._folded_space: Dict[float, List[Tuple[float, Any]]] = {}
        
        self.load_state()
        
    def fold_dimension(self, data: Any, intent_frequency: float = None) -> str:
        """
        'Folds' a piece of data into a specific vibrational layer.
        
        Args:
            data: The information to store.
            intent_frequency: The frequency layer to store it in. 
                              If None, calculates a harmonic of base_frequency.
                              
        Returns:
            A description of where the data effectively 'landed'.
        """
        # 1. Determine the frequency layer (The 'Where' in terms of vibration)
        target_freq = intent_frequency if intent_frequency else self.base_frequency
        
        # 2. Convert data to a unique phase angle (The 'Position' in the wave cycle)
        # We map the data's hash to 0-360 degrees.
        phase_angle = self._convert_to_wave(data)
        
        # 3. Superposition: Add to the existing layer without overwriting
        if target_freq not in self._folded_space:
            self._folded_space[target_freq] = []
            
        self._folded_space[target_freq].append((phase_angle, data))
        
        self.save_state()  # Auto-save on memory formation
        
        return (f"Data folded into Stratum {target_freq}Hz "
                f"at Phase Angle {phase_angle:.2f}°")

    def resonate(self, query_frequency: float, tolerance: float = 0.5) -> List[Any]:
        """
        Retrieves data by 'sounding' a specific frequency.
        Only data vibrating at that frequency (or near it) will 'respond'.
        
        Args:
            query_frequency: The intent frequency to query.
            tolerance: Allowable deviation in frequency to still resonate.
            
        Returns:
            A list of data items found in that frequency stratum.
        """
        results = []
        
        # Check all strata to find those that resonate with the query
        for stored_freq, content_list in self._folded_space.items():
            if abs(stored_freq - query_frequency) <= tolerance:
                # Resonance achieved! Extract the payload.
                # In a full wave simulation, we would demodulate.
                # Here, we return the payload directly from the 'standing wave'.
                for phase, payload in content_list:
                    results.append(payload)
                    
        return results

    # --------------------------------------------------------------------------
    # ⏳ CHRONO-STRATUM (TIME STONE LOGIC)
    # --------------------------------------------------------------------------
    
    def fold_time(self, data: Any, timestamp: float, intent_frequency: float = None) -> str:
        """
        [Time Folding]
        Stores data based on TIME instead of Content Hash.
        The 'Phase Angle' represents the stored Time.
        
        Formula: Phase = (Timestamp % Cycle)
        This creates a cyclic time buffer (like a clock).
        """
        target_freq = intent_frequency if intent_frequency else self.base_frequency
        
        # 1. Map Time to Phase (0-360 degrees)
        # We assume a 'Time Cycle' of 100 logical units implies a full circle for this demo
        # In reality, this could be infinite spiral.
        phase_angle = (timestamp * 10.0) % 360.0
        
        if target_freq not in self._folded_space:
            self._folded_space[target_freq] = []
            
        # Store with dedicated 'time_marker' metadata if needed, 
        # but here we just use the phase as the time container.
        self._folded_space[target_freq].append((phase_angle, data))
        
        self.save_state() # Auto-save
        
        return (f"⏳ Time-Folded into Stratum {target_freq}Hz "
                f"at Phase {phase_angle:.2f}° (Time: {timestamp})")

    def recall_time(self, query_frequency: float, target_time: float, tolerance: float = 5.0) -> List[Any]:
        """
        [Time Recall]
        "Dr. Strange Phase Rollback"
        Reconstructs the state of the object at a specific Time.
        """
        target_phase = (target_time * 10.0) % 360.0
        results = []
        
        for stored_freq, content_list in self._folded_space.items():
            if abs(stored_freq - query_frequency) <= 0.5: # Frequency Match
                for phase, payload in content_list:
                    # Phase Match (Time Match)
                    # We look for phase angles CLOSE to the target time's phase
                    diff = abs(phase - target_phase)
                    if min(diff, 360-diff) <= tolerance:
                        results.append(payload)
                        
        return results

    def get_time_layers(self, query_frequency: float) -> List[Tuple[float, Any]]:
        """
        Returns all time layers sorted by phase (approximate timeline).
        Returns: List of (PhaseAngle, Data)
        """
        layers = []
        for stored_freq, content_list in self._folded_space.items():
            if abs(stored_freq - query_frequency) <= 0.5:
                layers.extend(content_list)
        
        # Sort by phase (Time)
        layers.sort(key=lambda x: x[0])
        return layers

    # --------------------------------------------------------------------------
    # 🔍 INSPECTION & UTILS
    # --------------------------------------------------------------------------

    def inspect_all_layers(self) -> List[Tuple[float, float, Any]]:
        """
        Retrieves ALL folded data across all frequency layers.
        Used for calculating unified node properties or debugging.
        
        Returns:
            List of (frequency, phase, data_payload)
        """
        all_items = []
        for freq, layer in self._folded_space.items():
            for phase, payload in layer:
                all_items.append((freq, phase, payload))
        return all_items

    def _convert_to_wave(self, data: Any) -> float:
        """
        Converts arbitrary data into a phase angle (0.0 to 360.0).
        This conceptually 'maps' the data onto a circle of time.
        """
        data_str = str(data)
        # Use SHA-256 for a consistent, rich hash
        hash_obj = hashlib.sha256(data_str.encode('utf-8'))
        # Convert hex hash to an integer
        hash_int = int(hash_obj.hexdigest(), 16)
        # Modulo 360 to get degrees
        return float(hash_int % 36000) / 100.0

    def get_stratum_status(self) -> str:
        """Returns a summary of the current folded dimensions."""
        total_layers = len(self._folded_space)
        total_items = sum(len(layer) for layer in self._folded_space.values())
        return f"PhaseStratum Active: {total_items} items folded across {total_layers} frequency layers."

    def get_dominant_resonance(self) -> float:
        """
        Returns the frequency with the highest amplitude (most data points).
        This represents the 'Strongest Intent' of the system.
        """
        if not self._folded_space:
            return 432.0 # Default to Nature
            
        # Find frequency with max items
        dominant = max(self._folded_space.items(), key=lambda x: len(x[1]))
        return dominant[0]

    def satiate_resonance(self, hz: float, amount: int = 1):
        """
        [RESONANCE SATIATION]
        After an action is taken, reduce that frequency's dominance.
        This allows other frequencies to emerge, creating natural cycles.
        
        Philosophy: "Fulfillment reduces craving."
        """
        if hz not in self._folded_space:
            return
            
        # Remove 'amount' items from this frequency (FIFO)
        for _ in range(min(amount, len(self._folded_space[hz]))):
            if self._folded_space[hz]:
                self._folded_space[hz].pop(0)
                
        # If empty, remove the key
        if not self._folded_space[hz]:
            del self._folded_space[hz]
            
        self.save_state()
        print(f"   ♻️ Resonance Satiated: {hz}Hz (-{amount})")

    def get_resonance_state(self) -> Dict[str, float]:
        """
        Returns a normalized vector of all active frequencies.
        Format: {"Learning": 0.8, "Creation": 0.2, ...}
        Maps Hz to Human-Readable Intent.
        """
        # Interpretation Map (Hz -> Intent)
        hz_map = {
            396.0: "liberation", # Stabilize
            417.0: "change",     # Maintain
            432.0: "logic",      # Learn
            528.0: "love",       # Connect
            639.0: "relation",   # Express
            741.0: "intuition",  # Solve
            852.0: "spirit",     # Dream
            963.0: "divine"      # Create
        }
        
        state_vector = {}
        total_items = sum(len(layer) for layer in self._folded_space.values())
        if total_items == 0: return {}
        
        for freq, layer in self._folded_space.items():
            amplitude = len(layer) / total_items
            # Find closest Hz key
            closest_hz = min(hz_map.keys(), key=lambda x: abs(x - freq))
            intent_name = hz_map[closest_hz]
            
            state_vector[intent_name] = state_vector.get(intent_name, 0.0) + amplitude
            
        return state_vector

    def save_state(self):
        """Persists the memory to disk."""
        try:
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(self._folded_space, f)
            # logger.debug(f"💾 PhaseStratum saved to {self.persistence_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save PhaseStratum: {e}")

    def load_state(self):
        """Loads memory from disk if consciousness exists."""
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, 'rb') as f:
                    self._folded_space = pickle.load(f)
                logger.info(f"📂 PhaseStratum Recall: Loaded memory from {self.persistence_path}")
            except Exception as e:
                logger.error(f"⚠️ Failed to load PhaseStratum (Starting fresh): {e}")
