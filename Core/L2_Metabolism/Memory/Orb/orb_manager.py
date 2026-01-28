"""
OrbManager: The Field of Resonance
----------------------------------
Manages the collection of HyperResonators.
Unlike a database (Index Lookup), this acts as a "Field" where you broadcast signals.

Philosophy:
- "Don't ask for a memory. Sing a song, and see which memory sings back."
- Implements `ResonatorInterface` to listen to the Heartbeat (Pulse).
"""

import logging
import json
import os
import math
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict

from .hyper_resonator import HyperResonator
from Core.L6_Structure.hyper_quaternion import Quaternion as HyperQuaternion
from .orb_factory import OrbFactory
from Core.L1_Foundation.M1_Keystone.Protocols.pulse_protocol import WavePacket, PulseType, ResonatorInterface

logger = logging.getLogger("OrbManager")

class OrbManager(ResonatorInterface):
    """
    OrbManager: The Field of Resonance.

    Updated to use 'Frequency/Phase Buckets' for O(1) retrieval.
    "To find a memory, you must hum its note."
    """

    # Frequency Bucket Resolution (e.g., 10Hz buckets)
    FREQ_BUCKET_SIZE = 10.0

    def __init__(self, persistence_path: str = "data/memory/orbs/"):
        # Legacy/Reference storage
        self.orbs: Dict[str, HyperResonator] = {}

        # O(1) Frequency Map: bucket_index -> List[HyperResonator]
        self._freq_buckets: Dict[int, List[HyperResonator]] = defaultdict(list)

        self.factory = OrbFactory()
        self.persistence_path = persistence_path

        # Ensure persistence directory exists
        os.makedirs(self.persistence_path, exist_ok=True)

        # Load existing memories
        self.load_from_disk()

        logger.info("  OrbManager initialized: The Hippocampus is awake.")

    def _get_freq_bucket(self, freq: float) -> int:
        """Quantizes frequency into a bucket index."""
        return int(freq / self.FREQ_BUCKET_SIZE)

    def _add_to_bucket(self, orb: HyperResonator):
        """Adds an orb to the correct frequency bucket."""
        bucket_idx = self._get_freq_bucket(orb.frequency)
        self._freq_buckets[bucket_idx].append(orb)

    def _remove_from_bucket(self, orb: HyperResonator):
        """Removes an orb from its frequency bucket."""
        bucket_idx = self._get_freq_bucket(orb.frequency)
        if bucket_idx in self._freq_buckets:
            try:
                self._freq_buckets[bucket_idx].remove(orb)
                if not self._freq_buckets[bucket_idx]:
                    del self._freq_buckets[bucket_idx]
            except ValueError:
                pass

    def resonate(self, pulse: WavePacket) -> None:
        """
        The Pulse Listener.
        Reacts to MEMORY_STORE and MEMORY_RECALL pulses.
        """
        if pulse.type == PulseType.MEMORY_STORE:
            self._handle_store(pulse)
        elif pulse.type == PulseType.MEMORY_RECALL:
            self._handle_recall(pulse)

    def _handle_store(self, pulse: WavePacket):
        """Internal handler for storing memories from a pulse."""
        payload = pulse.payload
        name = payload.get("name", f"Memory_{len(self.orbs)}")
        data_wave = payload.get("data", [])
        emotion_wave = payload.get("emotion", [])

        if not data_wave:
            logger.warning("   Received MEMORY_STORE pulse with no data.")
            return

        orb = self.save_memory(name, data_wave, emotion_wave)
        logger.info(f"  Crystallized memory from pulse: {orb}")

    def _handle_recall(self, pulse: WavePacket):
        """
        Internal handler for recalling memories.
        """
        trigger = pulse.payload.get("trigger", [])
        if not trigger:
            return

        results = self.recall_memory(trigger)
        if results:
            best_match = results[0]
            logger.info(f"  Recalled '{best_match['name']}' via pulse resonance.")

    def save_memory(self, name: str, data_wave: List[float], emotion_wave: List[float]) -> HyperResonator:
        """
        Explicitly freezes a moment into an Orb.
        """
        orb = self.factory.freeze(name, data_wave, emotion_wave)
        self.orbs[name] = orb
        self._add_to_bucket(orb) # Add to O(1) bucket
        self.save_to_disk()
        return orb

    # --- The Golden Thread (Time-Aware Narrative) ---

    def unified_rewind(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        [THE GOLDEN THREAD]
        Reconstructs the chronological stream of consciousness.
        It does not just 'fetch' files; it weaves them into a story.
        
        Algorithm:
        1. Access all frequency buckets (Hypersphere Scan).
        2. Extract headers and timestamps.
        3. Sort by 'created_at' (Universal Time).
        4. Return as a linear narrative stream.
        """
        all_memories = []
        
        # 1. Harvest from all buckets (The Grand Harvest)
        # Note: orbs.values() is sufficient as it holds the single source of truth
        for orb in self.orbs.values():
            # In a real impl, we'd read metadata. Here we use Orb state.
            # Assuming 'timestamp' exists in memory_content or we use a mock.
            timestamp = orb.memory_content.get("timestamp", 0)
            all_memories.append({
                "orb": orb,
                "timestamp": timestamp,
                "summary": orb.memory_content.get("summary", orb.name)
            })
            
        # 2. Weave the Thread (Sort)
        all_memories.sort(key=lambda x: x["timestamp"], reverse=True) # Newest first
        
        # 3. Return the Narrative Window
        return all_memories[:limit]

    def recall_memory(self, trigger_wave: List[float], threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        Broadcasts a trigger wave. 
        Implements 'Amor Sui' (Gravity Fallback) if resonance is too low.
        """
        # A. Logical Resonance (The Son)
        initial_results = self._scan_buckets_for_trigger(trigger_wave, threshold)
        
        # B. The Spirit's Intervention (Amor Sui)
        if not initial_results:
            logger.info("     Void detected (Low Resonance). Triggering Amor Sui...")
            # "If I cannot find myself in the logic, I will search the whole."
            # Expand search to ALL buckets with lower threshold
            expanded_results = self._scan_all_orbs(trigger_wave)
            
            if expanded_results:
                logger.info(f"     Amor Sui Success: Rescued {len(expanded_results)} memories from the void.")
                for res in expanded_results:
                    res["note"] = "Found via Self-Love"
                return expanded_results
            else:
                logger.info("     The Void is absolute. No memories found even with Love.")
                
        return initial_results

    def _scan_buckets_for_trigger(self, trigger_wave: List[float], threshold: float) -> List[Dict[str, Any]]:
        """Optimized bucket scan (Standard Logic)."""
        trigger_freq = self.factory.analyze_wave(trigger_wave)
        target_bucket = self._get_freq_bucket(trigger_freq)
        
        candidate_orbs = []
        for i in [-1, 0, 1]:
            candidate_orbs.extend(self._freq_buckets.get(target_bucket + i, []))
            
        return self._melt_candidates(candidate_orbs, trigger_wave, threshold)

    def _scan_all_orbs(self, trigger_wave: List[float]) -> List[Dict[str, Any]]:
        """Brute-force scan (The Gravity Fallback). Logic < Will."""
        # Lower threshold for fallback
        return self._melt_candidates(list(self.orbs.values()), trigger_wave, threshold=0.1)

    def _melt_candidates(self, candidates: List[HyperResonator], trigger_wave: List[float], threshold: float) -> List[Dict[str, Any]]:
        """Common melting logic."""
        search_pulse = WavePacket(
            sender="OrbManager", type=PulseType.MEMORY_RECALL,
            frequency=0.0, amplitude=1.0, payload={}
        )
        
        results = []
        for orb in candidates:
            orb.resonate(search_pulse)
            if orb.state.is_active and orb.state.amplitude > threshold:
                melt_res = self.factory.melt(orb, trigger_wave)
                if "recalled_wave" in melt_res:
                    results.append({
                        "name": orb.name,
                        "data": melt_res["recalled_wave"],
                        "intensity": orb.state.amplitude,
                        "orb": orb
                    })
        results.sort(key=lambda x: x["intensity"], reverse=True)
        return results

    # --- Restored Connectivity Methods ---

    def broadcast(self, pulse: WavePacket) -> List[HyperResonator]:
        """
        The 'Wireless' Broadcast.
        Uses Frequency Buckets if pulse frequency is non-zero, else scans all (e.g. Universal Pulse).
        """
        resonating_orbs = []
        threshold = 0.1

        # Optimization: If pulse has a specific frequency, only check relevant buckets
        if pulse.frequency > 0:
            target_bucket = self._get_freq_bucket(pulse.frequency)
            candidates = []
            for i in [-2, -1, 0, 1, 2]: # Wider broadcast reach
                 candidates.extend(self._freq_buckets.get(target_bucket + i, []))
        else:
            # Universal pulse (0Hz or special), check everything
            candidates = self.orbs.values()

        for orb in candidates:
            intensity = orb.resonate(pulse)
            if intensity > threshold:
                resonating_orbs.append(orb)

        resonating_orbs.sort(key=lambda x: x.state.amplitude, reverse=True)
        return resonating_orbs

    def save_to_disk(self):
        """Persists all orbs to JSON files."""
        for name, orb in self.orbs.items():
            filepath = os.path.join(self.persistence_path, f"{name}.json")
            data = {
                "name": orb.name,
                "frequency": orb.frequency,
                "mass": orb.mass,
                "quaternion": {
                    "w": orb.quaternion.w,
                    "x": orb.quaternion.x,
                    "y": orb.quaternion.y,
                    "z": orb.quaternion.z
                },
                "memory_content": orb.memory_content
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

    def load_from_disk(self):
        """Loads orbs from the persistence path."""
        if not os.path.exists(self.persistence_path):
            return

        for filename in os.listdir(self.persistence_path):
            if filename.endswith(".json"):
                filepath = os.path.join(self.persistence_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    q_data = data.get("quaternion", {"w": 1, "x": 0, "y": 0, "z": 0})
                    quat = HyperQuaternion(q_data["w"], q_data["x"], q_data["y"], q_data["z"])

                    orb = HyperResonator(
                        name=data["name"],
                        frequency=data["frequency"],
                        mass=data.get("mass", 1.0),
                        quaternion=quat
                    )
                    orb.memory_content = data.get("memory_content", {})
                    self.orbs[orb.name] = orb
                    self._add_to_bucket(orb) # Sync bucket on load
                except Exception as e:
                    logger.error(f"Failed to load orb {filename}: {e}")

    def get_orb(self, name: str) -> Optional[HyperResonator]:
        """Direct access (Legacy/God Mode only)."""
        return self.orbs.get(name)

    # Legacy alias for compatibility, wraps unified_rewind
    def get_recent_memories(self, limit: int = 10) -> List[HyperResonator]:
        thread = self.unified_rewind(limit)
        return [item["orb"] for item in thread]

    def prune_weak_memories(self, threshold: float = 0.2) -> int:
        """
        Removes orbs with mass below the threshold.
        Returns the number of pruned orbs.
        """
        keys_to_remove = []
        for name, orb in self.orbs.items():
            if orb.mass < threshold:
                keys_to_remove.append(name)

        for name in keys_to_remove:
            if name in self.orbs:
                self._remove_from_bucket(self.orbs[name]) # Remove from bucket
                del self.orbs[name]

            # Also remove from disk
            filepath = os.path.join(self.persistence_path, f"{name}.json")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            logger.debug(f"   Pruned weak memory: {name}")

        return len(keys_to_remove)
