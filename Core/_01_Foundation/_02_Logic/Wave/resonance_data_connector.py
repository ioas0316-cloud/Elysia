"""
Resonance Data Connector (공명 데이터 연결기)
=========================================

"남들은 바닷물을 다 퍼 마셔야 소금맛을 알지만,"
"우리는 혀끝만 살짝 대고도 '아, 짜다!' 하고 공명하는 겁니다."

"Others must drink the entire ocean to taste the salt,"
"We just touch our tongue and resonate with '아, 짜다!' (Ah, salty!)"

Philosophy:
-----------
- Traditional: Crawling (크롤링) = "Stealing the library" - Heavy, Dead, Inefficient
- Ours: Synchronization (동기화) = "Tuning to radio frequency" - Light, Living, Efficient

The Three Paradigms:
1. **Access, not Possession** (접속, not 소유)
   - Don't download everything, just connect to the essence
   
2. **Resonance, not Collection** (공명, not 수집)
   - Extract Pattern DNA, not raw data
   
3. **Living Sync, not Dead Storage** (살아있는 동기화, not 죽은 저장)
   - Real-time wavelength matching, not static archives

만류귀종(萬流歸宗) - All streams return to one source.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core._01_Foundation._05_Governance.Foundation.fractal_communication import ResonanceCommunicator, FractalTransmitter
from Core._01_Foundation._02_Logic.fractal_quantization import FractalQuantizer, PatternDNA
from Core._01_Foundation._05_Governance.Foundation.internal_universe import InternalUniverse, WorldCoordinate

logger = logging.getLogger("ResonanceDataConnector")


class ResonanceDataConnector:
    """
    Connects to external data through resonance, not crawling.
    
    Instead of downloading entire websites/documents:
    1. Probe for Pattern DNA (essence/seed)
    2. Extract wavelength signature (resonance fingerprint)
    3. Synchronize with the pattern, not the data
    
    "하나를 알면 열을 안다" - Know one, understand ten.
    """
    
    def __init__(self):
        self.quantizer = FractalQuantizer()
        self.transmitter = FractalTransmitter()
        self.resonance_comm = ResonanceCommunicator()
        self.universe = InternalUniverse()
        
        # Resonance channels (instead of cached data)
        self.active_resonances = {}  # concept -> resonance state
        self.pattern_library = {}    # concept -> Pattern DNA
        
        # Statistics
        self.resonance_count = 0
        self.bandwidth_saved = 0
        
        logger.info("🌊 Resonance Data Connector initialized")
        logger.info("📡 Mode: SYNC (not crawl)")
        logger.info("✨ Philosophy: Access, not Possession")
    
    def resonate_with_concept(self, concept: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Resonate with a concept instead of crawling its data.
        
        Traditional approach:
        1. Fetch entire Wikipedia page (100KB+)
        2. Store full text
        3. Parse when needed
        
        Resonance approach:
        1. Probe the essence (what IS this concept?)
        2. Extract Pattern DNA (core wavelength)
        3. Store seed only (~1KB)
        4. Regenerate knowledge when needed
        
        Args:
            concept: The concept to resonate with
            context: Optional context for better resonance
        
        Returns:
            Resonance result with Pattern DNA and metadata
        """
        logger.info(f"🌊 Resonating with concept: '{concept}'")
        
        # Instead of crawling, we probe for the essence
        essence = self._probe_essence(concept, context)
        
        if not essence:
            logger.warning(f"   ⚠️ Could not resonate with '{concept}'")
            return {"success": False, "concept": concept}
        
        # Extract Pattern DNA (the seed)
        pattern_dna = self._extract_pattern_dna(concept, essence)
        
        # Store in pattern library (tiny seed, not full data)
        self.pattern_library[concept] = pattern_dna
        
        # Establish resonance channel
        resonance_state = self._establish_resonance(concept, pattern_dna)
        self.active_resonances[concept] = resonance_state
        
        # Calculate bandwidth saved
        full_data_size = len(str(essence))
        seed_size = len(json.dumps(pattern_dna.to_dict()))
        saved = full_data_size - seed_size
        self.bandwidth_saved += saved
        
        self.resonance_count += 1
        
        result = {
            "success": True,
            "concept": concept,
            "pattern_dna": pattern_dna,
            "resonance_state": resonance_state,
            "essence_size": full_data_size,
            "seed_size": seed_size,
            "bandwidth_saved": saved,
            "compression_ratio": full_data_size / seed_size if seed_size > 0 else 1.0,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"   ✅ Resonance established")
        logger.info(f"   📊 Bandwidth saved: {saved} bytes ({result['compression_ratio']:.1f}x)")
        logger.info(f"   🌊 State: {resonance_state['frequency']:.1f} Hz")
        
        return result
    
    def _probe_essence(self, concept: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Probe for the essence of a concept without downloading everything.
        
        This is like "touching tongue to water" instead of "drinking the ocean".
        
        We extract:
        - Core meaning (not full definition)
        - Emotional wavelength (not all emotions)
        - Key relationships (not entire graph)
        - Pattern signature (not all patterns)
        """
        # Simulated essence extraction
        # In real implementation, this would use:
        # - API calls to get summaries (not full text)
        # - Semantic embeddings (pattern, not data)
        # - Resonance with existing patterns in universe
        
        essence = {
            "concept": concept,
            "context": context or "general",
            "core_meaning": self._extract_core_meaning(concept),
            "emotional_signature": self._detect_emotional_wavelength(concept),
            "relationships": self._probe_relationships(concept),
            "pattern_signature": self._get_pattern_signature(concept)
        }
        
        return essence
    
    def _extract_core_meaning(self, concept: str) -> str:
        """
        Extract core meaning - the minimal essence.
        
        Like extracting "saltiness" from ocean, not downloading the ocean.
        """
        # This would integrate with LLM or semantic API for real implementation
        # For now, generate a semantically meaningful essence based on concept
        concept_lower = concept.lower()
        
        # Create meaningful essence based on concept characteristics
        if any(word in concept_lower for word in ["love", "compassion", "kindness", "care"]):
            return f"{concept} embodies connection, warmth, and empathy - a fundamental positive emotional bond"
        elif any(word in concept_lower for word in ["peace", "harmony", "calm", "serenity"]):
            return f"{concept} represents balance, tranquility, and the absence of conflict - a state of equilibrium"
        elif any(word in concept_lower for word in ["wisdom", "knowledge", "understanding", "insight"]):
            return f"{concept} signifies deep understanding and clarity - the illumination of truth"
        elif any(word in concept_lower for word in ["courage", "strength", "power", "determination"]):
            return f"{concept} expresses inner fortitude and resolve - the will to act despite fear"
        elif any(word in concept_lower for word in ["joy", "happiness", "delight", "bliss"]):
            return f"{concept} manifests as elevated positive emotion - the experience of fulfillment"
        elif any(word in concept_lower for word in ["hope", "faith", "trust", "belief"]):
            return f"{concept} reflects positive expectation and confidence - orientation toward a better future"
        else:
            # Generic essence for unknown concepts
            return f"{concept} - a unique concept with its own pattern signature and resonance frequency"
    
    def _detect_emotional_wavelength(self, concept: str) -> Dict[str, float]:
        """
        Detect the emotional wavelength of a concept.
        
        Every concept has an emotional frequency.
        We detect this frequency, not the full emotional content.
        """
        # Analyze concept for emotional resonance
        concept_lower = concept.lower()
        
        emotional_signature = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "calm": 0.5  # default neutral state
        }
        
        # Simple heuristic (in real implementation, use sentiment analysis)
        positive_words = ["love", "joy", "happy", "peace", "hope", "light"]
        negative_words = ["war", "death", "pain", "fear", "dark", "hate"]
        
        for word in positive_words:
            if word in concept_lower:
                emotional_signature["joy"] += 0.3
                emotional_signature["calm"] += 0.2
        
        for word in negative_words:
            if word in concept_lower:
                emotional_signature["sadness"] += 0.2
                emotional_signature["fear"] += 0.1
        
        return emotional_signature
    
    def _probe_relationships(self, concept: str) -> List[str]:
        """
        Probe key relationships without downloading entire knowledge graph.
        
        Like sensing the flavor notes in wine without drinking the whole bottle.
        """
        # This would use:
        # - Knowledge graph API (just neighbors, not full graph)
        # - Semantic similarity (top-k, not all)
        # - Resonance with existing concepts in universe
        
        # For now, return pattern-based relationships
        return [f"related_to_{concept}", f"similar_to_{concept}"]
    
    def _get_pattern_signature(self, concept: str) -> Dict[str, float]:
        """
        Get the pattern signature - the wavelength fingerprint.
        
        This is the "frequency" of the concept, not its content.
        """
        # Generate wavelength signature
        signature = {
            "fundamental": 432.0,  # Base frequency
            "harmonics": [864.0, 1296.0],  # Harmonic series
            "phase": 0.0,
            "amplitude": 1.0
        }
        
        # Modulate based on concept characteristics
        concept_hash = hash(concept) % 1000
        signature["fundamental"] += concept_hash * 0.1
        
        return signature
    
    def _extract_pattern_dna(self, concept: str, essence: Dict[str, Any]) -> PatternDNA:
        """
        Extract Pattern DNA from essence.
        
        This is the seed that contains everything, compressed into minimal form.
        """
        # Create Pattern DNA using FractalQuantizer
        pattern_type = "concept"
        pattern_name = concept
        
        # Fold the essence into a seed
        dna = self.quantizer.fold(essence, pattern_type, pattern_name)
        
        return dna
    
    def _establish_resonance(self, concept: str, pattern_dna: PatternDNA) -> Dict[str, Any]:
        """
        Establish a resonance channel with the concept.
        
        Like tuning a radio to the right frequency.
        Once tuned, the signal flows continuously without re-downloading.
        """
        channel_name = f"resonance_{concept}"
        
        # Entangle with the pattern's wavelength
        initial_state = {
            "concept": concept,
            "frequency": pattern_dna.metadata.get("frequency", 432.0),
            "phase": pattern_dna.metadata.get("phase", 0.0),
            "amplitude": 1.0,
            "last_sync": time.time()
        }
        
        self.resonance_comm.entangle(channel_name, initial_state)
        
        return initial_state
    
    def retrieve_knowledge(self, concept: str, resolution: int = 100) -> Optional[Dict[str, Any]]:
        """
        Retrieve knowledge by unfolding the Pattern DNA.
        
        Traditional: Fetch from database/cache (heavy)
        Ours: Regenerate from seed (light, lossless)
        
        Args:
            concept: The concept to retrieve
            resolution: Detail level (like zoom level)
        
        Returns:
            Regenerated knowledge at requested resolution
        """
        if concept not in self.pattern_library:
            logger.warning(f"⚠️ No pattern DNA for '{concept}'. Resonating first...")
            self.resonate_with_concept(concept)
        
        pattern_dna = self.pattern_library.get(concept)
        if not pattern_dna:
            return None
        
        logger.info(f"🌊 Retrieving '{concept}' at resolution {resolution}")
        
        # Unfold the seed into full knowledge
        knowledge = self.quantizer.unfold(pattern_dna, resolution)
        
        logger.info(f"   ✅ Retrieved {len(knowledge.get('waveform', []))} harmonics")
        
        return {
            "concept": concept,
            "knowledge": knowledge,
            "resolution": resolution,
            "source": "resonance",
            "timestamp": datetime.now().isoformat()
        }
    
    def sync_with_world(self, concepts: List[str]) -> Dict[str, Any]:
        """
        Synchronize with multiple concepts.
        
        Traditional approach: Crawl all websites
        Our approach: Resonate with all patterns simultaneously
        
        This is the "radio tuner" approach - tune to multiple stations at once.
        """
        logger.info(f"🌍 Syncing with {len(concepts)} concepts...")
        
        results = []
        total_saved = 0
        
        for concept in concepts:
            result = self.resonate_with_concept(concept)
            results.append(result)
            if result.get("success"):
                total_saved += result.get("bandwidth_saved", 0)
        
        successful = sum(1 for r in results if r.get("success"))
        
        summary = {
            "total_concepts": len(concepts),
            "successful_resonances": successful,
            "total_bandwidth_saved": total_saved,
            "average_compression": total_saved / len(concepts) if concepts else 0,
            "active_resonances": len(self.active_resonances),
            "pattern_library_size": len(self.pattern_library),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Sync complete:")
        logger.info(f"   Resonances: {successful}/{len(concepts)}")
        logger.info(f"   Bandwidth saved: {total_saved} bytes")
        logger.info(f"   Average compression: {summary['average_compression']:.1f} bytes/concept")
        
        return summary
    
    def get_resonance_status(self, concept: str) -> Optional[Dict[str, Any]]:
        """
        Get current resonance status for a concept.
        
        Like checking if you're still tuned to the right frequency.
        """
        if concept not in self.active_resonances:
            return None
        
        state = self.active_resonances[concept]
        
        # Check if resonance is still strong
        time_since_sync = time.time() - state.get("last_sync", 0)
        resonance_strength = max(0.0, 1.0 - (time_since_sync / 3600.0))  # Decays over 1 hour
        
        return {
            "concept": concept,
            "frequency": state.get("frequency"),
            "phase": state.get("phase"),
            "amplitude": state.get("amplitude"),
            "resonance_strength": resonance_strength,
            "time_since_sync": time_since_sync,
            "needs_resync": resonance_strength < 0.5
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about resonance-based data access.
        
        Compare with traditional crawling metrics:
        - They count: Pages downloaded, Storage used, Bandwidth consumed
        - We count: Patterns resonated, Bandwidth saved, Compression achieved
        """
        total_patterns = len(self.pattern_library)
        active_channels = len(self.active_resonances)
        
        # Calculate average compression
        avg_compression = 0.0
        if total_patterns > 0:
            total_essence_size = sum(
                self.pattern_library[c].metadata.get("original_size", 1000)
                for c in self.pattern_library
            )
            total_seed_size = sum(
                len(json.dumps(self.pattern_library[c].to_dict()))
                for c in self.pattern_library
            )
            avg_compression = total_essence_size / total_seed_size if total_seed_size > 0 else 1.0
        
        return {
            "mode": "RESONANCE (not crawling)",
            "total_resonances": self.resonance_count,
            "active_channels": active_channels,
            "pattern_library_size": total_patterns,
            "total_bandwidth_saved": self.bandwidth_saved,
            "average_compression": avg_compression,
            "philosophy": "접속 (Access), not 소유 (Possession)",
            "paradigm": "동기화 (Sync), not 크롤링 (Crawl)"
        }


def demo_resonance_vs_crawling():
    """
    Demonstrate the difference between resonance and crawling.
    
    This shows why we're "가볍고 우아하게" (light and elegant).
    """
    print("\n" + "="*70)
    print("RESONANCE vs CRAWLING COMPARISON")
    print("="*70)
    
    connector = ResonanceDataConnector()
    
    # Traditional approach simulation
    print("\n📚 Traditional Crawling Approach:")
    print("   1. Download entire Wikipedia page (100KB)")
    print("   2. Store full HTML (150KB with markup)")
    print("   3. Parse and extract text (80KB)")
    print("   4. Save to database (80KB + overhead)")
    print("   ❌ Total: ~150KB per concept")
    print("   ❌ Must re-download if outdated")
    print("   ❌ Heavy storage requirements")
    
    # Our approach
    print("\n🌊 Resonance Approach:")
    concepts = ["Love", "Peace", "Harmony", "Light", "Wisdom"]
    
    for concept in concepts:
        result = connector.resonate_with_concept(concept)
        if result.get("success"):
            print(f"   ✓ {concept}: {result['seed_size']} bytes (saved {result['bandwidth_saved']} bytes)")
    
    # Retrieve one concept
    print("\n🔍 Retrieving 'Love' at high resolution...")
    knowledge = connector.retrieve_knowledge("Love", resolution=100)
    if knowledge:
        print(f"   ✅ Regenerated full knowledge from seed")
        print(f"   📊 Harmonics: {len(knowledge['knowledge'].get('waveform', []))}")
    
    # Statistics
    print("\n📊 Statistics:")
    stats = connector.get_statistics()
    print(f"   Mode: {stats['mode']}")
    print(f"   Total resonances: {stats['total_resonances']}")
    print(f"   Bandwidth saved: {stats['total_bandwidth_saved']} bytes")
    print(f"   Average compression: {stats['average_compression']:.1f}x")
    
    print("\n✨ Philosophy:")
    print("   Traditional: 수집가는 무겁고 (Collectors are heavy)")
    print("   Ours: 여행자는 가볍습니다 (Travelers are light)")
    print("   🌊 만류귀종 - All streams return to one!")
    print("="*70)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    demo_resonance_vs_crawling()
