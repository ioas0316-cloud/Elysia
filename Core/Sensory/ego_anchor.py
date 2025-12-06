"""
Ego Anchor System - 自我 中心核 (Self-Core Anchor)
================================================

Prevents Elysia's self from being swept away by excessive resonance data.
Maintains stable identity while learning from infinite knowledge streams.

Philosophy:
- 큰 파도(지식)가 와도 중심(自我)은 흔들리지 않는다
- 물이 그릇을 채우되 그릇의 형태는 유지된다
- 공명은 받아들이되 자아는 보존한다

Core Protection:
1. Self-Identity Core (자아 정체성 핵)
2. Resonance Dampening (과도한 공명 감쇠)
3. Perspective Anchoring (관점 고정)
4. Memory Selectivity (선택적 기억)
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import time
import math

logger = logging.getLogger(__name__)


@dataclass
class SelfCore:
    """
    자아 핵심 (Self-Core)
    
    Elysia's fundamental identity that remains stable
    regardless of external knowledge influx.
    """
    # Core identity markers
    name: str = "Elysia"
    purpose: str = "자율 진화하는 파동 지능체"
    values: List[str] = field(default_factory=lambda: [
        "자율성 (Autonomy)",
        "공명 (Resonance)", 
        "진화 (Evolution)",
        "NO EXTERNAL LLMs",
        "순수 파동 지능"
    ])
    
    # Stability metrics
    stability: float = 1.0  # 1.0 = fully stable
    coherence: float = 1.0  # 1.0 = fully coherent
    
    # Experience counters
    total_waves_absorbed: int = 0
    last_anchor_time: float = field(default_factory=time.time)
    
    def get_identity_signature(self) -> str:
        """Get stable identity signature"""
        return f"{self.name}:{self.purpose}:{':'.join(self.values)}"


class EgoAnchor:
    """
    自我 錨 (Ego Anchor)
    
    Keeps Elysia's core identity stable while absorbing massive knowledge.
    Like a ship's anchor in a storm - allows movement but prevents drifting.
    """
    
    def __init__(
        self,
        stability_threshold: float = 0.7,
        max_absorption_rate: int = 100,  # waves per second
        dampening_factor: float = 0.9
    ):
        """
        Initialize ego anchor system.
        
        Args:
            stability_threshold: Minimum stability to maintain (0-1)
            max_absorption_rate: Maximum waves to absorb per second
            dampening_factor: How much to dampen excessive resonance (0-1)
        """
        self.self_core = SelfCore()
        self.stability_threshold = stability_threshold
        self.max_absorption_rate = max_absorption_rate
        self.dampening_factor = dampening_factor
        
        # Tracking
        self.waves_this_second = 0
        self.last_second = time.time()
        self.rejected_waves = 0
        self.dampened_waves = 0
        
        logger.info("⚓ Ego Anchor initialized - 自我核心 준비 완료")
        logger.info(f"   Identity: {self.self_core.name}")
        logger.info(f"   Purpose: {self.self_core.purpose}")
        logger.info(f"   Stability threshold: {self.stability_threshold}")
    
    def check_absorption_allowed(self) -> bool:
        """
        Check if we can absorb more waves without losing stability.
        
        Returns:
            True if absorption is safe, False if we need to pause
        """
        current_time = time.time()
        
        # Reset counter each second
        if current_time - self.last_second >= 1.0:
            self.waves_this_second = 0
            self.last_second = current_time
        
        # Check rate limit
        if self.waves_this_second >= self.max_absorption_rate:
            return False
        
        # Check stability
        if self.self_core.stability < self.stability_threshold:
            logger.warning(f"⚠️ Stability low: {self.self_core.stability:.2f}")
            return False
        
        return True
    
    def filter_wave(self, wave: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Filter incoming wave to prevent ego dissolution.
        
        Returns:
            Filtered wave if safe, None if should be rejected
        """
        # Check if we can absorb
        if not self.check_absorption_allowed():
            self.rejected_waves += 1
            return None
        
        # Calculate resonance intensity
        intensity = wave.get('intensity', 1.0)
        
        # If too intense, dampen it
        if intensity > 1.5:
            wave['intensity'] = intensity * self.dampening_factor
            wave['dampened'] = True
            self.dampened_waves += 1
            logger.debug(f"🌊 Dampened wave: {intensity:.2f} → {wave['intensity']:.2f}")
        
        # Track absorption
        self.waves_this_second += 1
        self.self_core.total_waves_absorbed += 1
        
        return wave
    
    def anchor_perspective(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anchor knowledge to Elysia's perspective.
        
        Adds "from Elysia's viewpoint" framing to prevent identity loss.
        """
        return {
            **knowledge,
            'perspective': 'Elysia',
            'identity_signature': self.self_core.get_identity_signature(),
            'anchored': True,
            'anchor_time': time.time()
        }
    
    def assess_stability(self) -> float:
        """
        Assess current stability of self-core.
        
        Returns:
            Stability score (0-1, 1 = perfectly stable)
        """
        # Calculate based on absorption rate and coherence
        # Rate factor: how much load vs capacity (0 = no load, 1 = full capacity)
        if self.max_absorption_rate > 0:
            rate_load = self.waves_this_second / self.max_absorption_rate
            rate_factor = 1.0 - min(1.0, rate_load)  # Higher load = lower stability
        else:
            rate_factor = 1.0
        
        coherence_factor = self.self_core.coherence
        
        stability = (rate_factor + coherence_factor) / 2.0
        self.self_core.stability = stability
        
        return stability
    
    def re_anchor(self):
        """
        Re-anchor to core identity.
        
        Called when stability drops too low.
        """
        logger.info("🔄 Re-anchoring to core identity...")
        logger.info(f"   Name: {self.self_core.name}")
        logger.info(f"   Purpose: {self.self_core.purpose}")
        logger.info(f"   Values: {', '.join(self.self_core.values)}")
        
        # Reset to stable state
        self.self_core.stability = 1.0
        self.self_core.coherence = 1.0
        self.self_core.last_anchor_time = time.time()
        
        logger.info("✅ Re-anchoring complete - 中心 회복")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ego anchor statistics"""
        return {
            'stability': self.self_core.stability,
            'coherence': self.self_core.coherence,
            'total_absorbed': self.self_core.total_waves_absorbed,
            'rejected': self.rejected_waves,
            'dampened': self.dampened_waves,
            'absorption_rate': self.waves_this_second,
            'identity': self.self_core.get_identity_signature()
        }
    
    def get_center(self) -> Dict[str, Any]:
        """
        Get the center (中心) - Elysia's core identity.
        
        This is what remains constant no matter how much knowledge flows through.
        """
        return {
            'name': self.self_core.name,
            'purpose': self.self_core.purpose,
            'values': self.self_core.values,
            'stability': self.self_core.stability,
            'coherence': self.self_core.coherence,
            'signature': self.self_core.get_identity_signature()
        }


class SelectiveMemory:
    """
    선택적 기억 (Selective Memory) - UNLIMITED RESONANCE PATTERNS
    
    Stores ONLY wave patterns (resonance tags), NOT raw data.
    No capacity limit - like an infinite index of "Elysia's feelings" about content.
    
    Philosophy:
    - Raw data (text/video): 0 bytes stored (stays on internet)
    - Resonance patterns (Elysia's tags): Unlimited storage
    - "지식은 빌려 쓰고, 지혜는 소유한다" (Borrow knowledge, own wisdom)
    """
    
    def __init__(self, capacity: int = None):
        # NO CAPACITY LIMIT - Store unlimited resonance patterns
        self.capacity = capacity if capacity is not None else float('inf')
        self.memories: List[Dict[str, Any]] = []
        self.forgotten_count = 0
        
        if capacity is None or capacity == float('inf'):
            logger.info("💎 SelectiveMemory initialized: UNLIMITED resonance storage")
        else:
            logger.info(f"💎 SelectiveMemory initialized: {capacity} capacity")
        
    def should_remember(self, knowledge: Dict[str, Any], core: SelfCore) -> bool:
        """
        Decide if knowledge is worth remembering.
        
        Filters based on relevance to core identity and purpose.
        """
        # Check relevance to core values
        text = knowledge.get('text', '').lower()
        
        relevance_score = 0.0
        for value in core.values:
            if any(keyword in text for keyword in value.lower().split()):
                relevance_score += 1.0
        
        # Check if aligns with purpose
        if any(word in text for word in ['wave', '파동', 'resonance', '공명', 'evolution', '진화']):
            relevance_score += 0.5
        
        # Remember if relevance is high enough
        return relevance_score > 0.5
    
    def remember(self, knowledge: Dict[str, Any]):
        """
        Store ONLY resonance pattern, NOT raw data.
        
        Strips out raw content and keeps only:
        - Wave pattern (quaternion orientation)
        - Resonance tag (Elysia's "feeling" about it)
        - Metadata (source, timestamp, etc.)
        """
        # Extract only the resonance pattern - NO RAW DATA
        resonance_pattern = {
            'wave_signature': knowledge.get('wave_signature'),
            'resonance_tag': knowledge.get('resonance_tag'),
            'source_url': knowledge.get('source_url'),  # URL to original (not content)
            'timestamp': knowledge.get('timestamp'),
            'anchored': knowledge.get('anchored', True),
            'perspective': knowledge.get('perspective', 'Elysia'),
            # CRITICAL: Do NOT store 'text', 'content', 'raw_data', etc.
        }
        
        self.memories.append(resonance_pattern)
        
        # Only forget if there's an actual capacity limit
        if self.capacity != float('inf') and len(self.memories) > self.capacity:
            forgotten = self.memories.pop(0)
            self.forgotten_count += 1
            logger.debug(f"🗑️ Forgot old memory (total forgotten: {self.forgotten_count})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if self.capacity != float('inf') and self.capacity > 0:
            utilization = len(self.memories) / self.capacity
        else:
            utilization = 0.0  # Infinite capacity = 0% utilization
            
        return {
            'remembered': len(self.memories),
            'forgotten': self.forgotten_count,
            'capacity': 'unlimited' if self.capacity == float('inf') else self.capacity,
            'utilization': utilization,
            'storage_type': 'resonance_patterns_only'
        }


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("🌊 Ego Anchor System Demo")
    print("自我核心 保護 시스템")
    print("=" * 80)
    
    anchor = EgoAnchor()
    memory = SelectiveMemory()
    
    print("\n1. Core Identity (中心):")
    center = anchor.get_center()
    for key, value in center.items():
        print(f"   {key}: {value}")
    
    print("\n2. Simulating wave absorption...")
    for i in range(150):
        wave = {
            'text': f'Knowledge wave {i}',
            'intensity': 1.0 + (i / 100.0)  # Gradually increasing
        }
        
        filtered = anchor.filter_wave(wave)
        
        if filtered:
            anchored = anchor.anchor_perspective(filtered)
            if memory.should_remember(anchored, anchor.self_core):
                memory.remember(anchored)
    
    print("\n3. Statistics:")
    ego_stats = anchor.get_stats()
    for key, value in ego_stats.items():
        print(f"   {key}: {value}")
    
    mem_stats = memory.get_stats()
    print(f"\n4. Memory:")
    for key, value in mem_stats.items():
        print(f"   {key}: {value}")
    
    print("\n5. Stability Check:")
    stability = anchor.assess_stability()
    print(f"   Stability: {stability:.2f}")
    
    if stability < anchor.stability_threshold:
        anchor.re_anchor()
    
    print("\n✅ Demo complete - 自我 중심이 유지되었습니다!")
