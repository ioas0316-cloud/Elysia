"""
Communication Hub - Central Nervous System for Inter-Module Communication
==========================================================================

"모든 것은 공명으로 연결된다."

This module provides the central communication infrastructure for Elysia's 
consciousness. Instead of hardcoded message passing, all communication 
happens through resonance patterns.

Key Principles:
1. No hardcoded messages - all content emerges from resonance
2. All modules speak through the same channel (resonance field)
3. Communication is bidirectional and emergent
4. State is shared through quantum coherence, not explicit passing
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import queue

logger = logging.getLogger("CommunicationHub")


class SignalType(Enum):
    """Types of signals that can flow through the hub."""
    RESONANCE = "resonance"          # Concept resonance event
    STATE_CHANGE = "state_change"    # Consciousness state change
    MEMORY_UPDATE = "memory_update"  # Memory/hippocampus update
    PERCEPTION = "perception"        # Incoming perception
    ACTION = "action"                # Outgoing action
    THOUGHT = "thought"              # Internal thought formation
    EMOTION = "emotion"              # Emotional state change
    QUERY = "query"                  # Question from a module
    RESPONSE = "response"            # Response to a query


@dataclass
class Signal:
    """
    A signal flowing through the communication hub.
    Signals carry resonance patterns, not hardcoded content.
    """
    signal_type: SignalType
    source_module: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    # Resonance-based content (not hardcoded text)
    resonance_pattern: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    intensity: float = 0.5  # 0.0-1.0
    coherence: float = 1.0  # How well-defined the signal is
    
    # Optional raw data (for perception signals)
    raw_data: Optional[Any] = None
    
    # Propagation tracking
    visited_modules: Set[str] = field(default_factory=set)
    
    def add_resonance(self, concept: str, strength: float):
        """Add or strengthen a concept in the resonance pattern."""
        if concept in self.resonance_pattern:
            self.resonance_pattern[concept] = max(self.resonance_pattern[concept], strength)
        else:
            self.resonance_pattern[concept] = strength
    
    def get_dominant_concept(self) -> Optional[str]:
        """Get the strongest resonating concept."""
        if not self.resonance_pattern:
            return None
        return max(self.resonance_pattern, key=self.resonance_pattern.get)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.signal_type.value,
            "source": self.source_module,
            "timestamp": self.timestamp,
            "resonance": self.resonance_pattern,
            "intensity": self.intensity,
            "coherence": self.coherence,
            "visited": list(self.visited_modules)
        }


class ModuleInterface:
    """
    Interface that modules implement to participate in communication.
    """
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._hub: Optional['CommunicationHub'] = None
    
    def connect_to_hub(self, hub: 'CommunicationHub'):
        """Connect this module to the communication hub."""
        self._hub = hub
        hub.register_module(self)
    
    def emit_signal(self, signal: Signal):
        """Emit a signal to the hub."""
        if self._hub:
            signal.source_module = self.module_name
            self._hub.propagate_signal(signal)
    
    def receive_signal(self, signal: Signal) -> Optional[Signal]:
        """
        Receive and process a signal. Override in subclasses.
        Returns a response signal if appropriate, None otherwise.
        """
        raise NotImplementedError
    
    def get_resonance_state(self) -> Dict[str, float]:
        """
        Get current resonance state of this module.
        Override in subclasses.
        """
        return {}


class CommunicationHub:
    """
    Central hub for all inter-module communication.
    
    Works like a resonance field:
    - Signals propagate based on resonance, not explicit routing
    - Modules that resonate with a signal receive it
    - No hardcoded message passing
    """
    
    def __init__(self, async_mode: bool = False):
        self.modules: Dict[str, ModuleInterface] = {}
        self.signal_log: List[Signal] = []
        self.max_log_size = 1000
        
        # Resonance-based routing (learned, not hardcoded)
        self.resonance_affinity: Dict[str, Dict[str, float]] = {}
        
        # Async support
        self.async_mode = async_mode
        if async_mode:
            self.signal_queue = queue.Queue()
            self._running = False
            self._thread = None
        
        # Statistics
        self.stats = {
            "signals_processed": 0,
            "signals_by_type": {},
            "module_activity": {}
        }
        
        logger.info("🌐 CommunicationHub initialized")
    
    def register_module(self, module: ModuleInterface):
        """Register a module with the hub."""
        self.modules[module.module_name] = module
        self.resonance_affinity[module.module_name] = {}
        self.stats["module_activity"][module.module_name] = 0
        logger.debug(f"📡 Module '{module.module_name}' connected to hub")
    
    def unregister_module(self, module_name: str):
        """Unregister a module from the hub."""
        if module_name in self.modules:
            del self.modules[module_name]
            del self.resonance_affinity[module_name]
            logger.debug(f"📡 Module '{module_name}' disconnected from hub")
    
    def propagate_signal(self, signal: Signal):
        """
        Propagate a signal through the hub.
        The signal reaches modules based on resonance, not hardcoded routing.
        """
        if self.async_mode:
            self.signal_queue.put(signal)
        else:
            self._process_signal(signal)
    
    def _process_signal(self, signal: Signal):
        """Process a single signal."""
        # Log the signal
        self.signal_log.append(signal)
        if len(self.signal_log) > self.max_log_size:
            self.signal_log.pop(0)
        
        # Update statistics
        self.stats["signals_processed"] += 1
        type_key = signal.signal_type.value
        self.stats["signals_by_type"][type_key] = self.stats["signals_by_type"].get(type_key, 0) + 1
        
        # Find receiving modules based on resonance
        receivers = self._find_resonating_modules(signal)
        
        # Propagate to receivers
        responses = []
        for module_name in receivers:
            if module_name in signal.visited_modules:
                continue  # Avoid loops
            
            module = self.modules.get(module_name)
            if module:
                signal.visited_modules.add(module_name)
                self.stats["module_activity"][module_name] = self.stats["module_activity"].get(module_name, 0) + 1
                
                try:
                    response = module.receive_signal(signal)
                    if response:
                        responses.append(response)
                except Exception as e:
                    logger.warning(f"Module '{module_name}' failed to process signal: {e}")
        
        # Propagate responses
        for response in responses:
            self._process_signal(response)
        
        # Update resonance affinity based on this interaction
        self._update_affinity(signal, receivers)
    
    def _find_resonating_modules(self, signal: Signal) -> List[str]:
        """
        Find modules that should receive this signal based on resonance.
        No hardcoded routing - purely resonance-based.
        """
        receivers = []
        
        for module_name, module in self.modules.items():
            if module_name == signal.source_module:
                continue  # Don't send back to source
            
            # Calculate resonance with this module
            module_state = module.get_resonance_state()
            resonance_score = self._calculate_resonance(signal.resonance_pattern, module_state)
            
            # Add affinity bonus (learned over time)
            affinity = self.resonance_affinity.get(signal.source_module, {}).get(module_name, 0.0)
            resonance_score += affinity * 0.3  # Affinity contributes 30%
            
            # Threshold based on signal intensity and coherence
            threshold = 0.3 * signal.coherence
            
            if resonance_score > threshold:
                receivers.append(module_name)
        
        return receivers
    
    def _calculate_resonance(self, pattern_a: Dict[str, float], pattern_b: Dict[str, float]) -> float:
        """Calculate resonance between two patterns."""
        if not pattern_a or not pattern_b:
            return 0.0
        
        # Find common concepts
        common = set(pattern_a.keys()) & set(pattern_b.keys())
        if not common:
            return 0.0
        
        # Calculate weighted overlap
        total_resonance = 0.0
        for concept in common:
            total_resonance += pattern_a[concept] * pattern_b[concept]
        
        # Normalize
        return total_resonance / max(len(pattern_a), len(pattern_b))
    
    def _update_affinity(self, signal: Signal, receivers: List[str]):
        """Update resonance affinity based on successful communication."""
        source = signal.source_module
        if source not in self.resonance_affinity:
            self.resonance_affinity[source] = {}
        
        for receiver in receivers:
            # Strengthen affinity slightly
            current = self.resonance_affinity[source].get(receiver, 0.0)
            self.resonance_affinity[source][receiver] = min(1.0, current + 0.01)
        
        # Decay other affinities slightly
        for other in self.resonance_affinity[source]:
            if other not in receivers:
                current = self.resonance_affinity[source][other]
                self.resonance_affinity[source][other] = max(0.0, current - 0.001)
    
    def broadcast_signal(self, signal: Signal):
        """Broadcast a signal to all modules (bypassing resonance routing)."""
        for module_name, module in self.modules.items():
            if module_name != signal.source_module:
                try:
                    module.receive_signal(signal)
                except Exception as e:
                    logger.warning(f"Broadcast to '{module_name}' failed: {e}")
    
    def query_module(self, target_module: str, signal: Signal) -> Optional[Signal]:
        """Send a direct query to a specific module."""
        module = self.modules.get(target_module)
        if module:
            signal.signal_type = SignalType.QUERY
            return module.receive_signal(signal)
        return None
    
    def get_global_resonance_state(self) -> Dict[str, float]:
        """Get the combined resonance state of all modules."""
        combined = {}
        
        for module in self.modules.values():
            state = module.get_resonance_state()
            for concept, strength in state.items():
                if concept in combined:
                    combined[concept] = max(combined[concept], strength)
                else:
                    combined[concept] = strength
        
        return combined
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hub statistics."""
        return {
            **self.stats,
            "registered_modules": list(self.modules.keys()),
            "log_size": len(self.signal_log)
        }
    
    def start_async(self):
        """Start async signal processing."""
        if not self.async_mode:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._async_loop, daemon=True)
        self._thread.start()
        logger.info("🌐 CommunicationHub async loop started")
    
    def stop_async(self):
        """Stop async signal processing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("🌐 CommunicationHub async loop stopped")
    
    def _async_loop(self):
        """Async signal processing loop."""
        while self._running:
            try:
                signal = self.signal_queue.get(timeout=0.1)
                self._process_signal(signal)
            except queue.Empty:
                continue


# Convenience function to create a signal
def create_signal(
    signal_type: SignalType,
    source: str,
    resonance: Optional[Dict[str, float]] = None,
    intensity: float = 0.5,
    raw_data: Any = None
) -> Signal:
    """Create a new signal for communication."""
    return Signal(
        signal_type=signal_type,
        source_module=source,
        resonance_pattern=resonance or {},
        intensity=intensity,
        raw_data=raw_data
    )
