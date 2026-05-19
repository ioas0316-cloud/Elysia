"""
Tests for Communication Hub - Central inter-module communication.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from Core.System.System.Integration.communication_hub import (
    CommunicationHub,
    ModuleInterface,
    Signal,
    SignalType,
    create_signal
)


class MockModule(ModuleInterface):
    """Mock module for testing."""
    
    def __init__(self, name: str, resonance_state: dict = None):
        super().__init__(name)
        self.resonance_state = resonance_state or {}
        self.received_signals = []
    
    def receive_signal(self, signal: Signal):
        self.received_signals.append(signal)
        return None
    
    def get_resonance_state(self):
        return self.resonance_state


class TestCommunicationHub:
    """Tests for CommunicationHub class."""
    
    def test_hub_creation(self):
        """Test hub initialization."""
        hub = CommunicationHub()
        assert hub is not None
        assert len(hub.modules) == 0
        assert hub.stats["signals_processed"] == 0
    
    def test_module_registration(self):
        """Test registering modules with the hub."""
        hub = CommunicationHub()
        module = MockModule("test_module")
        
        hub.register_module(module)
        
        assert "test_module" in hub.modules
        assert module._hub is None  # Module should use connect_to_hub
    
    def test_module_connection(self):
        """Test module connecting to hub."""
        hub = CommunicationHub()
        module = MockModule("test_module")
        
        module.connect_to_hub(hub)
        
        assert module._hub == hub
        assert "test_module" in hub.modules
    
    def test_signal_creation(self):
        """Test creating signals."""
        signal = create_signal(
            SignalType.RESONANCE,
            "source_module",
            resonance={"love": 0.8, "joy": 0.6},
            intensity=0.7
        )
        
        assert signal.signal_type == SignalType.RESONANCE
        assert signal.source_module == "source_module"
        assert signal.resonance_pattern["love"] == 0.8
        assert signal.intensity == 0.7
    
    def test_signal_propagation(self):
        """Test signal propagation based on resonance."""
        hub = CommunicationHub()
        
        # Create modules with different resonance states
        sender = MockModule("sender")
        receiver1 = MockModule("receiver1", {"love": 0.9})
        receiver2 = MockModule("receiver2", {"anger": 0.9})
        
        sender.connect_to_hub(hub)
        receiver1.connect_to_hub(hub)
        receiver2.connect_to_hub(hub)
        
        # Create a signal that resonates with "love"
        signal = create_signal(
            SignalType.RESONANCE,
            "sender",
            resonance={"love": 0.8},
            intensity=0.7
        )
        
        hub.propagate_signal(signal)
        
        # receiver1 should receive (resonates with love)
        assert len(receiver1.received_signals) == 1
        # receiver2 should not receive (no resonance with love)
        assert len(receiver2.received_signals) == 0
    
    def test_broadcast_signal(self):
        """Test broadcasting to all modules."""
        hub = CommunicationHub()
        
        receiver1 = MockModule("receiver1")
        receiver2 = MockModule("receiver2")
        
        receiver1.connect_to_hub(hub)
        receiver2.connect_to_hub(hub)
        
        signal = create_signal(
            SignalType.STATE_CHANGE,
            "system",
            intensity=1.0
        )
        
        hub.broadcast_signal(signal)
        
        # Both should receive
        assert len(receiver1.received_signals) == 1
        assert len(receiver2.received_signals) == 1
    
    def test_global_resonance_state(self):
        """Test getting combined resonance state."""
        hub = CommunicationHub()
        
        module1 = MockModule("m1", {"love": 0.8, "joy": 0.5})
        module2 = MockModule("m2", {"love": 0.6, "peace": 0.9})
        
        module1.connect_to_hub(hub)
        module2.connect_to_hub(hub)
        
        global_state = hub.get_global_resonance_state()
        
        # Should take max for overlapping concepts
        assert global_state["love"] == 0.8
        assert global_state["joy"] == 0.5
        assert global_state["peace"] == 0.9
    
    def test_statistics_tracking(self):
        """Test statistics are tracked correctly."""
        hub = CommunicationHub()
        
        module = MockModule("receiver", {"test": 1.0})
        module.connect_to_hub(hub)
        
        signal = create_signal(
            SignalType.THOUGHT,
            "sender",
            resonance={"test": 0.8}
        )
        
        hub.propagate_signal(signal)
        hub.propagate_signal(signal)
        
        stats = hub.get_statistics()
        assert stats["signals_processed"] == 2
        assert "thought" in stats["signals_by_type"]


class TestSignal:
    """Tests for Signal class."""
    
    def test_add_resonance(self):
        """Test adding resonance to a signal."""
        signal = Signal(
            signal_type=SignalType.THOUGHT,
            source_module="test"
        )
        
        signal.add_resonance("love", 0.8)
        signal.add_resonance("love", 0.9)  # Should update to max
        signal.add_resonance("joy", 0.5)
        
        assert signal.resonance_pattern["love"] == 0.9
        assert signal.resonance_pattern["joy"] == 0.5
    
    def test_get_dominant_concept(self):
        """Test getting the dominant concept."""
        signal = create_signal(
            SignalType.RESONANCE,
            "test",
            resonance={"love": 0.5, "joy": 0.8, "peace": 0.3}
        )
        
        dominant = signal.get_dominant_concept()
        assert dominant == "joy"
    
    def test_to_dict(self):
        """Test signal serialization."""
        signal = create_signal(
            SignalType.EMOTION,
            "source",
            resonance={"happy": 0.9},
            intensity=0.7
        )
        
        data = signal.to_dict()
        
        assert data["type"] == "emotion"
        assert data["source"] == "source"
        assert data["resonance"]["happy"] == 0.9
        assert data["intensity"] == 0.7


class TestSignalTypes:
    """Tests for signal type enumeration."""
    
    def test_all_types_exist(self):
        """Test all expected signal types are defined."""
        expected_types = [
            "RESONANCE",
            "STATE_CHANGE",
            "MEMORY_UPDATE",
            "PERCEPTION",
            "ACTION",
            "THOUGHT",
            "EMOTION",
            "QUERY",
            "RESPONSE"
        ]
        
        for type_name in expected_types:
            assert hasattr(SignalType, type_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
