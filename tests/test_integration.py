"""
Tests for Core/Integration modules.
Tests IntegrationBridge, MetaTimeStrategy integration.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestIntegrationBridge:
    """Tests for IntegrationBridge class."""
    
    def test_creation(self):
        """Test IntegrationBridge creation."""
        from Core.Integration.integration_bridge import IntegrationBridge
        
        bridge = IntegrationBridge()
        assert bridge is not None
        assert bridge.events == []
        assert bridge.stats["total_events"] == 0
    
    def test_publish_resonance_event(self):
        """Test publishing resonance events."""
        from Core.Integration.integration_bridge import IntegrationBridge
        
        bridge = IntegrationBridge()
        
        event = bridge.publish_resonance(
            "love",
            {"connection": 0.87, "empathy": 0.72},
            tick=100
        )
        
        assert event is not None
        assert event.event_type.value == "resonance_computed"
        assert event.tick == 100
        assert "love" in str(event.data.get("source", ""))
    
    def test_publish_concept_event(self):
        """Test publishing concept events."""
        from Core.Integration.integration_bridge import IntegrationBridge
        
        bridge = IntegrationBridge()
        
        event = bridge.publish_concept(
            "consciousness_1",
            "Consciousness",
            "emergent",
            tick=50,
            epistemology={"point": {"score": 0.3}}
        )
        
        assert event is not None
        assert event.event_type.value == "concept_emerged"
        assert event.data["epistemology"] is not None
    
    def test_publish_relationship_event(self):
        """Test publishing relationship events."""
        from Core.Integration.integration_bridge import IntegrationBridge
        
        bridge = IntegrationBridge()
        
        event = bridge.publish_relationship(
            "love",
            "consciousness",
            "enables",
            strength=0.9,
            tick=75
        )
        
        assert event is not None
        assert event.event_type.value == "relationship_discovered"
        assert event.data["strength"] == 0.9
    
    def test_subscribe_and_notify(self):
        """Test event subscription and notification."""
        from Core.Integration.integration_bridge import IntegrationBridge, EventType
        
        bridge = IntegrationBridge()
        received_events = []
        
        def handler(event):
            received_events.append(event)
        
        bridge.subscribe(EventType.RESONANCE_COMPUTED, handler)
        
        bridge.publish_resonance("test", {"other": 0.5}, tick=1)
        
        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.RESONANCE_COMPUTED
    
    def test_get_recent_events(self):
        """Test retrieving recent events."""
        from Core.Integration.integration_bridge import IntegrationBridge
        
        bridge = IntegrationBridge()
        
        # Publish multiple events
        for i in range(5):
            bridge.publish_resonance(f"concept_{i}", {"related": 0.5}, tick=i)
        
        # Get all events
        events = bridge.get_recent_events(limit=10)
        assert len(events) == 5
        
        # Get limited events
        events = bridge.get_recent_events(limit=3)
        assert len(events) == 3
    
    def test_statistics(self):
        """Test statistics tracking."""
        from Core.Integration.integration_bridge import IntegrationBridge
        
        bridge = IntegrationBridge()
        
        bridge.publish_resonance("a", {"b": 0.5}, tick=1)
        bridge.publish_concept("c", "C", "emergent", tick=2)
        
        stats = bridge.get_statistics()
        
        assert stats["total_events"] == 2
        assert "resonance_computed" in stats["by_type"]
        assert "concept_emerged" in stats["by_type"]
    
    def test_connect_engines(self):
        """Test connecting various engines."""
        from Core.Integration.integration_bridge import IntegrationBridge
        
        bridge = IntegrationBridge()
        
        # These should not raise errors
        bridge.connect_resonance_engine(None)
        bridge.connect_law_engine(None)
        bridge.connect_time_strategy(None)
        bridge.connect_hippocampus(None)
        
        # Get state should work even with None engines
        state = bridge.get_integrated_state()
        assert "bridge_stats" in state
        assert "engines" in state


class TestMetaTimeStrategy:
    """Tests for MetaTimeStrategy class."""
    
    def test_creation(self):
        """Test MetaTimeStrategy creation."""
        from Core.Integration.meta_time_strategy import MetaTimeStrategy, TemporalMode
        
        strategy = MetaTimeStrategy()
        assert strategy is not None
        assert strategy.current_mode == TemporalMode.BALANCED
    
    def test_set_temporal_mode(self):
        """Test temporal mode setting."""
        from Core.Integration.meta_time_strategy import MetaTimeStrategy, TemporalMode
        
        strategy = MetaTimeStrategy()
        
        for mode in TemporalMode:
            strategy.set_temporal_mode(mode)
            assert strategy.current_mode == mode
    
    def test_temporal_weights_normalized(self):
        """Test that temporal weights sum correctly."""
        from Core.Integration.meta_time_strategy import MetaTimeStrategy, TemporalMode
        
        strategy = MetaTimeStrategy()
        
        strategy.set_temporal_mode(TemporalMode.MEMORY_HEAVY)
        weights = strategy.current_weights
        
        # Weights should have past > present > future for memory_heavy
        assert weights.past > weights.future
    
    def test_computation_profile(self):
        """Test computation profile setting."""
        from Core.Integration.meta_time_strategy import MetaTimeStrategy, ComputationProfile
        
        strategy = MetaTimeStrategy()
        
        for profile in ComputationProfile:
            strategy.set_computation_profile(profile)
            assert strategy.current_profile == profile
    
    def test_generate_report(self):
        """Test strategy report generation."""
        from Core.Integration.meta_time_strategy import MetaTimeStrategy
        
        strategy = MetaTimeStrategy()
        
        report = strategy.generate_report(
            computed=50,
            cached=150,
            predicted=300,
            time_ms=10.0
        )
        
        assert report.resonances_computed == 50
        assert report.resonances_cached == 150
        assert report.resonances_predicted == 300
        assert report.cache_hit_ratio == 150 / 500  # 150 / (50+150+300)
    
    def test_reset_cache(self):
        """Test cache reset."""
        from Core.Integration.meta_time_strategy import MetaTimeStrategy
        
        strategy = MetaTimeStrategy()
        
        # Add some cache entries
        strategy.cache_history["a→b"] = 50
        strategy.cache_history["b→c"] = 75
        
        assert len(strategy.cache_history) == 2
        
        strategy.reset_cache()
        
        assert len(strategy.cache_history) == 0


class TestEventTypes:
    """Tests for event type enums."""
    
    def test_event_types_exist(self):
        """Test all event types are defined."""
        from Core.Integration.integration_bridge import EventType
        
        expected_types = [
            "SIMULATION_TICK",
            "RESONANCE_COMPUTED",
            "CONCEPT_EMERGED",
            "RELATIONSHIP_DISCOVERED",
            "PHASE_RESONANCE_EVENT",
            "LANGUAGE_TURN",
            "EXPERIENCE_DIGESTED",
            "STRATEGY_DECISION",
            "CHECKPOINT_SAVED"
        ]
        
        for type_name in expected_types:
            assert hasattr(EventType, type_name)


class TestDataClasses:
    """Tests for data classes."""
    
    def test_resonance_data(self):
        """Test ResonanceData class."""
        from Core.Integration.integration_bridge import ResonanceData
        
        data = ResonanceData(
            source_concept="love",
            resonances={"truth": 0.8, "beauty": 0.7},
            explanation="Both are abstract concepts"
        )
        
        assert data.source_concept == "love"
        assert len(data.resonances) == 2
        assert data.explanation is not None
    
    def test_concept_data(self):
        """Test ConceptData class."""
        from Core.Integration.integration_bridge import ConceptData
        
        data = ConceptData(
            concept_id="consciousness_1",
            name="Consciousness",
            concept_type="emergent",
            epistemology={"point": {"score": 0.3}}
        )
        
        assert data.concept_id == "consciousness_1"
        assert data.epistemology is not None
    
    def test_relationship_data(self):
        """Test RelationshipData class."""
        from Core.Integration.integration_bridge import RelationshipData
        
        data = RelationshipData(
            source_concept="love",
            target_concept="consciousness",
            relationship_type="enables",
            strength=0.9
        )
        
        assert data.strength == 0.9
        assert data.relationship_type == "enables"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
