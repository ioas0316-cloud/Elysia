"""
Test Synesthesia-Nervous Bridge

Tests the integration between synesthetic sensors and the nervous system.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def test_synesthesia_bridge_import():
    """Test that the synesthesia bridge can be imported"""
    from Core.Interface.synesthesia_nervous_bridge import SynesthesiaNervousBridge
    bridge = SynesthesiaNervousBridge()
    assert bridge is not None


def test_synesthesia_bridge_singleton():
    """Test that the bridge is a singleton"""
    from Core.Interface.synesthesia_nervous_bridge import get_synesthesia_bridge
    
    bridge1 = get_synesthesia_bridge()
    bridge2 = get_synesthesia_bridge()
    
    assert bridge1 is bridge2


def test_sense_and_map():
    """Test sensing and mapping sensory inputs"""
    from Core.Interface.synesthesia_nervous_bridge import get_synesthesia_bridge
    
    bridge = get_synesthesia_bridge()
    
    # Test with visual input
    inputs = {
        "visual": {
            "color": {
                "hue": 240, 
                "saturation": 0.8, 
                "brightness": 0.6, 
                "name": "blue"
            }
        }
    }
    
    snapshot = bridge.sense_and_map(inputs)
    
    # Verify snapshot structure
    assert snapshot is not None
    assert hasattr(snapshot, 'timestamp')
    assert hasattr(snapshot, 'sensory_inputs')
    assert hasattr(snapshot, 'spirit_states')
    assert hasattr(snapshot, 'active_pathways')
    
    # Verify we have at least one sensory input
    assert len(snapshot.sensory_inputs) > 0
    
    # Verify spirit states exist
    assert len(snapshot.spirit_states) > 0


def test_neural_map_visualization():
    """Test neural map visualization generation"""
    from Core.Interface.synesthesia_nervous_bridge import get_synesthesia_bridge
    
    bridge = get_synesthesia_bridge()
    
    # Get visualization data
    viz = bridge.get_neural_map_visualization()
    
    # Verify structure
    assert "nodes" in viz
    assert "edges" in viz
    assert "layers" in viz
    assert "metadata" in viz
    
    # Verify layers
    assert "external" in viz["layers"]
    assert "boundary" in viz["layers"]
    assert "internal" in viz["layers"]
    
    # Verify we have nodes in each layer
    assert len(viz["layers"]["external"]) > 0
    assert len(viz["layers"]["boundary"]) > 0
    assert len(viz["layers"]["internal"]) > 0
    
    # Verify nodes have required fields
    if len(viz["nodes"]) > 0:
        node = viz["nodes"][0]
        assert "id" in node
        assert "label" in node
        assert "type" in node
        assert "layer" in node


def test_multimodal_input():
    """Test with multiple sensory modalities"""
    from Core.Interface.synesthesia_nervous_bridge import get_synesthesia_bridge
    
    bridge = get_synesthesia_bridge()
    
    # Test with multiple inputs
    inputs = {
        "visual": {
            "color": {
                "hue": 120,
                "saturation": 0.7,
                "brightness": 0.8,
                "name": "green"
            }
        },
        "auditory": {
            "pitch": 523.25,  # C5
            "volume": 0.6,
            "duration": 1.0,
            "timbre": "bright"
        },
        "emotional": {
            "emotion": "joy",
            "valence": 0.8,
            "arousal": 0.7
        }
    }
    
    snapshot = bridge.sense_and_map(inputs)
    
    # Should have mappings for each input
    assert len(snapshot.sensory_inputs) >= 3
    
    # Verify different sensor types are present
    sensor_types = [s.sensor_type for s in snapshot.sensory_inputs]
    assert "visual" in sensor_types or "auditory" in sensor_types or "emotional" in sensor_types


def test_bridge_status():
    """Test bridge status reporting"""
    from Core.Interface.synesthesia_nervous_bridge import get_synesthesia_bridge
    
    bridge = get_synesthesia_bridge()
    
    status = bridge.get_status()
    
    assert "synesthesia_available" in status
    assert "nervous_system_available" in status
    assert "active_mappings" in status
    assert "pathway_activity" in status
    assert "recent_sensors" in status


def test_snapshot_to_dict():
    """Test snapshot serialization"""
    from Core.Interface.synesthesia_nervous_bridge import get_synesthesia_bridge
    
    bridge = get_synesthesia_bridge()
    
    inputs = {
        "visual": {
            "color": {
                "hue": 60,
                "saturation": 0.9,
                "brightness": 0.7,
                "name": "yellow"
            }
        }
    }
    
    snapshot = bridge.sense_and_map(inputs)
    snapshot_dict = snapshot.to_dict()
    
    # Verify it's a dictionary
    assert isinstance(snapshot_dict, dict)
    
    # Verify required keys
    assert "timestamp" in snapshot_dict
    assert "sensory_inputs" in snapshot_dict
    assert "spirit_states" in snapshot_dict
    assert "field_energy" in snapshot_dict
    assert "field_coherence" in snapshot_dict
    assert "active_pathways" in snapshot_dict


if __name__ == "__main__":
    print("Running Synesthesia-Nervous Bridge Tests...")
    print("=" * 60)
    
    # Run tests
    pytest.main([__file__, "-v"])
