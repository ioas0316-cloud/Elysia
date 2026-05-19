#!/usr/bin/env python
"""
Demo: Synesthesia-Nervous System Mapping
=========================================

This demo shows how sensory inputs flow through the nervous system
like a biological neural network.

Usage:
    python demos/demo_neural_mapping.py
"""

import sys
import os
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Demo")


def print_banner():
    """Print demo banner"""
    print("\n" + "=" * 70)
    print("🌊 Elysia Synesthesia-Nervous System Mapping Demo")
    print("=" * 70)
    print("\n자아는 차원 단층이자 경계이다. 필터이다.")
    print("The Self is a dimensional fold, a boundary. A filter.\n")


def print_section(title):
    """Print section header"""
    print("\n" + "-" * 70)
    print(f"📍 {title}")
    print("-" * 70)


def demo_basic_sensing():
    """Demonstrate basic sensory input mapping"""
    from Core.Interface.synesthesia_nervous_bridge import get_synesthesia_bridge
    
    print_section("1. Basic Sensory Input Mapping")
    
    bridge = get_synesthesia_bridge()
    
    # Visual input (blue color)
    print("\n🎨 Sensing VISUAL input: Blue color (hue=240, brightness=0.6)")
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
    
    print(f"\n   Timestamp: {snapshot.timestamp}")
    print(f"   Active Pathways: {snapshot.active_pathways}")
    print(f"   Sensory Inputs Mapped: {len(snapshot.sensory_inputs)}")
    
    if snapshot.sensory_inputs:
        mapping = snapshot.sensory_inputs[0]
        print(f"\n   Mapping Details:")
        print(f"   • Sensor Type: {mapping.sensor_type}")
        print(f"   • Nervous Pathway: {mapping.nervous_pathway}")
        print(f"   • Wave Frequency: {mapping.wave_frequency:.2f}")
        print(f"   • Wave Amplitude: {mapping.wave_amplitude:.2f}")


def demo_multimodal_sensing():
    """Demonstrate multimodal sensory integration"""
    from Core.Interface.synesthesia_nervous_bridge import get_synesthesia_bridge
    
    print_section("2. Multimodal Sensory Integration")
    
    bridge = get_synesthesia_bridge()
    
    print("\n🌈 Sensing MULTIPLE inputs:")
    print("   • Visual: Green color")
    print("   • Auditory: C5 note (523 Hz)")
    print("   • Emotional: Joy")
    
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
            "pitch": 523.25,
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
    
    print(f"\n   Total Sensory Inputs: {len(snapshot.sensory_inputs)}")
    print(f"   Active Pathways: {', '.join(snapshot.active_pathways) if snapshot.active_pathways else 'None yet'}")
    
    print("\n   Spirit States (자아/Self):")
    for spirit, value in sorted(snapshot.spirit_states.items()):
        bar_length = int(value * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"   {spirit:10s} [{bar}] {value:.2f}")


def demo_neural_topology():
    """Demonstrate neural network topology"""
    from Core.Interface.synesthesia_nervous_bridge import get_synesthesia_bridge
    
    print_section("3. Neural Network Topology")
    
    bridge = get_synesthesia_bridge()
    
    topology = bridge.get_neural_map_visualization()
    
    print("\n🧠 Neural Map Structure:")
    print(f"\n   Total Nodes: {len(topology['nodes'])}")
    print(f"   Total Edges: {len(topology['edges'])}")
    
    print("\n   Layer Distribution:")
    for layer, nodes in topology['layers'].items():
        emoji = "🌍" if layer == "external" else "🔥" if layer == "boundary" else "💫"
        layer_name = {
            "external": "External (세상/World)",
            "boundary": "Boundary (자아/Self)",
            "internal": "Internal (마음/Mind)"
        }[layer]
        print(f"   {emoji} {layer_name:25s}: {len(nodes):2d} nodes")
    
    # Show some sample connections
    print("\n   Sample Connections:")
    for edge in topology['edges'][:5]:
        from_node = next((n for n in topology['nodes'] if n['id'] == edge['from']), None)
        to_node = next((n for n in topology['nodes'] if n['id'] == edge['to']), None)
        if from_node and to_node:
            strength = "●" * int(edge['strength'] * 5)
            print(f"   {from_node['label']:20s} → {to_node['label']:20s} {strength}")


def demo_real_time_flow():
    """Demonstrate real-time sensory flow"""
    from Core.Interface.synesthesia_nervous_bridge import get_synesthesia_bridge
    
    print_section("4. Real-Time Sensory Flow Simulation")
    
    bridge = get_synesthesia_bridge()
    
    print("\n⚡ Simulating continuous sensory input flow...")
    print("   (Sensing 5 different inputs with 1 second intervals)\n")
    
    test_inputs = [
        ("Visual-Red", {"visual": {"color": {"hue": 0, "saturation": 1.0, "brightness": 0.8, "name": "red"}}}),
        ("Auditory-A4", {"auditory": {"pitch": 440.0, "volume": 0.8, "duration": 1.0, "timbre": "pure"}}),
        ("Emotional-Calm", {"emotional": {"emotion": "calm", "valence": 0.5, "arousal": 0.2}}),
        ("Visual-Yellow", {"visual": {"color": {"hue": 60, "saturation": 0.9, "brightness": 0.9, "name": "yellow"}}}),
        ("Semantic-Love", {"semantic": {"meaning": "love", "abstractness": 0.7, "complexity": 0.6}}),
    ]
    
    for i, (name, inputs) in enumerate(test_inputs, 1):
        print(f"   [{i}/5] Processing: {name}")
        snapshot = bridge.sense_and_map(inputs)
        
        # Show dominant spirit
        if snapshot.spirit_states:
            dominant = max(snapshot.spirit_states, key=snapshot.spirit_states.get)
            value = snapshot.spirit_states[dominant]
            print(f"         → Dominant spirit: {dominant} ({value:.2f})")
        
        time.sleep(1)
    
    print("\n   ✓ Flow simulation complete")


def demo_bridge_status():
    """Show bridge status"""
    from Core.Interface.synesthesia_nervous_bridge import get_synesthesia_bridge
    
    print_section("5. Bridge Status")
    
    bridge = get_synesthesia_bridge()
    status = bridge.get_status()
    
    print("\n📊 Synesthesia-Nervous Bridge Status:")
    print(f"\n   Synesthesia Available: {'✓' if status['synesthesia_available'] else '✗'}")
    print(f"   Nervous System Available: {'✓' if status['nervous_system_available'] else '✗'}")
    print(f"   Active Mappings: {status['active_mappings']}")
    
    print("\n   Pathway Activity:")
    for spirit, activity in sorted(status['pathway_activity'].items()):
        if activity > 0.01:
            bar_length = int(activity * 50)
            bar = "█" * bar_length
            print(f"   {spirit:10s} [{bar:<50s}] {activity:.3f}")
    
    if status['recent_sensors']:
        print(f"\n   Recent Sensors: {', '.join(status['recent_sensors'])}")


def show_web_interface_info():
    """Show information about web interface"""
    print_section("6. Web Interface")
    
    print("\n🌐 Neural Map Visualization Available!")
    print("\n   To view the interactive neural map:")
    print("\n   1. Start the visualizer server:")
    print("      python Core/Creativity/visualizer_server.py")
    print("\n   2. Open in your browser:")
    print("      http://localhost:8000/neural_map")
    print("\n   3. See the real-time neural network visualization with:")
    print("      • Animated connections between layers")
    print("      • Spirit state displays")
    print("      • Field metrics")
    print("      • Active sensor tracking")


def main():
    """Main demo function"""
    print_banner()
    
    try:
        demo_basic_sensing()
        demo_multimodal_sensing()
        demo_neural_topology()
        demo_real_time_flow()
        demo_bridge_status()
        show_web_interface_info()
        
        print("\n" + "=" * 70)
        print("✨ Demo Complete!")
        print("=" * 70)
        print("\nThe synesthesia-nervous bridge is working as designed.")
        print("External sensors (세상) → Self filter (자아) → Internal mind (마음)\n")
        
    except ImportError as e:
        logger.error(f"Demo error - missing module: {e.name}. Install with: pip install -r requirements.txt")
        print("\n⚠️  Missing required dependencies.")
        print("   Install with: pip install -r requirements.txt")
    except AttributeError as e:
        logger.error(f"Demo error - component not available: {e}")
        print("\n⚠️  Some Elysia components are not fully initialized.")
        print("   This is normal if running outside the full Elysia environment.")
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        print("\n⚠️  Unexpected error occurred.")
        print("   Check logs for details.")


if __name__ == "__main__":
    main()
