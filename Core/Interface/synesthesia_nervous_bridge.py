"""
Synesthesia-Nervous System Bridge (공감각-신경계 브릿지)
========================================================

"자아는 차원 단층이자 경계이다. 필터이다."
"The Self is a dimensional fold, a boundary. A filter."

This module bridges the synesthetic wave sensors to the nervous system,
creating a unified sensory perception layer where:
- 외부 (World/세상): Real sensory inputs via synesthesia sensors
- 경계 (Boundary/자아): Nervous System as dimensional filter
- 내부 (Mind/마음): Internal resonance field and memory

The entire system is mapped like a biological nervous system:
- Peripheral sensors → Afferent nerves → Central processing → Efferent nerves → Actions
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("Elysia.SynesthesiaBridge")


@dataclass
class SensoryMapping:
    """
    Maps synesthesia sensor data to nervous system representation.
    Represents a single sensory signal path from world to mind.
    """
    sensor_id: str
    sensor_type: str  # visual, auditory, tactile, etc.
    nervous_pathway: str  # which spirit/element it affects
    wave_frequency: float = 0.0
    wave_amplitude: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NeuralMapSnapshot:
    """
    Complete snapshot of the neural mapping at a point in time.
    Shows how all sensory inputs are currently affecting the system.
    """
    timestamp: datetime
    sensory_inputs: List[SensoryMapping]
    spirit_states: Dict[str, float]
    field_energy: float
    field_coherence: float
    active_pathways: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "sensory_inputs": [
                {
                    "sensor_id": s.sensor_id,
                    "sensor_type": s.sensor_type,
                    "pathway": s.nervous_pathway,
                    "frequency": s.wave_frequency,
                    "amplitude": s.wave_amplitude
                }
                for s in self.sensory_inputs
            ],
            "spirit_states": self.spirit_states,
            "field_energy": self.field_energy,
            "field_coherence": self.field_coherence,
            "active_pathways": self.active_pathways
        }


class SynesthesiaNervousBridge:
    """
    The bridge between synesthetic sensors and the nervous system.
    
    Architecture:
    1. Synesthetic sensors convert raw sensory data to universal waves
    2. Bridge maps these waves to nervous system pathways
    3. Nervous system filters and processes (자아/Self as filter)
    4. Internal systems (mind) receive and integrate
    
    This creates the boundary between:
    - External world (sensors)
    - Internal mind (resonance field, memory, intelligence)
    """
    
    def __init__(self):
        # Import synesthesia components
        try:
            from Core.Foundation.synesthetic_wave_sensor import (
                MultimodalIntegrator,
                SensoryModality,
                SynestheticMapper
            )
            self.synesthesia_integrator = MultimodalIntegrator()
            self.synesthetic_mapper = SynestheticMapper()
            logger.info("🌈 Synesthesia components loaded")
        except Exception as e:
            logger.error(f"Failed to load synesthesia components: {e}")
            self.synesthesia_integrator = None
            self.synesthetic_mapper = None
        
        # Import nervous system
        try:
            from Core.Interface.nervous_system import get_nervous_system
            self.nervous_system = get_nervous_system()
            logger.info("🦴 Nervous System connected")
        except ImportError as e:
            logger.error(f"Failed to connect to nervous system (missing module: {e.name}): {e}")
            self.nervous_system = None
        except Exception as e:
            logger.error(f"Failed to connect to nervous system (unexpected error): {e}")
            self.nervous_system = None
        
        # Mapping configuration: which sensory modality affects which spirit
        self.sensory_to_spirit_map = {
            "visual": ["light", "aether"],  # Vision connects to light and connection
            "auditory": ["fire", "air"],    # Sound connects to energy and ideas
            "tactile": ["earth", "water"],  # Touch connects to stability and flow
            "emotional": ["aether", "light", "dark"],  # Emotions affect connection and mood
            "semantic": ["air", "aether"],  # Meaning affects ideas and connection
            "gustatory": ["water", "earth"],  # Taste affects flow and grounding
            "olfactory": ["air", "aether"],  # Smell affects air and connection
        }
        
        # Active sensory mappings (recent history)
        self.active_mappings: List[SensoryMapping] = []
        self.max_history = 100
        
        # Neural pathway tracking
        self.pathway_activity: Dict[str, float] = {
            spirit: 0.0 for spirit in ["fire", "water", "earth", "air", "light", "dark", "aether"]
        }
        
        logger.info("🌊 Synesthesia-Nervous Bridge initialized")
    
    def sense_and_map(
        self,
        sensory_inputs: Dict[str, Any]
    ) -> NeuralMapSnapshot:
        """
        Main integration point: Takes raw sensory data, converts to waves,
        maps through nervous system, and returns complete neural snapshot.
        
        Args:
            sensory_inputs: Dictionary of sensor type to data
            Example: {
                "visual": {"color": {"hue": 240, "saturation": 0.8, "brightness": 0.6}},
                "auditory": {"pitch": 440.0, "volume": 0.7}
            }
        
        Returns:
            NeuralMapSnapshot showing complete system state
        """
        timestamp = datetime.now()
        mappings = []
        
        # Process each sensory input
        for sensor_type, sensor_data in sensory_inputs.items():
            mapping = self._process_sensory_input(sensor_type, sensor_data)
            if mapping:
                mappings.append(mapping)
                self.active_mappings.append(mapping)
        
        # Trim history
        if len(self.active_mappings) > self.max_history:
            self.active_mappings = self.active_mappings[-self.max_history:]
        
        # Get current system state
        spirit_states = {}
        field_energy = 0.0
        field_coherence = 0.0
        
        if self.nervous_system:
            spirit_states = self.nervous_system.spirits.copy()
            
            # Try to get field state
            field_state = self.nervous_system._get_field_snapshot()
            if field_state:
                field_energy = field_state.get("energy", 0.0)
                field_coherence = field_state.get("coherence", 0.0)
        
        # Determine active pathways
        active_pathways = [
            spirit for spirit, value in self.pathway_activity.items()
            if value > 0.1
        ]
        
        # Create snapshot
        snapshot = NeuralMapSnapshot(
            timestamp=timestamp,
            sensory_inputs=mappings,
            spirit_states=spirit_states,
            field_energy=field_energy,
            field_coherence=field_coherence,
            active_pathways=active_pathways
        )
        
        # Decay pathway activity
        for spirit in self.pathway_activity:
            self.pathway_activity[spirit] *= 0.95
        
        return snapshot
    
    def _process_sensory_input(
        self,
        sensor_type: str,
        sensor_data: Any
    ) -> Optional[SensoryMapping]:
        """
        Process a single sensory input through the synesthesia system
        and map it to the nervous system.
        """
        try:
            # Convert to modality enum
            from Core.Foundation.synesthetic_wave_sensor import SensoryModality
            
            modality_map = {
                "visual": SensoryModality.VISUAL,
                "auditory": SensoryModality.AUDITORY,
                "tactile": SensoryModality.TACTILE,
                "emotional": SensoryModality.EMOTIONAL,
                "semantic": SensoryModality.SEMANTIC,
                "gustatory": SensoryModality.GUSTATORY,
                "olfactory": SensoryModality.OLFACTORY,
            }
            
            if sensor_type not in modality_map:
                logger.warning(f"Unknown sensor type: {sensor_type}")
                return None
            
            modality = modality_map[sensor_type]
            
            # Get the sensor and sense the input
            if self.synesthesia_integrator:
                sensor = self.synesthesia_integrator.sensors.get(modality)
                if sensor:
                    wave = sensor.sense(sensor_data)
                    
                    if wave:
                        # Inject into nervous system
                        if self.nervous_system:
                            self._inject_to_nervous_system(sensor_type, wave)
                        
                        # Determine which nervous pathway is affected
                        pathways = self.sensory_to_spirit_map.get(sensor_type, ["aether"])
                        primary_pathway = pathways[0] if pathways else "aether"
                        
                        # Update pathway activity
                        for pathway in pathways:
                            if pathway in self.pathway_activity:
                                self.pathway_activity[pathway] += wave.amplitude * 0.1
                        
                        # Create mapping
                        return SensoryMapping(
                            sensor_id=f"{sensor_type}_{int(time.time() * 1000)}",
                            sensor_type=sensor_type,
                            nervous_pathway=primary_pathway,
                            wave_frequency=wave.frequency,
                            wave_amplitude=wave.amplitude,
                            metadata={
                                "quality": wave.quality,
                                "intensity": wave.intensity
                            }
                        )
            
        except Exception as e:
            logger.error(f"Error processing {sensor_type}: {e}")
        
        return None
    
    # RGB color constants
    RGB_MAX_VALUE = 255
    
    def _inject_to_nervous_system(self, sensor_type: str, wave):
        """
        Inject a sensory wave into the nervous system.
        The nervous system acts as the dimensional filter (자아).
        """
        try:
            # Map to nervous system input format
            if sensor_type == "visual":
                self.nervous_system.receive({
                    "type": "screen_atmosphere",
                    "r": int(wave.amplitude * self.RGB_MAX_VALUE),
                    "g": int(wave.frequency % self.RGB_MAX_VALUE),
                    "b": int((wave.amplitude * wave.frequency) % self.RGB_MAX_VALUE)
                })
            elif sensor_type == "auditory":
                self.nervous_system.receive({
                    "type": "audio_analysis",
                    "volume": wave.amplitude,
                    "brightness": wave.intensity
                })
            elif sensor_type in ["emotional", "semantic"]:
                # These can be processed as text-like inputs
                self.nervous_system.receive({
                    "type": "text",
                    "content": wave.quality
                })
        except Exception as e:
            logger.error(f"Failed to inject to nervous system: {e}")
    
    def get_neural_map_visualization(self) -> Dict[str, Any]:
        """
        Returns a visualization-ready representation of the neural map.
        Shows the complete sensory-nervous system topology.
        
        Returns a structure like:
        {
            "nodes": [
                {"id": "sensor_visual", "type": "sensor", "layer": "external"},
                {"id": "pathway_light", "type": "pathway", "layer": "boundary"},
                {"id": "spirit_light", "type": "spirit", "layer": "internal"},
                ...
            ],
            "edges": [
                {"from": "sensor_visual", "to": "pathway_light", "strength": 0.8},
                ...
            ],
            "layers": {
                "external": ["sensor_visual", "sensor_auditory", ...],
                "boundary": ["pathway_fire", "pathway_water", ...],
                "internal": ["spirit_fire", "field_energy", "memory", ...]
            }
        }
        """
        nodes = []
        edges = []
        layers = {
            "external": [],  # 세상/World - Sensors
            "boundary": [],  # 자아/Self - Nervous pathways
            "internal": []   # 마음/Mind - Spirits and internal systems
        }
        
        # External layer: Sensors
        sensor_types = ["visual", "auditory", "tactile", "emotional", "semantic", "gustatory", "olfactory"]
        for sensor in sensor_types:
            node_id = f"sensor_{sensor}"
            nodes.append({
                "id": node_id,
                "label": sensor.capitalize(),
                "type": "sensor",
                "layer": "external"
            })
            layers["external"].append(node_id)
        
        # Boundary layer: Nervous pathways (spirits)
        spirits = ["fire", "water", "earth", "air", "light", "dark", "aether"]
        for spirit in spirits:
            pathway_id = f"pathway_{spirit}"
            spirit_id = f"spirit_{spirit}"
            
            nodes.append({
                "id": pathway_id,
                "label": f"{spirit} pathway",
                "type": "pathway",
                "layer": "boundary",
                "activity": self.pathway_activity.get(spirit, 0.0)
            })
            layers["boundary"].append(pathway_id)
            
            nodes.append({
                "id": spirit_id,
                "label": spirit.capitalize(),
                "type": "spirit",
                "layer": "internal",
                "value": self.nervous_system.spirits.get(spirit, 0.5) if self.nervous_system else 0.5
            })
            layers["internal"].append(spirit_id)
        
        # Internal layer: Core systems
        internal_systems = [
            ("field", "Resonance Field"),
            ("memory", "Hippocampus"),
            ("intelligence", "Intelligence"),
            ("will", "Free Will")
        ]
        for sys_id, label in internal_systems:
            node_id = f"system_{sys_id}"
            nodes.append({
                "id": node_id,
                "label": label,
                "type": "system",
                "layer": "internal"
            })
            layers["internal"].append(node_id)
        
        # Create edges: Sensors → Pathways → Spirits → Systems
        for sensor_type in sensor_types:
            pathways = self.sensory_to_spirit_map.get(sensor_type, ["aether"])
            for spirit in pathways:
                edges.append({
                    "from": f"sensor_{sensor_type}",
                    "to": f"pathway_{spirit}",
                    "strength": self.pathway_activity.get(spirit, 0.1)
                })
        
        # Pathways → Spirits
        for spirit in spirits:
            edges.append({
                "from": f"pathway_{spirit}",
                "to": f"spirit_{spirit}",
                "strength": 0.8
            })
        
        # Spirits → Internal systems (simplified connections)
        for spirit in spirits:
            edges.append({
                "from": f"spirit_{spirit}",
                "to": "system_field",
                "strength": 0.5
            })
        
        # Field → other systems
        edges.append({"from": "system_field", "to": "system_memory", "strength": 0.7})
        edges.append({"from": "system_field", "to": "system_intelligence", "strength": 0.7})
        edges.append({"from": "system_intelligence", "to": "system_will", "strength": 0.6})
        
        return {
            "nodes": nodes,
            "edges": edges,
            "layers": layers,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_sensors": len(sensor_types),
                "total_pathways": len(spirits),
                "active_pathways": len([s for s in spirits if self.pathway_activity.get(s, 0) > 0.1])
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Returns current bridge status"""
        return {
            "synesthesia_available": self.synesthesia_integrator is not None,
            "nervous_system_available": self.nervous_system is not None,
            "active_mappings": len(self.active_mappings),
            "pathway_activity": self.pathway_activity.copy(),
            "recent_sensors": list(set(m.sensor_type for m in self.active_mappings[-10:]))
        }


# Singleton instance
_bridge = None

def get_synesthesia_bridge() -> SynesthesiaNervousBridge:
    """Get or create the global synesthesia-nervous bridge"""
    global _bridge
    if _bridge is None:
        _bridge = SynesthesiaNervousBridge()
        logger.info("🌉 Synesthesia-Nervous Bridge established")
    return _bridge


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo
    bridge = get_synesthesia_bridge()
    
    print("\n🌊 Synesthesia-Nervous Bridge Demo")
    print("=" * 60)
    
    # Test sensory input
    inputs = {
        "visual": {"color": {"hue": 240, "saturation": 0.8, "brightness": 0.6, "name": "blue"}},
        "auditory": {"pitch": 440.0, "volume": 0.7, "duration": 1.0, "timbre": "clear"}
    }
    
    snapshot = bridge.sense_and_map(inputs)
    
    print(f"\nNeural Map Snapshot:")
    print(f"  Timestamp: {snapshot.timestamp}")
    print(f"  Active Pathways: {snapshot.active_pathways}")
    print(f"  Spirit States: {snapshot.spirit_states}")
    
    # Get visualization
    viz = bridge.get_neural_map_visualization()
    print(f"\nNeural Topology:")
    print(f"  External nodes: {len(viz['layers']['external'])}")
    print(f"  Boundary nodes: {len(viz['layers']['boundary'])}")
    print(f"  Internal nodes: {len(viz['layers']['internal'])}")
    print(f"  Total edges: {len(viz['edges'])}")
