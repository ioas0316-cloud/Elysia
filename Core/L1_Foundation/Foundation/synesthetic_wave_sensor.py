"""
Synesthetic Wave Sensor (         )
========================================

  (  ,   ,   ,   ,   )                    
              .                     ,
          (     ,        )                   .

Architecture:
- SensoryModality:         
- WaveSensor:         
- SynestheticMapper:        
- MultimodalIntegrator:        
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json

logger = logging.getLogger("Elysia.SynestheticWaveSensor")


class SensoryModality(Enum):
    """     """
    VISUAL = "visual"  #   
    AUDITORY = "auditory"  #   
    TACTILE = "tactile"  #   
    GUSTATORY = "gustatory"  #   
    OLFACTORY = "olfactory"  #   
    PROPRIOCEPTIVE = "proprioceptive"  #        (     )
    VESTIBULAR = "vestibular"  #      (  )
    INTEROCEPTIVE = "interoceptive"  #       (     )
    TEMPORAL = "temporal"  #     
    SPATIAL = "spatial"  #     
    EMOTIONAL = "emotional"  #     
    SEMANTIC = "semantic"  #     


class WaveProperty(Enum):
    """     """
    FREQUENCY = "frequency"  #    
    AMPLITUDE = "amplitude"  #   
    PHASE = "phase"  #   
    WAVELENGTH = "wavelength"  #   
    VELOCITY = "velocity"  #   
    POLARIZATION = "polarization"  #   /   


@dataclass
class SensoryWave:
    """
          (Sensory Wave)
    
                        .
    """
    modality: SensoryModality
    timestamp: datetime = field(default_factory=datetime.now)
    
    #      
    frequency: float = 1.0  # Hz
    amplitude: float = 1.0  # 0.0 ~ 1.0
    phase: float = 0.0  # 0 ~ 2 
    
    #       
    waveform: np.ndarray = field(default_factory=lambda: np.array([]))
    
    #      
    duration: float = 0.0  # seconds
    intensity: float = 0.5  # 0.0 ~ 1.0
    quality: str = ""  #      ( : "bright", "sharp", "warm")
    
    #      
    spatial_location: Optional[Tuple[float, float, float]] = None  # (x, y, z)
    
    #      
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """       """
        return {
            "modality": self.modality.value,
            "timestamp": self.timestamp.isoformat(),
            "frequency": self.frequency,
            "amplitude": self.amplitude,
            "phase": self.phase,
            "duration": self.duration,
            "intensity": self.intensity,
            "quality": self.quality,
            "spatial_location": self.spatial_location,
            "metadata": self.metadata
        }


class WaveSensor:
    """
          (Wave Sensor)
    
                            .
    """
    
    def __init__(self, modality: SensoryModality):
        self.modality = modality
        self.is_active = True
        self.sensitivity = 1.0  #    (0.0 ~ 2.0)
        self.samples_collected = 0
        
        logger.info(f"  Wave sensor initialized: {modality.value}")
    
    def sense(self, input_data: Any) -> SensoryWave:
        """
                      
        
        Args:
            input_data:       (         )
            
        Returns:
            SensoryWave   
        """
        if not self.is_active:
            logger.warning(f"   Sensor {self.modality.value} is inactive")
            return None
        
        #       
        if self.modality == SensoryModality.VISUAL:
            wave = self._sense_visual(input_data)
        elif self.modality == SensoryModality.AUDITORY:
            wave = self._sense_auditory(input_data)
        elif self.modality == SensoryModality.TACTILE:
            wave = self._sense_tactile(input_data)
        elif self.modality == SensoryModality.GUSTATORY:
            wave = self._sense_gustatory(input_data)
        elif self.modality == SensoryModality.OLFACTORY:
            wave = self._sense_olfactory(input_data)
        elif self.modality == SensoryModality.EMOTIONAL:
            wave = self._sense_emotional(input_data)
        elif self.modality == SensoryModality.SEMANTIC:
            wave = self._sense_semantic(input_data)
        else:
            wave = self._sense_generic(input_data)
        
        self.samples_collected += 1
        return wave
    
    def _sense_visual(self, data: Any) -> SensoryWave:
        """          """
        #  :            
        if isinstance(data, dict) and "color" in data:
            color = data["color"]
            # RGB          (  =400THz,   =800THz)
            freq = 400 + (color.get("hue", 0) / 360) * 400  # THz
            amplitude = color.get("saturation", 0.5)
            intensity = color.get("brightness", 0.5)
            quality = color.get("name", "unknown")
        else:
            freq = 500.0
            amplitude = 0.5
            intensity = 0.5
            quality = "neutral"
        
        #       (   )
        t = np.linspace(0, 0.1, 100)
        waveform = amplitude * np.sin(2 * np.pi * freq * t)
        
        return SensoryWave(
            modality=SensoryModality.VISUAL,
            frequency=freq,
            amplitude=amplitude,
            waveform=waveform,
            intensity=intensity,
            quality=quality,
            metadata={"source": "visual_sensor"}
        )
    
    def _sense_auditory(self, data: Any) -> SensoryWave:
        """          """
        if isinstance(data, dict):
            freq = data.get("pitch", 440.0)  # Hz (A4 = 440Hz)
            amplitude = data.get("volume", 0.5)
            duration = data.get("duration", 1.0)
            quality = data.get("timbre", "pure")
        else:
            freq = 440.0
            amplitude = 0.5
            duration = 1.0
            quality = "pure"
        
        #      
        t = np.linspace(0, duration, int(duration * 44100))  # 44.1kHz sampling
        waveform = amplitude * np.sin(2 * np.pi * freq * t)
        
        return SensoryWave(
            modality=SensoryModality.AUDITORY,
            frequency=freq,
            amplitude=amplitude,
            waveform=waveform,
            duration=duration,
            intensity=amplitude,
            quality=quality,
            metadata={"source": "auditory_sensor"}
        )
    
    def _sense_tactile(self, data: Any) -> SensoryWave:
        """          """
        if isinstance(data, dict):
            pressure = data.get("pressure", 0.5)
            texture = data.get("texture", "smooth")
            temperature = data.get("temperature", 0.5)  # 0=cold, 1=hot
            location = data.get("location", (0, 0, 0))
        else:
            pressure = 0.5
            texture = "smooth"
            temperature = 0.5
            location = (0, 0, 0)
        
        #         ,            
        freq = 10.0 if texture == "smooth" else 50.0 if texture == "rough" else 30.0
        amplitude = pressure
        
        return SensoryWave(
            modality=SensoryModality.TACTILE,
            frequency=freq,
            amplitude=amplitude,
            intensity=pressure,
            quality=texture,
            spatial_location=location,
            metadata={"temperature": temperature}
        )
    
    def _sense_gustatory(self, data: Any) -> SensoryWave:
        """          """
        if isinstance(data, dict):
            taste = data.get("taste", "umami")  # sweet, sour, salty, bitter, umami
            intensity = data.get("intensity", 0.5)
        else:
            taste = "umami"
            intensity = 0.5
        
        #           
        taste_freq_map = {
            "sweet": 100.0,
            "sour": 200.0,
            "salty": 150.0,
            "bitter": 250.0,
            "umami": 175.0
        }
        freq = taste_freq_map.get(taste, 150.0)
        
        return SensoryWave(
            modality=SensoryModality.GUSTATORY,
            frequency=freq,
            amplitude=intensity,
            intensity=intensity,
            quality=taste,
            metadata={"taste_type": taste}
        )
    
    def _sense_olfactory(self, data: Any) -> SensoryWave:
        """          """
        if isinstance(data, dict):
            scent = data.get("scent", "neutral")
            intensity = data.get("intensity", 0.5)
            pleasantness = data.get("pleasantness", 0.5)  # -1=unpleasant, 1=pleasant
        else:
            scent = "neutral"
            intensity = 0.5
            pleasantness = 0.5
        
        #           
        freq = 50.0 + (pleasantness + 1) * 25.0  # 50-100 Hz
        
        return SensoryWave(
            modality=SensoryModality.OLFACTORY,
            frequency=freq,
            amplitude=intensity,
            intensity=intensity,
            quality=scent,
            metadata={"pleasantness": pleasantness}
        )
    
    def _sense_emotional(self, data: Any) -> SensoryWave:
        """          """
        if isinstance(data, dict):
            emotion = data.get("emotion", "neutral")
            valence = data.get("valence", 0.0)  # -1=negative, 1=positive
            arousal = data.get("arousal", 0.5)  # 0=calm, 1=excited
        else:
            emotion = "neutral"
            valence = 0.0
            arousal = 0.5
        
        #            
        freq = 1.0 + arousal * 10.0  # 1-11 Hz
        amplitude = abs(valence)
        phase = 0 if valence >= 0 else np.pi
        
        return SensoryWave(
            modality=SensoryModality.EMOTIONAL,
            frequency=freq,
            amplitude=amplitude,
            phase=phase,
            intensity=arousal,
            quality=emotion,
            metadata={"valence": valence, "arousal": arousal}
        )
    
    def _sense_semantic(self, data: Any) -> SensoryWave:
        """          """
        if isinstance(data, dict):
            meaning = data.get("meaning", "")
            abstractness = data.get("abstractness", 0.5)  # 0=concrete, 1=abstract
            complexity = data.get("complexity", 0.5)
        else:
            meaning = str(data)
            abstractness = 0.5
            complexity = 0.5
        
        #            
        freq = 5.0 + abstractness * 20.0  # 5-25 Hz
        amplitude = complexity
        
        return SensoryWave(
            modality=SensoryModality.SEMANTIC,
            frequency=freq,
            amplitude=amplitude,
            intensity=complexity,
            quality=meaning[:50],
            metadata={"abstractness": abstractness}
        )
    
    def _sense_generic(self, data: Any) -> SensoryWave:
        """             """
        return SensoryWave(
            modality=self.modality,
            frequency=1.0,
            amplitude=0.5,
            intensity=0.5,
            quality="generic",
            metadata={"raw_data": str(data)}
        )


class SynestheticMapper:
    """
           (Synesthetic Mapper)
    
                             .
     :         (      ),         (       )
    """
    
    def __init__(self):
        #           
        self.mapping_rules: Dict[Tuple[SensoryModality, SensoryModality], Callable] = {}
        self._initialize_default_mappings()
        
        logger.info("  Synesthetic Mapper initialized")
    
    def _initialize_default_mappings(self):
        """                """
        #         (      )
        self.mapping_rules[(SensoryModality.VISUAL, SensoryModality.AUDITORY)] = \
            self._map_visual_to_auditory
        
        #         (      )
        self.mapping_rules[(SensoryModality.AUDITORY, SensoryModality.VISUAL)] = \
            self._map_auditory_to_visual
        
        #         (       )
        self.mapping_rules[(SensoryModality.TACTILE, SensoryModality.AUDITORY)] = \
            self._map_tactile_to_auditory
        
        #         (      )
        self.mapping_rules[(SensoryModality.EMOTIONAL, SensoryModality.VISUAL)] = \
            self._map_emotional_to_visual
        
        #         (       )
        self.mapping_rules[(SensoryModality.SEMANTIC, SensoryModality.EMOTIONAL)] = \
            self._map_semantic_to_emotional
    
    def map(
        self, 
        source_wave: SensoryWave, 
        target_modality: SensoryModality
    ) -> SensoryWave:
        """
             
        
        Args:
            source_wave:         
            target_modality:         
            
        Returns:
                     
        """
        mapping_key = (source_wave.modality, target_modality)
        
        if mapping_key in self.mapping_rules:
            mapper_func = self.mapping_rules[mapping_key]
            result = mapper_func(source_wave)
            logger.debug(
                f"  Mapped {source_wave.modality.value}   {target_modality.value}"
            )
            return result
        else:
            #       (        )
            return self._generic_map(source_wave, target_modality)
    
    def _map_visual_to_auditory(self, wave: SensoryWave) -> SensoryWave:
        """           (      )"""
        #                     
        #  : 400-800 THz     : 20-20000 Hz
        audio_freq = (wave.frequency - 400) / 400 * 19980 + 20
        
        #        
        audio_amplitude = wave.intensity
        
        return SensoryWave(
            modality=SensoryModality.AUDITORY,
            frequency=audio_freq,
            amplitude=audio_amplitude,
            intensity=audio_amplitude,
            quality=f"sound_of_{wave.quality}",
            metadata={
                "source_modality": "visual",
                "original_frequency": wave.frequency
            }
        )
    
    def _map_auditory_to_visual(self, wave: SensoryWave) -> SensoryWave:
        """           (      )"""
        #                  
        #   : 20-20000 Hz    : 400-800 THz
        visual_freq = (wave.frequency - 20) / 19980 * 400 + 400
        
        #        
        visual_intensity = wave.amplitude
        
        return SensoryWave(
            modality=SensoryModality.VISUAL,
            frequency=visual_freq,
            amplitude=visual_intensity,
            intensity=visual_intensity,
            quality=f"color_of_{wave.quality}",
            metadata={
                "source_modality": "auditory",
                "original_frequency": wave.frequency
            }
        )
    
    def _map_tactile_to_auditory(self, wave: SensoryWave) -> SensoryWave:
        """           (       )"""
        #           
        audio_freq = wave.frequency * 20  #          
        audio_amplitude = wave.amplitude
        
        return SensoryWave(
            modality=SensoryModality.AUDITORY,
            frequency=audio_freq,
            amplitude=audio_amplitude,
            intensity=audio_amplitude,
            quality=f"sound_of_{wave.quality}_texture",
            metadata={"source_modality": "tactile"}
        )
    
    def _map_emotional_to_visual(self, wave: SensoryWave) -> SensoryWave:
        """           (      )"""
        #           
        #                (  /  ),             (  /  )
        valence = wave.metadata.get("valence", 0)
        
        if valence > 0:
            #    : 400-600 THz (  -  )
            visual_freq = 400 + valence * 200
            quality = "warm"
        else:
            #    : 600-800 THz (  -  )
            visual_freq = 600 + abs(valence) * 200
            quality = "cool"
        
        return SensoryWave(
            modality=SensoryModality.VISUAL,
            frequency=visual_freq,
            amplitude=wave.intensity,
            intensity=wave.intensity,
            quality=quality,
            metadata={"source_emotion": wave.quality}
        )
    
    def _map_semantic_to_emotional(self, wave: SensoryWave) -> SensoryWave:
        """          """
        #              
        abstractness = wave.metadata.get("abstractness", 0.5)
        
        #            
        arousal = wave.intensity
        
        return SensoryWave(
            modality=SensoryModality.EMOTIONAL,
            frequency=1.0 + arousal * 10.0,
            amplitude=wave.amplitude,
            intensity=arousal,
            quality="contemplative",
            metadata={
                "valence": 0.0,
                "arousal": arousal,
                "source_meaning": wave.quality
            }
        )
    
    def _generic_map(
        self, 
        source: SensoryWave, 
        target_modality: SensoryModality
    ) -> SensoryWave:
        """      (           )"""
        return SensoryWave(
            modality=target_modality,
            frequency=source.frequency,
            amplitude=source.amplitude,
            intensity=source.intensity,
            quality=f"mapped_from_{source.modality.value}",
            metadata={"source_modality": source.modality.value}
        )


class MultimodalIntegrator:
    """
             (Multimodal Integrator)
    
                                    .
    """
    
    def __init__(self):
        self.sensors: Dict[SensoryModality, WaveSensor] = {}
        self.mapper = SynestheticMapper()
        self.active_waves: List[SensoryWave] = []
        self.integration_history: List[Dict[str, Any]] = []
        
        #                   
        for modality in SensoryModality:
            self.sensors[modality] = WaveSensor(modality)
        
        logger.info("  Multimodal Integrator initialized")
    
    def sense_multimodal(
        self, 
        inputs: Dict[SensoryModality, Any]
    ) -> List[SensoryWave]:
        """
                     
        
        Args:
            inputs: {    :      }     
            
        Returns:
                        
        """
        waves = []
        
        for modality, input_data in inputs.items():
            if modality in self.sensors:
                wave = self.sensors[modality].sense(input_data)
                if wave:
                    waves.append(wave)
        
        self.active_waves = waves
        return waves
    
    def create_synesthetic_experience(
        self, 
        source_wave: SensoryWave,
        target_modalities: List[SensoryModality]
    ) -> List[SensoryWave]:
        """
                 
        
                                .
        
        Args:
            source_wave:         
            target_modalities:                 
            
        Returns:
                         
        """
        synesthetic_waves = [source_wave]  #      
        
        for target in target_modalities:
            if target != source_wave.modality:
                mapped_wave = self.mapper.map(source_wave, target)
                synesthetic_waves.append(mapped_wave)
        
        logger.info(
            f"  Created synesthetic experience: " +
            f"{source_wave.modality.value}   " +
            f"{', '.join(m.value for m in target_modalities)}"
        )
        
        return synesthetic_waves
    
    def integrate_waves(
        self, 
        waves: List[SensoryWave]
    ) -> Dict[str, Any]:
        """
                    
        
        Returns:
                     
        """
        if not waves:
            return {}
        
        #          
        by_modality = {}
        for wave in waves:
            modality = wave.modality.value
            if modality not in by_modality:
                by_modality[modality] = []
            by_modality[modality].append(wave)
        
        #          
        avg_frequency = np.mean([w.frequency for w in waves])
        avg_amplitude = np.mean([w.amplitude for w in waves])
        avg_intensity = np.mean([w.intensity for w in waves])
        
        #          (              )
        resonance_score = self._calculate_resonance(waves)
        
        integration = {
            "timestamp": datetime.now().isoformat(),
            "num_modalities": len(by_modality),
            "total_waves": len(waves),
            "modalities": list(by_modality.keys()),
            "waves_by_modality": {
                mod: [w.to_dict() for w in ws]
                for mod, ws in by_modality.items()
            },
            "integrated_metrics": {
                "average_frequency": avg_frequency,
                "average_amplitude": avg_amplitude,
                "average_intensity": avg_intensity,
                "resonance_score": resonance_score
            },
            "description": self._generate_integrated_description(waves)
        }
        
        self.integration_history.append(integration)
        return integration
    
    def _calculate_resonance(self, waves: List[SensoryWave]) -> float:
        """               """
        if len(waves) < 2:
            return 1.0
        
        #        
        frequencies = [w.frequency for w in waves]
        freq_std = np.std(frequencies)
        freq_resonance = 1.0 / (1.0 + freq_std)
        
        #       
        amplitudes = [w.amplitude for w in waves]
        amp_std = np.std(amplitudes)
        amp_resonance = 1.0 / (1.0 + amp_std)
        
        #         
        resonance = (freq_resonance + amp_resonance) / 2.0
        return resonance
    
    def _generate_integrated_description(self, waves: List[SensoryWave]) -> str:
        """            """
        modalities = [w.modality.value for w in waves]
        qualities = [w.quality for w in waves if w.quality]
        
        return (
            f"Integrated perception from {len(set(modalities))} modalities: " +
            f"{', '.join(set(modalities))}. " +
            f"Qualities: {', '.join(qualities[:3])}"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """      """
        return {
            "total_sensors": len(self.sensors),
            "active_waves": len(self.active_waves),
            "integration_count": len(self.integration_history),
            "sensors_status": {
                mod.value: {
                    "active": sensor.is_active,
                    "samples": sensor.samples_collected
                }
                for mod, sensor in self.sensors.items()
            }
        }


#      
def example_synesthetic_sensing():
    """            """
    integrator = MultimodalIntegrator()
    
    print("\n              ")
    print("=" * 60)
    
    # 1.        
    print("\n---            ---")
    inputs = {
        SensoryModality.VISUAL: {
            "color": {"hue": 240, "saturation": 0.8, "brightness": 0.6, "name": "blue"}
        },
        SensoryModality.AUDITORY: {
            "pitch": 440.0, "volume": 0.7, "duration": 1.0, "timbre": "clear"
        },
        SensoryModality.EMOTIONAL: {
            "emotion": "joy", "valence": 0.8, "arousal": 0.6
        }
    }
    
    waves = integrator.sense_multimodal(inputs)
    print(f"      : {len(waves)} ")
    for wave in waves:
        print(f"  - {wave.modality.value}: freq={wave.frequency:.2f}, amp={wave.amplitude:.2f}")
    
    # 2.          
    print("\n---        (       ,   ) ---")
    audio_wave = waves[1]  #      
    synesthetic = integrator.create_synesthetic_experience(
        audio_wave,
        [SensoryModality.VISUAL, SensoryModality.TACTILE]
    )
    print(f"          : {len(synesthetic)}    ")
    for wave in synesthetic:
        print(f"  - {wave.modality.value}: {wave.quality}")
    
    # 3.   
    print("\n---       ---")
    integration = integrator.integrate_waves(waves)
    print(f"     :")
    print(f"  -     : {integration['num_modalities']}")
    print(f"  -      : {integration['integrated_metrics']['resonance_score']:.3f}")
    print(f"  -   : {integration['description']}")


if __name__ == "__main__":
    example_synesthetic_sensing()
