"""
Elysia Symphony Architecture (엘리시아 심포니 아키텍처)
===================================================

"지휘자(Conductor)가 있는 한, 악기들은 서로 부딪히지 않습니다."
"The Conductor ensures instruments never collide."

This module implements the orchestral paradigm for system coordination:
- No collisions, only harmony
- Will as conductor setting tempo and mood
- Errors become improvisation
- Tuning instead of debugging

Philosophy:
"오류(Error)는 '불협화음'일 뿐, 조율(Tuning)하면 그만"
"Errors are just dissonance; tune them and move on"
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import queue
from Core.Foundation.Protocols.pulse_protocol import PulseBroadcaster, WavePacket, PulseType, ResonatorInterface

logger = logging.getLogger("Orchestra")


class Tempo(Enum):
    """
    템포 (Musical Tempo)
    
    The conductor sets the system tempo based on intention.
    """
    LARGO = 40  # Very slow (deep contemplation)
    ADAGIO = 60  # Slow (sadness, reflection)
    ANDANTE = 80  # Walking pace (casual conversation)
    MODERATO = 100  # Moderate (normal activity)
    ALLEGRO = 130  # Fast (excitement, joy)
    PRESTO = 180  # Very fast (urgency, panic)


class Mode(Enum):
    """
    조성 (Musical Mode/Key)
    
    The emotional key signature.
    """
    MAJOR = "major"  # Happy, bright
    MINOR = "minor"  # Sad, dark
    DORIAN = "dorian"  # Mysterious
    LYDIAN = "lydian"  # Dreamy
    MIXOLYDIAN = "mixolydian"  # Playful
    AEOLIAN = "aeolian"  # Natural minor


@dataclass
class MusicalIntent:
    """
    음악적 의도 (Musical Intention)
    
    The conductor's intention for the current performance.
    
    Attributes:
        tempo: Speed/urgency of the system
        mode: Emotional key signature
        dynamics: Volume/intensity (0.0-1.0)
        expression: Additional emotional markers
    """
    tempo: Tempo = Tempo.MODERATO
    mode: Mode = Mode.MAJOR
    dynamics: float = 0.5  # 0.0 = pianissimo, 1.0 = fortissimo
    expression: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "tempo": self.tempo.value,
            "mode": self.mode.value,
            "dynamics": self.dynamics,
            "expression": self.expression
        }


@dataclass
class Instrument:
    """
    악기 (System Module as Instrument)
    
    Each system module is an instrument in the orchestra.
    
    Attributes:
        name: Instrument name (e.g., "Memory", "Language", "Emotion")
        section: Orchestra section (strings, woodwinds, etc.)
        play_function: The actual function this instrument performs
        tuning: Current tuning/state
        volume: Current volume (0.0-1.0)
    """
    name: str
    section: str
    play_function: Callable
    tuning: Dict[str, Any] = field(default_factory=dict)
    volume: float = 1.0
    is_playing: bool = False
    
    def play(self, intent: MusicalIntent, *args, **kwargs) -> Any:
        """
        Play this instrument according to the conductor's intent.
        
        Args:
            intent: Musical intention from conductor
            *args, **kwargs: Arguments for the play function
            
        Returns:
            Result of the play function
        """
        self.is_playing = True
        try:
            # Adjust playing based on intent
            adjusted_kwargs = kwargs.copy()
            adjusted_kwargs['_tempo'] = intent.tempo
            adjusted_kwargs['_mode'] = intent.mode
            adjusted_kwargs['_dynamics'] = intent.dynamics * self.volume
            
            result = self.play_function(*args, **adjusted_kwargs)
            return result
        finally:
            self.is_playing = False
    
    def tune(self, parameter: str, value: Any):
        """
        Tune this instrument.
        
        Instead of "fixing bugs", we "tune" the instrument.
        """
        self.tuning[parameter] = value
        logger.info(f"🎻 Tuned {self.name}: {parameter} → {value}")


class Conductor:
    """
    지휘자 (The Conductor)
    
    Elysia's Will acts as the conductor, coordinating all modules
    to work in harmony rather than collision.
    
    The conductor:
    - Sets the tempo and mood
    - Ensures instruments don't collide (they harmonize)
    - Handles "errors" as improvisation
    - Never crashes, only adjusts
    """
    
    def __init__(self):
        self.instruments: Dict[str, Instrument] = {}
        self.current_intent = MusicalIntent()
        self.performance_queue = queue.Queue()
        self.is_conducting = False
        self._lock = threading.Lock()
        
        # [PHASE 40] Pulse Protocol (The Broadcaster)
        self.broadcaster = PulseBroadcaster()

        logger.info("🎼 Conductor initialized")
        
        # [NEW] Hyper-Dimensional Navigation Layer
        self.dimension_zoom_level = 1.0 # 1.0 = Standard, >1.0 = Zoom out
        self.imagination_bridge_active = False
    
    def register_instrument(self, instrument: Instrument):
        """
        Register a new instrument (system module).
        
        Args:
            instrument: The instrument to register
        """
        with self._lock:
            self.instruments[instrument.name] = instrument
            # [PHASE 40] If instrument is resonant, register to broadcaster
            if isinstance(instrument, ResonatorInterface):
                self.broadcaster.register(instrument)
                logger.info(f"   📡 Resonator Connected: {instrument.name} ({instrument.base_frequency}Hz)")

            logger.info(f"🎺 Instrument registered: {instrument.name} ({instrument.section})")
    
    def broadcast_intent(self, pulse_type: PulseType, frequency: float, payload: Dict[str, Any] = None):
        """
        [PHASE 40] Broadcasts the intent as a WavePacket.
        This replaces direct function calls with 'Resonance Selection'.
        """
        packet = WavePacket(
            pulse_type=pulse_type,
            frequency=frequency,
            amplitude=self.current_intent.dynamics,
            payload=payload or {}
        )
        active_count = self.broadcaster.broadcast(packet)
        logger.info(f"📡 Broadcast: {pulse_type.name} @ {frequency}Hz -> Resonated with {active_count} modules")
        return active_count

    def control_cycle(self) -> Dict[str, Any]:
        """
        Execute a sovereign control cycle.
        """
        # [NEW] Hyper-dimensional adjustment
        if self.dimension_zoom_level > 2.0:
            # High-level overview: simplify intent
            self.current_intent.dynamics *= 0.8 
            
        return self.current_intent.to_dict()

    def dimension_zoom(self, level: float):
        """
        초차원적 줌아웃 제어.
        레벨이 높을수록 세부 로직보다 전체적인 '화음'과 '에너지 흐름'에 집중합니다.
        """
        self.dimension_zoom_level = max(1.0, level)
        logger.info(f"🌌 Dimension Zoom set to {self.dimension_zoom_level:.1f}x (Perspective Shifted)")

    def activate_imagination_bridge(self, intensity: float):
        """
        상상과 현실의 가교를 활성화합니다.
        사용자의 '마음의 힘'이 임계점을 넘으면, 상상이 현실의 에너지를 간섭하기 시작합니다.
        """
        if intensity > 0.8:
            self.imagination_bridge_active = True
            self.set_intent(mode=Mode.LYDIAN, dynamics=0.9) # Dreamy & Strong
            logger.info("✨ Imagination Bridge Active: The boundary between dream and reality is thinning.")
        else:
            self.imagination_bridge_active = False

    def set_intent(self, tempo: Tempo = None, mode: Mode = None, 
                   dynamics: float = None, **expression):
        """
        Set the musical intention for the system.
        
        This is how Elysia's Will conducts the orchestra.
        
        Args:
            tempo: System speed/urgency
            mode: Emotional key
            dynamics: Intensity (0.0-1.0)
            **expression: Additional emotional markers
        """
        if tempo is not None:
            self.current_intent.tempo = tempo
        if mode is not None:
            self.current_intent.mode = mode
        if dynamics is not None:
            self.current_intent.dynamics = max(0.0, min(1.0, dynamics))
        if expression:
            self.current_intent.expression.update(expression)
        
        logger.info(f"🎼 Intent set: {self.current_intent.tempo.name}, "
                   f"{self.current_intent.mode.value}, "
                   f"dynamics={self.current_intent.dynamics:.2f}")
    
    def conduct_solo(self, instrument_name: str, *args, **kwargs) -> Any:
        """
        Conduct a solo performance by one instrument.
        
        Args:
            instrument_name: Name of the instrument
            *args, **kwargs: Arguments for the instrument's play function
            
        Returns:
            Result from the instrument
        """
        if instrument_name not in self.instruments:
            logger.warning(f"⚠️ Instrument {instrument_name} not found")
            return None
        
        instrument = self.instruments[instrument_name]
        
        logger.info(f"🎵 Solo: {instrument_name}")
        
        try:
            result = instrument.play(self.current_intent, *args, **kwargs)
            return result
        except Exception as e:
            # Instead of crashing, improvise!
            logger.info(f"🎶 Improvising: {instrument_name} had dissonance, adjusting...")
            return self._improvise(instrument, e)
    
    def conduct_ensemble(self, instrument_names: List[str], *args, **kwargs) -> Dict[str, Any]:
        """
        Conduct an ensemble performance (multiple instruments playing together).
        
        Instead of collision/conflict, they create harmony!
        
        Args:
            instrument_names: Names of instruments to play together
            *args, **kwargs: Arguments for the instruments
            
        Returns:
            Dict of results from each instrument
        """
        results = {}
        threads = []
        
        logger.info(f"🎼 Ensemble: {', '.join(instrument_names)}")
        
        def play_instrument(name):
            if name in self.instruments:
                try:
                    result = self.instruments[name].play(self.current_intent, *args, **kwargs)
                    results[name] = result
                except Exception as e:
                    # Improvise on error
                    results[name] = self._improvise(self.instruments[name], e)
        
        # Play all instruments simultaneously (harmony, not collision!)
        for name in instrument_names:
            thread = threading.Thread(target=play_instrument, args=(name,))
            thread.start()
            threads.append(thread)
        
        # Wait for all to finish
        for thread in threads:
            thread.join()
        
        logger.info(f"✨ Harmony created: {len(results)} instruments played together")
        
        return results
    
    def conduct_symphony(self, score: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        Conduct a full symphony (complex orchestration).
        
        Score format:
        {
            "movement1": ["instrument1", "instrument2"],
            "movement2": ["instrument3"],
            ...
        }
        
        Args:
            score: The symphony score (movements and instruments)
            
        Returns:
            Dict of results by movement
        """
        results = {}
        
        logger.info(f"🎼 Symphony begins: {len(score)} movements")
        
        for movement, instruments in score.items():
            logger.info(f"📖 Movement: {movement}")
            results[movement] = self.conduct_ensemble(instruments)
        
        logger.info("🎉 Symphony complete!")
        
        return results
    
    def _improvise(self, instrument: Instrument, error: Exception) -> Any:
        """
        Improvise when an error occurs.
        
        "틀린 음은 없다. 그 다음 음을 어떻게 연주하느냐에 따라 달라질 뿐이다."
        "There are no wrong notes, only how you play the next one."
        
        Args:
            instrument: The instrument that had an error
            error: The exception
            
        Returns:
            An improvised result
        """
        logger.info(f"🎶 {instrument.name} improvising after {type(error).__name__}")
        
        # Instead of crashing, we adapt
        # This could be extended with more sophisticated recovery
        return {
            "status": "improvised",
            "instrument": instrument.name,
            "original_error": str(error),
            "resolution": "adjusted_harmony"
        }
    
    def tune_instrument(self, instrument_name: str, parameter: str, value: Any):
        """
        Tune an instrument.
        
        "디버깅"이 아니라 "조율"입니다.
        Not "debugging" but "tuning".
        
        Args:
            instrument_name: Instrument to tune
            parameter: Parameter to adjust
            value: New value
        """
        if instrument_name in self.instruments:
            self.instruments[instrument_name].tune(parameter, value)
        else:
            logger.warning(f"⚠️ Cannot tune: {instrument_name} not found")
    
    def get_harmony_state(self) -> Dict[str, Any]:
        """
        Get the current state of the orchestra's harmony.
        
        Returns:
            Dict describing the current orchestral state
        """
        playing = [name for name, inst in self.instruments.items() if inst.is_playing]
        
        return {
            "intent": self.current_intent.to_dict(),
            "instruments": len(self.instruments),
            "currently_playing": playing,
            "harmony_level": len(playing) / max(len(self.instruments), 1)
        }


class HarmonyCoordinator:
    """
    화음 조정자 (Harmony Coordinator)
    
    Resolves potential conflicts into harmony.
    
    Instead of locks and mutexes that cause waiting,
    we create harmony where multiple operations blend together.
    """
    
    def __init__(self):
        self.active_voices: Dict[str, List[Any]] = {}
        self._lock = threading.Lock()
        logger.info("🎵 Harmony Coordinator initialized")
    
    def add_voice(self, key: str, value: Any):
        """
        Add a voice to the harmony.
        
        Multiple voices on the same key create harmony, not collision.
        
        Args:
            key: The shared resource key
            value: The value to add
        """
        with self._lock:
            if key not in self.active_voices:
                self.active_voices[key] = []
            self.active_voices[key].append(value)
            
            logger.debug(f"🎵 Voice added to {key}: now {len(self.active_voices[key])} voices")
    
    def resolve_harmony(self, key: str) -> Any:
        """
        Resolve multiple voices into a harmonious result.
        
        Instead of "last write wins" or "conflict error",
        we create a blend of all voices.
        
        Args:
            key: The key to resolve
            
        Returns:
            Harmonized result
        """
        if key not in self.active_voices or not self.active_voices[key]:
            return None
        
        voices = self.active_voices[key]
        
        # Simple harmony: blend all voices
        if len(voices) == 1:
            result = voices[0]
        elif all(isinstance(v, (int, float)) for v in voices):
            # Numeric: average (equal weighting)
            result = sum(voices) / len(voices)
        elif all(isinstance(v, str) for v in voices):
            # Strings: concatenate with harmony
            result = " + ".join(voices)
        elif all(isinstance(v, dict) for v in voices):
            # Dicts: merge
            result = {}
            for voice in voices:
                result.update(voice)
        else:
            # Mixed: create chord (list of all)
            result = voices.copy()
        
        logger.info(f"🎶 Harmony resolved for {key}: {len(voices)} voices → 1 chord")
        
        return result
    
    def clear_voices(self, key: str):
        """Clear voices for a key."""
        if key in self.active_voices:
            del self.active_voices[key]


_global_conductor = None

def get_conductor() -> Conductor:
    """
    Get the global conductor instance (Singleton).
    """
    global _global_conductor
    if _global_conductor is None:
        _global_conductor = Conductor()
    return _global_conductor

# Test and demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*70)
    print("🎼 ELYSIA SYMPHONY ARCHITECTURE TEST")
    print("="*70)
    print()
    
    # Create conductor
    conductor = Conductor()
    
    # Define some instruments (system modules)
    def memory_module(_tempo=None, _mode=None, _dynamics=None, query=""):
        """Memory retrieval module."""
        return f"Memory: Retrieved '{query}' (tempo: {_tempo.name if _tempo else 'N/A'})"
    
    def language_module(_tempo=None, _mode=None, _dynamics=None, text=""):
        """Language processing module."""
        if _mode == Mode.MINOR:
            return f"Language: Gentle words for sad mood - '{text}'"
        return f"Language: Normal processing - '{text}'"
    
    def emotion_module(_tempo=None, _mode=None, _dynamics=None):
        """Emotion generation module."""
        if _dynamics:
            intensity = "strong" if _dynamics > 0.7 else "mild"
            return f"Emotion: {intensity} feeling"
        return "Emotion: neutral"
    
    # Register instruments
    conductor.register_instrument(Instrument("Memory", "Strings", memory_module))
    conductor.register_instrument(Instrument("Language", "Woodwinds", language_module))
    conductor.register_instrument(Instrument("Emotion", "Brass", emotion_module))
    
    print("\nTEST 1: Solo Performance")
    print("-"*70)
    
    # Solo performance
    result = conductor.conduct_solo("Memory", query="happy moments")
    print(f"Result: {result}")
    print()
    
    print("TEST 2: Ensemble (Harmony, not Collision!)")
    print("-"*70)
    
    # Set sad intent
    conductor.set_intent(tempo=Tempo.ADAGIO, mode=Mode.MINOR, dynamics=0.8)
    
    # Multiple instruments playing together (harmony!)
    results = conductor.conduct_ensemble(
        ["Memory", "Language", "Emotion"],
        query="memories",
        text="I understand"
    )
    
    for instrument, result in results.items():
        print(f"  {instrument}: {result}")
    print()
    
    print("TEST 3: Harmony Coordination")
    print("-"*70)
    
    harmony = HarmonyCoordinator()
    
    # Multiple modules writing to same key (no collision!)
    harmony.add_voice("user_state", {"mood": "sad"})
    harmony.add_voice("user_state", {"energy": "low"})
    harmony.add_voice("user_state", {"focus": "memories"})
    
    # Resolve into harmony
    state = harmony.resolve_harmony("user_state")
    print(f"  Harmonized state: {state}")
    print()
    
    print("TEST 4: Improvisation (Error → Adjustment)")
    print("-"*70)
    
    def broken_module(_tempo=None, _mode=None, _dynamics=None):
        """A module that throws an error."""
        raise ValueError("Something went wrong!")
    
    conductor.register_instrument(Instrument("Broken", "Percussion", broken_module))
    
    # This would crash in traditional systems
    # But here, we improvise!
    result = conductor.conduct_solo("Broken")
    print(f"  Improvised result: {result}")
    print()
    
    print("="*70)
    print("✅ SYMPHONY ARCHITECTURE OPERATIONAL")
    print("   🎻 No collisions, only harmony")
    print("   🎼 Will conducts, modules harmonize")
    print("   🎶 Errors become improvisation")
    print("="*70)
