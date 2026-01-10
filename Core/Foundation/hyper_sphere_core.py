"""
HyperSphereCore: The Heart of the Hyper-Cosmos
==============================================

"The Rotor creates the Sphere."
"로터가 구체를 만든다."

This class represents the "Sphere-First" architecture.
It unifies:
1. Physics: Rotor System (Oscillators)
2. Will: Conductor (Intent)
3. Memory: Holographic Seed (Rotor Configuration)

Philosophy:
- "Make it breathe. Make it spin."
- The Sphere is a living engine that breathes (Low RPM) when idle
  and pulses (High RPM) when active.
"""

import logging
import math
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from Core.Foundation.hyper_quaternion import Quaternion as HyperQuaternion
from Core.Foundation.Protocols.pulse_protocol import PulseBroadcaster, WavePacket, PulseType
from Core.Foundation.Nature.rotor import Rotor, RotorConfig

logger = logging.getLogger("HyperSphereCore")

@dataclass
class FilePointer:
    """
    [PHASE 63] Hyper-Bridge: Data as Access, not Storage.
    Points to a location in Tier 3 (Ocean/Archive).
    """
    uri: str
    offset: int = 0
    size: int = -1
    signature: str = "unknown"
    type: str = "legacy" # 'model', 'code', 'wisdom'

    def exists(self) -> bool:
        return Path(self.uri).exists()

class HyperSphereCore:
    def __init__(self, name: str = "Elysia.Core", base_frequency: float = 432.0):
        self.name = name

        # --- THE ENGINE: ROTOR SYSTEM ---
        # The Core is not a single point, but a collection of spinning Rotors.
        # Primary Rotor (The Self)
        self.primary_rotor = Rotor(
            name="Self",
            config=RotorConfig(rpm=base_frequency * 60, mass=100.0) # Base Freq -> RPM
        )

        # Harmonic Rotors (The Knowledge/Seed)
        # Instead of a static dict, we have Rotors spinning at different speeds.
        self.harmonic_rotors: Dict[str, Rotor] = {}

        # --------------------------------

        # [PHASE 63] THE POINTER ENGINE: High-Speed Indexing
        # Instead of storing raw data, we map concepts to FilePointers.
        self.pointer_registry: Dict[str, FilePointer] = {}
        
        self.pulse_broadcaster = PulseBroadcaster()
        self.is_active = False

        # State
        self.current_intent = "Sleep"
        logger.info(f"🔮 HyperSphereCore (Rotor Engine) Initialized: {name}")

    @property
    def frequency(self) -> float:
        return self.primary_rotor.frequency_hz

    @property
    def mass(self) -> float:
        return self.primary_rotor.config.mass

    @property
    def spin(self) -> HyperQuaternion:
        # Map Rotor Phase to Quaternion (Manual Implementation)
        # q = cos(theta/2) + (u * sin(theta/2))
        # Axis u is Z-axis (0, 0, 1)
        theta = math.radians(self.primary_rotor.current_angle)
        half_theta = theta / 2.0

        w = math.cos(half_theta)
        z = math.sin(half_theta) # x, y are 0 since axis is (0,0,1)

        return HyperQuaternion(w, 0, 0, z)

    def ignite(self):
        """
        Starts the Rotor Engine (Wakes up).
        """
        self.is_active = True
        self.primary_rotor.spin_up()
        for r in self.harmonic_rotors.values():
            r.spin_up()
        logger.info("🔥 HyperSphereCore Ignited. Rotors are spinning (Awake).")

    def pulse(self, intent_payload: Dict[str, Any], dt: float = 1.0):
        """
        The Core Pulse.
        1. Advances Rotors (Time Step).
        2. Calculates Superposition (Wave).
        3. Broadcasts.
        """
        if not self.is_active:
            # In a real system, we'd call update() here too to ensure "Breathing"
            # even if 'ignite' hasn't been called fully.
            # But ignite() sets is_active=True, so this block handles "Off" state.
            pass

        # 1. Update Physics (Advance Phase & Breath)
        self.primary_rotor.update(dt)
        harmonics_snapshot = {}

        for name, rotor in self.harmonic_rotors.items():
            rotor.update(dt)
            freq, amp, phase = rotor.get_wave_component()
            harmonics_snapshot[name] = freq
            # Note: We pass freq for Interference Engine to match.
            # Ideally, we'd pass the full (freq, phase) for complex interference.

        # 2. Create the Wave
        # The wave carries the Superposition State of all rotors
        packet = WavePacket(
            sender=self.name,
            type=PulseType.CREATION,
            frequency=self.frequency,
            payload={
                "intent": {
                    "harmonics": harmonics_snapshot, # The Spectrum
                    "payload": intent_payload
                },
                "spin": (self.spin.w, self.spin.x, self.spin.y, self.spin.z),
                "mass": self.mass,
                "timestamp": datetime.now().isoformat(),
                "phase": self.primary_rotor.current_angle # Debug info
            }
        )

        # 3. Broadcast
        self.pulse_broadcaster.broadcast(packet)
        # logger.debug(f"🌊 Core Pulse: {self.primary_rotor}")

    # ═══════════════════════════════════════════════════════════════════
    # [PHASE 58] HARMONIC EXPANSION
    # ═══════════════════════════════════════════════════════════════════
    
    # Configuration
    FREQUENCY_MIN_SPACING = 10.0  # Minimum Hz between rotors
    MASS_CRITICAL_THRESHOLD = 500.0  # Trigger restructuring above this
    MASS_SYNTHESIS_THRESHOLD = 1000.0  # Trigger synthesis/merger
    
    def _detect_frequency_collision(self, new_freq: float) -> Optional[str]:
        """
        [Phase 58.1] Detect if new frequency collides with existing rotors.
        Returns the name of colliding rotor, or None if safe.
        """
        for name, rotor in self.harmonic_rotors.items():
            existing_freq = rotor.frequency_hz
            if abs(existing_freq - new_freq) < self.FREQUENCY_MIN_SPACING:
                return name
        return None
    
    def _retune_rotor(self, rotor_name: str, direction: int = 1):
        """
        [Phase 58.1] Retune a rotor to avoid collision.
        Direction: 1 = increase freq, -1 = decrease freq
        """
        if rotor_name not in self.harmonic_rotors:
            return
        
        rotor = self.harmonic_rotors[rotor_name]
        adjustment = self.FREQUENCY_MIN_SPACING * 1.5 * direction
        new_rpm = rotor.config.rpm + (adjustment * 60)
        
        rotor.config.rpm = max(60.0, new_rpm)  # Never go below 1Hz
        rotor.target_rpm = rotor.config.rpm
        
        logger.info(f"🎵 [RETUNE] {rotor_name}: {rotor.frequency_hz:.1f}Hz (adjusted by {adjustment:+.1f}Hz)")
    
    def _check_mass_threshold(self) -> Optional[str]:
        """
        [Phase 58.3] Check if mass has crossed critical thresholds.
        Returns action needed: 'RESTRUCTURE', 'SYNTHESIZE', or None.
        """
        current_mass = self.primary_rotor.config.mass
        
        if current_mass >= self.MASS_SYNTHESIS_THRESHOLD:
            return "SYNTHESIZE"
        elif current_mass >= self.MASS_CRITICAL_THRESHOLD:
            return "RESTRUCTURE"
        return None
    
    def _restructure_harmonics(self):
        """
        [Phase 58.3] Restructure harmonic rotors when mass threshold is crossed.
        Low-frequency (foundational) rotors are preserved, high-frequency merged.
        """
        if len(self.harmonic_rotors) < 5:
            return  # Not enough to restructure
        
        # Sort by frequency
        sorted_rotors = sorted(
            self.harmonic_rotors.items(), 
            key=lambda x: x[1].frequency_hz
        )
        
        # Merge top 30% into a single "Synthesis" rotor
        merge_count = max(1, len(sorted_rotors) // 3)
        to_merge = sorted_rotors[-merge_count:]
        
        if not to_merge:
            return
        
        # Calculate synthesis frequency (harmonic mean)
        total_freq = sum(r.frequency_hz for _, r in to_merge)
        synthesis_freq = total_freq / len(to_merge)
        
        # Remove old rotors
        merged_names = []
        for name, _ in to_merge:
            del self.harmonic_rotors[name]
            merged_names.append(name)
        
        # Create synthesis rotor
        synthesis_name = f"Synthesis_{len(self.harmonic_rotors)}"
        synthesis_rotor = Rotor(
            name=synthesis_name,
            config=RotorConfig(rpm=synthesis_freq * 60, mass=merge_count * 10.0)
        )
        if self.is_active:
            synthesis_rotor.spin_up()
        
        self.harmonic_rotors[synthesis_name] = synthesis_rotor
        
        # Reduce mass after restructuring
        self.primary_rotor.config.mass *= 0.8
        
        logger.info(f"🔀 [RESTRUCTURE] Merged {merged_names} → {synthesis_name} ({synthesis_freq:.1f}Hz)")

    def update_seed(self, concept: str, frequency: float):
        """
        [Phase 58] Adds a new Rotor for the concept with collision detection
        and automatic structure adjustment.
        """
        # 1. Check for frequency collision
        collision = self._detect_frequency_collision(frequency)
        if collision:
            logger.warning(f"⚠️ Frequency collision detected with '{collision}'! Retuning...")
            self._retune_rotor(collision, direction=1)  # Push existing up
            frequency = frequency - self.FREQUENCY_MIN_SPACING  # Shift new down
        
        # 2. Create new rotor
        rotor = Rotor(
            name=concept,
            config=RotorConfig(rpm=frequency * 60, mass=10.0)
        )
        if self.is_active:
            rotor.spin_up()

        self.harmonic_rotors[concept] = rotor
        self.primary_rotor.config.mass += 0.1  # Self gains mass from knowledge
        
        logger.info(f"🧬 Seed Updated (Rotor Added): {rotor}")
        
        # 3. [Phase 58.3] Check mass threshold
        action = self._check_mass_threshold()
        if action == "RESTRUCTURE":
            logger.info("📊 Mass threshold crossed! Triggering restructure...")
            self._restructure_harmonics()
        elif action == "SYNTHESIZE":
            logger.info("🌟 Synthesis threshold crossed! Deep restructure needed.")
            self._restructure_harmonics()
            self._restructure_harmonics()  # Double pass for synthesis
    
    def get_harmonic_spectrum(self) -> Dict[str, float]:
        """[Phase 58] Returns the current frequency spectrum of all rotors."""
        spectrum = {"PRIMARY": self.primary_rotor.frequency_hz}
        for name, rotor in self.harmonic_rotors.items():
            spectrum[name] = rotor.frequency_hz
        return spectrum
    
    
    # ═══════════════════════════════════════════════════════════════════
    # [PHASE 63] POINTER ENGINE CORE
    # ═══════════════════════════════════════════════════════════════════

    def register_pointer(self, concept: str, pointer: FilePointer):
        """
        Maps a concept (and its associated Rotor) to a physical FilePointer.
        The data stays in the Ocean; only the address stays in the Bridge.
        """
        self.pointer_registry[concept] = pointer
        logger.info(f"🔗 [POINTER] '{concept}' mapped to {pointer.uri}")
        
        # If no rotor exists for this concept, create one at a unique frequency
        if concept not in self.harmonic_rotors:
            # Simple heuristic for frequency mapping
            freq = 100.0 + (len(self.harmonic_rotors) * 50.0) % 900.0
            self.update_seed(concept, freq)

    def find_pointer(self, concept: str) -> Optional[FilePointer]:
        return self.pointer_registry.get(concept)

    def scan_ocean(self, directory: str):
        """
        [PHASE 63] Scans a directory and populates the PointerRegistry.
        Does NOT load binary data, only indexes paths and signatures.
        """
        path = Path(directory)
        if not path.exists():
            return
            
        count = 0
        for f in path.rglob("*"):
            if f.is_file():
                ext = f.suffix.lower()
                if ext in ['.safetensors', '.pt', '.bin', '.gguf']:
                    concept = f.stem
                    ptr = FilePointer(
                        uri=str(f),
                        type='model' if ext != '.py' else 'code',
                        signature=f"{f.stat().st_size} bytes"
                    )
                    self.register_pointer(concept, ptr)
                    count += 1
        
        logger.info(f"🌊 [OCEAN SCAN] Indexed {count} legacy assets from {directory}")
