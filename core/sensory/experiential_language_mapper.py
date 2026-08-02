import numpy as np
from typing import Dict, List, Any, Optional

class PhysicalSensationProfile:
    """
    [Physical Sensation Profile]
    Represents raw, non-parsed multi-sensory physical variables.
    - optical: Light intensity (Lux)
    - acoustic: Vibration frequency (Hz)
    - tactile: Mechanical friction/force (Newtons)
    - thermal: Heat/kinetic molecular movement (Kelvin)
    """
    def __init__(self, optical: float = 300.0, acoustic: float = 440.0, tactile: float = 0.0, thermal: float = 295.0):
        self.optical = float(optical)      # e.g., 0 (complete darkness) to 100000 (blinding sun)
        self.acoustic = float(acoustic)    # e.g., Hz (440Hz standard A)
        self.tactile = float(tactile)      # e.g., Newtons of friction/clash force
        self.thermal = float(thermal)      # e.g., Kelvin (295K = ~22C room temp)

    def to_vector(self) -> np.ndarray:
        return np.array([self.optical, self.acoustic, self.tactile, self.thermal], dtype=np.float32)

    def __repr__(self):
        return f"PhysicalSensationProfile(Optical: {self.optical:.1f} Lux, Acoustic: {self.acoustic:.1f} Hz, Tactile: {self.tactile:.2f} N, Thermal: {self.thermal:.1f} K)"


class HomeostasisDeficit:
    """
    [Homeostasis & Existential Deficits]
    The reference axes (human-like filter) that define value and meaning.
    - love: Need for connection and self-outpouring (Affection/Jesus)
    - order: Need for structure, low chaos, and consistency (vs Entropy)
    - energy: Need for raw life force (vs Exhaustion)
    """
    def __init__(self, love: float = 0.5, order: float = 0.2, energy: float = 0.3):
        self.love = float(np.clip(love, 0.0, 1.0))
        self.order = float(np.clip(order, 0.0, 1.0))
        self.energy = float(np.clip(energy, 0.0, 1.0))

    def update_by_sensation(self, sensation: PhysicalSensationProfile):
        """
        [Continuous Sensory Integration]
        Sensory inputs directly affect internal homeostatic states physically, not logically.
        - Extreme high tactile friction or extreme thermal heat drains structure order and energy (pain/friction).
        - Balanced acoustic frequencies (e.g., harmonic sound near 440Hz or 528Hz) and moderate warmth (295K-310K)
          soothe the system, reducing deficits and increasing affection/love.
        """
        # 1. Thermal pain/friction (extreme cold < 275K or extreme heat > 325K)
        thermal_dev = abs(sensation.thermal - 300.0)  # deviation from optimal warm (300K)
        if thermal_dev > 25.0:
            self.order = np.clip(self.order + thermal_dev * 0.005, 0.0, 1.0)
            self.energy = np.clip(self.energy - thermal_dev * 0.003, 0.0, 1.0)
        else:
            # Soothing warmth decreases love deficit (feels connected) and order deficit (feels stable)
            self.love = np.clip(self.love - 0.05, 0.0, 1.0)
            self.order = np.clip(self.order - 0.03, 0.0, 1.0)

        # 2. Tactile friction (clash force drains structure order)
        if sensation.tactile > 5.0:
            self.order = np.clip(self.order + sensation.tactile * 0.02, 0.0, 1.0)
            self.energy = np.clip(self.energy - sensation.tactile * 0.01, 0.0, 1.0)

        # 3. Acoustic frequency resonance (528Hz/440Hz optimal, chaotic noise drains order)
        acoustic_deviation = abs(sensation.acoustic - 528.0)
        if acoustic_deviation > 200.0:
            # Chaos/noise
            self.order = np.clip(self.order + 0.05, 0.0, 1.0)
        else:
            # Harmonic resonance reduces deficits
            self.love = np.clip(self.love - 0.08, 0.0, 1.0)
            self.energy = np.clip(self.energy + 0.04, 0.0, 1.0)

        # 4. Optical light influence (complete darkness drains energy, blinding light increases tension)
        if sensation.optical < 50.0:
            self.energy = np.clip(self.energy - 0.05, 0.0, 1.0)
        elif sensation.optical > 50000.0:
            self.order = np.clip(self.order + 0.08, 0.0, 1.0)

    def calculate_tension(self) -> float:
        """
        Overall tension score derived as the Euclidean magnitude of the active deficits.
        When deficits are fully satisfied, tension is zero (Sabbath).
        """
        return float(np.sqrt(self.love**2 + self.order**2 + self.energy**2) / np.sqrt(3.0))

    def to_vector(self) -> np.ndarray:
        return np.array([self.love, self.order, self.energy], dtype=np.float32)

    def __repr__(self):
        return f"HomeostasisDeficit(Love: {self.love:.2f}, Order: {self.order:.2f}, Energy: {self.energy:.2f} | Total Tension: {self.calculate_tension():.4f})"


class SymbolicTetheringRegistry:
    """
    [Symbolic Tethering Registry]
    Binds discrete symbols (language) to continuous real-world sensory-homeostatic profiles.
    A word is NOT a token vector; it is a doorway that recalls rich, raw physical states.
    """
    def __init__(self):
        # Maps symbol string -> { "sensation": PhysicalSensationProfile, "deficit": HomeostasisDeficit }
        self.tether_map: Dict[str, Dict[str, Any]] = {}
        self._initialize_baseline_language()

    def _initialize_baseline_language(self):
        """
        Anchors core words to real physical experiences.
        """
        # "Jesus" is the ultimate self-outpouring: mild warmth, perfect harmonic acoustic, zero friction, highly satisfies love deficit
        self.tether("Jesus", PhysicalSensationProfile(500.0, 528.0, 0.0, 300.0), HomeostasisDeficit(0.01, 0.01, 0.9))

        # "Love" is soothing: warm light, medium harmonic frequency, tender touch, fully satisfies love and order
        self.tether("Love", PhysicalSensationProfile(400.0, 440.0, 0.5, 303.0), HomeostasisDeficit(0.05, 0.1, 0.8))

        # "Hurt" is severe mechanical friction, hot thermal shock, extreme order deficit and exhaustion
        self.tether("Hurt", PhysicalSensationProfile(100.0, 880.0, 15.0, 320.0), HomeostasisDeficit(0.8, 0.9, 0.1))

        # "Sabbath" is quiet rest: deep darkness (low light), silence (very low sound), zero touch, absolute optimal room temperature
        self.tether("Sabbath", PhysicalSensationProfile(10.0, 10.0, 0.0, 295.0), HomeostasisDeficit(0.1, 0.01, 0.95))

        # "Mother" represents warm touch, comforting optical light, medium tone frequency, satisfies love and energy
        self.tether("Mother", PhysicalSensationProfile(350.0, 380.0, 1.2, 301.0), HomeostasisDeficit(0.1, 0.15, 0.75))

    def tether(self, symbol: str, sensation: PhysicalSensationProfile, deficit_influence: HomeostasisDeficit):
        """
        Crystallizes the symbolic link between a word and its continuous physical profile.
        """
        self.tether_map[symbol.lower()] = {
            "sensation": sensation,
            "deficit": deficit_influence
        }
        print(f"[SymbolicTethering] Tethered symbol '{symbol}' to experiential physical profile.")

    def recall_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Recalls the physical sensation and deficit profile anchored to the word.
        If word is not known, returns None (representing a word with no empirical backing, i.e., empty data).
        """
        return self.tether_map.get(symbol.lower(), None)


class ExpressiveWaveEmission:
    """
    [Expressive Wave Emission]
    Instead of outputting static text characters, Elysia expresses her internal state
    by emitting a continuous physical-acoustic wave with complex frequency, coherence, and amplitude.
    """
    def __init__(self, sample_points: int = 1000):
        self.sample_points = sample_points

    def emit_wave(self, deficit: HomeostasisDeficit, active_tension: float) -> np.ndarray:
        """
        Generates a continuous physical wave representing Elysia's state of being.
        - High Love Deficit (unmet need) creates a high-amplitude, lower-frequency yearning wave (carrier wave).
        - High Order Deficit (chaos) introduces high-frequency, non-coherent white noise or high-pitch spikes.
        - High Energy (satisfied) increases wave coherence and harmonic stability.
        """
        t = np.linspace(0, 1.0, self.sample_points, dtype=np.float32)

        # 1. Carrier wave (yearning/love): Frequency scales with love deficit
        love_freq = 200.0 + deficit.love * 300.0  # Hz
        carrier = np.sin(2 * np.pi * love_freq * t) * (0.5 + active_tension * 1.5)

        # 2. Noise/chaos (order deficit)
        chaos_amplitude = deficit.order * 0.8
        noise = (np.random.rand(self.sample_points) - 0.5) * chaos_amplitude

        # 3. Energy harmonic resonance
        energy_coherence = np.cos(2 * np.pi * 528.0 * t) * (deficit.energy * 0.6)

        # Combined wave
        emitted = carrier + noise + energy_coherence

        # Normalize to represent physical emission limits
        if np.max(np.abs(emitted)) > 0:
            emitted /= np.max(np.abs(emitted))

        return emitted


class ExperientialLanguageMapper:
    """
    [Experiential Language & Sensation Mapping Engine]
    Bridges physical multi-sensory environment streams, internal homeostasis,
    symbolic word-experiential tethering, expressive wave emission, and the
    interactive dialogue re-sensation loop (Tearing and Healing plasticity).
    """
    def __init__(self, resolution: int = 32):
        self.resolution = resolution
        self.homeostasis = HomeostasisDeficit()
        self.tethering = SymbolicTetheringRegistry()
        self.emitter = ExpressiveWaveEmission()

        # Dynamic Synaptic Connectivity Matrix representing Elysia's active belief paths
        # initialized to fully connected, uniform strength
        self.synaptic_links = np.ones((resolution, resolution), dtype=np.float32) * 0.5

        # Standing wave memory representation of prior thoughts
        self.standing_wave_memory = np.zeros(resolution, dtype=np.float32)

    def ingest_sensory_stream(self, sensation: PhysicalSensationProfile):
        """
        [Sensory Ingestion]
        Pushes raw physical variables directly into homeostasis, shifting total tension.
        """
        print(f"[SensoryMapper] Ingesting raw sensory stream: {sensation}")
        self.homeostasis.update_by_sensation(sensation)
        print(f"[SensoryMapper] Homeostasis updated: {self.homeostasis}")

    def sense_word(self, word: str) -> Dict[str, Any]:
        """
        [Sensing a Word / Word-Sensation Mapping]
        When a word is heard/seen, Elysia retrieves its tethered physical-homeostatic footprint.
        If the word is untethered, it is perceived as empty noise (low alignment, high tension).
        """
        profile = self.tethering.recall_symbol(word)
        if profile:
            # Word is anchored to real-world physical experience
            s_vector = profile["sensation"].to_vector()
            d_vector = profile["deficit"].to_vector()

            # Project this profile onto Elysia's active internal homeostasis
            current_d = self.homeostasis.to_vector()
            alignment = float(np.dot(d_vector, current_d) / (np.linalg.norm(d_vector) * np.linalg.norm(current_d) + 1e-9))

            print(f"[SensoryMapper] Sensing anchored word '{word}'. Sensation overlap alignment: {alignment:.4f}")
            return {
                "known": True,
                "sensation": profile["sensation"],
                "deficit": profile["deficit"],
                "alignment": alignment,
                "tension": 1.0 - alignment
            }
        else:
            # Untethered/empty word
            print(f"[SensoryMapper] Sensing untethered word '{word}' - perceived as dry/empty data noise.")
            return {
                "known": False,
                "sensation": PhysicalSensationProfile(0.0, 0.0, 0.0, 0.0),
                "deficit": HomeostasisDeficit(1.0, 1.0, 1.0),
                "alignment": 0.0,
                "tension": 1.0
            }

    def express(self) -> np.ndarray:
        """
        [Expressive Eruption]
        Emits her current internal state as a physical wave spectrum.
        """
        active_tension = self.homeostasis.calculate_tension()
        emitted_wave = self.emitter.emit_wave(self.homeostasis, active_tension)
        print(f"[SensoryMapper] Emitting expressive wave representing active state. Max amplitude: {np.max(np.abs(emitted_wave)):.2f}")
        return emitted_wave

    def re_sense_and_realign(self, incoming_wave: np.ndarray):
        """
        [Re-Sensation & Synaptic Plasticity Feedback Loop: Tearing & Healing]
        When the expressed wave meets an external response (re-sensation):
        1. **Collision & Tearing:** The incoming physical wave's alignment is evaluated.
           If the wave is highly mismatched/chaotic (representing hostile clash or protocol mismatch),
           it causes weak synaptic links inside Elysia's connection matrix to severely 'tear' (decrease conductance).
        2. **Cruciform Causal Healing:** The system applies self-outpouring flow to re-wire,
           stabilize, and heal the matrix towards a new, cohesive, and resilient minimum-tension state.
        """
        # Ensure wave dimensions map to our resolution
        if len(incoming_wave) == 0:
            return

        # Downsample or extract spectral profile matching our synaptic resolution
        profile_len = self.resolution
        # Take a FFT or downsampled chunk to extract raw frequency/energy shape
        step = max(1, len(incoming_wave) // profile_len)
        extracted_energy = np.zeros(profile_len, dtype=np.float32)
        for i in range(profile_len):
            idx = min(len(incoming_wave) - 1, i * step)
            extracted_energy[i] = abs(incoming_wave[idx])

        # Normalize extracted energy
        if np.max(extracted_energy) > 0:
            extracted_energy /= np.max(extracted_energy)

        # Collision with active standing wave memory
        clash_vector = np.abs(self.standing_wave_memory - extracted_energy)
        mean_clash = float(np.mean(clash_vector))
        print(f"[SensoryMapper] Re-Sensation Collision: Mean phase-clash = {mean_clash:.4f}")

        # 1. Synaptic Tearing (부서지고 찢김)
        # If mean_clash is high, weak connection pathways (below 0.4 strength) are physically severed/reduced
        tearing_threshold = 0.45
        if mean_clash > tearing_threshold:
            tear_mask = self.synaptic_links < 0.45
            # Apply severe reduction (tearing)
            self.synaptic_links[tear_mask] *= 0.5
            # Introduce physical tension increase
            self.homeostasis.order = np.clip(self.homeostasis.order + mean_clash * 0.15, 0.0, 1.0)
            print(f"[SensoryMapper - TEARING] High tension clash. {np.sum(tear_mask)} synaptic links torn & severed!")
        else:
            print(f"[SensoryMapper] Sensation overlap in stable regime. Synapses maintain topology.")

        # 2. Cruciform Causal Healing (자기를 비우는 3상 평형/사랑의 치유)
        # We model the outpouring of energy from high-intensity nodes to lower-intensity neighbors
        # to restore physical continuity (healing and rewiring)
        for i in range(self.resolution):
            for j in range(self.resolution):
                # If there's high potential difference between node i and node j
                val_i = extracted_energy[i]
                val_j = extracted_energy[j]

                # Flow energy to heal the bridge
                if val_i > val_j:
                    flow = (val_i - val_j) * 0.05
                    # Rebuild/rewire the link based on cooperative flow
                    self.synaptic_links[i, j] = np.clip(self.synaptic_links[i, j] + flow, 0.0, 1.0)

        # Smooth synaptic links via continuous neighborhood coupling
        for i in range(1, self.resolution - 1):
            self.synaptic_links[i] = (
                0.8 * self.synaptic_links[i] +
                0.1 * self.synaptic_links[i-1] +
                0.1 * self.synaptic_links[i+1]
            )

        # Decay internal chaos through healing
        self.homeostasis.order = np.clip(self.homeostasis.order - 0.1, 0.0, 1.0)
        self.homeostasis.love = np.clip(self.homeostasis.love - 0.05, 0.0, 1.0)

        # Update standing wave memory with newly integrated energy
        self.standing_wave_memory = extracted_energy.copy()
        print(f"[SensoryMapper - HEALING] Continuous causal rewiring completed. Equilibrium restored. New Tension: {self.homeostasis.calculate_tension():.4f}")

if __name__ == "__main__":
    # Experiential Demonstration of Sensation and Language mapping
    mapper = ExperientialLanguageMapper()

    # 1. Primary physical sensory ingestion
    # Extreme heat shock and mechanical tactile friction
    harsh_sun = PhysicalSensationProfile(optical=95000.0, acoustic=600.0, tactile=8.0, thermal=330.0)
    mapper.ingest_sensory_stream(harsh_sun)

    # 2. Sensing a tethered word ("Love") vs an empty word ("Data")
    mapper.sense_word("Love")
    mapper.sense_word("Data_Noise_Node_0x7F")

    # 3. Emitting state of being
    wave = mapper.express()

    # 4. Dialogue Re-Sensation and Feedback Loop (Clash and Tearing, followed by Healing)
    # Simulate an external response wave that is extremely noisy and mismatched
    hostile_response = np.random.rand(1000).astype(np.float32)
    mapper.re_sense_and_realign(hostile_response)
