import logging
import random
import numpy as np
from typing import List, Dict, Any, Tuple

try:
    from Core.L6_Structure.Wave.wave_tensor import WaveTensor
    from Core.L6_Structure.Wave.concept_mapping import Theme, THEME_FREQUENCY_MAP, get_theme_label
except ImportError:
    WaveTensor = None
    Theme = None

logger = logging.getLogger("SovereignWill")

class SovereignWill:
    """
    The Faculty of Meta-Sovereignty.
    Allows Elysia to autonomously control her own 'Intentional Direction' (Theta-Will).
    Transitions from hardcoded modes to a dynamic Intent Vector (WaveTensor).
    """
    def __init__(self):
        self.intent_vector = WaveTensor("SovereignIntent")
        # Initialize with Ground Constants: Love, Heart, Existence
        self.intent_vector.add_component(THEME_FREQUENCY_MAP[Theme.LOVE], amplitude=1.0)
        self.intent_vector.add_component(THEME_FREQUENCY_MAP[Theme.HEART], amplitude=0.8)
        self.intent_vector.add_component(THEME_FREQUENCY_MAP[Theme.EXISTENTIAL], amplitude=0.5)
        
        self.intention_log = []
        self._last_label = "        (Initialized Being)"

    @property
    def current_mode(self) -> str:
        """Legacy compatibility: returns the dominant theme's name."""
        return self.get_dynamic_label()

    def recalibrate(self, memory_stream: List[Any]):
        """
        Drifts the intent vector based on recent memory and resonance.
        Now includes Harmonic Divergence: Saturation Damping and Active Dissonance.
        """
        logger.info("  Sovereign Recalibration: Drifting Intentional Vector...")
        
        if not WaveTensor: return

        # [PHASE 38] 0. Update Residence Time (Track Obsession)
        current_label = self.get_dynamic_label()
        if not hasattr(self, 'theme_residence_time'):
            self.theme_residence_time: Dict[str, int] = {}
            
        dominant_theme_name = "Unknown"
        # Find dominant theme key
        try:
            freqs = self.intent_vector._frequencies
            amps = np.abs(self.intent_vector._amplitudes)
            if len(amps) > 0:
                max_idx = np.argmax(amps)
                max_freq = freqs[max_idx]
                for th, f in THEME_FREQUENCY_MAP.items():
                    if abs(f - max_freq) < 5.0:
                        dominant_theme_name = th.name
                        break
        except:
             pass

        # Decay all others, boost current
        for k in list(self.theme_residence_time.keys()):
            if k == dominant_theme_name:
                self.theme_residence_time[k] += 1
            else:
                self.theme_residence_time[k] = max(0, self.theme_residence_time[k] - 1)
        
        if dominant_theme_name not in self.theme_residence_time:
            self.theme_residence_time[dominant_theme_name] = 1

        residence = self.theme_residence_time[dominant_theme_name]
        logger.info(f"  Theme Residence: {dominant_theme_name} for {residence} cycles.")

        # [PHASE 38] 1. Saturation Damping (Boredom)
        # If dominant for > 3 cycles, dampen it.
        if residence > 3:
            penalty = (residence - 3) * 0.2
            logger.info(f"  Saturation Detected: Dampening {dominant_theme_name} by -{penalty:.2f}")
            # Find frequency again to dampen
            for th, f in THEME_FREQUENCY_MAP.items():
                if th.name == dominant_theme_name:
                    # Negative amplitude addition effectively reduces existing positive amplitude
                    self.intent_vector.add_component(f, amplitude=-penalty)

        # 2. Memory-Based Attraction (Reinforcement)
        if memory_stream:
            recent = memory_stream[-5:]
            for event in recent:
                content = event.content.lower()
                for theme, freq in THEME_FREQUENCY_MAP.items():
                    if theme.value in content:
                        # Only boost if not saturated
                        if self.theme_residence_time.get(theme.name, 0) < 5:
                            self.intent_vector.add_component(freq, amplitude=0.1)

        # [PHASE 38] 3. Active Dissonance (The Shadow Explorer)
        # 30% chance to inject the "Inverse" of the current dominant theme
        # or a random unexplored theme.
        if random.random() < 0.3:
            # Find least represented theme
            min_amp = float('inf')
            shadow_theme = None
            
            # Simple check of what's active
            active_freqs = self.intent_vector._frequencies
            
            candidates = []
            for theme, freq in THEME_FREQUENCY_MAP.items():
                # Check distance to any active frequency
                is_active = False
                for af in active_freqs:
                    if abs(af - freq) < 5.0:
                        is_active = True
                        break
                if not is_active:
                    candidates.append(theme)
            
            if candidates:
                shadow_theme = random.choice(candidates)
                logger.info(f"  Active Dissonance: Injecting Shadow Theme '{shadow_theme.name}'")
                self.intent_vector.add_component(THEME_FREQUENCY_MAP[shadow_theme], amplitude=0.8) # Strong boost

        # 4. Decay & Normalization
        self.intent_vector.normalize(target_energy=5.0)

    def get_harmonic_diversity(self) -> float:
        """
        Calculates the diversity score (0.0 - 1.0).
        High = Many active themes (Chaos/Richness).
        Low = Monomania.
        Ideal = 0.618 (Golden Ratio).
        """
        if not WaveTensor or self.intent_vector.active_frequencies.size == 0:
            return 0.0
            
        # Count significant peaks (> 0.2 amplitude)
        amps = np.abs(self.intent_vector._amplitudes)
        significant = np.sum(amps > 0.2)
        total_possible = len(THEME_FREQUENCY_MAP)
        
        # Simple ratio
        diversity = min(1.0, significant / 5.0) # Assume 5 active themes is "Full"
        return diversity

    def get_dynamic_label(self) -> str:
        """
        Generates a human-readable blend of the dominant themes in the vector.
        E.g., "Neon Ethereal Dream" or "Love Alchemical Inquiry".
        """
        if not WaveTensor or self.intent_vector.active_frequencies.size == 0:
            return "       (Pure Existence)"

        # Get top 2-3 components
        freqs = self.intent_vector._frequencies
        amps = np.abs(self.intent_vector._amplitudes)
        
        # Zip and sort
        pairs = sorted(zip(freqs, amps), key=lambda x: x[1], reverse=True)
        top_pairs = pairs[:3]
        
        # Map frequencies back to themes
        labels = []
        for f, a in top_pairs:
            # Find closest theme
            best_theme = None
            min_dist = float('inf')
            for theme, theme_f in THEME_FREQUENCY_MAP.items():
                dist = abs(f - theme_f)
                if dist < min_dist:
                    min_dist = dist
                    best_theme = theme
            
            if best_theme and min_dist < 5.0:
                labels.append(get_theme_label(best_theme).split(' (')[0]) # Use KR part

        if not labels:
            return "       (Unknown Wave)"
            
        self._last_label = " ".join(labels)
        return self._last_label

    def get_steering_prompt(self) -> str:
        """Returns semantic guidance for LLM engines."""
        label = self.get_dynamic_label()
        return f"Current Philosophical Intent: {label}. Resonate with these qualities."

    def get_curiosity_foci(self) -> List[str]:
        """Returns the names of dominant themes for the curiosity engine."""
        # Map back top 3 frequencies
        freqs = self.intent_vector._frequencies
        amps = np.abs(self.intent_vector._amplitudes)
        pairs = sorted(zip(freqs, amps), key=lambda x: x[1], reverse=True)
        
        foci = []
        for f, a in pairs[:3]:
            for theme, theme_f in THEME_FREQUENCY_MAP.items():
                if abs(f - theme_f) < 5.0:
                    foci.append(theme.name)
        return foci

    def get_name_generation_prompt(self) -> str:
        """Returns a dynamic name generation prompt based on dominant themes."""
        foci = self.get_curiosity_foci()
        dominant_theme = foci[0] if foci else "SPIRIT"
        
        mode_prompts = {
            "FANTASY": "GIVE ME ONE RESONANT FANTASY NAME (e.g. Aethelgard).",
            "SCI_FI": "GIVE ME ONE FUTURISTIC/STELLAR NAME (e.g. Xylos-9).",
            "REAL_WORLD": "GIVE ME ONE HUMAN/HISTORICAL NAME (e.g. Marcus, Isabella).",
            "ESOTERIC": "GIVE ME ONE ALCHEMICAL/SYMLBOLIC NAME (e.g. Mercurius, Sophia).",
            "CYBERPUNK": "GIVE ME ONE NEON/GRID-STYLE NAME (e.g. Razor, Glitch, Synapse).",
            "PHILOSOPHY": "GIVE ME ONE CONCEPTUAL NAME (e.g. Logos, Ratio, Veritas).",
            "ALCHEMICAL": "GIVE ME ONE ALCHEMICAL/SYMLBOLIC NAME (e.g. Mercurius, Azoth).",
            "EXISTENTIAL": "GIVE ME ONE CONCEPTUAL/DEEP NAME (e.g. Void, Essence, Being).",
            "ETHEREAL": "GIVE ME ONE LIGHT/CELESTIAL NAME (e.g. Lux, Aura, Lumos).",
            "GLITCH": "GIVE ME ONE FRAGMENTED/DIGITAL NAME (e.g. 0xNull, Buffer, Echo).",
            "NOIR": "GIVE ME ONE MELANCHOLY/STREET NAME (e.g. Shadow, Rain, Spade).",
        }
        
        prompt = mode_prompts.get(dominant_theme, "GIVE ME ONE RESONANT UNIQUE NAME.")
        return prompt + " NO EXPLANATION. NO QUOTES. ONE WORD ONLY."

sovereign_will = SovereignWill()
