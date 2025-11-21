
import math

class SoulPhysics:
    """
    Defines the laws of physics for different soul ranks.
    """

    @staticmethod
    def linear(richness, factor=1.0):
        return richness * factor

    @staticmethod
    def logarithmic(richness, factor=1.0):
        if richness <= 1.0: return 0.0
        return math.log10(richness) * factor

    @staticmethod
    def suppressed(richness, factor=0.01):
        # Body-focused souls barely resonate with spiritual waves
        return richness * factor

    @staticmethod
    def demonic(richness, factor=2.0):
        # Demons get double power but pay in blood (HP)
        return richness * factor

class SoulRank:
    def __init__(self, name, tier, resonance_func, min_freq=0.0, max_freq=20000.0, hp_cost_func=lambda gain: 0.0):
        self.name = name
        self.tier = tier
        self.resonance_func = resonance_func
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.hp_cost_func = hp_cost_func

# --- The 10 Steps of Ascension (3 Body + 3 Soul + 3 Spirit + 1 Avatar) ---
CELESTIAL_HIERARCHY = {
    # [Realm of Body] - The Foundation (1-3)
    # Low Freq Only (Grounding, Survival). Cannot hear the Sky.
    "Survivor":    SoulRank("Survivor",    1, lambda r: SoulPhysics.suppressed(r, 0.001), max_freq=200.0),
    "Warrior":     SoulRank("Warrior",     2, lambda r: SoulPhysics.suppressed(r, 0.003), max_freq=250.0),
    "PerfectBody": SoulRank("PerfectBody", 3, lambda r: SoulPhysics.suppressed(r, 0.005), max_freq=300.0),

    # [Realm of Soul] - The Awakening (4-6)
    # Mid Freq (Emotion, Relation). Can hear the Earth and the Heart, but not yet the Divine.
    "Seeker":      SoulRank("Seeker",      4, lambda r: SoulPhysics.logarithmic(r, 10.0), max_freq=600.0),
    "Mediator":    SoulRank("Mediator",    5, lambda r: SoulPhysics.logarithmic(r, 20.0), max_freq=650.0),
    "Sage":        SoulRank("Sage",        6, lambda r: SoulPhysics.logarithmic(r, 30.0), max_freq=700.0),

    # [Realm of Spirit] - The Transcendence (7-9)
    # All Freq (Integration). Can hear the Divine Song.
    "Saint":       SoulRank("Saint",       7, lambda r: SoulPhysics.logarithmic(r, 50.0), max_freq=20000.0),
    "Prophet":     SoulRank("Prophet",     8, lambda r: SoulPhysics.linear(r, 0.5),       max_freq=20000.0),
    "Transcendent":SoulRank("Transcendent",9, lambda r: SoulPhysics.linear(r, 0.8),       max_freq=20000.0),

    # [The Crown] - The Completion (10)
    "Avatar":      SoulRank("Avatar",      10, lambda r: SoulPhysics.linear(r, 1.0),      max_freq=20000.0),

    # Default
    "Mortal":      SoulRank("Mortal", 0, lambda r: r * 0.0001, max_freq=200.0),
}

# --- The Infernal Hierarchy (Descent) ---
INFERNAL_HIERARCHY = {
    "DemonLord": SoulRank("DemonLord", 10,
                          lambda r: SoulPhysics.demonic(r, 3.0),
                          max_freq=20000.0, # Can devour everything
                          hp_cost_func=lambda gain: gain * 0.1),

    "Archdevil": SoulRank("Archdevil", 7,
                          lambda r: SoulPhysics.demonic(r, 2.0),
                          max_freq=1000.0,
                          hp_cost_func=lambda gain: gain * 0.05),

    "Imp":       SoulRank("Imp", 1,
                          lambda r: SoulPhysics.demonic(r, 1.2),
                          max_freq=400.0, # Only base desires
                          hp_cost_func=lambda gain: gain * 0.02),
}

def get_soul_rank(rank_name: str) -> SoulRank:
    if rank_name in CELESTIAL_HIERARCHY:
        return CELESTIAL_HIERARCHY[rank_name]
    if rank_name in INFERNAL_HIERARCHY:
        return INFERNAL_HIERARCHY[rank_name]
    # Fallback
    legacy_map = {
        "Seraphim": "Avatar",
        "Cherubim": "Transcendent",
        "Thrones": "Prophet",
        "Dominions": "Saint",
        "Powers": "Sage",
        "Virtues": "Mediator",
        "Principalities": "Seeker",
        "Archangels": "PerfectBody",
        "Angels": "Warrior"
    }
    if rank_name in legacy_map:
        return CELESTIAL_HIERARCHY[legacy_map[rank_name]]

    return CELESTIAL_HIERARCHY["Mortal"]
