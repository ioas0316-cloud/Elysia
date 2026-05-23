"""
Elysia Core Fractal Rotor Logic
===============================
Implements the continuous phase-locking physical equations:
- Phase offset relative to parent rotor
- Pulling force for phase synchronization
- Mechanical tension and collapse/realignment thresholds
"""

import math

def normalize_phase(phase: float) -> float:
    """Normalize phase angle to the range [-pi, pi]."""
    phase = phase % (2.0 * math.pi)
    if phase > math.pi:
        phase -= 2.0 * math.pi
    return phase

class Rotor:
    """
    Rotor representing a phase coordinate in a hierarchical scale.
    """
    def __init__(self, id_tag: str, level: int = 0, parent: 'Rotor' = None, initial_phase_offset: float = 0.0):
        self.id = id_tag
        self.level = level
        self.parent = parent
        self.phase_offset = normalize_phase(initial_phase_offset)
        self.tension = 0.0
        self.sub_rotors = []
        self._global_phase = 0.0

        # Dynamic dimensional state variables
        self.active_axes = 3
        self.MAX_AXES = 8
        self.MIN_AXES = 1
        self.stable_ticks = 0

    def bifurcate(self):
        """Expands active axes (Dimension Split) when local tension is too high."""
        if self.active_axes < self.MAX_AXES:
            self.active_axes += 1
            # Distribute shock: reduce current phase offset as energy flows into new dimension
            self.phase_offset = normalize_phase(self.phase_offset * 0.5)
            self.tension = abs(self.phase_offset)
            self.stable_ticks = 0

    def compress(self):
        """Locks unused dimensions and reduces active axes when long-term stability is detected."""
        if self.active_axes > self.MIN_AXES:
            self.active_axes -= 1
            self.stable_ticks = 0

    @property
    def tension_limit(self) -> float:
        """Dynamically scales tension limit based on rotor hierarchical level."""
        return (math.pi / 2.0) / (self.level + 1.0)

    @property
    def current_phase(self) -> float:
        """Dynamically resolves absolute phase by summing up the hierarchy."""
        if self.parent:
            return normalize_phase(self.parent.current_phase + self.phase_offset)
        return normalize_phase(self._global_phase)

    def attach_child(self, child_rotor: 'Rotor'):
        self.sub_rotors.append(child_rotor)

    def observe(self, global_rotation_delta: float = 0.0):
        """
        Pulling force is exerted by parent to align child phases (Phase-Locking).
        Tension builds up in case of high phase differences, triggering bifurcation or collapse.
        """
        if self.parent is None:
            self._global_phase = normalize_phase(self._global_phase + global_rotation_delta)

        for sub in self.sub_rotors:
            # Dynamic coupling strength based on parent-child alignment coherence
            alignment = math.cos(sub.phase_offset)
            pull_strength = 0.05 + 0.15 * max(0.0, alignment)
            
            pull_force = math.sin(sub.phase_offset) * pull_strength
            sub.phase_offset = normalize_phase(sub.phase_offset - pull_force)
            sub.tension = abs(sub.phase_offset)

            # Relieve stress / Dimensional control
            if sub.tension > sub.tension_limit:
                # 1. Try to bifurcate first to distribute shock into high dimension
                if sub.active_axes < sub.MAX_AXES:
                    sub.bifurcate()
                else:
                    # 2. If already at MAX_AXES and still exceeding limit, trigger collapse
                    sub.collapse_and_realign()
            else:
                # Track stability ticks for dimensional compression
                if sub.tension < sub.tension_limit * 0.2:
                    sub.stable_ticks += 1
                    if sub.stable_ticks >= 5:
                        sub.compress()
                else:
                    sub.stable_ticks = 0

            # Recursively observe descendants
            sub.observe()

    def collapse_and_realign(self):
        """
        Releases accumulated stress energy into the parent and collapses
        to the nearest stable state (0 or pi).
        """
        if self.parent:
            impact = self.tension * 0.5
            if self.phase_offset > 0:
                self.parent.phase_offset = normalize_phase(self.parent.phase_offset - impact)
            else:
                self.parent.phase_offset = normalize_phase(self.parent.phase_offset + impact)

        # Drop to stable attractor point (0 or pi)
        if abs(self.phase_offset) > math.pi / 2.0:
            self.phase_offset = math.pi if self.phase_offset > 0 else -math.pi
        else:
            self.phase_offset = 0.0
            
        self.tension = 0.0

def phase_bar(phase: float, tension: float) -> str:
    normalized = (phase + math.pi) / (2 * math.pi)
    width = 20
    pos = int(normalized * width)
    bar = ['-'] * width
    marker = 'O' if tension < 0.5 else ('X' if tension < 1.0 else '⚡')
    if 0 <= pos < width:
        bar[pos] = marker
    return "".join(bar)

def display_rotors(rotor: Rotor, prefix=""):
    phase_deg = math.degrees(rotor.current_phase)
    offset_deg = math.degrees(rotor.phase_offset)
    bar = phase_bar(rotor.phase_offset, rotor.tension)

    if rotor.parent is None:
        print(f"│ {prefix}{rotor.id:<5} [CORE] Phase: {phase_deg:6.1f}° | Global: {bar} │")
    else:
        print(f"│ {prefix}{rotor.id:<5} [T: {rotor.tension:4.2f}] Offset: {offset_deg:6.1f}° | Wave: [{bar}] │")
    
    for i, sub in enumerate(rotor.sub_rotors):
        branch = "├─" if i < len(rotor.sub_rotors)-1 else "└─"
        display_rotors(sub, prefix + branch)
