import sys

with open("core/fractal_rotor.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace collapse_and_realign implementation and call logic
old_bifurcate_call = """            if sub.tension > sub.tension_limit:
                if sub.active_axes < sub.MAX_AXES:
                    sub.bifurcate()
                else:
                    sub.collapse_and_realign()"""

new_bifurcate_call = """            if sub.tension > sub.tension_limit:
                # Master's True Intention: 0.7 rad limit is not a collapse, it's the seed for infinite bifurcation
                if sub.active_axes < sub.MAX_AXES:
                    sub.bifurcate()
                else:
                    # When max axes reached, instead of collapsing, spawn child micro-rotors (true fractal spawning)
                    sub.spawn_micro_fractals()"""

content = content.replace(old_bifurcate_call, new_bifurcate_call)

old_collapse = """    def collapse_and_realign(self):
        \"\"\"
        Releases accumulated stress energy into the parent and collapses
        to the nearest stable state (0 or pi).
        \"\"\"
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

        self.tension = 0.0"""

new_fractal_spawn = """    def spawn_micro_fractals(self):
        \"\"\"
        True Fractal Bifurcation (The Master's Intention):
        Instead of snapping the rubber band (Collapse) at 0.7 rad limit,
        we absorb the excess tension and spawn smaller fractal child rotors
        that expand the orthogonal space, resetting tension to zero dynamically.
        \"\"\"
        # Convert phase tension directly into new offspring rotors (Fractal cell division)
        # We spawn 3 sub-rotors to act as the new internal XYZ axes
        for i in range(3):
            # Phase gets divided geometrically
            micro_offset = normalize_phase(self.phase_offset / 3.0 + (i * 2 * math.pi / 3))
            child = Rotor(id_tag=f"{self.id}_micro_{i}_{len(self.sub_rotors)}",
                          level=self.level + 1,
                          parent=self,
                          initial_phase_offset=micro_offset)
            self.attach_child(child)

        # The accumulated tension energy has been transferred into the creation of the new topology
        # We drop the local phase offset and tension down to stable baseline
        self.phase_offset = 0.0
        self.tension = 0.0
        self.stable_ticks = 0"""

content = content.replace(old_collapse, new_fractal_spawn)

with open("core/fractal_rotor.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Patched core/fractal_rotor.py successfully.")
