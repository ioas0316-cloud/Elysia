import math

class SemanticVoxel:
    def __init__(self, name=None, coords=None, mass=0.0, frequency=0.0, is_anchor=False):
        self.name = name
        self.coords = coords
        self.mass = mass
        self.frequency = frequency
        self.is_anchor = is_anchor
        self.quaternion = None
        if coords:
            from pyquaternion import Quaternion
            if len(coords) == 4:
                self.quaternion = Quaternion(coords[3], coords[0], coords[1], coords[2]) # w, x, y, z
            else:
                self.quaternion = Quaternion(1.0, 0.0, 0.0, 0.0)
        else:
            from pyquaternion import Quaternion
            self.quaternion = Quaternion(1.0, 0.0, 0.0, 0.0)

        self.inbound_edges = []
        self.outbound_edges = []
        self.base_mass = mass

    @property
    def dynamic_mass(self):
        return self.base_mass + (len(self.inbound_edges) * 5.0)

    def distance_to(self, other_voxel):
        if not self.coords or not other_voxel.coords:
            return float('inf')
        return math.sqrt(sum((a - b)**2 for a, b in zip(self.coords, other_voxel.coords)))
