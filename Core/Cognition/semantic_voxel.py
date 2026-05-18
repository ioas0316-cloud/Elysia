import math

class SemanticVoxel:
    def __init__(self, name=None, coords=None, mass=0.0, frequency=0.0, is_anchor=False):
        self.name = name
        self.coords = coords
        self.mass = mass
        self.frequency = frequency
        self.is_anchor = is_anchor
        self.quaternion = None
        self.inbound_edges = []
        self.outbound_edges = []

    def distance_to(self, other_voxel):
        if not self.coords or not other_voxel.coords:
            return float('inf')
        return math.sqrt(sum((a - b)**2 for a, b in zip(self.coords, other_voxel.coords)))
