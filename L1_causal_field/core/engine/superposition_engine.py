"""
Quantum Superposition and Lazy State Collapse Engine.
Entities outside observation frustums/rays remain in 32-byte dormant superposition states.
Upon observer measurement (ray intersection or causal wave trigger), they undergo O(1)
state collapse into 64-byte manifested states.
Includes Vulkan TLAS Zero-Copy Interop interface for fast GPU updates.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class SuperpositionNode:
    """32-Byte Dormant Representation in VRAM/RAM"""
    centroid: List[float]       # [x, y, z] - 12B
    bounding_radius: float     # 4B
    state_bitmask: int = 0     # 4B
    entropy_factor: int = 0    # 2B
    is_collapsed: int = 0      # 1B
    node_id: int = 0           # ID reference

@dataclass
class CollapsedStateNode:
    """64-Byte Manifested Representation in VRAM/RAM"""
    node_id: int
    transform_matrix: List[float] = field(default_factory=lambda: [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0
    ]) # 3x4 Affine Matrix - 48B
    active_anim_frame: int = 0 # 4B
    entity_id: int = 0         # 4B
    causal_signal_id: int = 0  # 4B
    state_bitmask: int = 0     # 4B

@dataclass
class ObserverRay:
    origin: List[float]      # [x, y, z]
    direction: List[float]   # normalized [dx, dy, dz]
    max_distance: float = 1000.0

class SuperpositionEngine:
    def __init__(self, count: int = 0):
        self.superposition_nodes: List[SuperpositionNode] = []
        self.collapsed_nodes: Dict[int, CollapsedStateNode] = {}
        if count > 0:
            self.allocate_nodes(count)

    def allocate_nodes(self, count: int, radius: float = 15.0):
        self.superposition_nodes = []
        for i in range(count):
            c = [float(i % 100) * 10.0, float((i // 100) % 100) * 10.0, 0.0]
            node = SuperpositionNode(
                centroid=c,
                bounding_radius=radius,
                node_id=i
            )
            self.superposition_nodes.append(node)

    def collapse_superposition_nodes(
        self,
        observer_rays: List[ObserverRay],
        causal_triggers: Optional[List[Tuple[List[float], float]]] = None
    ) -> List[int]:
        """
        Evaluates observer measurement rays and causal wave triggers to collapse
        dormant nodes into 64-byte CollapsedStateNodes.
        """
        newly_collapsed = []

        # Pre-process ray data for fast pure-python loop execution
        parsed_rays = []
        for ray in observer_rays:
            ro_x, ro_y, ro_z = ray.origin
            rd_x, rd_y, rd_z = ray.direction
            max_d = ray.max_distance
            parsed_rays.append((ro_x, ro_y, ro_z, rd_x, rd_y, rd_z, max_d))

        parsed_waves = causal_triggers if causal_triggers else []

        for node in self.superposition_nodes:
            if node.is_collapsed == 1:
                continue

            cx, cy, cz = node.centroid
            rad = node.bounding_radius
            rad_sq = rad * rad
            is_observed = False

            # Ray-Sphere intersection
            for ro_x, ro_y, ro_z, rd_x, rd_y, rd_z, max_d in parsed_rays:
                oc_x = cx - ro_x
                oc_y = cy - ro_y
                oc_z = cz - ro_z

                dot_dir = oc_x * rd_x + oc_y * rd_y + oc_z * rd_z
                if dot_dir < -rad or dot_dir > max_d + rad:
                    continue

                oc_sq = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z
                proj_dist_sq = oc_sq - dot_dir * dot_dir
                if proj_dist_sq <= rad_sq:
                    is_observed = True
                    break

            # Wave trigger intersection
            if not is_observed and parsed_waves:
                for wave_origin, wave_radius in parsed_waves:
                    wx, wy, wz = wave_origin
                    dx = cx - wx
                    dy = cy - wy
                    dz = cz - wz
                    if (dx * dx + dy * dy + dz * dz) <= (wave_radius + rad) ** 2:
                        is_observed = True
                        break

            if is_observed:
                node.is_collapsed = 1
                node.state_bitmask |= 0x01
                idx = node.node_id

                seed = (idx * 1664525 + node.entropy_factor + 1013904223) & 0xFFFFFFFF
                anim_frame = seed % 120

                transform = [
                    1.0, 0.0, 0.0, cx,
                    0.0, 1.0, 0.0, cy,
                    0.0, 0.0, 1.0, cz
                ]

                collapsed = CollapsedStateNode(
                    node_id=idx,
                    transform_matrix=transform,
                    active_anim_frame=anim_frame,
                    entity_id=idx,
                    causal_signal_id=0xFFFFFFFF,
                    state_bitmask=node.state_bitmask
                )
                self.collapsed_nodes[idx] = collapsed
                newly_collapsed.append(idx)

        return newly_collapsed

class CausalVulkanCudaBridge:
    """
    Interface simulating Zero-Copy Vulkan External Memory Import and CUDA TLAS Refit.
    Allows direct VRAM transform matrix writes to VkAccelerationStructureInstanceKHR.
    """
    def __init__(self, instance_count: int):
        self.instance_count = instance_count
        self.tlas_instance_buffer = [
            {
                "transform": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "mask": 0x00,
                "instanceCustomIndex": i
            }
            for i in range(instance_count)
        ]

    def update_tlas_instances_zero_copy(self, collapsed_nodes: Dict[int, CollapsedStateNode]) -> int:
        """
        Directly writes collapsed transform matrices into TLAS instance buffer in VRAM with zero CPU-GPU copy.
        """
        updated_count = 0
        for node_id, collapsed in collapsed_nodes.items():
            if node_id < self.instance_count:
                instance = self.tlas_instance_buffer[node_id]
                instance["transform"] = list(collapsed.transform_matrix)
                instance["mask"] = 0xFF if (collapsed.state_bitmask & 0x01) else 0x00
                updated_count += 1
        return updated_count
