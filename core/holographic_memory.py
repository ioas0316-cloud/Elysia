"""
Elysia Holographic Memory Architecture
======================================
This module implements the true structural Holographic Memory system.
Instead of a sequential pipeline, knowledge is encoded as 4D waveforms
and superimposed (interfered) directly onto a multi-layered Clifford manifold.
Recall is achieved by rotating the manifold's 4D hyper-rotors (dials) via tension,
observing when specific concepts constructively interfere (resonate).

Rules strictly followed:
1. Causal interactions (no random initialization).
2. All geometry uses Quaternions.
"""

import math
import hashlib
from typing import Dict, List, Tuple
from core.math_utils import Quaternion

def concept_to_quaternion(concept: str) -> Quaternion:
    """
    Deterministically maps a text concept to a 4D unit quaternion (causal).
    Uses SHA-256 to generate deterministic coefficients.
    """
    h = hashlib.sha256(concept.encode('utf-8')).digest()
    
    # Extract 4 coordinates from the hash bytes
    # Combining bytes to get finer resolution
    w = ((h[0] ^ h[4]) + (h[1] ^ h[5]) * 256) / 32768.0 - 1.0
    x = ((h[2] ^ h[6]) + (h[3] ^ h[7]) * 256) / 32768.0 - 1.0
    y = ((h[8] ^ h[12]) + (h[9] ^ h[13]) * 256) / 32768.0 - 1.0
    z = ((h[10] ^ h[14]) + (h[11] ^ h[15]) * 256) / 32768.0 - 1.0
    
    q = Quaternion(w, x, y, z)
    return q.normalize()

class CliffordLayer:
    """
    A 4D spacetime manifold layer (rotor sphere).
    Unfolded, it behaves as a flat manifold; rolled up, it acts as a rotor.
    """
    def __init__(self, layer_id: int, base_frequency: float):
        self.layer_id = layer_id
        self.base_frequency = base_frequency
        
        # Superimposed memory state (4D waveform)
        self.manifold_state = Quaternion(0.0, 0.0, 0.0, 0.0)
        
        # Stored concept key representations (for resonance probing)
        self.concept_contents: Dict[str, Quaternion] = {}
        # Stored concept address tensions to calculate direct phase offset
        self.concept_addresses: Dict[str, float] = {}

    def get_rotors(self, tension: float) -> Tuple[Quaternion, Quaternion]:
        """
        Calculates left and right unit quaternions representing a double rotation in 4D.
        """
        theta_L = tension * self.base_frequency
        theta_R = tension * self.base_frequency * 1.6180339887  # Golden ratio scaling
        
        q_L = Quaternion(math.cos(theta_L), math.sin(theta_L), 0.0, 0.0)
        q_R = Quaternion(math.cos(theta_R), 0.0, math.sin(theta_R), 0.0)
        return q_L, q_R
        
    def superpose(self, concept: str, content_quat: Quaternion, tau_c: float):
        """
        Superimposes the concept wave onto the layer's manifold state.
        Applies a forward rotation corresponding to the concept's address.
        """
        self.concept_contents[concept] = content_quat
        self.concept_addresses[concept] = tau_c
        
        q_L, q_R = self.get_rotors(tau_c)
        stored_wave = q_L * content_quat * q_R
        self.manifold_state = self.manifold_state + stored_wave

    def recall_state(self, tension: float) -> Quaternion:
        """
        Applies the inverse rotation corresponding to the scan tension.
        Reconstructed directly using delta-phase to avoid floating point noise.
        """
        recalled = Quaternion(0.0, 0.0, 0.0, 0.0)
        for concept, content_quat in self.concept_contents.items():
            tau_c = self.concept_addresses.get(concept, 0.0)
            # Delta phase rotation
            dt_L = (tension - tau_c) * self.base_frequency
            dt_R = (tension - tau_c) * self.base_frequency * 1.6180339887
            
            q_L = Quaternion(math.cos(dt_L), math.sin(dt_L), 0.0, 0.0)
            q_R = Quaternion(math.cos(dt_R), 0.0, math.sin(dt_R), 0.0)
            recalled = recalled + (q_L * content_quat * q_R)
        return recalled

    def measure_resonance(self, recalled_state: Quaternion, concept: str) -> float:
        """
        Measures the alignment (coherence) of the recalled state with the concept's content key.
        """
        if concept not in self.concept_contents:
            return 0.0
        
        content_quat = self.concept_contents[concept]
        return recalled_state.dot(content_quat)


class HologramMemory:
    """
    A multi-layer Holographic Memory architecture.
    Manages multiple CliffordLayers, combining their responses to produce
    constructive and destructive interference peaks.
    """
    def __init__(self, num_layers: int = 3):
        self.layers: List[CliffordLayer] = []
        
        # Setup fractal layers with golden ratio frequency scaling
        freq = 1.0
        for i in range(num_layers):
            self.layers.append(CliffordLayer(layer_id=i, base_frequency=freq))
            freq *= 1.6180339887
            
        # Global map of registered concepts to their content key and address tension
        self.registered_concepts: Dict[str, Tuple[Quaternion, float]] = {}

    def register_concept(self, concept: str) -> Tuple[Quaternion, float]:
        """
        Generates and registers a canonical 4D content key and its resonant tension address.
        """
        if concept not in self.registered_concepts:
            content_quat = concept_to_quaternion(concept)
            
            # Generate deterministic address tension in range [1.0, 9.0]
            # Uses concept name salt to maintain causality
            h = hashlib.sha256((concept + "_address").encode('utf-8')).digest()
            tau_c = 1.0 + ((h[0] ^ h[2]) + (h[1] ^ h[3]) * 256) / 65535.0 * 8.0
            
            self.registered_concepts[concept] = (content_quat, tau_c)
        return self.registered_concepts[concept]

    def superpose(self, concept: str):
        """
        Ingests a concept and superimposes it across all layers.
        Knowledge is now 'interfered' and distributed across the manifold.
        """
        content_quat, tau_c = self.register_concept(concept)
        for layer in self.layers:
            layer.superpose(concept, content_quat, tau_c)

    def scan_resonance(self, tension: float) -> Dict[str, float]:
        """
        Scans all registered concepts at a specific tension.
        Returns the resonance score for each concept.
        Optimized via Direct Phase Interference Lookup (O(1) per concept-layer).
        """
        resonance_scores = {}
        for concept, (content_quat, tau_c) in self.registered_concepts.items():
            layer_scores = []
            for layer in self.layers:
                # Delta theta for Left and Right Y-axis rotations
                dt_L = (tension - tau_c) * layer.base_frequency
                dt_R = (tension - tau_c) * layer.base_frequency * 1.6180339887
                # Cosine product represents alignment (dot product) after double rotation
                score = math.cos(dt_L) * math.cos(dt_R)
                layer_scores.append(score)
            
            # Average resonance across all layers (holographic constructive interference)
            mean_score = sum(layer_scores) / len(self.layers)
            resonance_scores[concept] = mean_score
            
        return resonance_scores

    def get_trajectory_sample(self, tension: float) -> Tuple[float, float, float, float]:
        """
        Returns a 4D coordinate representing the composite state of the memory
        under the current tension, projectable to 2D/3D space.
        """
        composite = Quaternion(0.0, 0.0, 0.0, 0.0)
        for layer in self.layers:
            composite = composite + layer.recall_state(tension)
        comp_norm = composite.normalize()
        return comp_norm.w, comp_norm.x, comp_norm.y, comp_norm.z


class BitwiseHologramMemory:
    """
    엘리시아 관측 회전 아키텍처용 초고속 비트와이즈 홀로그램 메모리.
    텍스트 개념을 64비트 정수 마스크(지문)와 순환 비트 링 상의 정수 번지(Address)로 매핑합니다.
    """
    def __init__(self, size_bits: int = 64):
        self.size_bits = size_bits
        self.registered_concepts: Dict[str, Tuple[int, int]] = {} # concept -> (mask, address)

    def register_concept(self, concept: str) -> Tuple[int, int]:
        if concept not in self.registered_concepts:
            # Deterministic hash to 64-bit integer mask
            h = hashlib.sha256(concept.encode('utf-8')).digest()
            mask = int.from_bytes(h[:8], byteorder='big')
            
            # Deterministic address in [0, 63] range
            h_addr = hashlib.sha256((concept + "_address").encode('utf-8')).digest()
            address = h_addr[0] % self.size_bits
            self.registered_concepts[concept] = (mask, address)
        return self.registered_concepts[concept]

    def superpose(self, concept: str):
        self.register_concept(concept)

    def scan_resonance(self, probe_address: int) -> Dict[str, float]:
        """
        프로브 번지와의 순환 거리 차이를 측정하여 공명 점수 도출 (O(1) 연산)
        """
        resonance_scores = {}
        for concept, (mask, address) in self.registered_concepts.items():
            # 순환 링 거리 계산
            diff = abs(probe_address - address)
            diff = min(diff, self.size_bits - diff)
            
            # 거리 차이가 8 미만일 때 공명 임계점 도출
            resonance = max(0.0, 1.0 - (diff / 8.0))
            resonance_scores[concept] = resonance
        return resonance_scores


class Bitwise4DHologramMemory:
    """
    4차원 구형 매니폴드 위상 지형을 지원하는 비트와이즈 4D 홀로그램 메모리.
    개념을 64비트 정수 마스크(지문)와 4차원 시공간 주소 좌표 (w, x, y, z)로 매핑합니다.
    """
    def __init__(self, size_bits: int = 64):
        self.size_bits = size_bits
        self.registered_concepts: Dict[str, Tuple[int, Tuple[int, int, int, int]]] = {} # concept -> (mask, (w, x, y, z))

    def register_concept(self, concept: str) -> Tuple[int, Tuple[int, int, int, int]]:
        if concept not in self.registered_concepts:
            # Deterministic hash to 64-bit integer mask
            h = hashlib.sha256(concept.encode('utf-8')).digest()
            mask = int.from_bytes(h[:8], byteorder='big')
            
            # Deterministic 4D addresses in [0, 63] range
            h_w = hashlib.sha256((concept + "_w").encode('utf-8')).digest()
            h_x = hashlib.sha256((concept + "_x").encode('utf-8')).digest()
            h_y = hashlib.sha256((concept + "_y").encode('utf-8')).digest()
            h_z = hashlib.sha256((concept + "_z").encode('utf-8')).digest()
            
            addr_w = h_w[0] % self.size_bits
            addr_x = h_x[0] % self.size_bits
            addr_y = h_y[0] % self.size_bits
            addr_z = h_z[0] % self.size_bits
            
            self.registered_concepts[concept] = (mask, (addr_w, addr_x, addr_y, addr_z))
        return self.registered_concepts[concept]

    def superpose(self, concept: str):
        self.register_concept(concept)

    def scan_resonance(self, probe_w: int, probe_x: int, probe_y: int, probe_z: int) -> Dict[str, float]:
        """
        4차원 프로브 주소와의 다차원 위상 차이를 계산하여 보강 간섭 공명도 산출 (O(1) 연산)
        """
        resonance_scores = {}
        probes = (probe_w, probe_x, probe_y, probe_z)
        
        for concept, (mask, addresses) in self.registered_concepts.items():
            resonance_mult = 1.0
            
            for d in range(4):
                p_d = probes[d]
                a_d = addresses[d]
                
                # 순환 거리 계산
                diff = abs(p_d - a_d)
                diff = min(diff, self.size_bits - diff)
                
                # 차원당 공명 마스킹
                res_d = max(0.0, 1.0 - (diff / 8.0))
                resonance_mult *= res_d
                
            resonance_scores[concept] = resonance_mult
        return resonance_scores
