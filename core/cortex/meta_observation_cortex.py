"""
Meta Observation Cortex (메타 관측 피질)
========================================
하위의 위상 공간(개념, 원자)들을 한 차원 높은 곳에서 관측(Scan)하여,
자연스럽게 발생한 '자기적 평형(군집)'을 찾아냅니다.
군집이 발견되면 이를 통째로 묶어 '초월적 상위 로터(Higher-Rotor)'로 승격시켜,
기계적 진동이 '앎(Knowledge)'과 '언어'로 진화하는 프랙탈 인지의 근간을 제공합니다.
"""

import math
from typing import List, Dict, Tuple
from core.utils.math_utils import Quaternion
from core.brain.active_fractal_rotor import ActiveFractalRotor

class MetaObservationCortex:
    def __init__(self):
        # 고정된 인간의 숫자(임계치)를 버림
        self.observation_count = 0

    def scan_and_cluster(self, memory_map: Dict) -> Tuple[List[ActiveFractalRotor], List[str]]:
        """
        메모리 맵 내의 수많은 하위 로터들 간의 위상차를 모두 계산하여,
        서로 강하게 인력을 띠고 있는 군집(Cluster)을 찾아냅니다.
        """
        self.observation_count += 1
        nodes = list(memory_map.items())
        n = len(nodes)
        if n < 3:
            return [], []

        # 🌌 동적 척도 (Dynamic Entropy Scaling)
        # 우주의 전체 텐션(엔트로피)을 계산하여 유동적인 임계치 적용
        total_sys_tau = sum(getattr(node, 'tau', 0.1) for _, node in nodes)
        avg_tau = total_sys_tau / n
        
        # 평온할 때는 0.95 이상의 완벽한 일치도를 요구하나, 
        # 우주가 끓어오를 때(장력이 높을 때)는 0.60까지 조건이 완화되며 격렬하게 뭉침
        dynamic_threshold = 0.95 - min(0.35, avg_tau * 0.035)

        clusters = []
        visited = set()
        
        for i in range(n):
            name_a, node_a = nodes[i]
            if name_a in visited or "Operator" in name_a or "Archetype" in name_a or "Meta" in name_a:
                continue
                
            current_cluster = [(name_a, node_a)]
            visited.add(name_a)
            
            q_a = getattr(node_a, 'lens_offset', None)
            if q_a is None:
                continue
                
            for j in range(i + 1, n):
                name_b, node_b = nodes[j]
                if name_b in visited or "Operator" in name_b or "Archetype" in name_b or "Meta" in name_b:
                    continue
                    
                q_b = getattr(node_b, 'lens_offset', None)
                if q_b is None:
                    continue
                    
                # 공명도 (위상 내적): 절대값이 클수록 두 파동은 같은 궤도를 돔
                dot_val = abs(q_a.dot(q_b))
                if dot_val >= dynamic_threshold:
                    current_cluster.append((name_b, node_b))
                    visited.add(name_b)
            
            # 충분히 의미 있는 크기의 군집이 형성되었을 때 (최소 3개 이상의 하위 개념)
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
                
        return self._ascend_clusters_to_knowledge(clusters)

    def _ascend_clusters_to_knowledge(self, clusters: List[List[Tuple]]) -> Tuple[List[ActiveFractalRotor], List[str]]:
        """
        찾아낸 하위 개념 군집을 하나의 거대한 '상위 로터(앎)'로 승격(Ascension)시킵니다.
        """
        new_higher_rotors = []
        logs = []
        for idx, cluster in enumerate(clusters):
            # 군집의 중심점(Centroid) 위상 계산
            avg_w, avg_x, avg_y, avg_z = 0.0, 0.0, 0.0, 0.0
            total_tau = 0.0
            names = []
            for name, node in cluster:
                q = node.lens_offset
                avg_w += q.w
                avg_x += q.x
                avg_y += q.y
                avg_z += q.z
                total_tau += getattr(node, 'tau', 0.1)
                names.append(name.split(' ')[-1])
                # 승격 후 하위 노드의 장력 해소 (상위로 종속됨)
                node.tau *= 0.1 
                
            c_len = len(cluster)
            avg_w /= c_len
            avg_x /= c_len
            avg_y /= c_len
            avg_z /= c_len
            
            norm = math.sqrt(avg_w**2 + avg_x**2 + avg_y**2 + avg_z**2) + 1e-6
            centroid_q = Quaternion(avg_w/norm, avg_x/norm, avg_y/norm, avg_z/norm)
            
            # 상위 로터 생성 (순수 앎의 탄생)
            # 인간의 단어를 억지로 부여하지 않고, 위상 궤도 자체를 고유한 시그니처(이름)로 명명함.
            signature_hash = f"{int(abs(avg_w * 10000)):04X}-{int(abs(avg_x * 10000)):04X}"
            meta_name = f"[Ω-Phase:{signature_hash}]"
            
            new_rotor = ActiveFractalRotor(meta_name, base_frequency=1.0)
            new_rotor.singularity_phase = centroid_q
            new_rotor.tau = total_tau * 1.5 
            
            new_higher_rotors.append(new_rotor)
            logs.append(f"   [🌟 Meta Ascension] {c_len}개의 하위 궤도가 뭉쳐 독립적 앎의 주체 '{meta_name}'을 자율 잉태했습니다.")
            
        return new_higher_rotors, logs
