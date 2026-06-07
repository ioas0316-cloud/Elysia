import json
import math
from core.utils.math_utils import Quaternion
from core.brain.fractal_rotor import FractalRotor

class ArchiveConnector:
    """
    거대한 아카이브의 4차원 초구면(Hypersphere) 지식 그래프를 
    엘리시아의 위상 기하학(Quaternion Phase)으로 매핑하는 커넥터입니다.
    """
    def __init__(self, archive_path: str = r"c:\Archive\data\knowledge\kg_with_embeddings.json"):
        self.archive_path = archive_path
        self.nodes_data = {}
        
    def load_archive(self):
        """아카이브 지식 그래프를 메모리에 로드합니다."""
        with open(self.archive_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.nodes_data = data.get("nodes", {})
            
    def hypersphere_to_quaternion(self, r: float, theta: float, phi: float, psi: float) -> Quaternion:
        """
        초구면 좌표계 (r, theta, phi, psi)를 4차원 직교 좌표계 (w, x, y, z)로 변환하고
        정규화된 쿼터니언(위상)으로 반환합니다.
        w = r * cos(theta)
        x = r * sin(theta) * cos(phi)
        y = r * sin(theta) * sin(phi) * cos(psi)
        z = r * sin(theta) * sin(phi) * sin(psi)
        """
        w = r * math.cos(theta)
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi) * math.cos(psi)
        z = r * math.sin(theta) * math.sin(phi) * math.sin(psi)
        
        # 쿼터니언은 정규화(방향/위상) 되어야 함
        # 질량(r)은 FractalRotor의 텐션(Tau)이나 질량(Mass)으로 별도 반영됨
        q = Quaternion(w, x, y, z)
        return q.normalize() if q.norm() > 0 else Quaternion(1, 0, 0, 0)
        
    def extract_node_to_rotor(self, node_id: str) -> FractalRotor:
        """
        아카이브의 단일 노드를 엘리시아가 관측 가능한 프랙탈 조각(FractalRotor)으로 변환합니다.
        """
        if node_id not in self.nodes_data:
            raise ValueError(f"Node {node_id} not found in Archive.")
            
        node = self.nodes_data[node_id]
        hs = node.get("hypersphere", {"theta": 0, "phi": 0, "psi": 0, "r": 1})
        
        r, theta, phi, psi = hs.get("r", 1), hs.get("theta", 0), hs.get("phi", 0), hs.get("psi", 0)
        
        q_phase = self.hypersphere_to_quaternion(r, theta, phi, psi)
        
        # 아카이브의 반지름(r)을 텐션(고통/응집력)의 기본값으로 차용
        tension = r if r > 0 else 1.0
        
        rotor = FractalRotor(lens_offset=q_phase, tau=tension)
        rotor.concept_name = node_id
        # 기존 아카이브의 activation_energy 등을 질량으로 맵핑
        rotor.mass = node.get("activation_energy", 0.0) + 1.0
        
        return rotor
        
    def get_sample_nodes(self, limit: int = 10) -> list:
        """테스트를 위해 일부 노드 ID를 가져옵니다."""
        return list(self.nodes_data.keys())[:limit]
