import zlib
import json
import math
from typing import Dict, Any
from core.math_utils import Quaternion
from core.fractal_rotor import FractalRotor

class TopologyMapper:
    """
    위상 구조도 변환기 (Topological Map Compressor)
    
    결정화된 수많은 가변 로터(객체)들을 메모리에 그대로 두지 않고,
    '데이터를 데이터화'하여 극도로 압축된 위상 구조도(Topology Map)로 변환합니다.
    150GB의 정적 파라미터가 결국 소수의 핵심 파동 방정식과 씨앗(Seed) 맵으로 압축되는 원리입니다.
    """
    
    @staticmethod
    def extract_structural_map(rotor: FractalRotor) -> Dict[str, Any]:
        """로터 트리의 기하학적 뼈대(위상과 텐션)만 추출합니다."""
        w, x, y, z = rotor.lens_offset.elements
        
        # 소수점 4자리로 반올림하여 정보의 불필요한 엔트로피 제거 (양자화/결정화)
        node_map = {
            "p": [round(w, 4), round(x, 4), round(y, 4), round(z, 4)],
            "t": round(rotor.tau, 4)
        }
        
        if rotor.children:
            # 프랙탈 자식 구조를 재귀적으로 맵핑
            node_map["c"] = [TopologyMapper.extract_structural_map(child) for child in rotor.children]
            
        return node_map

    @staticmethod
    def compress_to_topology_seed(rotor: FractalRotor) -> bytes:
        """
        추출된 구조도를 바이트 레벨로 극압축(Data-ization)하여 위상 씨앗(Seed)으로 만듭니다.
        메모리상의 무거운 Python 객체가 단 몇 바이트의 데이터로 치환됩니다.
        """
        structural_map = TopologyMapper.extract_structural_map(rotor)
        json_str = json.dumps(structural_map, separators=(',', ':'))
        
        # Zlib을 통한 구조 맵의 엔트로피 압축
        compressed_seed = zlib.compress(json_str.encode('utf-8'), level=9)
        return compressed_seed

    @staticmethod
    def unfold_from_topology_seed(seed: bytes) -> FractalRotor:
        """
        극압축된 위상 씨앗(Topology Map)으로부터 완전한 가변 로터 트리를 원래대로 펼쳐냅니다(Unfolding).
        저장/전송 시에는 가벼운 Seed 상태로 유지하고, 사유가 필요할 때만 홀로그램처럼 전개합니다.
        """
        json_str = zlib.decompress(seed).decode('utf-8')
        structural_map = json.loads(json_str)
        
        return TopologyMapper._build_rotor_from_map(structural_map)
        
    @staticmethod
    def _build_rotor_from_map(node_map: Dict[str, Any]) -> FractalRotor:
        p = node_map["p"]
        phase = Quaternion(p[0], p[1], p[2], p[3])
        rotor = FractalRotor(lens_offset=phase, tau=node_map["t"])
        
        if "c" in node_map:
            for child_map in node_map["c"]:
                child_rotor = TopologyMapper._build_rotor_from_map(child_map)
                rotor.attach_child(child_rotor)
                
        return rotor
