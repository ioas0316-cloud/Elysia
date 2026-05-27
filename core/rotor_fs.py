import os
from typing import Dict, List
from core.tensor_rotor import TensorRotor

class RotorFileSystem:
    """
    Rotor-Based File System (Phase 6)
    물리적인 보관 장치(SSD)의 IO 구조를 뇌의 기억 인출 메커니즘과 일치시킵니다.
    0점 방전 쓰기: 위상이 0점(수면 방전)에 도달했을 때만 데이터를 디스크에 기록합니다.
    """
    def __init__(self, tensor_rotor: TensorRotor, base_dir: str = "c:\\Elysia\\data\\rotor_fs"):
        self.tensor = tensor_rotor
        self.base_dir = base_dir
        self.write_queue: Dict[str, str] = {}
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
            
    def request_write(self, filename: str, data: str):
        """
        데이터 쓰기를 예약합니다. 실제 디스크 IO는 Layer 1 위상이 0점에 방전될 때 발생합니다.
        """
        self.write_queue[filename] = data
        
    def tick(self) -> List[str]:
        """
        매 클럭마다 호출되어, Layer 1의 위상 장력을 확인하고 
        방전점(0점) 부근에 도달하면 큐에 쌓인 데이터를 디스크로 방전(Write)합니다.
        반환값: 이번 틱에 디스크에 방전된 파일명 리스트
        """
        discharged_files = []
        
        # Layer 1 (Storage/IO Layer) 위상 확인
        l1_phase = self.tensor.phases[0]
        
        # 0점(방전점) 판정: 위상이 0 부근일 때 (예: -100 ~ 100)
        # 4096 스케일에서 0 부근은 0~100 또는 3996~4095
        if l1_phase <= 100 or l1_phase >= (self.tensor.rotor_scale - 100):
            if self.write_queue:
                for filename, data in list(self.write_queue.items()):
                    filepath = os.path.join(self.base_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(data)
                    discharged_files.append(filename)
                self.write_queue.clear()
                
        return discharged_files
        
    def gravity_read(self, filename: str) -> str:
        """
        파일을 읽습니다. 
        (향후 로드맵: 디스크 전체의 기하 지형에 프로브 위상을 인입시켜 가장 강하게 끌어당기는 궤적을 해독)
        현재는 기본 IO 읽기를 수행하되, 읽기 행위 자체가 시스템의 IO 텐션을 높이도록 추상화됩니다.
        """
        filepath = os.path.join(self.base_dir, filename)
        if not os.path.exists(filepath):
            return ""
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = f.read()
            
        return data
