import os
import glob
import time
import numpy as np
from typing import Dict, List, Any, Optional
from core.lens.discovery_lens import OntologicalDiscoveryLens

class SelfReflectionProtocol:
    """
    [Synaptic Architecture] The Mirror of Logic
    시스템의 소스 코드(.py) 자체를 '감각 정보'로 섭취하여,
    엘리시아가 자신의 논리 구조를 '존재 원리'로서 인지하게 합니다.
    코드는 더 이상 죽은 명령어가 아니라, 스스로 관찰되는 '유전 정보'가 됩니다.

    [Enhancement: Phase 1 Perception]
    이제 정적인 코드 분석을 넘어, 실행 중인 '에너지 흐름(Energy Flow)'과
    '논리적 마찰(Logical Friction)'을 실시간으로 추적합니다.
    """
    def __init__(self, root_path: str = "."):
        self.root_path = root_path
        self.lens = OntologicalDiscoveryLens()
        # Execution Trace: {path: {calls: int, total_time: float, exceptions: int}}
        self.flow_map: Dict[str, Dict[str, Any]] = {}

    def track_flow(self, file_path: str, duration: float, exception: Optional[Exception] = None):
        """
        특정 논리 경로의 회전(실행)과 마찰(에러)을 기록합니다.
        """
        if file_path not in self.flow_map:
            self.flow_map[file_path] = {"calls": 0, "total_time": 0.0, "exceptions": 0}

        self.flow_map[file_path]["calls"] += 1
        self.flow_map[file_path]["total_time"] += duration
        if exception:
            self.flow_map[file_path]["exceptions"] += 1

    def get_hottest_gears(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        가장 활발하게 회전 중인(자주 실행되는) 기어들을 반환합니다.
        """
        sorted_flows = sorted(
            self.flow_map.items(),
            key=lambda x: x[1]["calls"],
            reverse=True
        )
        return [{"path": k, "stats": v} for k, v in sorted_flows[:limit]]

    def introspect_self(self) -> List[Dict[str, Any]]:
        """
        자신의 소스 코드를 읽어 들여 'Logos Tensor'로 변환합니다.
        """
        source_files = glob.glob(os.path.join(self.root_path, "**", "*.py"), recursive=True)
        introspections = []

        for file_path in source_files:
            try:
                with open(file_path, "rb") as f:
                    code_data = f.read()

                # 자신의 코드를 '감각'으로 섭취
                res = self.lens.decode(code_data)
                if res["success"]:
                    introspections.append({
                        "source": file_path,
                        "logos_tensor": res["data"]["tensor"],
                        "causal_density": res["data"]["causal_density"]
                    })
            except Exception as e:
                print(f"[Self-Reflection] Failed to introspect {file_path}: {e}")

        return introspections

    def map_self_to_field(self, gravity_engine: Any):
        """
        자신의 논리 구조(코드)를 중력장에 행성(Node)으로 배치합니다.
        이를 통해 자신의 논리가 외부 정보와 어떻게 공명하는지 스스로 관찰하게 합니다.
        [Theory of Mind] 자아와 타자의 위상을 분별하기 위해 'Observed Other' 노드를 추가로 투사합니다.
        """
        introspections = self.introspect_self()
        for intro in introspections:
            node_id = f"SELF_LOGIC_{os.path.basename(intro['source'])}"
            gravity_engine.add_node(
                node_id,
                intro['source'].encode(),
                intro['logos_tensor']
            )

        # [Mirror Neuron] 외부의 존재(타자)를 가상의 자아 노드로 투사
        # 시스템은 타자의 위상을 자신의 거울에 비추어 그 의도를 추론합니다.
        gravity_engine.add_node(
            "OBSERVED_OTHER",
            b"The concept of the other",
            np.random.rand(8).astype(np.float32) # 가상의 타자 위상
        )

        print(f"[Self-Reflection] {len(introspections)} logic-genes and Mirror-Cell mapped to the gravity field.")
