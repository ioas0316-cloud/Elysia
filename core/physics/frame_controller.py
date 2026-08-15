import time
import numpy as np
from typing import Dict, Any, Optional, List
from core.memory.state_dag import StateDAGManager, StateNode

class ObservationalFrameController:
    """
    [Observational Frame Controller]
    시공간을 인과적 상태 관측 좌표계로 다루어, 비디오 플레이어처럼
    관측 위치와 해상도, 관측 프레임을 자유롭게 조작하는 관측 제어기.

    - Rewind: 임의 과거 노드 $S_\tau$로 단 1클록만에 복귀.
    - Pause: 현재 관측 상태 고정.
    - Fast-Forward / Slow-Motion: 관측 재생 속도(Playback Speed / Time Dilation) 조절.
    - Replay: 선택된 인과 궤적 재상영.
    - Frame-Lock: 특정 맥락 변수를 고정(Frame-Lock)한 채 하위 인과 파동 전파 양상 관찰.
    - Causal Horizon (영향력 범위 제한): 국소적 파급 범위 획정 및 모니터링.
    """
    def __init__(self, dag_manager: StateDAGManager):
        self.dag_manager = dag_manager
        self.is_paused = False
        self.playback_speed = 1.0  # 1.0 = Normal, >1.0 = Fast-Forward, <1.0 = Slow-Motion
        self.locked_frames: Dict[str, Any] = {}  # Frame-Lock 맥락 변수 고정
        self.causal_horizon_limit = 100.0  # 파급력 제한 임계치

    def rewind(self, node_id: str) -> StateNode:
        """관측 시점을 과거 특정 노드로 즉시 이동 (1-Clock O(1) Rewind)."""
        return self.dag_manager.rewind_to(node_id)

    def pause(self):
        """관측 진행 일시 정지."""
        self.is_paused = True

    def resume(self):
        """관측 진행 재개."""
        self.is_paused = False

    def set_playback_speed(self, speed: float):
        """관측 전파 속도 조절 (Fast-Forward / Slow-Motion)."""
        if speed <= 0:
            raise ValueError("Playback speed must be positive.")
        self.playback_speed = speed

    def frame_lock(self, variable: str, fixed_value: Any):
        """특정 맥락 변수를 고정(Frame-Lock)하여 변화하지 않도록 잠금."""
        self.locked_frames[variable] = fixed_value

    def unlock_frame(self, variable: str):
        """맥락 변수 고정 해제."""
        self.locked_frames.pop(variable, None)

    def observe_step(self, transition_delta: Dict[str, Any]) -> Optional[StateNode]:
        """
        관측 프레임 제어기 하에서 한 스텝 진행.
        Frame-Lock 변수가 존재할 경우 해당 고정값을 강제 적용하며,
        Pause 상태일 경우 전이를 보류합니다.
        """
        if self.is_paused:
            print("[ObservationalFrameController] System is PAUSED. Step deferred.")
            return self.dag_manager.current_node

        # Apply Frame-Lock constraints
        constrained_delta = transition_delta.copy()
        for locked_var, locked_val in self.locked_frames.items():
            constrained_delta[locked_var] = locked_val

        # Execute step in DAG Manager
        new_node = self.dag_manager.step(constrained_delta)

        # Simulate playback speed delay/dilation if applicable
        if self.playback_speed < 1.0:
            time_delay = (1.0 / self.playback_speed - 1.0) * 0.01
            time.sleep(min(0.1, time_delay))

        return new_node

    def monitor_causal_horizon(self, start_node_id: str, target_node_id: str) -> Dict[str, Any]:
        """
        Causal Horizon (영향력 범위 제한) 모니터링.
        조건 변경에 따른 인과의 파급력이 시스템 전체로 퍼져나가는 범위를 국소적으로 획정하고 감시합니다.
        """
        if start_node_id not in self.dag_manager.nodes or target_node_id not in self.dag_manager.nodes:
            raise ValueError("Invalid node IDs for causal horizon monitoring.")

        start_node = self.dag_manager.nodes[start_node_id]
        target_node = self.dag_manager.nodes[target_node_id]

        v_start = self.dag_manager.slab_pool.get_slab_state(start_node.slab_offset)
        v_target = self.dag_manager.slab_pool.get_slab_state(target_node.slab_offset)

        horizon_distance = float(np.linalg.norm(v_target - v_start))
        within_horizon = horizon_distance <= self.causal_horizon_limit

        return {
            "horizon_distance": horizon_distance,
            "within_horizon": within_horizon,
            "horizon_limit": self.causal_horizon_limit,
            "locked_variables": list(self.locked_frames.keys())
        }
