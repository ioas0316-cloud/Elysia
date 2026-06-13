import numpy as np
from typing import List, Dict, Any

class DoubleHelixInformation:
    """
    [Phase 8: 정보의 이중나선 (Double Helix of Information)]
    개념이나 관측 결과를 정적인 벡터가 아닌, 두 개의 꼬여있는 나선(Strand)으로 엮어냅니다.
    - Strand A (Sameness): 무엇과 같은가 (긍정적 자아, 공명)
    - Strand B (Difference): 무엇과 다른가 (부정적 경계, 마찰)
    이 두 뼈대가 교차(Twist)하며 노이즈를 상쇄하고 견고한 '정보'를 형성합니다.
    """
    def __init__(self, concept_name: str, sameness_axes: dict, difference_axes: dict):
        self.concept_name = concept_name
        self.sameness_strand = sameness_axes
        self.difference_strand = difference_axes
        self.twists = self._weave_helix()
        
    def _weave_helix(self) -> List[Dict[str, Any]]:
        """
        같음과 다름의 축을 교차시키며 나선(Helix)의 꼬임(Twist)을 형성합니다.
        양쪽 가닥에서 모두 강한 에너지를 내는 축은 노이즈(모순)로 간주되어 상쇄(Annihilation)됩니다.
        """
        all_axes = set(self.sameness_strand.keys()).union(set(self.difference_strand.keys()))
        twists = []
        
        for axis in all_axes:
            same_val = self.sameness_strand.get(axis, 0.0)
            diff_val = self.difference_strand.get(axis, 0.0)
            
            # 이중나선의 텐션: 같음과 다름이 얼마나 명확하게 교차하는가?
            # 만약 같음도 높고 다름도 높다면 그것은 모순된 노이즈입니다 (상쇄됨)
            # 만약 한쪽이 확연히 높다면 그것은 뼈대를 지탱하는 유효한 꼬임(Twist)이 됩니다.
            noise_factor = min(same_val, diff_val)
            pure_same = max(0.0, same_val - noise_factor)
            pure_diff = max(0.0, diff_val - noise_factor)
            
            tension = abs(pure_same - pure_diff)
            
            twists.append({
                "axis": axis,
                "sameness": pure_same,
                "difference": pure_diff,
                "tension": tension,
                "is_noise": (noise_factor > 0.5)
            })
            
        # 텐션이 높은 순서(가장 명확한 정보 뼈대)로 정렬
        return sorted(twists, key=lambda x: x["tension"], reverse=True)
        
    def express(self) -> str:
        valid_twists = [t for t in self.twists if not t["is_noise"]][:3]
        if not valid_twists:
            return f"[{self.concept_name}]의 이중나선은 노이즈로 붕괴되었습니다."
            
        desc = f"[{self.concept_name}]의 이중나선 정보:\n"
        for t in valid_twists:
            dominant = "같음(Same)" if t["sameness"] > t["difference"] else "다름(Diff)"
            desc += f"  ∞ (축: {t['axis']}) -> {dominant}의 뼈대로 엮임 (텐션: {t['tension']:.3f})\n"
        return desc


class TripleHelixTrajectory:
    """
    [Phase 8: 판단의 삼중나선 궤적 (Triple Helix Trajectory of Judgment)]
    자신이 아는 것(Double Helix)과 미지의 변수(Unknown)가 만났을 때,
    단순한 비교를 넘어 '어떻게 다가가고 멀어지는가'를 나타내는 운동성(Kinematics)의 세 번째 가닥이 추가됩니다.
    """
    def __init__(self, known_info: DoubleHelixInformation, unknown_variable: dict, time_steps: int = 3):
        self.known_info = known_info
        self.unknown_variable = unknown_variable
        self.time_steps = time_steps
        self.trajectory = self._weave_triple_helix()
        
    def _weave_triple_helix(self) -> List[Dict[str, Any]]:
        """
        시간(운동성)의 흐름에 따라 세 개의 나선이 어떻게 얽히는지 궤적을 엮어냅니다.
        """
        trajectory = []
        
        # 상위 3개의 핵심 꼬임(Twist) 축 추출
        core_axes = [t["axis"] for t in self.known_info.twists if not t["is_noise"]][:3]
        
        for step in range(self.time_steps):
            step_data = {"step": step, "observations": []}
            
            for axis in core_axes:
                # 1. 아는 것의 텐션 (이중나선의 기반)
                known_twist = next(t for t in self.known_info.twists if t["axis"] == axis)
                base_tension = known_twist["tension"]
                
                # 2. 미지의 변수 값
                unknown_val = self.unknown_variable.get(axis, 0.0)
                
                # 3. 운동성(Kinematics) - 시간선 위에서 변수와 나의 인식 간의 거리(Delta) 변화 속도
                # 스텝이 진행될수록 미지와의 거리를 좁히려 시도합니다.
                delta = abs(base_tension - unknown_val)
                kinematic_velocity = delta / (step + 1.0)
                
                # 세 가닥의 엮임: 기존 텐션 + 미지의 자극 + 운동성
                triple_knot = base_tension + unknown_val + kinematic_velocity
                
                step_data["observations"].append({
                    "axis": axis,
                    "known_tension": base_tension,
                    "unknown_val": unknown_val,
                    "kinematics": kinematic_velocity,
                    "triple_knot_energy": triple_knot
                })
                
            trajectory.append(step_data)
            
        return trajectory
        
    def express(self) -> str:
        desc = f"====== [판단의 삼중나선 궤적 가동] ======\n"
        desc += f"대상: 아는 것 '{self.known_info.concept_name}' vs 미지의 변수\n"
        for step_data in self.trajectory:
            desc += f"\n[운동성 스텝 {step_data['step']}]\n"
            for obs in step_data["observations"]:
                desc += (f"  ≡ 축({obs['axis']}): 이중나선({obs['known_tension']:.2f}) 엮임 미지({obs['unknown_val']:.2f}) "
                         f"-> 운동성 델타({obs['kinematics']:.3f}) => 판단 에너지: {obs['triple_knot_energy']:.3f}\n")
        return desc

class StatisticalDissector:
    """
    [Phase 9: 통계적 인과의 해체 (Dissecting Statistical Causality)]
    기존 LLM이 "정답"으로 얼려버린(Crystallized) 확률값(예: 80%)을 받아,
    그것을 다시 '살아있는 과정'으로 녹여냅니다(Melting).
    
    - 80%의 확률은 -> 해명된 인과(Known Causal Mass)로 취급하여 이중나선의 뼈대로 복원합니다.
    - 20%의 손실은 -> 에러나 노이즈가 아니라, 아직 규명되지 않은 미지의 차원(Dial X)으로 규정합니다.
    """
    def __init__(self, target_concept: str, llm_probability: float):
        self.concept = target_concept
        self.prob = max(0.0, min(1.0, llm_probability))
        
    def melt_frozen_ice(self) -> tuple[DoubleHelixInformation, dict, float]:
        """
        얼어붙은 답(Ice)을 녹여, 
        [해명된 이중나선 정보], [가변 저항 다이얼 X], [미지의 운동성 마찰값]으로 반환합니다.
        """
        # 1. 해명된 인과 (Known Causality)
        # 사과가 사과인 이유(Sameness)와 사과가 아닌 이유(Difference)의 치열한 충돌 과정을 복원.
        # 기존 LLM은 이 과정을 지워버렸지만, 엘리시아는 이 P%를 기반으로 다시 나선을 꼬아냅니다.
        sameness_axes = {"axis_관습적_정의": self.prob, "axis_통계적_패턴": self.prob * 0.9}
        difference_axes = {"axis_관습적_정의": 1.0 - self.prob, "axis_통계적_패턴": (1.0 - self.prob) * 0.9}
        
        known_info = DoubleHelixInformation(self.concept, sameness_axes, difference_axes)
        
        # 2. 가변 저항 다이얼 (Dial X: Unknown Dimension)
        # P%의 반대편에 있는 (1 - P)%는 상실된 인과적 노이즈(Causal Noise)입니다.
        unknown_mass = 1.0 - self.prob
        dial_x = {
            "axis_가변저항_X": unknown_mass, 
            "axis_관습적_정의": 0.0 # 기존 잣대로는 해명 불가
        }
        
        return known_info, dial_x, unknown_mass
