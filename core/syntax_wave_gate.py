"""
Elysia Syntax-to-Wave Gate (SyntaxWaveGate)
===========================================
코드를 연속적인 위상 기하학적 궤적으로 사상하고, 
괄호 매칭을 위상의 비틀림과 풀림(Rotor Twist & Untwist)으로 모델링하며,
오차 발생 시 가장 가까운 정상 어휘(Attractor)로 위상을 정렬시키는
물리적 중력 법칙(Syntax Gravity) 기반 구문 분석기입니다.
"""

import re
import hashlib
import math
from typing import Dict, List, Tuple, Optional

class SyntaxGravityCollapse(Exception):
    """구문 위상이 어떤 정상 중력권(Attractor)에도 안착하지 못하고 붕괴했을 때 발생합니다."""
    pass

class SyntaxWaveGate:
    def __init__(self, rotor_scale: int = 4096, gravity_G: float = 1.0, collapse_threshold: float = 0.8, memory=None):
        self.rotor_scale = rotor_scale
        self.rotor_mask = rotor_scale - 1
        
        self.G = gravity_G
        self.collapse_threshold = collapse_threshold
        
        # [Phase 96] 하드코딩된 어휘집(Lexicon) 전면 삭제
        # 더 이상 'if', 'def' 등에 임의의 질량과 위상을 강제 부여하지 않습니다.
        # 오직 엘리시아의 실제 기억망(memory.ui_concept_map)의 동적 텐션만을 중력원(Attractor)으로 삼습니다.
        self.memory = memory
        
        # 2. 괄호쌍 트위스트 오프셋
        self.bracket_twists = {
            "(": 512,  # 45도 회전
            ")": -512,
            "{": 1024, # 90도 회전
            "}": -1024,
            "[": 256,  # 22.5도 회전
            "]": -256
        }

    def tokenize(self, code: str) -> List[str]:
        """정규식을 사용하여 구문 요소를 토큰 스트림으로 분리합니다."""
        # 단어 문자 혹은 괄호 및 연산자 매칭
        pattern = r"\w+|[(){}\[\]+\-*/=:]"
        return re.findall(pattern, code)

    def _hash_token_phase(self, token: str) -> int:
        """모든 어휘를 차별 없이 결정론적 위상각(해시)으로 순수 사상합니다."""
        # [Phase 96] 하드코딩된 사전(Lexicon) 검사 제거. 무조건 기하학적 해시 변환.
        h = hashlib.sha256(token.encode('utf-8')).digest()
        val = int.from_bytes(h[:4], byteorder='big')
        return val & self.rotor_mask

    def calculate_trajectory(self, code: str) -> Tuple[int, float, List[int]]:
        """
        코드 스트림을 순회하며 위상 궤적을 계산합니다.
        반환값: (최종 위상, 누적 괄호 비틀림 장력, 궤적 히스토리)
        """
        tokens = self.tokenize(code)
        current_phase = 0
        bracket_tension = 0.0
        trajectory = [current_phase]
        
        bracket_stack = []

        for token in tokens:
            if token in self.bracket_twists:
                # 괄호 비틀림 적용 (Rotor Twist & Untwist)
                twist = self.bracket_twists[token]
                current_phase = (current_phase + twist) & self.rotor_mask
                
                # 괄호 정합성 스택 관리 및 장력(Tension) 누적
                if twist > 0:
                    bracket_stack.append(token)
                    bracket_tension += 0.1
                else:
                    # 닫는 괄호 매칭 검사
                    matching_brackets = {")": "(", "}": "{", "]": "["}
                    expected = matching_brackets.get(token)
                    if bracket_stack and bracket_stack[-1] == expected:
                        bracket_stack.pop()
                        bracket_tension = max(0.0, bracket_tension - 0.1)
                    else:
                        # 괄호 불일치 오류 -> 텐션 대폭 가산 (모순/Difference)
                        bracket_tension += 0.5
            else:
                # 일반 토큰 위상 중합
                token_phase = self._hash_token_phase(token)
                current_phase = (current_phase + token_phase) & self.rotor_mask
                
            trajectory.append(current_phase)

        # [Phase 5] 미닫힌 괄호 존재 시 장력 가산: 인위적인 스칼라(0.3)가 아닌 기하학적 비틀림(Twist) 잔향 합산
        leftover_twist = 0
        for b in bracket_stack:
            leftover_twist += self.bracket_twists.get(b, 0)
        bracket_tension += abs(leftover_twist)
        
        return current_phase, bracket_tension, trajectory

    def calculate_circular_distance(self, phase1: int, phase2: int) -> float:
        """원형 링 상에서의 두 위상 간의 궤적 거리 및 방향각 차이를 라디안으로 반환합니다."""
        diff = (phase2 - phase1) & self.rotor_mask
        if diff > (self.rotor_scale // 2):
            # 음의 방향
            diff_rad = (diff - self.rotor_scale) * (2 * math.pi / self.rotor_scale)
        else:
            # 양의 방향
            diff_rad = diff * (2 * math.pi / self.rotor_scale)
        return diff_rad

    def evaluate_gravity(self, code: str) -> Dict[str, any]:
        """
        중력 포텐셜 및 끌어당기는 힘(Gravity Torque)을 계산합니다.
        자가 치유(Self-healing) 기능:
        만약 최종 위상이 특정 Attractor의 중력 인력 반경(150 위상 단위) 이내에 있다면,
        그 키워드로 위상이 강제 인입(Locking)되어 자가 교정이 수행됩니다.
        """
        final_phase, bracket_tension, trajectory = self.calculate_trajectory(code)
        
        closest_word = None
        min_dist_abs = float('inf')
        total_gravity_torque = 0.0
        
        attractors_data = {}

        # [Phase 96] 동적 어휘망 기반 중력 계산
        dynamic_attractors = {}
        if self.memory and hasattr(self.memory, 'ui_concept_map'):
            for word, node in self.memory.ui_concept_map.items():
                phase = int((node.lens_offset.w + 1.0) / 2.0 * self.rotor_scale) % self.rotor_scale
                mass = max(1.0, abs(node.tau))
                dynamic_attractors[word] = {"phase": phase, "mass": mass}

        # 각 키워드 Attractor의 중력 계산
        for word, info in dynamic_attractors.items():
            target_phase = info["phase"]
            mass = info["mass"]
            
            # 위상각 거리 계산 (라디안)
            dist_rad = self.calculate_circular_distance(final_phase, target_phase)
            abs_dist_rad = abs(dist_rad)
            
            if abs_dist_rad < min_dist_abs:
                min_dist_abs = abs_dist_rad
                closest_word = word
                
            # 만유인력 포텐셜 우물 공식: V = - (G * M) / (r + epsilon)
            epsilon = 0.1
            potential = - (self.G * mass) / (abs_dist_rad + epsilon)
            
            # 중력 끌림 토크
            torque = (self.G * mass * math.sin(dist_rad)) / ((abs_dist_rad + epsilon) ** 2)
            total_gravity_torque += torque
            
            attractors_data[word] = {
                "distance_rad": dist_rad,
                "potential": potential,
                "torque": torque
            }

        # 자가 치유(Self-healing) 판단: 가장 가까운 키워드와의 거리 검사
        # 150 위상 단위 이내(라디안 기준 약 0.23)일 때 포착(Capture)되어 보정됨
        capture_radius_rad = (150 / self.rotor_scale) * 2 * math.pi
        is_captured = (min_dist_abs <= capture_radius_rad)
        
        healed_phase = final_phase
        healed_word = None
        if is_captured and closest_word and closest_word in dynamic_attractors:
            healed_phase = dynamic_attractors[closest_word]["phase"]
            healed_word = closest_word
            
        # 총 구문 텐션 계산 (괄호 장력 + 가장 가까운 포텐셜의 부재 비율)
        # 포텐셜이 깊을수록(낮을수록) 인지 정합성(Sameness)이 높으므로 텐션이 낮아짐
        # 최대로 가능한 잠재력 대비 현재 최적 포텐셜 크기 비율 산출
        closest_potential = attractors_data[closest_word]["potential"] if closest_word else 0.0
        # 텐션 공식: Bracket Tension + (1.0 - Resonance)
        # Resonance는 포텐셜 우물의 깊이에 비례
        resonance = min(1.0, max(0.0, abs(closest_potential) / 150.0))
        total_syntax_tension = bracket_tension + (1.0 - resonance)

        # 중력 붕괴(오류) 판정
        is_collapsed = total_syntax_tension > self.collapse_threshold

        return {
            "final_phase": final_phase,
            "bracket_tension": bracket_tension,
            "trajectory": trajectory,
            "closest_attractor": closest_word,
            "is_captured": is_captured,
            "healed_phase": healed_phase,
            "healed_word": healed_word,
            "total_gravity_torque": total_gravity_torque,
            "total_syntax_tension": total_syntax_tension,
            "is_collapsed": is_collapsed,
            "attractors": attractors_data
        }

    def parse_with_gravity(self, code: str) -> str:
        """코드를 파싱하여 자가 치유된 결과를 반환하거나, 임계 텐션을 넘을 경우 붕괴 예외를 냅니다."""
        res = self.evaluate_gravity(code)
        if res["is_collapsed"]:
            raise SyntaxGravityCollapse(
                f"구문 중력 붕괴 발생! 총 텐션 {res['total_syntax_tension']:.4f}가 "
                f"붕괴 임계치 {self.collapse_threshold}를 초과했습니다. "
                f"(가장 가까운 어휘: {res['closest_attractor']}, 괄호 장력: {res['bracket_tension']:.2f})"
            )
        
        # 자가 치유에 성공했다면 치유된 단어 반환, 아니면 원본 문자열 반환
        return res["healed_word"] if res["healed_word"] else "UNKNOWN_SEQUENCE"
