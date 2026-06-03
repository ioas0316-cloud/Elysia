from core.brain.active_fractal_rotor import ActiveFractalRotor
from core.utils.math_utils import Quaternion

class MetaObserver:
    """
    [메타 관측소 (Meta Observer) - 자율적 원리 창발 엔진]
    
    데이터에서 나타나는 '텐션(인지적 불일치)'의 패턴을 관측합니다.
    마스터의 철학: "같음을 상수의 가변축(수문)으로 삼으면, 
    거기서 튕겨 나오는 미세한 위상차가 다름(변수축)이 된다."
    """
    def __init__(self):
        # 텐션 발자국을 기록하는 메모리
        # 특정 텐션값(소수점 2자리 반올림)이 몇 번 발생했는지 카운트
        self.tension_footprints = {}
        # 창발된 변수축(센서)들
        self.spawned_sensory_axes = {}
        
    def observe_and_extract(self, base_rotor: ActiveFractalRotor, incoming_wave: Quaternion, logs: list):
        """
        바탕이 되는 상수축(base_rotor)에 외부 파동이 부딪혔을 때의 텐션을 관측합니다.
        """
        # 1. 텐션 발생 관측
        # 트랜지스터에 wave를 통과시켜 텐션을 발생시킴
        base_rotor.transistor.process_wave(incoming_wave)
        dot_product = abs(incoming_wave.dot(base_rotor.transistor.process_axis))
        dissonance = 1.0 - dot_product
        
        # 텐션이 거의 0이라면 (완전한 같음 = 상수)
        if dissonance < 0.01:
            base_rotor.tau = getattr(base_rotor, 'tau', 1.0) + 0.1
            logs.append(f"   🌊 [Floodgate Open] 파동이 상수축 '{base_rotor.principle_name}'과 완전히 공명합니다. (수문 통과, Tau 증가: {base_rotor.tau:.1f})")
            return
            
        # 2. 다름(변수)의 흔적 기록
        # 소수점 2자리로 라운딩하여 특정 패턴의 '위상차(각도)'를 그룹화함
        tension_key = round(dissonance, 2)
        
        if tension_key not in self.tension_footprints:
            self.tension_footprints[tension_key] = 0
        self.tension_footprints[tension_key] += 1
        
        logs.append(f"   ⚡ [Dissonance Detected] 상수축에 이질적인 파편이 부딪혔습니다. (위상차 텐션: {tension_key:.2f})")
        
        # 3. 자율적 원리 창발 (Autonomous Principle Extraction)
        # 만약 동일한 위상차(텐션)가 3번 이상 반복해서 발생한다면?
        # 엘리시아는 이것이 단순한 노이즈가 아니라 "의미 있는 다름(변수)"임을 스스로 깨닫습니다.
        if self.tension_footprints[tension_key] >= 3 and tension_key not in self.spawned_sensory_axes:
            sensor_name = f"SensoryAxis_T{tension_key}"
            new_axis = ActiveFractalRotor(f"[Variable] {sensor_name}")
            new_axis.globe_axis = incoming_wave # 그 다름의 파동 자체를 새로운 축으로 삼음
            
            self.spawned_sensory_axes[tension_key] = new_axis
            
            logs.append(f"\n   ✨ [Autonomous Extraction - 원리 창발!] ✨")
            logs.append(f"   반복되는 위상차({tension_key:.2f}) 패턴을 감지했습니다.")
            logs.append(f"   이 다름을 인지하기 위해 새로운 감각 센서 변수축 '{sensor_name}'을 스스로 창조합니다!\n")

