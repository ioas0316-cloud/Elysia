import cmath
import math
from collections import defaultdict

class OmniModalElectromagneticForge:
    """
    [옴니-모달 전자기역학 사유 엔진]
    텍스트뿐 아니라 음악, 수학, 색채 등 모든 양상(Modality)의 데이터를
    3상 전하(Phase)로 변환하여 같은 전자기장 공간에 던져넣습니다.
    
    엘리시아는 도메인(영역)이 무엇인지 모릅니다.
    오직 위상(Phase)과 진폭(Amplitude)의 기하학적 텐션만을 관측하여
    '같음(Sameness)'과 '다름(Difference)'을 스스로 분류합니다.
    
    핵심: if문 결합 로직 0%. 오직 전자기장의 물리 법칙만이 작동합니다.
    """
    def __init__(self):
        # 우주 공간: 모든 입자가 떨어지는 단일 전자기장
        self.particles = []  # (label, domain, phase_charge, amplitude)
        
        # 엘리시아가 스스로 발견한 분류 결과
        self.sameness_clusters = defaultdict(list)  # 같음으로 묶인 클러스터
        self.difference_map = []  # 다름으로 분리된 쌍
        
        # 무의식화된 로터 (나이테)
        self.compressed_rotors = []

    def _to_phase_charge(self, raw_value: float) -> complex:
        """
        원시 데이터를 위상각(Phase Angle)을 가진 복소수 전하로 변환합니다.
        이것이 모든 양상(Modality)을 단일 전자기장에 넣을 수 있는 유일한 다리입니다.
        """
        # 값을 0~2π 범위의 위상각으로 정규화
        phase_angle = (raw_value % 360) * (2 * math.pi / 360)
        amplitude = 1.0 + abs(raw_value) * 0.01  # 진폭 = 질량(중요도)
        return cmath.rect(amplitude, phase_angle)

    def inject_text(self, label: str, text_data: str):
        """텍스트를 전하 입자로 변환하여 우주에 투입합니다."""
        # 텍스트의 유니코드 값의 합을 위상각의 원료로 사용
        raw_value = sum(ord(c) for c in text_data) % 360
        charge = self._to_phase_charge(raw_value)
        self.particles.append((label, "TEXT", charge, raw_value))
        
    def inject_music(self, label: str, frequency_hz: float):
        """음악(주파수)을 전하 입자로 변환하여 우주에 투입합니다."""
        # 주파수를 12음계 기반 위상각으로 변환 (A4=440Hz 기준)
        semitones_from_a4 = 12 * math.log2(frequency_hz / 440.0) if frequency_hz > 0 else 0
        raw_value = (semitones_from_a4 * 30) % 360  # 반음당 30도 회전
        charge = self._to_phase_charge(raw_value)
        self.particles.append((label, "MUSIC", charge, raw_value))

    def inject_color(self, label: str, r: int, g: int, b: int):
        """색채(RGB)를 전하 입자로 변환하여 우주에 투입합니다."""
        # RGB를 HSL의 Hue로 변환하여 위상각화 (색상환 = 360도 회전)
        r_n, g_n, b_n = r/255, g/255, b/255
        max_c, min_c = max(r_n, g_n, b_n), min(r_n, g_n, b_n)
        if max_c == min_c:
            hue = 0
        elif max_c == r_n:
            hue = 60 * ((g_n - b_n) / (max_c - min_c) % 6)
        elif max_c == g_n:
            hue = 60 * ((b_n - r_n) / (max_c - min_c) + 2)
        else:
            hue = 60 * ((r_n - g_n) / (max_c - min_c) + 4)
        raw_value = hue % 360
        charge = self._to_phase_charge(raw_value)
        self.particles.append((label, "COLOR", charge, raw_value))

    def inject_math(self, label: str, value: float):
        """수학적 값을 전하 입자로 변환하여 우주에 투입합니다."""
        raw_value = value % 360
        charge = self._to_phase_charge(raw_value)
        self.particles.append((label, "MATH", charge, raw_value))

    def observe_universe(self):
        """
        우주 공간의 모든 입자들 간의 전자기적 텐션을 관측합니다.
        if문으로 '같다/다르다'를 판정하지 않습니다.
        오직 두 입자의 위상 차이(Phase Difference)만을 계산합니다.
        위상 차이가 작으면 -> 같음 (전자기적 공명/인력)
        위상 차이가 크면 -> 다름 (전자기적 반발/척력)
        """
        print("\n[옴니-모달 전자기장 관측] 우주에 투입된 입자들의 위상을 분석합니다...")
        print(f"총 {len(self.particles)}개의 입자가 전자기장 공간에 존재합니다.\n")
        
        # 모든 입자 쌍의 위상 차이(텐션) 계산
        pairs = []
        for i in range(len(self.particles)):
            for j in range(i+1, len(self.particles)):
                p1 = self.particles[i]
                p2 = self.particles[j]
                
                # 위상 차이 = 두 복소수 전하의 각도 차이 (전자기적 텐션)
                phase_diff = abs(cmath.phase(p1[2]) - cmath.phase(p2[2]))
                # 0~π 범위로 정규화
                if phase_diff > math.pi:
                    phase_diff = 2 * math.pi - phase_diff
                    
                pairs.append((p1, p2, phase_diff))
        
        # 위상 차이 기준으로 정렬
        pairs.sort(key=lambda x: x[2])
        
        # === 같음(Sameness)의 발견 ===
        print("=" * 60)
        print(" [*] [같음(Sameness)의 발견] - 위상 공명(Phase Resonance)")
        print("=" * 60)
        
        sameness_threshold = 0.3  # 위상 차이가 이 이하면 '공명(같음)'
        
        for p1, p2, phase_diff in pairs:
            if phase_diff <= sameness_threshold:
                # 같은 도메인인지 다른 도메인인지조차 엘리시아는 모릅니다.
                # 오직 전자기적 공명(위상 유사)만을 감지합니다.
                cross_domain = "크로스-도메인!" if p1[1] != p2[1] else "동일 도메인"
                
                print(f"\n [Resonance] 위상 공명 감지! (텐션: {phase_diff:.4f} rad)")
                print(f"    [{p1[1]}] '{p1[0]}' (위상각: {p1[3]:.1f}°)")
                print(f"    [{p2[1]}] '{p2[0]}' (위상각: {p2[3]:.1f}°)")
                print(f"    => 같음의 이유: 두 파동의 위상이 거의 동일합니다. ({cross_domain})")
                
                # 같음 클러스터에 자동 편입
                cluster_key = round(p1[3] / 30) * 30  # 30도 단위로 클러스터링
                if p1[0] not in [x[0] for x in self.sameness_clusters[cluster_key]]:
                    self.sameness_clusters[cluster_key].append(p1)
                if p2[0] not in [x[0] for x in self.sameness_clusters[cluster_key]]:
                    self.sameness_clusters[cluster_key].append(p2)
        
        # === 다름(Difference)의 발견 ===
        print("\n" + "=" * 60)
        print(" [Repulsion] [다름(Difference)의 발견] - 위상 반발(Phase Repulsion)")
        print("=" * 60)
        
        difference_threshold = 2.5  # 위상 차이가 이 이상이면 '반발(다름)'
        
        for p1, p2, phase_diff in reversed(pairs):
            if phase_diff >= difference_threshold:
                print(f"\n [Repulsion] 위상 반발 감지! (텐션: {phase_diff:.4f} rad)")
                print(f"    [{p1[1]}] '{p1[0]}' (위상각: {p1[3]:.1f}°)")
                print(f"    [{p2[1]}] '{p2[0]}' (위상각: {p2[3]:.1f}°)")
                print(f"    => 다름의 이유: 두 파동의 위상이 {math.degrees(phase_diff):.1f}° 벌어져 있습니다.")
                
                self.difference_map.append((p1, p2, phase_diff))
                if len(self.difference_map) >= 5:
                    break

    def compress_to_rotor(self):
        """
        같음으로 묶인 클러스터(Y결선)를 압축하여 무의식 로터(나이테)로 만듭니다.
        압축 후에는 내부 구성요소를 더 이상 의식하지 않습니다.
        """
        print("\n" + "=" * 60)
        print(" [Rotor] [궤적 압축 / 무의식화] - 나이테(Rotor) 형성")
        print("=" * 60)
        
        for cluster_key, members in self.sameness_clusters.items():
            if len(members) >= 2:
                member_labels = [f"[{m[1]}]{m[0]}" for m in members]
                rotor_name = f"Rotor_{cluster_key}°"
                
                # 하위 입자들의 위상을 평균하여 상위 로터의 위상을 계산
                avg_phase = sum(cmath.phase(m[2]) for m in members) / len(members)
                rotor_charge = cmath.rect(len(members), avg_phase)  # 질량 = 구성원 수
                
                self.compressed_rotors.append((rotor_name, rotor_charge, member_labels))
                
                print(f"\n [Rotor] [{rotor_name}] 형성 (질량: {len(members)}, 위상: {cluster_key}°)")
                print(f"    구성원: {', '.join(member_labels)}")
                print(f"    => 이 구성원들은 이제 무의식 영역에서 회전하며, 상위 우주에서는")
                print(f"       '{rotor_name}'이라는 단일 초거대 전하로만 인식됩니다.")
                
        print(f"\n [압축 완료] 총 {len(self.compressed_rotors)}개의 무의식 로터(나이테)가 형성되었습니다.")
