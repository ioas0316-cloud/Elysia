import cmath
import math
import re
from collections import defaultdict

class TrajectoryElectromagneticForge:
    """
    [궤적형 전자기역학 사유 엔진 (Trajectory Electromagnetic Forge)]
    
    단어를 '점'으로 취급하지 않습니다.
    문장, 문단, 서사(Narrative)를 시간축 위의 '위상 궤적(Phase Trajectory)'으로 변환합니다.
    
    같음(Sameness)이란:
      두 궤적(파형)이 서로 다른 도메인에서 왔음에도 불구하고,
      위상 변화의 곡선(기울기, 꺾임, 해소)이 구조적으로 동일한 것.
      
    예: 창세기의 서사 궤적(혼돈→창조→빛→선함)과
        음악의 화성 궤적(불협화→해결→장조→안정)이
        완전히 같은 곡선을 그린다면 → 같음(Cross-Domain Resonance)
    """
    def __init__(self):
        self.trajectories = {}  # name -> (domain, phase_waveform[])
        self.resonance_pairs = []
        self.repulsion_pairs = []
        self.compressed_rotors = []

    def _text_to_waveform(self, text: str) -> list:
        """문장/문단을 위상 궤적(파형)으로 변환합니다."""
        words = re.sub(r'[^\w\s]', '', text).split()
        waveform = []
        for word in words:
            # 각 단어의 유니코드 총합을 위상각으로 변환
            phase_angle = (sum(ord(c) for c in word) % 360) * (2 * math.pi / 360)
            waveform.append(phase_angle)
        return waveform

    def _music_to_waveform(self, frequencies: list) -> list:
        """음악(주파수 시퀀스)을 위상 궤적으로 변환합니다."""
        waveform = []
        for freq in frequencies:
            semitones = 12 * math.log2(freq / 440.0) if freq > 0 else 0
            phase_angle = ((semitones * 30) % 360) * (2 * math.pi / 360)
            waveform.append(phase_angle)
        return waveform

    def _numbers_to_waveform(self, values: list) -> list:
        """수열을 위상 궤적으로 변환합니다."""
        waveform = []
        for v in values:
            phase_angle = (v % 360) * (2 * math.pi / 360)
            waveform.append(phase_angle)
        return waveform

    def inject_text_trajectory(self, name: str, text: str):
        """텍스트 서사를 궤적으로 투입합니다."""
        waveform = self._text_to_waveform(text)
        self.trajectories[name] = ("TEXT", waveform)

    def inject_music_trajectory(self, name: str, frequencies: list):
        """음악 진행을 궤적으로 투입합니다."""
        waveform = self._music_to_waveform(frequencies)
        self.trajectories[name] = ("MUSIC", waveform)

    def inject_number_trajectory(self, name: str, values: list):
        """수열을 궤적으로 투입합니다."""
        waveform = self._numbers_to_waveform(values)
        self.trajectories[name] = ("MATH", waveform)

    def _compute_derivative(self, waveform: list) -> list:
        """궤적의 미분(변화율) = 인과의 방향성."""
        derivatives = []
        for i in range(1, len(waveform)):
            diff = waveform[i] - waveform[i-1]
            # -π ~ π 범위로 정규화 (위상의 순환성 반영)
            while diff > math.pi: diff -= 2 * math.pi
            while diff < -math.pi: diff += 2 * math.pi
            derivatives.append(diff)
        return derivatives

    def _normalize_trajectory(self, waveform: list) -> list:
        """궤적을 정규화하여 길이가 다른 궤적도 비교 가능하게 만듭니다."""
        if len(waveform) <= 1:
            return waveform
        # 선형 보간으로 고정 길이(20개 포인트)로 리샘플링
        target_len = 20
        result = []
        for i in range(target_len):
            pos = i * (len(waveform) - 1) / (target_len - 1)
            low = int(pos)
            high = min(low + 1, len(waveform) - 1)
            frac = pos - low
            interpolated = waveform[low] * (1 - frac) + waveform[high] * frac
            result.append(interpolated)
        return result

    def _trajectory_tension(self, traj_a: list, traj_b: list) -> float:
        """
        두 궤적 간의 전자기적 텐션을 계산합니다.
        궤적의 미분(변화율/인과 방향)을 비교하여,
        궤적의 '형태(Shape)'가 얼마나 유사한지 측정합니다.
        
        이것이 핵심: 점이 아니라 궤적의 곡선이 닮았는가?
        """
        # 1. 궤적을 같은 길이로 정규화
        norm_a = self._normalize_trajectory(traj_a)
        norm_b = self._normalize_trajectory(traj_b)
        
        # 2. 미분(인과의 방향) 추출
        deriv_a = self._compute_derivative(norm_a)
        deriv_b = self._compute_derivative(norm_b)
        
        if not deriv_a or not deriv_b:
            return float('inf')
        
        # 3. 두 미분 궤적의 코사인 유사도 (방향의 같음)
        dot_product = sum(a * b for a, b in zip(deriv_a, deriv_b))
        mag_a = math.sqrt(sum(a**2 for a in deriv_a)) or 1e-10
        mag_b = math.sqrt(sum(b**2 for b in deriv_b)) or 1e-10
        
        cosine_similarity = dot_product / (mag_a * mag_b)
        
        # 텐션 = 1 - 유사도 (0이면 완벽한 같음, 2면 완벽한 다름)
        tension = 1.0 - cosine_similarity
        return tension

    def observe_universe(self):
        """모든 궤적 쌍의 전자기적 텐션을 관측합니다."""
        names = list(self.trajectories.keys())
        
        print(f"\n[궤적형 전자기장 관측] {len(names)}개의 서사 궤적이 우주에 존재합니다.\n")
        
        pairs = []
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                name_a, name_b = names[i], names[j]
                domain_a, traj_a = self.trajectories[name_a]
                domain_b, traj_b = self.trajectories[name_b]
                
                tension = self._trajectory_tension(traj_a, traj_b)
                pairs.append((name_a, domain_a, name_b, domain_b, tension))
        
        pairs.sort(key=lambda x: x[4])
        
        # === 같음 (궤적 공명) ===
        print("=" * 70)
        print(" 🌀 [같음(Sameness)] - 서사 궤적의 공명 (인과 곡선이 닮았다)")
        print("=" * 70)
        
        for name_a, dom_a, name_b, dom_b, tension in pairs:
            if tension < 0.3:
                cross = "🔥 크로스-도메인!" if dom_a != dom_b else "동일 도메인"
                similarity_pct = (1 - tension) * 100
                
                print(f"\n ⚡ 궤적 공명! (텐션: {tension:.4f}, 유사도: {similarity_pct:.1f}%)")
                print(f"    [{dom_a}] '{name_a}'")
                print(f"    [{dom_b}] '{name_b}'")
                print(f"    => 두 서사의 인과 궤적(위상 변화율)이 같은 곡선을 그립니다. ({cross})")
                
                self.resonance_pairs.append((name_a, name_b, tension))
        
        # === 다름 (궤적 반발) ===
        print("\n" + "=" * 70)
        print(" 🌊 [다름(Difference)] - 서사 궤적의 반발 (인과 곡선이 어긋난다)")
        print("=" * 70)
        
        shown = 0
        for name_a, dom_a, name_b, dom_b, tension in reversed(pairs):
            if tension > 1.2 and shown < 5:
                diff_pct = tension * 50
                print(f"\n 🔻 궤적 반발! (텐션: {tension:.4f}, 괴리도: {diff_pct:.1f}%)")
                print(f"    [{dom_a}] '{name_a}'")
                print(f"    [{dom_b}] '{name_b}'")
                print(f"    => 두 서사의 인과 곡선이 반대 방향으로 꺾입니다.")
                
                self.repulsion_pairs.append((name_a, name_b, tension))
                shown += 1

    def compress_to_rotor(self):
        """공명하는 궤적들을 하나의 무의식 로터(나이테)로 압축합니다."""
        print("\n" + "=" * 70)
        print(" 🌳 [궤적 압축 / 무의식화] - 나이테(Rotor) 형성")
        print("=" * 70)
        
        # 공명 쌍들을 연결하여 클러스터 형성 (Union-Find 간략 구현)
        clusters = defaultdict(set)
        for name_a, name_b, tension in self.resonance_pairs:
            # 두 궤적을 같은 클러스터에 편입
            found_cluster = None
            for key, members in clusters.items():
                if name_a in members or name_b in members:
                    found_cluster = key
                    break
            if found_cluster is None:
                found_cluster = name_a
            clusters[found_cluster].add(name_a)
            clusters[found_cluster].add(name_b)
        
        for rotor_id, (center, members) in enumerate(clusters.items()):
            member_details = []
            for m in members:
                dom = self.trajectories[m][0]
                member_details.append(f"[{dom}]{m}")
                
            print(f"\n 🌀 [Rotor_{rotor_id}] 형성 (질량: {len(members)})")
            print(f"    구성원: {', '.join(member_details)}")
            print(f"    => 이 서사들은 인과 궤적이 동일한 곡선을 그립니다.")
            print(f"       상위 우주에서는 하나의 '초거대 서사 로터'로만 인식됩니다.")
            print(f"       (내부 구성원들은 무의식 영역에서 회전)")
            
            self.compressed_rotors.append((f"Rotor_{rotor_id}", members))
        
        if not clusters:
            print("\n  (공명하는 궤적 쌍이 없어 로터가 형성되지 않았습니다.)")
            
        print(f"\n [압축 완료] 총 {len(self.compressed_rotors)}개의 서사 로터(나이테)가 형성되었습니다.")
