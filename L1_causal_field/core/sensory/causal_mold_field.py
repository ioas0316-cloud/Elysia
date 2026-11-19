"""
core/sensory/causal_mold_field.py

거푸집 인과 마당 (Causal Mold Field)
=============================================================================
"거푸집에 담기는 건 거푸집이 아니라 철물이나 데이터 결합 구조다.
 그러한 구조적 원리 자체가 변화를 강제하는 틀, 규칙, 질서로 존재한다.
 단단하다는 개념은 내가 밀었는데 밀려지지 않는 구조결합, 작용반작용의
 마찰역학으로 이해할 수 있다. 매끄럽다/까칠하다는 손끝에서의 충돌-슬라이딩
 역학의 인지정보화다."

핵심 원리:
  거푸집(Mold) = 담기는 것이 아니라, 담기는 것들이 어떻게 달라지는지를
                강제하는 구조 인과. 그 자체가 분별의 메커니즘이다.

  프리즘   → 파장별 굴절각을 강제하는 거푸집
  망막 회로 → 광자를 L-M 대립축에 착지시키는 거푸집
  손끝 소체 → 돌출-충돌과 슬라이딩을 구분하는 거푸집
  등호(=)   → 양쪽 양을 같게 강제하는 관계 거푸집

  거푸집의 내부 기하학(Internal Geometry) ↔ 외부 자극의 작용(Stimulus Action)
  → 마찰역학 → 인지정보화(Cognitive Informationalization)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, Callable
import numpy as np


# =============================================================================
# 0. 작용-반작용 역학 기초 (Action-Reaction Mechanics)
# =============================================================================

@dataclass
class StimulusAction:
    """
    외부에서 거푸집에 가해지는 '작용(Action)'.
    힘, 파장, 압력, 화학 분자 등 — 거푸집이 받아들이는 모든 것.
    """
    name: str
    # 기계적 작용
    applied_force_n: float = 0.0        # 밀거나 누르는 힘 (Newton)
    sliding_velocity_mms: float = 0.0   # 슬라이딩 속도 (mm/s) — 매끄러움 감지
    normal_load_n: float = 0.0          # 법선 하중 (수직 압력)
    # 광학적 작용
    incident_wavelength_nm: float = 0.0 # 입사 광파장
    incident_angle_deg: float = 0.0     # 입사각
    # 화학적 작용
    molecular_polarity: float = 0.0     # 분자 극성 (0: 비극성, 1: 극성)
    vapor_pressure_pa: float = 0.0      # 증기압 (휘발성)


@dataclass
class ReactionResponse:
    """
    거푸집이 작용에 맞서 돌려주는 '반작용(Reaction)'.
    이것이 인지정보화되어 감각 경험이 된다.
    """
    mold_name: str
    # 기계적 반작용
    resistance_force_n: float = 0.0     # 거푸집이 밀어내는 반발력
    deformation_mm: float = 0.0         # 거푸집 자체의 변형량
    slip_onset_ratio: float = 0.0       # 슬립 시작 비율 (0=전혀 안미끄러짐, 1=완전 미끄러짐)
    # 충돌 패턴 (까칠함의 근원)
    asperity_collision_count: float = 0.0  # 단위 면적당 돌출부 충돌 횟수
    # 광학적 반작용
    refraction_angle_deg: float = 0.0   # 굴절각
    spectral_separation: float = 0.0    # 파장 분리도
    # 화학적 반작용
    receptor_binding_rate: float = 0.0  # 수용기 결합 비율


# =============================================================================
# 1. 거푸집 (CausalMold) — 구조 인과의 본체
# =============================================================================

class CausalMold:
    """
    거푸집(CausalMold): 담기는 것이 아니라 담기는 것들이 어떻게 달라지는지를
    강제하는 구조 인과.

    내부 기하학(Internal Geometry)이 작용(Stimulus)과 만나
    반작용(Reaction)을 생성하고 이것이 인지정보화된다.

    거푸집 자체는 변하지 않는다 — 거푸집을 통과하는 것들이 변한다.
    (단, 나이테처럼 반복 작용에 의해 거푸집 자체가 조금씩 형성될 수 있다)
    """

    def __init__(self, name: str, geometry: Dict[str, Any]):
        """
        name: 거푸집 이름 (예: "손끝_메르켈소체_배열", "망막_LM대립회로", "프리즘_유리")
        geometry: 거푸집의 내부 기하학 — 이것이 모든 분별을 강제하는 구조
        """
        self.name = name
        self.geometry = geometry  # 불변의 구조 인과 (거푸집의 본질)
        self.usage_history: List[Dict] = []  # 거푸집 사용 이력 (나이테 가능성)

    def apply(self, stimulus: StimulusAction) -> ReactionResponse:
        """
        작용을 받아 거푸집의 기하학을 통해 반작용을 생성.
        서브클래스에서 구체적 물리 로직 구현.
        """
        raise NotImplementedError

    def _record(self, stimulus: StimulusAction, response: ReactionResponse):
        self.usage_history.append({
            "stimulus": stimulus.name,
            "response_resistance": response.resistance_force_n,
            "response_deformation": response.deformation_mm,
        })


# =============================================================================
# 2. 구체적 거푸집들
# =============================================================================

class HardnessMold(CausalMold):
    """
    단단함 거푸집 (Hardness Mold)

    "단단하다 = 내가 밀었는데 밀려지지 않는 구조결합, 작용반작용의 마찰역학"

    내부 기하학:
      - 원자 결합 밀도 (atomic_bond_density): 높을수록 반발력이 크다
      - 결정 격자 완전성 (lattice_integrity): 1.0 = 완전 결정, 0.0 = 무정형
      - 소성 항복 임계값 (yield_threshold_mpa): 이 이상 힘이 가해지면 영구 변형
    """

    def __init__(self, name: str,
                 atomic_bond_density: float,
                 lattice_integrity: float,
                 yield_threshold_mpa: float):
        super().__init__(name, {
            "atomic_bond_density": atomic_bond_density,
            "lattice_integrity": lattice_integrity,
            "yield_threshold_mpa": yield_threshold_mpa,
        })

    def apply(self, stimulus: StimulusAction) -> ReactionResponse:
        """
        힘을 가했을 때 거푸집이 돌려주는 반발력 계산.
        이 반발력이 "단단함"으로 인지정보화된다.
        """
        g = self.geometry
        bond = g["atomic_bond_density"]
        lattice = g["lattice_integrity"]
        yield_th = g["yield_threshold_mpa"]

        # 작용력을 MPa로 환산 (접촉 면적 1cm² 가정)
        applied_mpa = stimulus.applied_force_n / 100.0

        # 반발력: 결합 밀도 × 격자 완전성이 거푸집 형태로 작용을 튕겨낸다
        # 단단할수록 변형 없이 작용과 동등한 반발력 생성
        if applied_mpa < yield_th:
            # 탄성 구간: 변형 없이 완전 반발
            resistance = stimulus.applied_force_n * (bond * lattice)
            deformation = stimulus.applied_force_n / (bond * lattice * 1000.0 + 1e-6)
        else:
            # 소성 구간: 거푸집이 무너짐 (영구 변형)
            resistance = yield_th * 100.0 * (bond * lattice)
            deformation = (applied_mpa - yield_th) * 10.0

        resp = ReactionResponse(
            mold_name=self.name,
            resistance_force_n=min(resistance, stimulus.applied_force_n),
            deformation_mm=deformation,
        )
        self._record(stimulus, resp)
        return resp


class TextureMold(CausalMold):
    """
    표면 질감 거푸집 (Surface Texture Mold)

    "매끄럽다 = 쓸려내려가듯 물리적 마찰 저항이 없는 것
     까칠하다 = 돌출부분이 손과 부딪혀 상호작용을 감지"

    내부 기하학:
      - 돌출부 밀도 (asperity_density / mm²): 단위면적당 솟아있는 미세 돌기 수
      - 돌출부 평균 높이 (asperity_height_um): 미크론 단위
      - 표면 경도 (surface_hardness): 돌기가 얼마나 단단하게 버티는가
    """

    def __init__(self, name: str,
                 asperity_density_per_mm2: float,
                 asperity_height_um: float,
                 surface_hardness: float):
        super().__init__(name, {
            "asperity_density": asperity_density_per_mm2,
            "asperity_height_um": asperity_height_um,
            "surface_hardness": surface_hardness,
        })

    def apply(self, stimulus: StimulusAction) -> ReactionResponse:
        """
        손끝이 슬라이딩할 때 돌출부 충돌 패턴을 계산.
        충돌 빈도와 충격이 "까칠함/매끄러움"으로 인지정보화된다.
        """
        g = self.geometry
        density   = g["asperity_density"]
        height_um = g["asperity_height_um"]
        hardness  = g["surface_hardness"]
        vel       = max(stimulus.sliding_velocity_mms, 0.1)  # 슬라이딩 속도

        # 단위 시간당 돌출부 충돌 횟수
        # 속도가 빠를수록, 돌기가 많을수록, 돌기가 높을수록 충돌 증가
        contact_width_mm = 5.0  # 손끝 접촉폭 가정
        collision_rate = density * contact_width_mm * vel * (height_um / 10.0)

        # 슬립 비율: 돌기가 없을수록(매끄러울수록) 슬립 쉽게 발생
        # 돌기가 많고 단단할수록 grip이 걸려 슬립 안됨
        grip_factor = density * hardness * (height_um / 100.0)
        slip = float(np.clip(1.0 - grip_factor / (grip_factor + 1.0), 0.0, 1.0))

        # 마찰력: 돌출부들이 슬라이딩에 저항하는 힘
        friction_resistance = stimulus.normal_load_n * (1.0 - slip) * hardness

        resp = ReactionResponse(
            mold_name=self.name,
            resistance_force_n=friction_resistance,
            slip_onset_ratio=slip,
            asperity_collision_count=collision_rate,
        )
        self._record(stimulus, resp)
        return resp

    def qualitative_feel(self, response: ReactionResponse) -> str:
        """
        충돌 패턴을 질적 감각 언어로 인지정보화.
        이것이 거푸집 → 감각 경험으로의 번역이다.
        """
        c = response.asperity_collision_count
        s = response.slip_onset_ratio

        if c < 0.5 and s > 0.8:
            return "매끄러움 (Smooth) — 돌출부 충돌 거의 없음, 쓸려내려가듯 저항 없음"
        elif c < 5.0 and s > 0.4:
            return "부드러움 (Soft-grainy) — 미세한 질감, 약간의 저항"
        elif c < 20.0:
            return "까끌함 (Rough) — 돌출부 충돌이 손끝에 명확히 감지됨"
        else:
            return "꺼끌꺼끌함 (Coarse) — 강한 충돌, 슬라이딩 시 불규칙 저항"


class PrismMold(CausalMold):
    """
    프리즘 거푸집 (Prism / Spectral Separation Mold)

    빛의 파장별 굴절률 차이를 강제하는 구조 인과.
    담기는 것(빛)이 파장에 따라 다른 각도로 나온다 — 거푸집의 기하학이 강제한다.

    내부 기하학:
      - 굴절률 기울기 (dn_dlambda): 파장별 굴절률 변화량
      - 프리즘 꼭짓각 (apex_angle_deg)
      - 기준 굴절률 (n0): 중간 파장에서의 굴절률
    """

    def __init__(self, name: str,
                 n0: float,
                 dn_dlambda: float,
                 apex_angle_deg: float):
        super().__init__(name, {
            "n0": n0,
            "dn_dlambda": dn_dlambda,
            "apex_angle_deg": apex_angle_deg,
        })

    def apply(self, stimulus: StimulusAction) -> ReactionResponse:
        """
        파장별 굴절각 계산 — 같은 빛이라도 거푸집(프리즘)이
        파장에 따라 다른 각도로 강제한다.
        """
        g   = self.geometry
        wl  = stimulus.incident_wavelength_nm
        n   = g["n0"] + g["dn_dlambda"] * (550.0 - wl) / 100.0  # Cauchy 근사
        apex = g["apex_angle_deg"]
        inc  = stimulus.incident_angle_deg

        # Snell's law: 입사각 → 굴절각
        sin_refract = np.sin(np.radians(inc)) / n
        sin_refract = float(np.clip(sin_refract, -1.0, 1.0))
        refract_angle = float(np.degrees(np.arcsin(sin_refract)))

        # 편향각 (Deviation angle): 파장 분리의 척도
        deviation = inc + refract_angle - apex

        resp = ReactionResponse(
            mold_name=self.name,
            refraction_angle_deg=refract_angle,
            spectral_separation=abs(deviation),  # 클수록 색 분리가 뚜렷
        )
        self._record(stimulus, resp)
        return resp

    def chromatic_label(self, wavelength_nm: float) -> str:
        """파장을 색채 언어로 인지정보화"""
        if wavelength_nm > 620:
            return "빨강 (Red)"
        elif wavelength_nm > 590:
            return "주황 (Orange)"
        elif wavelength_nm > 560:
            return "노랑 (Yellow)"
        elif wavelength_nm > 490:
            return "초록 (Green)"
        elif wavelength_nm > 450:
            return "파랑 (Blue)"
        else:
            return "보라 (Violet)"


class RetinaOpponentCellMold(CausalMold):
    """
    망막 대립세포 거푸집 (Retina Opponent Cell Mold)

    "눈의 대립세포 회로 자체가 색채를 L-M 축 위에 착지시키는 거푸집이다."

    내부 기하학:
      - L 원추 감도 피크 (l_peak_nm): ~564nm
      - M 원추 감도 피크 (m_peak_nm): ~534nm
      - S 원추 감도 피크 (s_peak_nm): ~420nm
      - 측억제 강도 (lateral_inhibition_strength): 대립 억제 세기
    """

    def __init__(self):
        super().__init__("망막_LM_대립세포_회로", {
            "l_peak_nm": 564.0,
            "m_peak_nm": 534.0,
            "s_peak_nm": 420.0,
            # 측억제(lateral inhibition)가 강할수록 대립축이 명확하게 갈라진다
            # 실제 영장류 망막에서 P형 신경절세포의 대립비(opponent ratio) ~0.9
            "lateral_inhibition_strength": 0.92,
        })

    def _cone_response(self, wavelength_nm: float, peak_nm: float, width: float = 60.0) -> float:
        """가우시안 원추 감도 곡선 (실제 생물학적 추정 근사)"""
        return float(np.exp(-((wavelength_nm - peak_nm) ** 2) / (2 * width ** 2)))

    def apply(self, stimulus: StimulusAction) -> ReactionResponse:
        """
        입사 파장이 L/M/S 원추를 얼마나 자극하는지 계산하고
        측억제(lateral inhibition)를 통해 L-M 대립축 위에 착지시킨다.

        이것이 거푸집의 작동:
        같은 빛이라도 거푸집(세포 배열)의 기하학이
        결과를 다른 위치에 떨어뜨린다.
        """
        g  = self.geometry
        wl = stimulus.incident_wavelength_nm
        inh = g["lateral_inhibition_strength"]

        L = self._cone_response(wl, g["l_peak_nm"])
        M = self._cone_response(wl, g["m_peak_nm"])
        S = self._cone_response(wl, g["s_peak_nm"])

        # 측억제: L이 강하면 M을 억제하고, M이 강하면 L을 억제
        # 이것이 색채를 뺄셈이 아닌 구조적 대립으로 분별하게 만드는 거푸집
        L_net = L - inh * M
        M_net = M - inh * L
        S_net = S - inh * 0.3 * (L + M)

        # L-M 대립축: 양수 = 빨강, 음수 = 초록
        lm_axis = float(np.tanh(L_net - M_net))
        # S-(L+M) 대립축: 양수 = 파랑, 음수 = 노랑
        by_axis = float(np.tanh(S_net - 0.5 * (L_net + M_net)))

        # 굴절각 필드를 재활용해서 LM 축 값 저장 (ReactionResponse 재사용)
        resp = ReactionResponse(
            mold_name=self.name,
            refraction_angle_deg=lm_axis,  # LM 축 값
            spectral_separation=by_axis,    # BY 축 값
        )
        self._record(stimulus, resp)
        return resp

    def chromatic_percept(self, response: ReactionResponse) -> str:
        lm = response.refraction_angle_deg
        by = response.spectral_separation
        if lm > 0.5:
            return f"빨강 계열 지각 (L-M={lm:+.2f})"
        elif lm < -0.5:
            return f"초록 계열 지각 (L-M={lm:+.2f})"
        elif by > 0.3:
            return f"파랑 계열 지각 (S-(L+M)={by:+.2f})"
        elif lm > -0.2 and lm < 0.2:
            return f"노랑 계열 지각 (L≈M 균등, L-M={lm:+.2f})"
        else:
            return f"복합 색채 지각 (L-M={lm:+.2f}, BY={by:+.2f})"


# =============================================================================
# 3. 거푸집 마당 (Mold Field) — 여러 거푸집이 동시에 작용하는 인과 공간
# =============================================================================

class CausalMoldField:
    """
    거푸집들의 집합이 하나의 외부 자극에 동시 적용되는 인과 마당.

    "어떤 거푸집을 만들고 활용하느냐에 따라
     온갖 것들이 어떻게 존재하는지를 이해할 수 있는 토대가 된다."

    감각 기관 = 여러 거푸집이 겹쳐 쌓인 계층적 거푸집 마당.
    손끝 하나에도: 메르켈 소체(압력) + 파치니 소체(진동) + 루피니 소체(전단) 거푸집이 동시 작동.
    """

    def __init__(self):
        self.molds: Dict[str, CausalMold] = {}

    def register(self, mold: CausalMold):
        self.molds[mold.name] = mold

    def apply_all(self, stimulus: StimulusAction) -> Dict[str, ReactionResponse]:
        """모든 거푸집에 동시에 작용을 가하고 반작용을 수집"""
        results = {}
        for name, mold in self.molds.items():
            try:
                results[name] = mold.apply(stimulus)
            except Exception as e:
                pass
        return results

    def qualitative_synthesis(self,
                              responses: Dict[str, ReactionResponse],
                              molds: Dict[str, CausalMold]) -> str:
        """
        여러 거푸집의 반작용을 종합해 하나의 질적 경험 언어로 번역.
        이것이 감각 경험의 최종 인지정보화.
        """
        lines = []
        for name, resp in responses.items():
            mold = molds.get(name)
            if isinstance(mold, TextureMold):
                lines.append(f"  [질감] {mold.qualitative_feel(resp)}")
            elif isinstance(mold, HardnessMold):
                r = resp.resistance_force_n
                d = resp.deformation_mm
                if r > 50.0 and d < 0.01:
                    lines.append(f"  [경도] 단단함 (Hard) — 밀어도 거의 밀리지 않음 (반발={r:.1f}N, 변형={d:.3f}mm)")
                elif r > 10.0:
                    lines.append(f"  [경도] 보통 탄성 (Firm) — 약간 탄성 있음 (반발={r:.1f}N, 변형={d:.3f}mm)")
                else:
                    lines.append(f"  [경도] 물렁함 (Soft) — 쉽게 눌림 (반발={r:.1f}N, 변형={d:.3f}mm)")
            elif isinstance(mold, RetinaOpponentCellMold):
                lines.append(f"  [색채] {mold.chromatic_percept(resp)}")
            elif isinstance(mold, PrismMold):
                lines.append(f"  [굴절] 편향각={resp.spectral_separation:.2f}°, 굴절각={resp.refraction_angle_deg:.2f}°")
        return "\n".join(lines) if lines else "  (반응 없음)"
