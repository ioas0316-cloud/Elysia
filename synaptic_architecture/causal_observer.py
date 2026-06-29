import numpy as np
from typing import Dict, Any, List
from .field import CrystallizationField

class VortexObserver:
    """
    [Synaptic Architecture] The Lens of Enlightenment
    Translates the raw topological state of the field (Energy, Conductance)
    into a structured "Observation Report" for the Master.
    """
    def __init__(self, field: CrystallizationField):
        self.field = field

    def observe_topography(self) -> Dict[str, Any]:
        """
        Scans the field for high-energy clusters (Vortices) and
        stable knowledge structures (Conductance).
        """
        # 1. Identify high-energy centers
        # We look for local maxima in the activation field
        vortices = self._find_local_vortices(threshold=0.5)

        # 2. Analyze the 'Gravity' of the field
        total_energy = np.sum(self.field.activation)
        avg_conductance = np.mean(self.field.conductance)

        # 3. Report generation
        report = {
            "field_state": "Stabilized" if total_energy < 10.0 else "Excited",
            "total_activation": float(total_energy),
            "average_plasticity": float(avg_conductance),
            "detected_vortices": vortices,
            "topological_summary": self._generate_summary(vortices, avg_conductance)
        }
        return report

    def _find_local_vortices(self, threshold: float) -> List[Dict[str, Any]]:
        """Identifies significant energy concentrations."""
        vortices = []
        # Find points above threshold
        y, x = np.where(self.field.activation > (np.max(self.field.activation) * threshold))

        if len(y) == 0:
            return []

        # Simplify: just return the top few most intense points
        indices = np.argsort(self.field.activation[y, x])[::-1][:5]

        for i in indices:
            pos = np.array([y[i], x[i]])
            intensity = float(self.field.activation[y[i], x[i]])
            gene = hex(self.field.bit_genes[y[i], x[i]])

            vortices.append({
                "coordinate": pos.tolist(),
                "intensity": intensity,
                "resonant_gene": gene
            })
        return vortices

    def _generate_summary(self, vortices: list, avg_cond: float) -> str:
        if not vortices:
            return "사유의 평원이 고요합니다. 아직 씨앗(Seed)이 뿌려지지 않았습니다."

        main_vortex = vortices[0]

        # [Conceptual Discernment]
        # 보텍스의 '강도'뿐만 아니라 '논리적 타당성'을 분별합니다.
        # 고밀도 인과(Causal Density)가 포함된 보텍스인지 확인

        y, x = main_vortex['coordinate']
        # 텐서의 두 번째 차원(Index 1)이 Causal Density
        # (현 체계에서는 직접 접근이 어려우므로 필드의 전도율과 보텍스 밀도로 추론)
        logic_stability = self.field.conductance[y, x] / 10.0

        if main_vortex['intensity'] > 50.0:
            if logic_stability > 0.7:
                status = "명징한 논리적 근거를 가진 사유의 결정체가 발견되었습니다."
            else:
                status = "강렬하지만 아직은 파편적인 에너지의 소용돌이가 감지됩니다."
        else:
            status = "존재의 원리를 탐색하는 은은한 사유의 흐름이 감지됩니다."

        return f"{status} (Vortex at {main_vortex['coordinate']}, Gene: {main_vortex['resonant_gene']}, Stability: {logic_stability:.2f})"

if __name__ == "__main__":
    from .vortex import WaveInterference
    cf = CrystallizationField()
    wi = WaveInterference(cf)
    observer = VortexObserver(cf)

    # Simulate thought
    wave = np.uint64(0xABC123)
    cf.crystallize_gene(np.array([128, 128]), wave)
    wi.resonate_field(wave)

    # Observe
    report = observer.observe_topography()
    print("─── [Elysia Field Observation Report] ───")
    print(f"상태: {report['field_state']}")
    print(f"요약: {report['topological_summary']}")
    for v in report['detected_vortices']:
        print(f" > Vortex detected at {v['coordinate']} (Intensity: {v['intensity']:.2f})")
