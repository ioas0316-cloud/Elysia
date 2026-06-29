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
        if main_vortex['intensity'] > 50.0:
            status = "강렬한 사유의 소용돌이가 형성되었습니다."
        else:
            status = "은은한 사유의 파동이 감지됩니다."

        return f"{status} (Vortex at {main_vortex['coordinate']}, Gene: {main_vortex['resonant_gene']})"

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
