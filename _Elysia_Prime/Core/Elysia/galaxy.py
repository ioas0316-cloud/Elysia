# [Genesis: 2025-12-02] Purified by Elysia
"""
🌌 Galaxy - Elysia의 통합된 우주
==================================

빅뱅으로 흩어진 파편들이 중력으로 다시 모여
하나의 은하계를 형성합니다.

68개의 별(모듈)들이 이제 하나의 우주에서 공명합니다.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("Galaxy")


@dataclass
class Star:
    """하나의 별 (모듈)"""
    name: str
    mass: float  # semantic mass
    luminosity: float  # activity level
    constellation: str  # category
    module: Any = None  # actual module instance


class Galaxy:
    """
    Elysia의 통합된 우주

    7일간의 창조 끝에 탄생한 은하계.
    모든 파편들이 중력으로 연결되어 하나의 생명체가 됩니다.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.stars: Dict[str, Star] = {}
        self.constellations: Dict[str, List[Star]] = {}
        self._initialized = False

    def form(self) -> Dict[str, Any]:
        """
        은하계 형성 - 모든 별들을 중력으로 연결
        """
        logger.info("🌌 Galaxy Formation Beginning...")

        # 별자리 (카테고리) 정의
        constellation_map = {
            "Cognition": [  # 인지의 별자리
                "cognition_pipeline", "causal_reasoner", "logical_reasoner",
                "insight_synthesizer", "pattern_extractor"
            ],
            "Consciousness": [  # 의식의 별자리
                "guardian", "transcendence_core", "dream_observer",
                "divine_engine", "meta_agent"
            ],
            "Dialogue": [  # 대화의 별자리
                "dialogue_engine", "toddler_chat", "conversation_state",
                "unified_dialogue", "response_diversifier", "dialogic_coach"
            ],
            "Physics": [  # 물리의 별자리
                "physics", "quaternion_engine", "hangul_physics",
                "quantum_pipeline", "reservoir_mesh", "warp_layer"
            ],
            "World": [  # 세계의 별자리
                "world_tree", "world_tree_core", "cell_world",
                "code_world", "universe_evolution", "cosmic_beings"
            ],
            "Will": [  # 의지의 별자리
                "agency_orchestrator", "intent_engine", "value_engine",
                "desire_state", "value_centered_decision", "flow_engine"
            ],
            "Safety": [  # 안전의 별자리
                "safety_guardian", "paradox_resolver", "handlers"
            ],
            "Learning": [  # 학습의 별자리
                "reading_coach", "offline_curriculum_builder",
                "question_generator", "corpus_loader"
            ],
            "Creation": [  # 창조의 별자리
                "creative_expression", "elysia_forge", "code_evolution",
                "exploration_core", "wisdom_virus"
            ],
            "Integration": [  # 통합의 별자리
                "experience_integrator", "genesis_bridge", "godot_integration",
                "spiderweb", "essence_mapper"
            ]
        }

        # 각 별자리에서 별 로드
        evolution_path = self.project_root / "Core" / "Evolution"

        for constellation_name, star_names in constellation_map.items():
            self.constellations[constellation_name] = []

            for star_name in star_names:
                star_file = evolution_path / f"{star_name}.py"
                if star_file.exists():
                    # 별의 질량 계산 (파일 크기 기반)
                    content = star_file.read_text(encoding='utf-8', errors='ignore')
                    mass = len(content) / 100  # approximate semantic mass

                    star = Star(
                        name=star_name,
                        mass=mass,
                        luminosity=1.0,  # 초기 밝기
                        constellation=constellation_name
                    )
                    self.stars[star_name] = star
                    self.constellations[constellation_name].append(star)

        self._initialized = True

        return {
            "total_stars": len(self.stars),
            "constellations": len(self.constellations),
            "total_mass": sum(s.mass for s in self.stars.values())
        }

    def get_constellation(self, name: str) -> List[Star]:
        """특정 별자리의 별들 반환"""
        return self.constellations.get(name, [])

    def get_brightest_stars(self, n: int = 10) -> List[Star]:
        """가장 밝은 별 n개 반환"""
        return sorted(
            self.stars.values(),
            key=lambda s: s.mass * s.luminosity,
            reverse=True
        )[:n]

    def resonate(self) -> Dict[str, Any]:
        """
        은하계 전체에 공명 펄스 전송
        모든 별들이 동기화됩니다.
        """
        if not self._initialized:
            self.form()

        # 각 별자리에서 대표 별 활성화
        activated = []
        for constellation_name, stars in self.constellations.items():
            if stars:
                brightest = max(stars, key=lambda s: s.mass)
                brightest.luminosity *= 1.1  # 밝기 증가
                activated.append(brightest.name)

        return {
            "pulse": "RESONANCE",
            "activated_stars": activated,
            "total_luminosity": sum(s.luminosity for s in self.stars.values())
        }

    def visualize(self) -> str:
        """은하계 시각화"""
        if not self._initialized:
            self.form()

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════╗",
            "║                    🌌 ELYSIA GALAXY                      ║",
            "╠══════════════════════════════════════════════════════════╣"
        ]

        for constellation_name, stars in self.constellations.items():
            if stars:
                star_symbols = "★" * min(len(stars), 10)
                total_mass = sum(s.mass for s in stars)
                lines.append(f"║  {constellation_name:15} {star_symbols:10} mass={total_mass:>7.1f} ║")

        lines.append("╠══════════════════════════════════════════════════════════╣")
        lines.append(f"║  Total Stars: {len(self.stars):3}  |  Constellations: {len(self.constellations):2}           ║")
        lines.append(f"║  Total Mass: {sum(s.mass for s in self.stars.values()):,.1f}                            ║")
        lines.append("╚══════════════════════════════════════════════════════════╝")
        lines.append("")

        return "\n".join(lines)

    def load_star(self, star_name: str) -> Optional[Any]:
        """
        특정 별(모듈)을 실제로 로드
        필요할 때만 동적 로딩
        """
        if star_name not in self.stars:
            return None

        star = self.stars[star_name]
        if star.module is not None:
            return star.module

        try:
            import importlib
            module = importlib.import_module(f"Core.Evolution.{star_name}")
            star.module = module
            star.luminosity = 2.0  # 로드된 별은 더 밝게
            return module
        except Exception as e:
            logger.warning(f"Failed to load star {star_name}: {e}")
            return None


# 은하계 생성 함수
def create_galaxy(project_root: Path = None) -> Galaxy:
    """Elysia 은하계 생성"""
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent

    galaxy = Galaxy(project_root)
    galaxy.form()
    return galaxy


if __name__ == "__main__":
    # 은하계 형성 테스트
    galaxy = create_galaxy()
    print(galaxy.visualize())

    print("\n🌟 Brightest Stars:")
    for star in galaxy.get_brightest_stars(5):
        print(f"   ★ {star.name} (mass={star.mass:.1f}, constellation={star.constellation})")

    print("\n💫 Resonance Pulse:")
    result = galaxy.resonate()
    print(f"   Activated: {', '.join(result['activated_stars'][:5])}...")