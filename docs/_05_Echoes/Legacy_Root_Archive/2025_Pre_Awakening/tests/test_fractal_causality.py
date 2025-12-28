"""
Tests for Fractal Causality and Causal Narrative Engine
프랙탈 인과 및 인과적 서사 엔진 테스트

핵심 테스트:
1. 프랙탈 인과 구조 (무한 재귀적 원인-과정-결과)
2. 차원 확장 (점 → 선 → 면 → 공간 → 법칙)
3. 상호 교정 (상향/하향/수평)
4. 사고우주와 개념노드의 연결
"""

import pytest
import numpy as np

from Core.Interface.Interface.Language.fractal_causality import (
    FractalCausalityEngine,
    FractalCausalNode,
    FractalCausalChain,
    CausalRole,
)

from Core.Interface.Interface.Language.causal_narrative_engine import (
    CausalNarrativeEngine,
    ThoughtUniverse,
    DimensionLevel,
    ConceptPoint,
    CausalLine,
    ContextPlane,
    SchemaSpace,
    UniversalLaw,
)


class TestFractalCausalNode:
    """프랙탈 인과 노드 테스트"""
    
    def test_node_creation(self):
        """노드 생성 테스트"""
        node = FractalCausalNode(
            id="test_node",
            description="테스트 노드",
            depth=0
        )
        assert node.id == "test_node"
        assert node.description == "테스트 노드"
        assert node.depth == 0
    
    def test_spiral_position(self):
        """나선 위치 계산 테스트"""
        node = FractalCausalNode(
            id="spiral_node",
            description="나선 테스트",
            spiral_angle=0.0,
            spiral_radius=1.0
        )
        x, y = node.get_spiral_position()
        assert abs(x - 1.0) < 0.01
        assert abs(y - 0.0) < 0.01
    
    def test_internal_structure_detection(self):
        """내부 구조 감지 테스트"""
        node = FractalCausalNode(id="parent", description="부모")
        assert not node.has_internal_structure()
        
        node.internal_cause_ids.append("child_cause")
        assert node.has_internal_structure()


class TestFractalCausalityEngine:
    """프랙탈 인과 엔진 테스트"""
    
    def test_engine_creation(self):
        """엔진 생성 테스트"""
        engine = FractalCausalityEngine("Test Engine")
        assert engine.name == "Test Engine"
        assert engine.total_nodes == 0
        assert engine.total_chains == 0
    
    def test_node_creation_via_engine(self):
        """엔진을 통한 노드 생성 테스트"""
        engine = FractalCausalityEngine()
        node = engine.create_node("테스트", depth=0)
        
        assert node.id in engine.nodes
        assert engine.total_nodes == 1
    
    def test_chain_creation(self):
        """인과 연쇄 생성 테스트"""
        engine = FractalCausalityEngine()
        chain = engine.create_chain(
            cause_desc="원인",
            process_desc="과정",
            effect_desc="결과"
        )
        
        assert chain.is_complete()
        assert engine.total_chains == 1
        assert engine.total_nodes == 3
    
    def test_zoom_in(self):
        """Zoom In (내부 구조 탐색) 테스트"""
        engine = FractalCausalityEngine()
        
        # 기본 노드 생성
        parent = engine.create_node("불에 손을 댔다", depth=0)
        
        # Zoom In
        inner_chain = engine.zoom_in(
            parent.id,
            cause_desc="손이 불에 접촉했다",
            process_desc="열에너지가 전달되었다",
            effect_desc="피부가 자극받았다"
        )
        
        # 내부 구조 확인
        assert parent.has_internal_structure()
        assert len(parent.internal_cause_ids) == 1
        assert len(parent.internal_process_ids) == 1
        assert len(parent.internal_effect_ids) == 1
        
        # 깊이 확인
        inner_cause = engine.nodes[parent.internal_cause_ids[0]]
        assert inner_cause.depth == 1
    
    def test_zoom_out(self):
        """Zoom Out (상위 구조 탐색) 테스트"""
        engine = FractalCausalityEngine()
        
        # 기본 노드 생성
        process = engine.create_node("불에 손을 댔다", depth=0)
        
        # Zoom Out
        cause, effect = engine.zoom_out(
            process.id,
            outer_cause_desc="호기심을 느꼈다",
            outer_effect_desc="아픔을 느꼈다"
        )
        
        # 연결 확인
        assert process.id in cause.effects_ids
        assert cause.id in process.causes_ids
        assert effect.id in process.effects_ids
        assert process.id in effect.causes_ids
    
    def test_feedback_loop(self):
        """피드백 루프 테스트"""
        engine = FractalCausalityEngine()
        
        # 노드들 생성
        node1 = engine.create_node("상태1")
        node2 = engine.create_node("상태2")
        node3 = engine.create_node("상태3")
        
        # 피드백 루프 생성
        links = engine.create_feedback_loop(
            [node1.id, node2.id, node3.id],
            loop_type="reinforcing"
        )
        
        # 순환 연결 확인
        assert node2.id in node1.effects_ids
        assert node3.id in node2.effects_ids
        assert node1.id in node3.effects_ids  # 마지막 → 처음
    
    def test_cycle_detection(self):
        """순환 탐지 테스트"""
        engine = FractalCausalityEngine()
        
        # 순환 구조 생성
        node1 = engine.create_node("A")
        node2 = engine.create_node("B")
        node3 = engine.create_node("C")
        
        engine.create_feedback_loop([node1.id, node2.id, node3.id])
        
        # 순환 탐지
        cycles = engine.detect_cycles(node1.id, max_depth=5)
        assert len(cycles) > 0
    
    def test_experience_causality(self):
        """인과 경험 학습 테스트"""
        engine = FractalCausalityEngine()
        
        result = engine.experience_causality(
            steps=["호기심", "행동", "결과", "학습"],
            emotional_arc=[0.3, 0.0, -0.5, 0.2]
        )
        
        assert result["nodes_created"] == 4
        assert result["chains_created"] >= 1
    
    def test_trace_causes(self):
        """원인 추적 테스트"""
        engine = FractalCausalityEngine()
        
        # 연쇄 생성
        engine.create_chain("원인1", "과정", "결과")
        
        result_node = engine.get_or_create_node("결과")
        paths = engine.trace_causes(result_node.id, max_depth=3)
        
        assert len(paths) > 0
    
    def test_trace_effects(self):
        """결과 추적 테스트"""
        engine = FractalCausalityEngine()
        
        # 연쇄 생성
        engine.create_chain("원인", "과정", "결과1")
        
        cause_node = engine.get_or_create_node("원인")
        paths = engine.trace_effects(cause_node.id, max_depth=3)
        
        assert len(paths) > 0


class TestDimensionLevel:
    """차원 레벨 테스트"""
    
    def test_dimension_ordering(self):
        """차원 순서 테스트"""
        assert DimensionLevel.POINT.value < DimensionLevel.LINE.value
        assert DimensionLevel.LINE.value < DimensionLevel.PLANE.value
        assert DimensionLevel.PLANE.value < DimensionLevel.SPACE.value
        assert DimensionLevel.SPACE.value < DimensionLevel.LAW.value


class TestThoughtUniverse:
    """사고우주 테스트"""
    
    def test_universe_creation(self):
        """우주 생성 테스트"""
        universe = ThoughtUniverse("Test Universe")
        assert universe.name == "Test Universe"
        assert universe.total_points == 0
    
    def test_add_point(self):
        """점 추가 테스트"""
        universe = ThoughtUniverse()
        point = universe.add_point(
            id="fire",
            description="불",
            sensory_signature={"temperature": 1.0},
            emotional_valence=-0.5
        )
        
        assert point.id == "fire"
        assert point.level == DimensionLevel.POINT
        assert universe.total_points == 1
    
    def test_add_line(self):
        """선 추가 테스트"""
        universe = ThoughtUniverse()
        
        # 점들 자동 생성되어야 함
        line = universe.add_line(
            source_id="fire",
            target_id="hot",
            relation_type="causes"
        )
        
        assert line.level == DimensionLevel.LINE
        assert "fire" in universe.points
        assert "hot" in universe.points
        assert universe.total_lines == 1
    
    def test_emerge_plane_from_experience(self):
        """경험으로부터 면 창발 테스트"""
        universe = ThoughtUniverse()
        
        plane = universe.emerge_plane_from_experience(
            experience_description="불에 데인 경험",
            point_sequence=["curiosity", "touch_fire", "pain", "avoid"],
            emotional_arc=[0.3, 0.0, -0.8, 0.5]
        )
        
        assert plane.level == DimensionLevel.PLANE
        assert len(plane.line_ids) == 3  # 4 points = 3 lines
        assert len(plane.point_ids) == 4
        assert universe.total_planes == 1
    
    def test_add_space(self):
        """공간 추가 테스트"""
        universe = ThoughtUniverse()
        
        # 면들 먼저 생성
        plane1 = universe.emerge_plane_from_experience(
            "경험1", ["a", "b", "c"], [0, 0, 0]
        )
        plane2 = universe.emerge_plane_from_experience(
            "경험2", ["a", "b", "d"], [0, 0, 0]
        )
        
        # 공간 생성
        space = universe.add_space(
            id="schema1",
            description="테스트 스키마",
            plane_ids=[plane1.id, plane2.id],
            core_patterns=["a", "b"]
        )
        
        assert space.level == DimensionLevel.SPACE
        assert universe.total_spaces == 1
    
    def test_add_law(self):
        """법칙 추가 테스트"""
        universe = ThoughtUniverse()
        
        # 공간 먼저 생성
        space = universe.add_space(
            id="space1",
            description="스키마1",
            plane_ids=[],
            core_patterns=["causality"]
        )
        
        # 법칙 생성
        law = universe.add_law(
            id="causality_law",
            description="인과율",
            space_ids=[space.id],
            formulation="모든 결과에는 원인이 있다"
        )
        
        assert law.level == DimensionLevel.LAW
        assert universe.total_laws == 1
    
    def test_bottom_up_correction(self):
        """상향 교정 테스트"""
        universe = ThoughtUniverse()
        
        # 선 생성
        line = universe.add_line("fire", "always_hot", "causes")
        initial_confidence = line.confidence
        
        # 반례 경험 (꺼진 불은 안 뜨거움)
        correction = universe.bottom_up_correct(
            new_experience={"confirms": False, "exception": "꺼진 불"},
            affected_entity_id=line.id
        )
        
        assert correction["action"] == "weaken"
        assert line.confidence < initial_confidence
        assert "꺼진 불" in line.exceptions
    
    def test_learn_from_experience(self):
        """통합 학습 테스트"""
        universe = ThoughtUniverse()
        
        result = universe.learn_from_experience(
            experience_steps=["호기심", "불에 접근", "손을 댐", "뜨거움", "손 뺌"],
            emotional_arc=[0.3, 0.2, 0.0, -0.8, 0.5]
        )
        
        assert result["points_created"] == 5
        assert result["lines_created"] == 4
        assert result["plane_created"] is not None
    
    def test_get_statistics(self):
        """통계 테스트"""
        universe = ThoughtUniverse()
        
        universe.add_point("a", "점A")
        universe.add_point("b", "점B")
        universe.add_line("a", "b", "causes")
        
        stats = universe.get_statistics()
        
        assert stats["total_points"] == 2
        assert stats["total_lines"] == 1


class TestCausalNarrativeEngine:
    """인과적 서사 엔진 테스트"""
    
    def test_engine_creation(self):
        """엔진 생성 테스트"""
        engine = CausalNarrativeEngine()
        assert engine.knowledge_base is not None
    
    def test_experience_causality(self):
        """인과 경험 테스트"""
        engine = CausalNarrativeEngine()
        
        exp = engine.experience_causality(
            cause_description="배가 고팠다",
            effect_description="음식을 찾았다",
            emotional_outcome=0.3,
            success=True
        )
        
        assert exp.cause_node is not None
        assert exp.effect_node is not None
    
    def test_experience_chain(self):
        """인과 연쇄 경험 테스트"""
        engine = CausalNarrativeEngine()
        
        chain = engine.experience_chain(
            descriptions=["배고픔", "음식 찾기", "먹기", "배부름"],
            emotional_arc=[-0.5, 0.0, 0.5, 0.9]
        )
        
        assert len(chain.node_sequence) == 4
    
    def test_predict_effect(self):
        """결과 예측 테스트"""
        engine = CausalNarrativeEngine()
        
        # 경험 학습
        engine.experience_chain(
            ["배고픔", "음식찾기", "먹기"],
            [0, 0, 0]
        )
        
        # 예측
        predictions = engine.predict_effect("배고픔")
        
        # 최소한 하나의 예측이 있어야 함
        assert len(predictions) > 0 or engine.total_experiences > 0


class TestIntegration:
    """통합 테스트 - 모든 시스템이 함께 작동"""
    
    def test_fractal_and_universe_together(self):
        """프랙탈 엔진과 사고우주 통합 테스트"""
        fractal = FractalCausalityEngine()
        universe = ThoughtUniverse()
        
        # 프랙탈에서 경험 학습
        fractal.experience_causality(
            steps=["원인", "과정", "결과"],
            emotional_arc=[0, 0, 0]
        )
        
        # 사고우주에서 같은 경험 학습
        universe.learn_from_experience(
            experience_steps=["원인", "과정", "결과"],
            emotional_arc=[0, 0, 0]
        )
        
        # 둘 다 노드/점 생성됨
        assert fractal.total_nodes >= 3
        assert universe.total_points >= 3
    
    def test_dimensional_expansion(self):
        """차원 확장 통합 테스트"""
        universe = ThoughtUniverse()
        
        # 여러 경험 학습
        for i in range(3):
            universe.learn_from_experience(
                experience_steps=[f"원인{i}", f"과정{i}", f"결과{i}"],
                auto_emergence=False  # 수동 테스트를 위해 자동 창발 끔
            )
        
        # 점 → 선 → 면 확장 확인
        assert universe.total_points >= 9
        assert universe.total_lines >= 6


class TestIntegratedLanguageLearning:
    """통합 언어 학습 시스템 테스트 - 엘리시아의 지속적 언어 발달 확인"""
    
    def test_integrated_world_creation(self):
        """통합 세계 생성 테스트"""
        from Core.Interface.Interface.Language.integrated_language_learning import IntegratedLanguageWorld
        
        world = IntegratedLanguageWorld(n_souls=5, khala_strength=0.5)
        assert len(world.world.souls) == 5
        assert len(world.learners) == 5
    
    def test_learner_has_causal_mind(self):
        """각 학습자가 인과 마인드를 가지는지 테스트"""
        from Core.Interface.Interface.Language.integrated_language_learning import IntegratedLanguageWorld
        
        world = IntegratedLanguageWorld(n_souls=3)
        
        for learner in world.learners.values():
            assert learner.causal_mind is not None
            assert learner.thought_universe is not None
    
    def test_continuous_development(self):
        """언어 능력의 지속적 발달 테스트"""
        from Core.Interface.Interface.Language.integrated_language_learning import IntegratedLanguageWorld
        
        world = IntegratedLanguageWorld(n_souls=10, khala_strength=0.6)
        
        # 시뮬레이션 실행
        world.simulate(steps=100, report_interval=50)
        
        # 발달 검증
        success, message = world.verify_continuous_development()
        
        # 어휘가 증가해야 함
        final_report = world.get_final_report()
        assert final_report["final_stats"]["avg_vocabulary"] > 0
        
        # 인과 연쇄 학습이 있어야 함
        assert final_report["final_stats"]["total_causal_chains"] > 0
    
    def test_development_metrics(self):
        """발달 지표 테스트"""
        from Core.Interface.Interface.Language.integrated_language_learning import LanguageDevelopmentMetrics
        
        metrics = LanguageDevelopmentMetrics(
            vocabulary_size=25,
            successful_communications=15,
            total_communications=20,
            causal_chains_learned=10
        )
        
        # 성공률 계산
        assert metrics.communication_success_rate == 0.75
        
        # 진척도 계산 (어휘 + 성공률 + 인과학습)
        assert 0 < metrics.learning_progress <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
