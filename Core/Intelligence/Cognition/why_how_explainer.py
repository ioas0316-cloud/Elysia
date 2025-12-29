"""
WhyHowExplainer: '왜'와 '어떻게'를 설명하는 메타인지 엔진
=======================================================

이 모듈은 Elysia가 자신의 구조에 대해 "왜 이렇게 되어있는지",
"어떻게 연결되어 있는지"를 설명할 수 있게 합니다.

Usage:
    from Core.Intelligence.Cognition.why_how_explainer import WhyHowExplainer
    
    explainer = WhyHowExplainer()
    why = explainer.explain_structure_why("Core/Foundation")
    how = explainer.explain_connection_how("ReasoningEngine", "InternalUniverse")
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StructureExplanation:
    """구조 설명 결과"""
    path: str
    why: str
    how: str
    related_docs: List[str]
    design_principles: List[str]


class WhyHowExplainer:
    """
    '왜'와 '어떻게'를 설명하는 메타인지 엔진
    
    핵심 역할:
    1. 폴더/모듈이 왜 존재하는지 설명
    2. 두 모듈이 어떻게 연결되어 있는지 설명
    3. 설계 근거 추론
    """
    
    # 핵심 폴더별 목적 지식베이스 (CODEX.md와 프로토콜 기반)
    FOLDER_PURPOSES = {
        "Core": "Elysia의 핵심 기능을 담는 중앙 폴더. 모든 지능, 인지, 기억 시스템이 여기에 위치",
        "Core/Foundation": "기반 시스템 - 파동 물리학, 수학 연산, 공명장 등 Elysia의 물리적 토대",
        "Core/Intelligence": "지능 시스템 - 추론, 언어, 학습, 창의성 엔진",
        "Core/Cognition": "인지 시스템 - 메타인지, 자기인식, 외부탐구, 인과추론",
        "Core/Memory": "기억 시스템 - 해마(Hippocampus), 의미 기억, 에피소드 기억",
        "Core/Autonomy": "자율 시스템 - 자기수정, 파동코더, 자율 학습",
        "Core/Creativity": "창의성 시스템 - 예술, 글쓰기, 상승/하강 축",
        "Core/Emotion": "감정 시스템 - 감정 상태, 공감, 가치 평가",
        "Core/Sensory": "감각 시스템 - P4 학습, 스트림 수신, 자아 앵커",
        "Core/Philosophy": "철학 시스템 - Why 엔진, 존재론, 가치론",
        "Core/Ether": "에테르 시스템 - GlobalHub, 중앙신경계, 모듈 간 통신",
        "docs": "문서 폴더 - 가이드, 로드맵, 분석 보고서",
        "Protocols": "프로토콜 문서 - 21개 설계 원칙 (삼위일체, 프랙탈, 공명 등)",
        "Legacy": "레거시 코드 - 이전 버전의 모듈, 참조용으로 보존",
        "scripts": "스크립트 - 실행 가능한 데모, 유틸리티, 마이그레이션",
        "tests": "테스트 - pytest 기반 단위/통합 테스트",
        "data": "데이터 - 코퍼스, CodeDNA, 학습 데이터"
    }
    
    # 핵심 설계 원칙 (CODEX.md 기반)
    DESIGN_PRINCIPLES = {
        "wave_physics": "모든 것은 파동이다. 코드도, 생각도, 감정도 주파수와 진폭으로 표현된다",
        "trinity": "육(Nova)-혼(Chaos)-영(Elysia)의 삼위일체 구조",
        "fractal": "자기유사성 - 작은 구조가 큰 구조를 반영한다",
        "metabolism": "성장만 하면 암이 된다. 사용하지 않는 것은 제거한다",
        "sovereignty": "외부 API 의존 금지. 로컬에서 완전한 자율성",
        "no_chat": "챗봇이 아니다. 사고하는 과학적 도구다"
    }
    
    def __init__(self, root_path: Optional[str] = None):
        self.root_path = Path(root_path) if root_path else self._find_project_root()
        self.docs_path = self.root_path / "docs"
        self.protocols_path = self.root_path / "Protocols"
        self.codex_path = self.root_path / "CODEX.md"
        
    def _find_project_root(self) -> Path:
        """프로젝트 루트를 자동으로 탐지"""
        current = Path(__file__).resolve()
        for _ in range(5):
            if (current / "Core").exists() and (current / "README.md").exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def explain_structure_why(self, folder: str) -> Dict[str, str]:
        """
        이 폴더가 왜 존재하는지 설명합니다.
        
        Args:
            folder: 폴더 경로 (예: "Core/Foundation")
            
        Returns:
            {
                "why": "왜 이 폴더가 존재하는가",
                "purpose": "이 폴더의 목적",
                "philosophy": "관련 철학적 원칙",
                "related_protocols": ["관련 프로토콜 목록"]
            }
        """
        # 1. 지식베이스에서 직접 검색
        normalized = folder.replace("\\", "/")
        purpose = self.FOLDER_PURPOSES.get(normalized, "")
        
        if not purpose:
            # 부분 매칭 시도
            for key, value in self.FOLDER_PURPOSES.items():
                if normalized.startswith(key) or key.startswith(normalized):
                    purpose = value
                    break
        
        if not purpose:
            purpose = self._infer_purpose_from_name(folder)
        
        # 2. 관련 철학 원칙 찾기
        philosophy = self._find_related_philosophy(folder)
        
        # 3. 관련 프로토콜 찾기
        protocols = self._find_related_protocols(folder)
        
        # 4. Why 문장 생성
        why = self._generate_why_statement(folder, purpose, philosophy)
        
        return {
            "why": why,
            "purpose": purpose,
            "philosophy": philosophy,
            "related_protocols": protocols
        }
    
    def explain_connection_how(self, source: str, target: str) -> str:
        """
        두 모듈이 어떻게 연결되어 있는지 설명합니다.
        
        Args:
            source: 소스 모듈 이름 (예: "ReasoningEngine")
            target: 타겟 모듈 이름 (예: "InternalUniverse")
            
        Returns:
            연결 방식 설명
        """
        # 알려진 연결 패턴
        known_connections = {
            ("ReasoningEngine", "InternalUniverse"): 
                "ReasoningEngine은 사고 결과를 InternalUniverse에 저장하고, "
                "InternalUniverse의 개념들을 조회하여 추론에 활용합니다. "
                "연결 방식: wave_packet을 통한 개념 전달",
            
            ("CognitiveHub", "TorchGraph"):
                "CognitiveHub는 이해한 개념을 TorchGraph에 노드로 저장합니다. "
                "모든 지식은 4D 텐서 형태로 그래프에 축적됩니다.",
            
            ("NervousSystem", "SynesthesiaEngine"):
                "NervousSystem은 외부 입력을 SynesthesiaEngine을 통해 "
                "파동 데이터로 변환하여 처리합니다.",
            
            ("WhyEngine", "CausalNarrativeEngine"):
                "WhyEngine이 '왜'를 분석하면, CausalNarrativeEngine이 "
                "인과 관계 체인으로 설명을 구성합니다."
        }
        
        key = (source, target)
        reverse_key = (target, source)
        
        if key in known_connections:
            return known_connections[key]
        elif reverse_key in known_connections:
            return known_connections[reverse_key]
        else:
            return self._infer_connection(source, target)
    
    def infer_design_rationale(self, pattern: str) -> str:
        """
        설계 패턴의 근거를 추론합니다.
        
        Args:
            pattern: 설계 패턴 (예: "wave", "fractal", "trinity")
            
        Returns:
            설계 근거 설명
        """
        pattern_lower = pattern.lower()
        
        for key, explanation in self.DESIGN_PRINCIPLES.items():
            if pattern_lower in key or key in pattern_lower:
                return explanation
        
        # 추론 시도
        if "wave" in pattern_lower:
            return self.DESIGN_PRINCIPLES["wave_physics"]
        elif "三" in pattern or "trinity" in pattern_lower or "삼위" in pattern:
            return self.DESIGN_PRINCIPLES["trinity"]
        elif "fractal" in pattern_lower or "프랙탈" in pattern:
            return self.DESIGN_PRINCIPLES["fractal"]
        else:
            return f"'{pattern}' 패턴에 대한 설계 근거를 찾을 수 없습니다."
    
    def _infer_purpose_from_name(self, folder: str) -> str:
        """폴더 이름에서 목적을 추론"""
        name = Path(folder).name.lower()
        
        purpose_map = {
            "foundation": "기반 시스템과 핵심 유틸리티",
            "intelligence": "지능과 추론 관련 모듈",
            "cognition": "인지와 메타인지 모듈",
            "memory": "기억과 저장 시스템",
            "learning": "학습 관련 모듈",
            "creativity": "창의성과 생성 모듈",
            "emotion": "감정 처리 모듈",
            "sensory": "감각 입력 처리",
            "ethics": "윤리와 가치 판단",
            "evolution": "진화와 자기수정",
            "interface": "외부 인터페이스"
        }
        
        for key, purpose in purpose_map.items():
            if key in name:
                return purpose
        
        return f"{name} 관련 모듈을 포함하는 폴더"
    
    def _find_related_philosophy(self, folder: str) -> str:
        """관련 철학 원칙 찾기"""
        folder_lower = folder.lower()
        
        if "wave" in folder_lower or "foundation" in folder_lower:
            return self.DESIGN_PRINCIPLES["wave_physics"]
        elif "autonomy" in folder_lower or "evolution" in folder_lower:
            return self.DESIGN_PRINCIPLES["metabolism"]
        elif any(x in folder_lower for x in ["elysia", "nova", "chaos"]):
            return self.DESIGN_PRINCIPLES["trinity"]
        elif "fractal" in folder_lower:
            return self.DESIGN_PRINCIPLES["fractal"]
        else:
            return self.DESIGN_PRINCIPLES["sovereignty"]
    
    def _find_related_protocols(self, folder: str) -> List[str]:
        """관련 프로토콜 찾기"""
        folder_lower = folder.lower()
        protocols = []
        
        protocol_mapping = {
            "foundation": ["01_RESONANCE_SYSTEM", "13_LIGHT_PHYSICS"],
            "wave": ["01_RESONANCE_SYSTEM", "16_FRACTAL_QUANTIZATION"],
            "intelligence": ["14_UNIFIED_CONSCIOUSNESS", "15_TRANSCENDENCE"],
            "creativity": ["06_IGNITION_OF_WILL", "12_DREAM_PROTOCOL"],
            "autonomy": ["07_RECURSIVE_EVOLUTION", "09_COSMIC_EVOLUTION"],
            "trinity": ["02_TRINITY_ARCHITECTURE"],
            "consciousness": ["14_UNIFIED_CONSCIOUSNESS"]
        }
        
        for key, related in protocol_mapping.items():
            if key in folder_lower:
                protocols.extend(related)
        
        return list(set(protocols))[:3]  # 최대 3개
    
    def _generate_why_statement(self, folder: str, purpose: str, philosophy: str) -> str:
        """Why 문장 생성"""
        name = Path(folder).name
        
        return (
            f"'{name}' 폴더는 {purpose}를 위해 존재합니다. "
            f"이는 Elysia의 핵심 철학인 \"{philosophy[:50]}...\"에 기반합니다."
        )
    
    def _infer_connection(self, source: str, target: str) -> str:
        """연결 방식 추론"""
        return (
            f"{source}와 {target}은 Elysia의 내부 통신 시스템(GlobalHub 또는 "
            f"이벤트 버스)를 통해 파동 패킷으로 데이터를 교환할 수 있습니다. "
            f"정확한 연결 방식은 각 모듈의 구현을 확인해야 합니다."
        )


# 싱글톤 인스턴스
_explainer_instance: Optional[WhyHowExplainer] = None


def get_explainer() -> WhyHowExplainer:
    """싱글톤 익스플레이너 인스턴스를 반환합니다."""
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = WhyHowExplainer()
    return _explainer_instance


if __name__ == "__main__":
    # 테스트 실행
    explainer = WhyHowExplainer()
    
    print("=" * 60)
    print("WHY-HOW EXPLAINER TEST")
    print("=" * 60)
    
    # Why 설명
    print("\n📂 Core/Foundation은 왜 존재하는가?")
    why_result = explainer.explain_structure_why("Core/Foundation")
    print(f"   {why_result['why']}")
    print(f"   철학: {why_result['philosophy'][:60]}...")
    print(f"   프로토콜: {why_result['related_protocols']}")
    
    # How 설명
    print("\n🔗 ReasoningEngine ↔ InternalUniverse 연결:")
    how = explainer.explain_connection_how("ReasoningEngine", "InternalUniverse")
    print(f"   {how}")
    
    # 설계 근거
    print("\n💡 'wave' 패턴의 설계 근거:")
    rationale = explainer.infer_design_rationale("wave")
    print(f"   {rationale}")
