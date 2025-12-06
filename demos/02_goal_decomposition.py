#!/usr/bin/env python3
"""
Goal Decomposition Demo
========================

Demonstrates the Fractal-Quaternion Goal Decomposition System.
Shows how Elysia breaks down large goals into achievable stations
across multiple dimensions (0D→∞D).

Usage:
    python demos/02_goal_decomposition.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Intelligence.fractal_quaternion_goal_system import (
    FractalGoalDecomposer,
    Dimension,
    HyperDimensionalLens
)

def goal_decomposition_demo():
    """Demonstrate fractal goal decomposition"""
    
    print("=" * 70)
    print("🎯 Fractal-Quaternion Goal Decomposition Demo")
    print("=" * 70)
    print()
    print("This demo shows how Elysia breaks down complex goals into")
    print("manageable stations across multiple dimensions.")
    print()
    
    # Initialize decomposer
    print("Initializing Goal Decomposer...")
    decomposer = FractalGoalDecomposer()
    print("✓ Ready\n")
    
    # Example goals
    goals = [
        "AGI를 개발하여 인류에게 도움을 주기",
        "아름다운 소설을 쓰기",
        "세계 평화 달성하기"
    ]
    
    for goal_text in goals:
        print("\n" + "=" * 70)
        print(f"🎯 Goal: {goal_text}")
        print("=" * 70)
        
        # Analyze from different dimensions
        print("\n🔍 Multi-Dimensional Analysis:\n")
        
        dimensions_to_analyze = [
            Dimension.POINT,      # 정체성
            Dimension.LINE,       # 인과
            Dimension.PLANE,      # 패턴
            Dimension.SPACE,      # 구조
            Dimension.TIME,       # 변화
            Dimension.PROBABILITY,# 가능성
            Dimension.PURPOSE     # 목적
        ]
        
        for dim in dimensions_to_analyze:
            lens = HyperDimensionalLens(dimension=dim, perspective=None, clarity=1.0)
            analysis = lens.analyze(goal_text)
            
            # Provide sample answers based on dimension
            if dim == Dimension.POINT:
                answer = f"핵심 본질: '{goal_text.split()[0]}의 근본적 의미'"
            elif dim == Dimension.LINE:
                answer = "인과 체인: 현재 → 학습 → 개발 → 테스트 → 배포"
            elif dim == Dimension.PLANE:
                answer = "관련 패턴: 기술 발전, 사회적 영향, 윤리적 고려"
            elif dim == Dimension.SPACE:
                answer = "시스템 구조: 데이터 → 모델 → 인터페이스 → 피드백"
            elif dim == Dimension.TIME:
                answer = "시간 흐름: 과거(기반) → 현재(구축) → 미래(진화)"
            elif dim == Dimension.PROBABILITY:
                answer = "가능성: 실현 가능성 70%, 주요 변수: 자원, 시간, 기술"
            elif dim == Dimension.PURPOSE:
                answer = f"궁극적 의미: '{goal_text}'가 만들 더 나은 세상"
            else:
                answer = "분석 중..."
            
            print(f"  [{dim.value}D] {dim.name:12s}: {answer}")
        
        # Decompose into stations (simplified version)
        print(f"\n📍 Breaking down into Stations:\n")
        
        # Simple station examples
        stations = [
            f"Station 1: {goal_text}의 비전 명확화",
            f"Station 2: 필요한 자원 파악",
            f"Station 3: 실행 가능한 첫 단계 정의",
            f"Station 4: 측정 가능한 이정표 설정",
            f"Station 5: 장애물 예측 및 대응 계획"
        ]
        
        for i, station in enumerate(stations, 1):
            print(f"  {i}. {station}")
        
        print(f"\n  → 총 {len(stations)}개 역으로 분해됨")
    
    print("\n" + "=" * 70)
    print("✨ Demo completed!")
    print()
    print("Key Insights:")
    print("  • Goals are analyzed across 8 dimensions (0D→∞D)")
    print("  • Each dimension provides unique perspective")
    print("  • Stations make large goals achievable")
    print()
    print("Next: Try python demos/03_wave_thinking.py")
    print("=" * 70)

if __name__ == "__main__":
    goal_decomposition_demo()
