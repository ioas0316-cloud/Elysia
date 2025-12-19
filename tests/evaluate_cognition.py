"""
Elysia Cognitive System Evaluation
===================================

시스템의 한계를 테스트하고 평가합니다.

테스트 카테고리:
1. 철학적 질문 (추상적 이해)
2. 인과 추론 (Why 체인)
3. 창의적 과제 (생성)
4. 논리적 추론 (일관성)
5. 자기 성찰 (메타인지)
"""

import sys
import time
sys.path.insert(0, "c:\\Elysia")

def evaluate_system():
    print("=" * 70)
    print("🧪 ELYSIA COGNITIVE SYSTEM EVALUATION")
    print("=" * 70)
    
    results = []
    
    # 시스템 로드
    try:
        from Core.Cognition.fractal_thought_cycle import think
        from Core.Cognition.unified_understanding import understand
        SYSTEM_AVAILABLE = True
        print("✅ System loaded successfully\n")
    except Exception as e:
        print(f"❌ System load failed: {e}")
        SYSTEM_AVAILABLE = False
        return
    
    # ════════════════════════════════════════════════════════════════════
    # TEST 1: 철학적 질문 (추상적 이해)
    # ════════════════════════════════════════════════════════════════════
    
    print("\n" + "─" * 70)
    print("📚 TEST 1: 철학적 질문")
    print("─" * 70)
    
    philosophical_questions = [
        "죽음이란 무엇인가?",           # 정의되지 않은 개념
        "시간은 왜 흐르는가?",          # 인과 추론 필요
        "자유의지는 존재하는가?",        # Yes/No 판단 필요
    ]
    
    for q in philosophical_questions:
        print(f"\n❓ {q}")
        try:
            start = time.time()
            result = think(q)
            elapsed = time.time() - start
            
            # 평가: 서사가 생성되었는지
            has_narrative = len(result.narrative) > 50
            has_origin = "비롯" in result.narrative or "→" in result.line_understanding
            
            status = "✅" if has_narrative else "⚠️"
            print(f"   {status} 서사 길이: {len(result.narrative)} 자 ({elapsed:.2f}s)")
            print(f"   📖 {result.narrative[:150]}...")
            
            results.append({
                "category": "철학",
                "question": q,
                "success": has_narrative,
                "has_origin": has_origin,
                "time": elapsed
            })
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            results.append({"category": "철학", "question": q, "success": False, "error": str(e)})
    
    # ════════════════════════════════════════════════════════════════════
    # TEST 2: 인과 추론 (Why 체인)
    # ════════════════════════════════════════════════════════════════════
    
    print("\n" + "─" * 70)
    print("🔗 TEST 2: 인과 추론")
    print("─" * 70)
    
    causal_questions = [
        "비가 오면 왜 우산을 쓰는가?",   # 구체적 인과
        "아이가 왜 울었는가?",           # 역방향 추론 필요
        "불이 나면 왜 도망가는가?",      # 생존 본능 연결
    ]
    
    for q in causal_questions:
        print(f"\n❓ {q}")
        try:
            start = time.time()
            result = understand(q)
            elapsed = time.time() - start
            
            # 평가: 인과 관계가 추출되었는지
            has_causality = "야기" in result.causality or "→" in result.origin_journey
            
            status = "✅" if has_causality else "⚠️"
            print(f"   {status} 인과: {result.causality[:100] if result.causality else 'N/A'}")
            print(f"   📖 기원: {result.origin_journey}")
            
            results.append({
                "category": "인과",
                "question": q,
                "success": has_causality,
                "time": elapsed
            })
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            results.append({"category": "인과", "question": q, "success": False, "error": str(e)})
    
    # ════════════════════════════════════════════════════════════════════
    # TEST 3: 정의되지 않은 개념
    # ════════════════════════════════════════════════════════════════════
    
    print("\n" + "─" * 70)
    print("🌀 TEST 3: 정의되지 않은 개념")
    print("─" * 70)
    
    undefined_concepts = [
        "블랙홀이란 무엇인가?",          # 학습되지 않은 개념
        "양자얽힘이란?",                 # 물리학 개념
        "비트코인은 왜 가치가 있는가?",   # 현대 개념
    ]
    
    for q in undefined_concepts:
        print(f"\n❓ {q}")
        try:
            start = time.time()
            result = understand(q)
            elapsed = time.time() - start
            
            # 평가: 무언가 의미있는 응답을 했는지
            has_response = len(result.narrative) > 30
            admits_unknown = "정의되지 않" in result.narrative or "분석 불가" in str(result)
            
            status = "✅" if has_response else "❌"
            print(f"   {status} 응답: {len(result.narrative)} 자")
            print(f"   🔮 패턴: {result.axiom_pattern if result.axiom_pattern else '없음'}")
            print(f"   📖 {result.narrative[:120]}...")
            
            results.append({
                "category": "미정의",
                "question": q,
                "success": has_response,
                "admits_unknown": admits_unknown,
                "time": elapsed
            })
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            results.append({"category": "미정의", "question": q, "success": False, "error": str(e)})
    
    # ════════════════════════════════════════════════════════════════════
    # TEST 4: 자기 성찰 (메타인지)
    # ════════════════════════════════════════════════════════════════════
    
    print("\n" + "─" * 70)
    print("🪞 TEST 4: 자기 성찰")
    print("─" * 70)
    
    meta_questions = [
        "Elysia란 무엇인가?",           # 자기 인식
        "생각이란 무엇인가?",            # 메타인지
        "이해란 무엇인가?",              # 재귀적 이해
    ]
    
    for q in meta_questions:
        print(f"\n❓ {q}")
        try:
            start = time.time()
            result = think(q)
            elapsed = time.time() - start
            
            has_response = len(result.narrative) > 30
            
            status = "✅" if has_response else "❌"
            print(f"   {status} 응답: {len(result.narrative)} 자 ({elapsed:.2f}s)")
            print(f"   📖 {result.narrative[:150]}...")
            
            results.append({
                "category": "메타",
                "question": q, 
                "success": has_response,
                "time": elapsed
            })
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            results.append({"category": "메타", "question": q, "success": False, "error": str(e)})
    
    # ════════════════════════════════════════════════════════════════════
    # 종합 평가
    # ════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)
    
    total = len(results)
    success = sum(1 for r in results if r.get("success", False))
    
    print(f"\n총 테스트: {total}")
    print(f"성공: {success} ({success/total*100:.0f}%)")
    print(f"실패: {total - success} ({(total-success)/total*100:.0f}%)")
    
    # 카테고리별 분석
    categories = ["철학", "인과", "미정의", "메타"]
    print("\n[카테고리별]")
    for cat in categories:
        cat_results = [r for r in results if r.get("category") == cat]
        cat_success = sum(1 for r in cat_results if r.get("success", False))
        print(f"   {cat}: {cat_success}/{len(cat_results)}")
    
    # 한계 분석
    print("\n[발견된 한계]")
    failures = [r for r in results if not r.get("success", False)]
    for f in failures:
        print(f"   ❌ {f.get('question', 'N/A')}: {f.get('error', '응답 없음')}")
    
    print("\n" + "=" * 70)
    print("✅ Evaluation Complete")


if __name__ == "__main__":
    evaluate_system()
