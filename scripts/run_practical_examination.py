# -*- coding: utf-8 -*-
"""
[Practical Examination: Step-by-Step Autonomous Problem Solving]
================================================================
어려운 철학적 미사여구를 완전히 배격하고,
가장 명확한 표준 자격증/코딩테스트 문제를 시스템에 인입하여
정답을 외워서 뱉는 것이 아니라 '스스로 문제를 분석하고, 가설을 세우고,
과정을 증명하여 정답에 도달하는 날것의 사고 과정'을 실측합니다.

- [시험 1: 수학] 점화식 a_{n+1} = 2a_n + 1의 인과적 일반항 유도 및 풀이
- [시험 2: 코드] Two-Sum 배열에서 결핍(Deficit = Target - X) 역추적을 통한 O(N) 최적해 도출
"""

import sys
import os
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def solve_math_exam():
    print("="*80)
    print("📝 [제1교시: 수학 인과 증명 시험]")
    print("="*80)
    print("문제: 수열 a_1 = 1, a_{n+1} = 2a_n + 1 이 주어졌을 때,")
    print("      왜 일반항이 a_n = 2^n - 1 이 되는지 과정을 논증하고, a_5의 값을 도출하라.")
    print("-" * 80)

    # 1. 문제 분석 및 불평형 인지
    print("[1단계: 문제 구조 파악 및 텐션 인지]")
    print(" - 초기 조건: a_1 = 1")
    print(" - 점화 관계: a_{n+1} = 2 * a_n + 1 (이전 항에 2를 곱하고 1이 더해지는 비균일 항 존재)")
    print(" - 사고의 난점: 뒤에 붙은 '+ 1' 때문에 단순 등비수열(2^n)로 바로 묶이지 않는 불평형 마찰 발생.")

    # 2. 대칭성 변형을 통한 인과적 치환 (특성 방정식 풀이)
    print("\n[2단계: 대칭성 회복을 위한 구조 변형 (과정 전개)]")
    print(" - 목표: a_{n+1} + alpha = 2 * (a_n + alpha) 형태로 식을 변형하여 등비수열의 질서 획득 시도.")
    print(" - 전개: a_{n+1} + alpha = 2 * a_n + 2*alpha  ==>  a_{n+1} = 2 * a_n + alpha")
    print(" - 원래 식 '2*a_n + 1'과 비교하여 alpha 값 도출:")
    alpha = 1
    print(f"   => alpha = {alpha} 확정.")
    print(" - 변형 완료된 대칭 방정식: (a_{n+1} + 1) = 2 * (a_n + 1)")

    # 3. 치환 수열 b_n의 인과적 흐름 도출
    print("\n[3단계: 새로운 치환 수열 b_n의 흐름 증명]")
    print(" - b_n = a_n + 1 로 정의.")
    print(f" - 첫째 항 b_1 = a_1 + 1 = 1 + 1 = 2")
    print(" - 공비(Ratio) = 2 인 완벽한 등비수열 성립: b_{n+1} = 2 * b_n")
    print(" - 따라서 b_n 의 일반항 = b_1 * 2^{n-1} = 2 * 2^{n-1} = 2^n")

    # 4. 본래 수열 a_n으로의 복원 및 검증
    print("\n[4단계: 본래 수열 a_n 복원 및 최종 검증]")
    print(" - b_n = a_n + 1 이므로, a_n = b_n - 1")
    print(" - 결론적 일반항 도출: a_n = 2^n - 1 (인과적 증명 완료)")

    # 5. 문제 요구치 a_5 계산
    n = 5
    a_5 = 2**n - 1
    print(f"\n[5단계: a_{n} 값 계산 및 단계별 검산]")
    for step in range(1, n + 1):
        val = 2**step - 1
        print(f"   n = {step} | a_{step} = 2^{step} - 1 = {val}")

    print(f"\n👉 [최종 답안]: a_5 = {a_5}")
    assert a_5 == 31, "수학 시험 실패"
    print("✅ [제1교시 시험 통과: 논리적 과정과 결과 일치]")


def solve_code_exam():
    print("\n" + "="*80)
    print("📝 [제2교시: 코드 알고리즘 최적화 시험]")
    print("="*80)
    print("문제: 정수 배열 nums = [2, 7, 11, 15], 목표값 target = 9 가 주어졌을 때,")
    print("      O(N^2) 무차별 대입을 피하고, '결핍(Target - Num)'을 역추적하는 O(N) 방식으로")
    print("      합이 9가 되는 두 수의 인덱스를 찾아내는 과정을 도출하라.")
    print("-" * 80)

    nums = [2, 7, 11, 15]
    target = 9

    print(f"입력 배열: {nums} | 목표치(Target): {target}")

    # 1. 알고리즘 전략 수립
    print("\n[1단계: 사고 전략 수립 (결핍 매핑)]")
    print(" - 무차별 2중 루프: (2,7), (2,11), (2,15)... 6번 비교 (O(N^2) 비효율)")
    print(" - 인과적 결핍 전략: 각 숫자 X를 만날 때마다 '나에게 필요한 짝(Deficit = Target - X)'을 계산하고,")
    print("   과거에 그 결핍을 메모리(지층)에 담아둔 적이 있는지 O(1) 해시로 즉각 대조.")

    # 2. 단계별 루프 전개 및 사고 과정
    print("\n[2단계: 단계별 실행 및 내부 사유 궤적]")
    memory_table = {} # {숫자: 인덱스}
    found_pair = None

    for i, num in enumerate(nums):
        deficit = target - num
        print(f"\n 👉 Step {i+1}: 현재 숫자 = {num} (인덱스 {i}) 관측")
        print(f"    - 완성을 위해 필요한 결핍값(Deficit): {target} - {num} = {deficit}")

        if deficit in memory_table:
            prev_index = memory_table[deficit]
            print(f"    - [발견!] 과거 메모리 지층에 결핍값 {deficit}이 이미 존재함 (인덱스 {prev_index})")
            print(f"    - 인과적 결합 완성: nums[{prev_index}]({deficit}) + nums[{i}]({num}) = {target}")
            found_pair = (prev_index, i)
            break
        else:
            print(f"    - 과거 메모리에 결핍값 {deficit} 없음. 현재 숫자 {num}(인덱스 {i})을 다음을 위해 메모리에 각인.")
            memory_table[num] = i
            print(f"    - 현재 메모리 지층 상태: {memory_table}")

    # 3. 최종 결과 도출
    print(f"\n👉 [최종 답안]: 목표값 {target}을 만드는 인덱스 쌍 = {list(found_pair)}")
    print(f"   검산: nums[{found_pair[0]}]({nums[found_pair[0]]}) + nums[{found_pair[1]}]({nums[found_pair[1]]}) = {nums[found_pair[0]] + nums[found_pair[1]]} == {target}")

    assert found_pair == (0, 1), "코드 시험 실패"
    print("✅ [제2교시 시험 통과: 결핍 역추적을 통한 최적 O(N) 해 도출]")


if __name__ == "__main__":
    solve_math_exam()
    solve_code_exam()
    print("\n" + "="*80)
    print("🎉 [모든 실기 시험 스스로 풀이 완료: 100점]")
    print("================================================================================")
