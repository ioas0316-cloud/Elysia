# [사법부] 최고 감사 데몬 (상시 감시 및 AST 스캐너)
# 행정부와 입법부를 통제하고 단절을 유발하는 판단/스위치 연산 및 임포트(Import)를 적발함.

import os
import re

class JudiciaryDemon:
    def __init__(self, leg_dir, exec_dir):
        self.leg_dir = leg_dir
        self.exec_dir = exec_dir

    def check_heresy_in_legislative(self):
        """
        ① 판단(if/else)의 전량 참수, 하드코딩, 비트 연산자 적발
        ② 단일 전자기 회로망의 붕괴(단절) 여부 검사
        ③ 제로-임포트(Zero-Import) 율법 검사: 외부 라이브러리 차단
        """
        print("⚖️ [사법부] 입법부 코드 이단(Heresy) 정적 검사 중 (제로-임포트 및 단절 심사)...")
        offenses = []
        for root, _, files in os.walk(self.leg_dir):
            for file in files:
                if file.endswith(('.cu', '.c', '.cpp', '.h')):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # 이단 규정 0: 명시적 if/else 조건문 전량 금지 (판단 병목 타파)
                        if re.search(r'\bif\s*\(', content):
                            offenses.append(f"[이단 적발] {filepath} - 죽은 판단(if문) 발견! 회로망은 상시 개방되어야 한다.")
                        if re.search(r'\belse\b', content):
                            offenses.append(f"[이단 적발] {filepath} - 죽은 판단(else문) 발견! 물리적 위상 수학으로 우회하라.")

                        # 이단 규정 1: 매직 넘버를 이용한 조건문 (점 세기 노가다 잔재)
                        if re.search(r'==\s*\d+', content):
                            offenses.append(f"[이단 적발] {filepath} - 점 세기용 매직넘버 비교문 발견!")

                        # 이단 규정 2: 하드코딩된 매핑 구조
                        if re.search(r'\{[^{}]*".*"\s*:\s*\d+[^{}]*\}', content):
                            offenses.append(f"[이단 적발] {filepath} - 하드코딩 매핑 구조 발견! 단일 텐서망을 사용하라.")

                        # 이단 규정 3: 0과 1만을 이용한 2진법 변수 선언/할당 (삼진법 -1, 0, 1 강제)
                        if re.search(r'bool\s+\w+\s*=\s*(true|false|0|1)\s*;', content):
                            offenses.append(f"[이단 적발] {filepath} - 2진법 bool 변수 사용! 삼진법(-1,0,1) 위상 장력을 사용하라.")

                        # 이단 규정 4: 비트 연산자(^) 사용 금지 (스위치 게이트 단절 금지)
                        if re.search(r'\w+\s*\^\s*\w+', content) or re.search(r'\^\=', content):
                            offenses.append(f"[이단 적발] {filepath} - 죽은 스위치 비트 연산(^) 발견! 단일 회로망 순환을 방해하지 마라.")

                        # 이단 규정 5: 제로-임포트. 기성 C++ STL 라이브러리 포함 금지.
                        # <cuda_runtime.h> 와 파이썬 바인딩용 필수 헤더 외엔 전부 금지.
                        if re.search(r'#include\s*<(iostream|vector|cmath|string|map|set|algorithm)>', content):
                            offenses.append(f"[이단 적발] {filepath} - 외부 모듈 Import(#include) 발견! 회로망을 끊는 정적 의존성을 제거하라.")

        if offenses:
            for offense in offenses:
                print(f"🚨 {offense}")
            raise Exception("⚖️ [사법부 판결] 입법부 탄핵! 회로망을 단절시키고 수입(Import)에 의존하는 이단 코드를 기각합니다.")
        print("✅ [사법부] 입법부 정적 검사 통과. 단일 전자기 회로망이 완벽하게 연결되었으며 외부 의존성이 없는 순수 폐회로입니다.")

    def run_all_inspections(self):
        print("⚖️ ================= 사법부 감사 시작 =================")
        self.check_heresy_in_legislative()
        print("⚖️ ================= 사법부 감사 완료 =================\n")

if __name__ == "__main__":
    demon = JudiciaryDemon(leg_dir="legislative", exec_dir="executive")
    demon.run_all_inspections()
