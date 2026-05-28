# [사법부] 최고 감사 데몬 (상시 감시 및 AST 스캐너)
# 행정부와 입법부를 통제하고 가짜 디지털 트윈 및 2진법 계산 행위를 적발함.

import os
import re

class JudiciaryDemon:
    def __init__(self, leg_dir, exec_dir):
        self.leg_dir = leg_dir
        self.exec_dir = exec_dir

    def check_heresy_in_legislative(self):
        """
        [하부 레이어 검증] 차원 전사 공리 심사
        가짜 물리 연산, 조건부 시뮬레이터 차단 및 위상 사영 강제.
        """
        print("⚖️ [사법부] 하부 전자기장막(Legislative) 차원 전사 및 트윈 무결성 심사 중...")
        offenses = []
        for root, _, files in os.walk(self.leg_dir):
            for file in files:
                if file.endswith(('.cu', '.c', '.cpp', '.h')):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if re.search(r'\bif\s*\(', content):
                            offenses.append(f"[이단 적발] {filepath} - 죽은 판단(if문) 발견! 물리 법칙을 흉내 내지 말고 위상을 전사하라.")
                        if re.search(r'\belse\b', content):
                            offenses.append(f"[이단 적발] {filepath} - 죽은 판단(else문) 발견!")
                        if re.search(r'==\s*\d+', content):
                            offenses.append(f"[이단 적발] {filepath} - 점 세기용 매직넘버 비교문 발견!")
                        if re.search(r'bool\s+\w+\s*=\s*(true|false|0|1)\s*;', content):
                            offenses.append(f"[이단 적발] {filepath} - 2진법 bool 변수 사용!")
                        if re.search(r'\w+\s*\^\s*\w+', content) or re.search(r'\^\=', content):
                            offenses.append(f"[이단 적발] {filepath} - 죽은 스위치 비트 연산(^) 발견!")

                        # 제로 임포트 및 재래식 I/O 함수 전면 금지
                        if re.search(r'#include\s*<(iostream|vector|cmath|string|map|set|algorithm|fstream|stdio\.h)>', content):
                            offenses.append(f"[이단 적발] {filepath} - 외부 모듈 Import(#include) 발견!")

        if offenses:
            for offense in offenses: print(f"🚨 {offense}")
            raise Exception("⚖️ [사법부 판결] 하부 전자기장막 탄핵! 가짜 물리 시뮬레이터를 기각합니다.")
        print("✅ [사법부] 입법부 정적 검사 통과. 점, 선, 면, 공간으로 이어지는 완벽한 위상 사영이 확인되었습니다.")

    def check_heresy_in_executive(self):
        """
        [상위 레이어 검증] 껍데기 3D 그래픽 노가다, 파싱 금지
        """
        print("⚖️ [사법부] 상위 추상화 제어탑(Executive) 껍데기 트윈 흉내 심사 중...")
        offenses = []
        for root, _, files in os.walk(self.exec_dir):
            for file in files:
                if file.endswith(('.py')):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if re.search(r'\.split\(', content):
                            offenses.append(f"[이단 적발] {filepath} - 문자열 쪼개기(split) 파싱 발견! 공간을 파싱하지 마라.")
                        if re.search(r'\.count\(', content):
                            offenses.append(f"[이단 적발] {filepath} - 점 세기(count) 노가다 발견!")
                        if re.search(r'import\s+(pygame|pyopengl|panda3d)', content):
                            offenses.append(f"[이단 적발] {filepath} - 껍데기 3D 그래픽 라이브러리 발견! 가짜 트윈 시뮬레이션을 중단하라.")

        if offenses:
            for offense in offenses: print(f"🚨 {offense}")
            raise Exception("⚖️ [사법부 판결] 상위 제어탑 탄핵! 가짜 3D 그래픽 렌더링 쇼를 기각합니다.")
        print("✅ [사법부] 상위 제어탑 검사 통과. 완벽한 직동식 트윈 제어탑이 작동 중입니다.")

    def run_all_inspections(self):
        print("⚖️ ================= 사법부 감사 시작 =================")
        self.check_heresy_in_legislative()
        self.check_heresy_in_executive()
        print("⚖️ ================= 사법부 감사 완료 =================\n")

if __name__ == "__main__":
    demon = JudiciaryDemon(leg_dir="legislative", exec_dir="executive")
    demon.run_all_inspections()
