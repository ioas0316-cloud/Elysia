# -*- coding: utf-8 -*-
"""
[Live Verification: Autonomous Intent Sprouting & Self-Governed Parameter Tuning]
================================================================================
인간이 "무엇을 해라"라고 지시하지 않고,
날것의 환경 스트림(시스템 저항, 외부 텍스트, 이질적 신호)을 연속 방류했을 때:
1. 시스템이 스스로 내부 기대치와 현실의 위상차(ΔΦ)를 감지하는가?
2. 그 마찰로부터 스스로 '탐구 의도(Intent)'를 자발적으로 발행하는가?
3. 발행된 의도를 스스로 역메커니즘(Θ)과 웻지 지층으로 해결·갱신하는가?
"""

import sys
import os
import time
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.consciousness.autonomous_intent_generator import AutonomousIntentGenerator
from core.memory.causal_controller import CausalMemoryController


def run_autonomous_daemon_simulation():
    print("="*80)
    print("🌟 [LIVE VERIFICATION: AUTONOMOUS INTENT SPROUTING & SELF-MOLDING]")
    print("    외부 지시 없이 환경 요동에서 스스로 의도와 목적을 도출하는 실증")
    print("="*80)

    controller = CausalMemoryController()
    daemon = AutonomousIntentGenerator(controller)

    # 연속적으로 유입되는 다양한 실세계 환경 스트림 시뮬레이션
    ambient_streams = [
        (b"LOW_CPU_IDLE_HARMONIC_STREAM", "system_hardware"),
        (b"SUDDEN_HIGH_NETWORK_SPIKE_DISCORDANT_ANOMALY_0xFE", "network_sensor"),
        (b"TEXT_STREAM_THE_KING_WAS_BETRAYED_AND_SADNESS_OVERWHELMED", "external_dialogue"),
        (b"UNKNOWN_HIGH_ENTROPY_QUANTUM_NOISE_PACKET_0xAA_0x55", "unstructured_raw_feed"),
        (b"EQUILIBRIUM_RESTORED_STEADY_STATE_PEACE_SIGNAL", "ambient_field")
    ]

    print("\n[1단계: 외부 환경 스트림 연속 방류 및 자발적 의도(Intent) 발아 관측]")
    
    for idx, (stream_data, source_tag) in enumerate(ambient_streams):
        log = daemon.observe_ambient_stream(stream_data, source_tag=source_tag)
        print(f"\n 👉 [Stream {idx+1:02d} | 출처: {source_tag}]")
        print(f"    - 유입 신호 위상: {np.round(log['observed_phase'], 3).tolist()}")
        print(f"    - 내부 기대치(예측): {np.round(log['internal_expectation'], 3).tolist()}")
        print(f"    - 위상차 (Phase Gap ΔΦ): {log['phase_gap_norm']:.4f}")
        
        if log['intent_sprouted']:
            intent = log['sprouted_intent']
            print(f"    - 🔥 [자발적 의도 발아!] Intent ID: {intent['intent_id']}")
            print(f"       * 동기: {intent['motivation']}")
            print(f"       * 텐션 강도: {intent['tension_intensity']:.4f}")
        else:
            print(f"    - [평온 상태] 위상차가 임계치 이하로 평형 유지.")

    total_sprouted = len(daemon.intent_queue)
    print(f"\n[2단계: 누적된 자발적 의도 큐(Queue) 확인]")
    print(f" - 시스템 스스로 발행한 총 자발적 탐구 의도 수: {total_sprouted}개")
    for item in daemon.intent_queue:
        print(f"   * [{item['intent_id']}] 텐션: {item['tension_intensity']:.4f} | 출처: {item['source']}")

    # 3. 의도들의 자율 해결 및 매개변수 Θ 갱신
    print("\n[3단계: 인간 개입 없는 의도 자율 해결 및 생성 파라미터 Θ 갱신]")
    resolved_results = daemon.resolve_intents_autonomously()

    for r in resolved_results:
        print(f"   * 해결 완료: Intent [{r['intent_id']}] ──► 갱신된 메커니즘 [{r['derived_mechanism_id']}] (MDL 복잡도: {r['mdl_complexity']:.4f}, Engram: {r['resolved_engram_id'][:14]}...)")

    # 4. 최종 실증 결과 판정
    final_rings = float(np.linalg.norm(daemon.plasticity.annual_rings))
    print("\n[4단계: 자아 주조 진화 실증 결과]")
    print(f" - 자발적 의도 생성 성공 여부       : {total_sprouted > 0} ({total_sprouted}개 생성)")
    print(f" - 자율 매개변수 Θ 역산 및 해결 여부 : {len(resolved_results) == total_sprouted} (100% 자율 해결)")
    print(f" - 렌즈 변형 및 나이테 누적 지층    : {final_rings:.4f}")

    assert total_sprouted >= 2, "자발적 의도 발아 실패"
    assert len(resolved_results) == total_sprouted, "자율 의도 해결 실패"
    assert final_rings > 0.0, "나이테 각인 실패"

    print("\n" + "="*80)
    print("🎉 [완전 자립형 자율 의도 생성 및 매개변수 갱신 실증 100% 통과]")
    print("   인간이 '무엇을 해라'라고 코딩해주지 않아도,")
    print("   환경과의 어긋남(ΔΦ)을 스스로 감지하여 탐구 목적(Intent)을 자발적으로 발행하고,")
    print("   자신의 매개변수 Θ를 스스로 역산하여 지층에 안착시키는 자율 동역학을 확인했습니다.")
    print("="*80)


if __name__ == "__main__":
    run_autonomous_daemon_simulation()
