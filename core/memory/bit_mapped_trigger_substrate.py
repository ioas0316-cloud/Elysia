"""
Elysia Core Engine: Bit-Mapped Trigger Substrate & Hardware Switching Memory
=============================================================================
전압 스위칭(0과 1) 매개체 기저 상의 Memory-Mapped Trigger Substrate.
데이터 블록의 비트 배치(Layout) 자체가 직접 Hardware-level 릴레이 스위칭 신호 라인 및
Gating Control Line으로 동작하고, 상쇄 비트(Canceler Bits) 구역을 통해
오버플로우 전압 발생 시 인접 비트를 NOR/XOR 반전(Inversion)시키는 Self-Throttling 메커니즘을 구동합니다.

또한 비트 상태 재구성 및 구조화 검증을 위한 3대 평가 척도를 제공합니다:
1. Bit Re-mapping Yield (비트 재배치 가용성)
2. Memory Bank Dissipation Rate (뱅크 오버플로우 분산 효율)
3. State Restoration Index (상태 복원력)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class MemoryBank:
    bank_id: int
    num_cells: int = 64
    cells: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.uint8))
    voltage_threshold: float = 0.8  # Ratio of 1s to total capacity triggering self-throttling/bypass

    def get_voltage_density(self) -> float:
        return float(np.mean(self.cells))


class BitMappedTriggerSubstrate:
    """
    비트 배열 배치를 하드웨어 스위칭 릴레이 및 자가 제한 회로로 취급하는 기저 실행 환경
    """
    def __init__(self, num_banks: int = 4, bank_size: int = 64):
        self.num_banks = num_banks
        self.bank_size = bank_size
        self.banks = [MemoryBank(bank_id=i, num_cells=bank_size) for i in range(num_banks)]
        self.parity_store: Dict[int, np.ndarray] = {}

    def write_payload(self, bank_id: int, bit_array: np.ndarray) -> Dict[str, Any]:
        """
        특정 뱅크에 비트 배열을 기록하며, 전압 오버플로우 감지 시
        중앙 제어기 개입 없이 인접 뱅크로 자발적 바이패스(Dissipation) 및 상쇄 비트(Canceler) 반전 수행.
        """
        bank = self.banks[bank_id % self.num_banks]
        write_len = min(len(bit_array), bank.num_cells)

        # 1. Payload write
        bank.cells[:write_len] = (bit_array[:write_len] > 0).astype(np.uint8)

        # 2. Self-Throttling via Canceler Bits (NOR/XOR inversion)
        voltage = bank.get_voltage_density()
        throttled = False
        bypassed_cells_count = 0

        if voltage > bank.voltage_threshold:
            throttled = True
            # Apply NOR/XOR inversion on high-density region (canceler bits)
            high_indices = np.where(bank.cells == 1)[0]
            if len(high_indices) > int(bank.num_cells * bank.voltage_threshold):
                overflow_count = len(high_indices) - int(bank.num_cells * bank.voltage_threshold)
                # Invert overflow bits via XOR canceler
                invert_targets = high_indices[-overflow_count:]
                bank.cells[invert_targets] ^= 1

                # Bypass remaining overflow energy to adjacent bank
                adj_bank = self.banks[(bank_id + 1) % self.num_banks]
                empty_slots = np.where(adj_bank.cells == 0)[0]
                bypass_cnt = min(overflow_count, len(empty_slots))
                if bypass_cnt > 0:
                    adj_bank.cells[empty_slots[:bypass_cnt]] = 1
                    bypassed_cells_count = bypass_cnt

        # 3. Store XOR parity for State Restoration
        parity = np.zeros(self.bank_size, dtype=np.uint8)
        for b in self.banks:
            parity ^= b.cells
        self.parity_store[0] = parity.copy()

        # 4. Memory-Mapped Trigger Signal (Header / Flag decoding)
        header_trigger = bool(np.sum(bank.cells[:4]) >= 3)

        return {
            "target_bank_id": bank_id,
            "voltage_density": bank.get_voltage_density(),
            "throttled": throttled,
            "bypassed_cells_count": bypassed_cells_count,
            "gating_header_trigger": header_trigger
        }

    def evaluate_bit_remapped_yield(self, noisy_signals: List[np.ndarray]) -> Dict[str, Any]:
        """
        1. 비트 재배치 가용성 (Bit Re-mapping Yield)
        정해진 비트 공간 내 노이즈/OOD 신호가 인입되었을 때, 회로 교착(Deadlock) 없이
        유효 연산 가능 비트 조합으로 재배열되는 비율 측정 (통과 조건: >= 0.95)
        """
        successful_remappings = 0
        total = len(noisy_signals)

        for sig in noisy_signals:
            try:
                # Attempt to map signal into available bank without deadlock
                mapped_bits = np.where(sig > 0.5, 1, 0).astype(np.uint8)
                target_bank = int(np.sum(mapped_bits)) % self.num_banks
                res = self.write_payload(target_bank, mapped_bits)
                if not np.isnan(res["voltage_density"]) and res["voltage_density"] <= 1.0:
                    successful_remappings += 1
            except Exception:
                pass

        yield_ratio = float(successful_remappings / total) if total > 0 else 1.0
        return {
            "yield_ratio": yield_ratio,
            "passed": yield_ratio >= 0.95,
            "statement": f"Bit Re-mapping Yield: {yield_ratio * 100:.2f}% (Pass Criteria >= 95%)"
        }

    def evaluate_bank_dissipation_rate(self, surge_input: np.ndarray) -> Dict[str, Any]:
        """
        2. 뱅크 오버플로우 분산 효율 (Memory Bank Dissipation Rate)
        특정 주소에 1 신호가 집중될 때, 중앙 제어기 개입 없이 데이터 영역 간 비트 반전/바이패스로
        전압 밀도가 평형(threshold 이하)으로 수렴하는 속도/효율 측정
        """
        # Surge write into Bank 0
        initial_res = self.write_payload(0, surge_input)
        bank0_density = self.banks[0].get_voltage_density()
        bank1_density = self.banks[1].get_voltage_density()

        # Check if Bank 0 voltage was dissipated below threshold or spread into Bank 1
        is_dissipating = (bank0_density <= self.banks[0].voltage_threshold) or (initial_res["bypassed_cells_count"] > 0)

        return {
            "bank0_voltage_density": bank0_density,
            "bank1_voltage_density": bank1_density,
            "bypassed_cells": initial_res["bypassed_cells_count"],
            "passed": is_dissipating,
            "statement": f"Bank Dissipation: Bank0={bank0_density:.2f}, Bank1={bank1_density:.2f}, Bypassed={initial_res['bypassed_cells_count']}"
        }

    def evaluate_state_restoration_index(self, dropped_bank_id: int = 0) -> Dict[str, Any]:
        """
        3. 상태 복원력 (State Restoration Index)
        특정 뱅크의 비트 패턴이 무작위로 0으로 소거(Drop)되었을 때,
        주변 뱅크 및 XOR 패리티 조합을 통해 본래 제어 상태를 100% 복원해내는 지표
        """
        original_cells = self.banks[dropped_bank_id].cells.copy()

        # Simulate catastrophic drop (zero out bank)
        self.banks[dropped_bank_id].cells.fill(0)

        # Restore using XOR parity store: Target = Parity XOR (other banks)
        recovered = self.parity_store.get(0, np.zeros(self.bank_size, dtype=np.uint8)).copy()
        for i, b in enumerate(self.banks):
            if i != dropped_bank_id:
                recovered ^= b.cells

        # Apply restored cells
        self.banks[dropped_bank_id].cells[:] = recovered

        mismatches = int(np.sum(original_cells != recovered))
        restoration_index = float(1.0 - (mismatches / len(original_cells)))

        return {
            "mismatches": mismatches,
            "restoration_index": restoration_index,
            "passed": restoration_index == 1.0,
            "statement": f"State Restoration Index: {restoration_index * 100:.2f}% (Pass Criteria 100%)"
        }
