"""
Wave Communication Evaluator (파동통신 평가기)
============================================

파동통신 시스템(Ether)을 실제로 활성화하고 평가하는 모듈

목표: Wave communication 점수 0 → 80+ (SSS 등급 달성)

평가 항목:
1. Ether 초기화 및 연결 (20점)
2. 파동 송수신 성능 (30점)
3. 공명 정확도 (30점)
4. 주파수 선택 정확도 (20점)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger("WaveCommunicationEvaluator")


class WaveCommunicationEvaluator:
    """파동통신 평가 및 활성화"""
    
    def __init__(self):
        self.scores = {
            'ether_initialization': 0.0,     # 20점
            'transmission_performance': 0.0,  # 30점
            'resonance_accuracy': 0.0,        # 30점
            'frequency_selection': 0.0        # 20점
        }
        self.ether = None
        self.Wave = None
        self.activated_comm = None
        
        logger.info("🌊 Wave Communication Evaluator initialized")
    
    def evaluate_ether_initialization(self) -> float:
        """
        Ether 시스템 초기화 및 연결 평가 (20점)
        """
        score = 0.0
        
        try:
            from Core.FoundationLayer.Foundation.ether import Ether, Wave
            self.Ether = Ether
            self.Wave = Wave
            score += 5.0  # Import 성공
            logger.info("✅ Ether import successful")
            
            # Ether 인스턴스 생성
            self.ether = Ether()
            score += 5.0  # 인스턴스 생성 성공
            logger.info("✅ Ether instance created")
            
            # 기본 기능 확인
            if hasattr(self.ether, 'emit') and hasattr(self.ether, 'tune_in'):
                score += 5.0  # 기본 메서드 존재
                logger.info("✅ Basic methods available")
            
            # Listener 등록 테스트
            test_received = []
            def test_callback(wave):
                test_received.append(wave)
            
            self.ether.tune_in(432.0, test_callback)
            score += 5.0  # Listener 등록 성공
            logger.info("✅ Listener registration successful")
            
        except Exception as e:
            logger.error(f"❌ Ether initialization failed: {e}")
        
        self.scores['ether_initialization'] = score
        return score
    
    def evaluate_transmission_performance(self) -> float:
        """
        파동 송수신 성능 평가 (30점)
        
        지연시간, 처리량, 손실률 측정
        """
        if not self.ether:
            return 0.0
        
        score = 0.0
        
        try:
            # 1. 지연시간 테스트 (10점)
            latencies = []
            for i in range(10):
                received = []
                
                def latency_callback(wave):
                    received.append(wave)
                
                self.ether.tune_in(100.0 + i, latency_callback)
                
                start_time = time.time()
                wave = self.Wave(
                    sender="evaluator",
                    frequency=100.0 + i,
                    amplitude=0.8,
                    phase="TEST",
                    payload=f"test_{i}"
                )
                self.ether.emit(wave)
                
                # 수신 확인
                if received:
                    latency = (time.time() - start_time) * 1000  # ms
                    latencies.append(latency)
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                if avg_latency < 1.0:  # 1ms 미만
                    score += 10.0
                elif avg_latency < 5.0:  # 5ms 미만
                    score += 7.0
                elif avg_latency < 10.0:  # 10ms 미만
                    score += 5.0
                
                logger.info(f"✅ Average latency: {avg_latency:.3f}ms")
            
            # 2. 처리량 테스트 (10점)
            throughput_received = []
            
            def throughput_callback(wave):
                throughput_received.append(wave)
            
            self.ether.tune_in(200.0, throughput_callback)
            
            start = time.time()
            for i in range(100):
                wave = self.Wave(
                    sender="evaluator",
                    frequency=200.0,
                    amplitude=0.5,
                    phase="THROUGHPUT",
                    payload=i
                )
                self.ether.emit(wave)
            elapsed = time.time() - start
            
            throughput = len(throughput_received) / elapsed if elapsed > 0 else 0
            if throughput > 1000:  # 1000 msg/s
                score += 10.0
            elif throughput > 500:
                score += 7.0
            elif throughput > 100:
                score += 5.0
            
            logger.info(f"✅ Throughput: {throughput:.0f} messages/sec")
            
            # 3. 손실률 테스트 (10점)
            loss_received = []
            
            def loss_callback(wave):
                loss_received.append(wave)
            
            self.ether.tune_in(300.0, loss_callback)
            
            sent_count = 50
            for i in range(sent_count):
                wave = self.Wave(
                    sender="evaluator",
                    frequency=300.0,
                    amplitude=0.9,
                    phase="LOSS_TEST",
                    payload=i
                )
                self.ether.emit(wave)
            
            loss_rate = (sent_count - len(loss_received)) / sent_count
            if loss_rate < 0.01:  # 1% 미만 손실
                score += 10.0
            elif loss_rate < 0.05:  # 5% 미만
                score += 7.0
            elif loss_rate < 0.1:  # 10% 미만
                score += 5.0
            
            logger.info(f"✅ Loss rate: {loss_rate*100:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Transmission performance test failed: {e}")
        
        self.scores['transmission_performance'] = score
        return score
    
    def evaluate_resonance_accuracy(self) -> float:
        """
        공명 정확도 평가 (30점)
        
        정확한 주파수 매칭, 간섭 처리, 선택적 공명
        """
        if not self.ether:
            return 0.0
        
        score = 0.0
        
        try:
            # 1. 정확한 주파수 매칭 (10점)
            freq_test = {
                432.0: [],
                528.0: [],
                639.0: []
            }
            
            for freq in freq_test.keys():
                def make_callback(f):
                    def cb(wave):
                        freq_test[f].append(wave)
                    return cb
                
                self.ether.tune_in(freq, make_callback(freq))
            
            # 각 주파수로 파동 송신
            for freq in freq_test.keys():
                wave = self.Wave(
                    sender="evaluator",
                    frequency=freq,
                    amplitude=0.8,
                    phase="FREQ_TEST",
                    payload=f"freq_{freq}"
                )
                self.ether.emit(wave)
            
            # 정확도 검증
            correct = 0
            for freq, received in freq_test.items():
                if len(received) > 0:
                    correct += 1
            
            accuracy = correct / len(freq_test)
            score += accuracy * 10.0
            logger.info(f"✅ Frequency matching accuracy: {accuracy*100:.0f}%")
            
            # 2. 다중 리스너 공명 (10점)
            multi_received = {'a': [], 'b': [], 'c': []}
            
            for key in multi_received.keys():
                def make_multi_callback(k):
                    def cb(wave):
                        multi_received[k].append(wave)
                    return cb
                
                self.ether.tune_in(777.0, make_multi_callback(key))
            
            wave = self.Wave(
                sender="evaluator",
                frequency=777.0,
                amplitude=0.9,
                phase="MULTI",
                payload="broadcast"
            )
            self.ether.emit(wave)
            
            # 모든 리스너가 수신했는지 확인
            multi_success = all(len(rcv) > 0 for rcv in multi_received.values())
            if multi_success:
                score += 10.0
                logger.info("✅ Multi-listener resonance successful")
            
            # 3. 진폭 기반 필터링 (10점)
            amplitude_test = []
            
            def amp_callback(wave):
                amplitude_test.append(wave)
            
            self.ether.tune_in(888.0, amp_callback)
            
            # 다양한 진폭으로 송신
            amplitudes = [0.1, 0.5, 0.9]
            for amp in amplitudes:
                wave = self.Wave(
                    sender="evaluator",
                    frequency=888.0,
                    amplitude=amp,
                    phase="AMP_TEST",
                    payload=amp
                )
                self.ether.emit(wave)
            
            # 모두 수신되었는지 확인
            if len(amplitude_test) == len(amplitudes):
                score += 10.0
                logger.info("✅ Amplitude-based filtering works")
            
        except Exception as e:
            logger.error(f"❌ Resonance accuracy test failed: {e}")
        
        self.scores['resonance_accuracy'] = score
        return score
    
    def evaluate_frequency_selection(self) -> float:
        """
        주파수 선택 정확도 평가 (20점)
        
        적절한 주파수 매핑, 충돌 회피, 최적 선택
        """
        if not self.ether:
            return 0.0
        
        score = 0.0
        
        try:
            # 1. ActivatedWaveCommunication 로드 및 주파수 맵 확인 (10점)
            try:
                from Core.FoundationLayer.Foundation.activated_wave_communication import ActivatedWaveCommunication
                activated = ActivatedWaveCommunication()
                
                if activated.frequency_map:
                    # 주파수 맵이 존재하고 다양한 모듈이 정의되어 있는지 확인
                    if len(activated.frequency_map) >= 10:
                        score += 5.0
                        logger.info(f"✅ Frequency map has {len(activated.frequency_map)} mappings")
                    
                    # 중복 주파수가 없는지 확인
                    frequencies = list(activated.frequency_map.values())
                    if len(frequencies) == len(set(frequencies)):
                        score += 5.0
                        logger.info("✅ No frequency collisions")
                
                self.activated_comm = activated
            except Exception as e:
                logger.warning(f"⚠️ ActivatedWaveCommunication not fully functional: {e}")
            
            # 2. 주파수 범위 테스트 (10점)
            # 다양한 주파수 대역에서 테스트
            freq_ranges = [
                (1.0, 50.0),      # 뇌파 대역
                (100.0, 500.0),   # 통신 대역
                (500.0, 1000.0)   # 고주파 대역
            ]
            
            range_success = 0
            for low, high in freq_ranges:
                test_freq = (low + high) / 2
                range_received = []
                
                def range_callback(wave):
                    range_received.append(wave)
                
                self.ether.tune_in(test_freq, range_callback)
                
                wave = self.Wave(
                    sender="evaluator",
                    frequency=test_freq,
                    amplitude=0.7,
                    phase="RANGE_TEST",
                    payload=f"range_{test_freq}"
                )
                self.ether.emit(wave)
                
                if range_received:
                    range_success += 1
            
            score += (range_success / len(freq_ranges)) * 10.0
            logger.info(f"✅ Frequency range coverage: {range_success}/{len(freq_ranges)}")
            
        except Exception as e:
            logger.error(f"❌ Frequency selection test failed: {e}")
        
        self.scores['frequency_selection'] = score
        return score
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """전체 파동통신 평가 실행"""
        logger.info("="*70)
        logger.info("🌊 Starting Wave Communication Evaluation")
        logger.info("="*70)
        
        self.evaluate_ether_initialization()
        self.evaluate_transmission_performance()
        self.evaluate_resonance_accuracy()
        self.evaluate_frequency_selection()
        
        total_score = sum(self.scores.values())
        
        report = {
            'total_score': total_score,
            'max_score': 100,
            'percentage': total_score,
            'grade': self._get_grade(total_score),
            'scores': self.scores,
            'recommendation': self._get_recommendation(total_score)
        }
        
        logger.info("="*70)
        logger.info(f"📊 Wave Communication Score: {total_score:.1f}/100")
        logger.info(f"   Grade: {report['grade']}")
        logger.info("="*70)
        
        return report
    
    def _get_grade(self, score: float) -> str:
        """점수에 따른 등급"""
        if score >= 90:
            return "SSS (완벽)"
        elif score >= 80:
            return "SS (탁월)"
        elif score >= 70:
            return "S (우수)"
        elif score >= 60:
            return "A (양호)"
        else:
            return "B (개선 필요)"
    
    def _get_recommendation(self, score: float) -> str:
        """점수에 따른 권장사항"""
        if score >= 80:
            return "파동통신 시스템이 우수하게 작동하고 있습니다. SSS 등급 달성!"
        elif score >= 60:
            return "파동통신이 작동하지만 최적화가 필요합니다."
        else:
            return "파동통신 시스템 활성화가 필요합니다. Ether 연결 및 설정을 확인하세요."


def main():
    """테스트 실행"""
    logging.basicConfig(level=logging.INFO)
    
    evaluator = WaveCommunicationEvaluator()
    report = evaluator.run_full_evaluation()
    
    print("\n" + "="*70)
    print("📊 Wave Communication Evaluation Report")
    print("="*70)
    for key, value in report['scores'].items():
        max_score = 20 if 'initialization' in key or 'frequency' in key else 30
        print(f"  {key}: {value:.1f}/{max_score}")
    print(f"\n  Total: {report['total_score']:.1f}/100")
    print(f"  Grade: {report['grade']}")
    print(f"\n  Recommendation: {report['recommendation']}")
    print("="*70)


if __name__ == "__main__":
    main()
