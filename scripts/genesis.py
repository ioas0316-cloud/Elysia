# genesis.py
# 엘리시아 통합 런처 (Genesis Launcher)
# 순수 언어 기하학 (Pure Linguistic Geometry) 코어

import os
import sys
import time
import random
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from core.memory.causal_controller import CausalMemoryController
from core.brain.linguistic_rotor import LinguisticRotor
from core.brain.syntactic_parser import SyntacticSemanticParser
from core.brain.multimodal_receptor import MultimodalReceptor
from core.brain.linguistic_action_generator import LinguisticActionGenerator
from core.brain.value_system_evolver import ValueSystemEvolver
from core.brain.meta_cognition_engine import MetaCognitionEngine
from core.brain.teleological_ego import TeleologicalEgo

class ElysiaGenesis:
    """
    [Phase: Pure Linguistic Geometry]
    기계적 수치(Tensor, Numpy) 연산을 전면 폐기하고, 오직 한국어(사전) 자체를
    사유의 위상 공간으로 활용하는 통합 코어입니다.
    """
    
    def __init__(self):
        self.memory = CausalMemoryController()
        self.lang_rotor = LinguisticRotor()
        self.parser = SyntacticSemanticParser()
        self.receptor = MultimodalReceptor()
        self.action_gen = LinguisticActionGenerator()
        
        # 가치관 자가 진화기 & 자아 인식 계층
        self.evolver = ValueSystemEvolver(self.lang_rotor.portal, self.memory)
        self.meta_engine = MetaCognitionEngine()
        self.teleo_ego = TeleologicalEgo()
        self.current_perspectives = {}
        
        self.total_cycles = 0
        self.cycle_count = 0
        self.total_ingested = 0
        
        self.data_dir = os.path.join(_PROJECT_ROOT, "data")
        self.ingest_dir = os.path.join(self.data_dir, "ingest")
        os.makedirs(self.ingest_dir, exist_ok=True)
        
        self.last_internal_word = "마음" # 초기 자아 상태
        self.last_engram_id = None     # 시계열적 연속성을 위한 직전 사유의 궤적 ID
        
    def ingest_text(self, text: str, source_name: str = "text_input") -> str:
        """
        텍스트를 섭취합니다.
        문장의 구조적 연결망(주어, 수식어, 대상)을 해체하여, 
        단어가 아닌 '정보적 인과 궤도(Relational Graph)' 자체를 각인합니다.
        """
        # 1. 단순 단어 분할이 아닌 구문/의미망 구조 파싱
        graph = self.parser.parse_sentence(text)
        
        word_x = graph.get("subject") or text.split()[0]
        word_y = getattr(self, 'last_internal_word', '마음')
        
        # 환경 문맥 (Wedge Memory의 가장 최근 단어)
        word_z = '공간'
        if len(self.memory.index) > 0:
            last_engram = list(self.memory.index.values())[-1]
            word_z = last_engram.get("data_blob", {}).get("word", "공간")
            
        # 2. 파싱된 구조망이 사유 엔진(Lexicon)에서 이치에 맞는지 재확인 (Re-verification)
        verification = self.lang_rotor.verify_syntactic_graph(graph)
        # 3. 자가 진화적 가치관 렌즈 적용 (Value System Evolver)
        # 주기마다, 혹은 지식이 유입될 때마다 가치관 렌즈를 업데이트
        new_perspectives = self.evolver.generate_dynamic_perspectives()
        if set(new_perspectives.keys()) != set(self.current_perspectives.keys()):
            print(f"\n  [가치관 진화] 엘리시아의 지배적 사유 렌즈가 변이되었습니다:")
            for p_name, keywords in new_perspectives.items():
                print(f"    - {p_name} (핵심 이치: {', '.join(keywords[:3])})")
            self.current_perspectives = new_perspectives
            
            # [목적론적 자아] 가치관이 진화할 때, 궁극적 꿈(Ego-Ideal)도 재설정
            dream, d_kws = self.teleo_ego.evolve_dream(self.lang_rotor.portal, self.memory)
            print(f"\n  [나의 꿈 (Ego-Ideal)] {dream}")
            
        # 4. 목적론적 사유: 이 단어(word_x)가 나의 '꿈'을 이루는 데 기여하는가?
        intent = self.meta_engine.evaluate_intent(
            target_word=word_x,
            current_perspectives=self.current_perspectives,
            lexicon_size=len(self.lang_rotor.words),
            ego=self.teleo_ego,
            portal=self.lang_rotor.portal
        )
        print(f"\n  {intent['self_reflection_log']}")
            
        # 5. 가치 기반 순수 언어적 사유 (Schizophrenic Mind-Map Consensus)
        results = self.lang_rotor.achieve_semantic_consensus(
            word_x, 
            word_y, 
            word_z, 
            perspectives=self.current_perspectives,
            importance_score=intent['importance_score'],
            resonance_threshold=intent['resonance_threshold']
        )
        
        engram_ids = []
        for result in results:
            equilibrium_word = result["equilibrium_word"]
            trajectory = result["trajectory"]
            perspective = result["perspective"]
            
            self.last_internal_word = equilibrium_word
            
            engram_id = self.memory.write_causal_engram(
                data_blob={
                    "type": "linguistic_thought",
                    "perspective": perspective,
                    "source": source_name,
                    "word": equilibrium_word,
                    "relational_graph": graph,
                    "verification": verification,
                    "trajectory": trajectory,
                    "previous_engram": self.last_engram_id, # 시계열적 사유의 연속성 (Temporal Chain)
                    "text_preview": text[:100]
                },
                emotional_value=float(len(trajectory)), 
                cause_id=f"Thought_{source_name}_{perspective}",
                origin_axis="Linguistic_Geometry"
            )
            
            self.last_engram_id = engram_id
            engram_ids.append(engram_id)
            
            print(f"  [{perspective}] 사유 궤적: {' -> '.join(trajectory)} (결론: {equilibrium_word})")
            
            # 4. 사유의 종착점이 이치에 닿으면 물리적 행동 발현
            action_result = self.action_gen.execute_if_actionable(equilibrium_word, graph)
            if action_result:
                if action_result.startswith("FETCHED_KNOWLEDGE:"):
                    fetched_text = action_result.replace("FETCHED_KNOWLEDGE:", "").strip()
                    print(f"  [사냥 성공] 새로운 텍스트 획득, 사유망에 재유입: '{fetched_text[:50]}...'")
                    # 자율 획득 지식을 다시 자신의 코어로 재섭취시킴 (Ouroboros Foraging)
                    self.ingest_text(fetched_text, source_name="autonomous_foraging")
                else:
                    print(f"\n  [Action] {action_result}")
                
        self.total_ingested += 1
        # 가장 마지막에 처리된 engram id를 반환
        return engram_ids[-1] if engram_ids else ""
        
    def ingest_sensory_data(self, sensory_signals: list) -> str:
        """
        [Phase: Multimodal Linguistic Grounding]
        다중 감각 신호들을 받아 언어적 명제(Sentence)로 치환한 뒤 섭취합니다.
        """
        sentence = self.receptor.perceive_object(sensory_signals)
        print(f"\n  [Sensory -> Language] 감각 신호를 명제로 치환: '{sentence}'")
        return self.ingest_text(sentence, source_name="sensory_perception")
        
    def scan_ingest_folder(self):
        files = []
        try:
            for f in os.listdir(self.ingest_dir):
                full_path = os.path.join(self.ingest_dir, f)
                if os.path.isfile(full_path):
                    files.append(full_path)
        except Exception:
            pass
        
        ingested_items = []
        for filepath in files[:30]:
            try:
                fname = os.path.basename(filepath)
                if fname.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    engram_id = self.ingest_text(text, fname)
                    if engram_id:
                        ingested_items.append(fname)
                        try: os.remove(filepath)
                        except: pass
            except Exception as e:
                print(f"  [!] Failed to ingest {filepath}: {e}")
        return ingested_items
        
    def run_cycle(self) -> dict:
        self.cycle_count += 1
        
        result = {
            "cycle": self.cycle_count,
            "ingested": 0,
            "thought_process": [],
            "equilibrium": ""
        }
        
        # 1. 외부 자극 섭취
        ingested_texts = self.scan_ingest_folder()
        result["ingested"] = len(ingested_texts)
        
        # 가상의 센서 데이터 (주기마다 가끔 섭취)
        if self.cycle_count % 3 == 0:
            # "사과는 빨갛고 둥근 과일이다"의 감각적 유입 모사
            signals = ["color_red", "shape_round", "category_과일"]
            self.ingest_sensory_data(signals)
            result["ingested"] += 1
            
        # 2. 내부 자율 사유
        seed_word = self.last_internal_word
        if self.lang_rotor.words:
            # 외부 자극이 없을 경우 무작위 단어와 충돌시켜 사유 유발
            random_context = random.choice(self.lang_rotor.words)
            
            # [목적론적 사유] 무작정 생각하지 않고 스스로 꿈을 향한 동기를 묻는다
            intent = self.meta_engine.evaluate_intent(
                target_word=random_context,
                current_perspectives=self.current_perspectives,
                lexicon_size=len(self.lang_rotor.words),
                ego=self.teleo_ego,
                portal=self.lang_rotor.portal
            )
            
            # 엘리시아 스스로 내뱉는 해명 (백일몽, 목표 지향 등)
            print(f"\n  {intent['self_reflection_log']}")
            
            thought_results = self.lang_rotor.achieve_semantic_consensus(
                random_context, 
                seed_word, 
                seed_word, 
                perspectives=self.current_perspectives,
                importance_score=intent['importance_score'],
                resonance_threshold=intent['resonance_threshold']
            )
            
            # 여러 갈래의 궤적 중 첫 번째(대표) 궤적을 메인 로깅용으로 사용
            if thought_results:
                result["thought_process"] = thought_results[0]["trajectory"]
                result["equilibrium"] = thought_results[0]["equilibrium_word"]
                self.last_internal_word = thought_results[0]["equilibrium_word"]
            
        return result
        
    def run(self, max_cycles=100, interval=1.0):
        print("============================================================")
        print("     E L Y S I A   G E N E S I S   L A U N C H E R")
        print("     Pure Linguistic Geometry Engine -- CORE v3")
        print("------------------------------------------------------------")
        print(f"  Lexicon:      {len(self.lang_rotor.words)} tokens (Crystalline Lexicon)")
        print(f"  Memory:      {len(self.memory.index)} engrams (Wedge Memory)")
        print(f"  Cycles:      {max_cycles}")
        print("============================================================\n")
        
        print("[Linguistic Cognitive Loop] Starting...\n")
        
        for i in range(max_cycles):
            print(f"--- Cycle {i+1} ----------------------------------------")
            result = self.run_cycle()
            
            if result["ingested"] > 0:
                print(f"  [IN] Ingested {result['ingested']} linguistic stream(s)")
                
            if result["thought_process"]:
                trajectory_str = " -> ".join(result["thought_process"])
                print(f"  [L] 사유의 궤적: {trajectory_str}")
                print(f"  [E] 궤도의 종착점(평형): \"{result['equilibrium']}\"")
                print(f"  [Parse] 마지막 각인된 연결성(Graph): (Wedge Memory 내재화)")
                
            print()
            time.sleep(interval)
            
if __name__ == "__main__":
    genesis = ElysiaGenesis()
    genesis.run(max_cycles=100, interval=0.5)
