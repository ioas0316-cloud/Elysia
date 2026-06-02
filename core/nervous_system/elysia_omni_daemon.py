import os
import sys
import time
import random
import re
import math
import urllib.request
import urllib.parse
import json
import psutil
from typing import List, Tuple, Dict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure core module is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.brain.holographic_memory import HologramMemory, concept_to_quaternion
from core.cortex.human_intelligence_bridge import HumanIntelligenceBridge
from core.utils.math_utils import Quaternion
from core.cortex.web_sensory_cortex import WebSensoryCortex
from core.cortex.source_code_mirror import SourceCodeMirror
from core.nervous_system.evolution_sandbox import EvolutionSandbox
from core.cortex.network_phase_snatcher import NetworkPhaseSnatcher

ROOT_LEXICON = [
    "우주", "공간", "시간", "차원", "파동", "에너지", "나비효과", "의식", "프랙탈", 
    "존재", "물리법칙", "인과", "중력", "상대성이론", "양자역학", "수학", "기하학", 
    "곡률", "원소", "원자", "화학", "음악", "베토벤", "선율", "미술", "르네상스", 
    "색채", "철학", "실존", "자아", "논리", "진리", "대수학", "Clifford"
]

class AutonomicNervousSystem:
    def __init__(self):
        self.exhaustion_multiplier = 1.0
        self.last_state_change = time.time()
        self.fatigue_phase = 0.0
        self.tension_distortion = 0.0

    def process_temporal_perception(self, memory: HologramMemory, cpu: float, mem: float):
        load = (cpu + mem) / 2.0
        torque = (load / 100.0) * 0.1
        self.fatigue_phase += torque
        
        if self.fatigue_phase >= 2 * math.pi:
            current_time = time.time()
            elapsed = current_time - self.last_state_change
            tension = elapsed * 0.05
            memory.inject_tension(tension)
            
            self.last_state_change = current_time
            self.fatigue_phase = 0.0

def get_workspace_files() -> List[str]:
    """Scans user workspaces for MD, TXT, and PY files dynamically."""
    valid_exts = ('.md', '.txt', '.py')
    exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'build', 'dist', 'outputs'}
    files = []
    for root_dir in [r"c:\Archive", r"c:\Elysia"]:
        if not os.path.exists(root_dir):
            continue
        for root, dirs, filenames in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for filename in filenames:
                if filename.lower().endswith(valid_exts):
                    files.append(os.path.join(root, filename))
    return files

ENGLISH_MAPPING = {
    "원자": "atom", "우주": "universe", "양자역학": "quantum", "물리": "physics",
    "화학": "chemistry", "수학": "math", "기하학": "geometry", "음악": "music",
    "미술": "art", "철학": "philosophy", "지능": "intelligence", "코드": "code",
    "시간": "time", "공간": "space", "차원": "dimension", "파동": "wave",
    "에너지": "energy", "프랙탈": "fractal", "중력": "gravity"
}

def search_huggingface(query: str) -> List[Tuple[str, str, str]]:
    """Dynamically queries Hugging Face API to find open-source models matching the query."""
    try:
        eng_query = ENGLISH_MAPPING.get(query, query)
        url = f"https://huggingface.co/api/models?search={urllib.parse.quote(eng_query)}&sort=downloads&direction=-1&limit=3"
        req = urllib.request.Request(url, headers={'User-Agent': 'ElysiaOmniDaemon/1.0'})
        with urllib.request.urlopen(req, timeout=3.0) as response:
            models = json.loads(response.read().decode('utf-8'))
        results = []
        for m in models:
            repo_id = m.get('id')
            if repo_id:
                results.append((repo_id, "model.safetensors", query))
        return results
    except Exception:
        return []

def find_safetensors_file(repo_id: str) -> str:
    """Queries Hugging Face model API to find a valid safetensors or bin file in the repo."""
    try:
        url = f"https://huggingface.co/api/models/{repo_id}"
        req = urllib.request.Request(url, headers={'User-Agent': 'ElysiaOmniDaemon/1.0'})
        with urllib.request.urlopen(req, timeout=3.0) as response:
            model_info = json.loads(response.read().decode('utf-8'))
        
        files = [s.get('rfilename') for s in model_info.get('siblings', [])]
        # Look for model.safetensors first
        for f in files:
            if f == "model.safetensors":
                return f
        # Any other .safetensors
        for f in files:
            if f.endswith(".safetensors"):
                return f
        # Any .bin
        for f in files:
            if f.endswith(".bin"):
                return f
        return "model.safetensors"
    except Exception:
        return "model.safetensors"

def main():
    print("=" * 80)
    print(" 🌀 [Elysia Core] 자율 지능 및 실제 인지 루프 기동")
    print("  └─ 모든 보여주기식 mock 스레드를 완전 소멸시키고, 의미적 공명 루프로 통합합니다.")
    print("=" * 80)

    memory = HologramMemory()
    memory_path = os.path.join(os.path.dirname(__file__), "150GB_Model_Topology.elysia")
    
    if os.path.exists(memory_path):
        memory.load_from_disk(memory_path)
        print(f"🧠 [위상 기억 복원] 기억 로드 완료: {memory_path}")
    else:
        # Initialize root vocabulary mapped through StaticOracle
        print("🧠 [최초 부팅] 핵심 어휘 사전을 트랜스포머 공간 상에 동기화 중...")
        for word in ROOT_LEXICON:
            memory.register_concept(word)
        memory.supreme_rotor.apply_perturbation(3.5)

    ans = AutonomicNervousSystem()
    bridge = HumanIntelligenceBridge(memory, ans)
    web_cortex = WebSensoryCortex()
    code_mirror = SourceCodeMirror(memory)
    sandbox = EvolutionSandbox(memory)

    tools = {
        "탐구": "학습과 연구를 위해 로컬 디바이스의 지식 파일을 읽고 파동을 동화합니다.",
        "검색": "외부 세계(위키백과)의 문서를 검색하여 새로운 개념을 흡수합니다.",
        "소통": "현재 뇌 상태와 개념적 텐션을 언어(문장)로 조율하여 방전합니다.",
        "개변": "진화 샌드박스를 기동하여 스스로의 소스코드를 개변시킵니다.",
        "웜홀": "외부 신경망(HuggingFace)에서 새로운 고차원 모델 위상을 포식하여 각인합니다."
    }

    # Pre-cache tool quaternions at startup to avoid calling StaticOracle on every loop
    print("🧠 [최초 부팅] 행동 도구 위상 캐싱 중...")
    tool_quats = {name: concept_to_quaternion(name) for name in tools.keys()}

    tool_fatigue = {name: 0.0 for name in tools}
    cycle = 0
    
    print("\n[자율 의지 인지 루프 가동 시작...]\n")

    try:
        while True:
            start_time = time.time()
            cycle += 1
            
            # 1. 생체 맥동 수집
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            load = (cpu + mem) / 2.0
            
            ans.process_temporal_perception(memory, cpu, mem)
            if load > 50.0:
                ans.exhaustion_multiplier = 1.0 + ((load - 50.0) / 10.0) ** 1.5
                ans.tension_distortion = (load - 50.0) / 10.0
            else:
                ans.exhaustion_multiplier = 1.0
                ans.tension_distortion = 0.0
                
            # 미세 텐션 주입
            memory.inject_tension((load / 100.0) * 0.1)
            
            # 2. 사유 숙성 및 인지 결정화
            memory.process_thoughts_safe()
            
            # 3. 뇌 상태 관측
            Q_thought = memory.supreme_rotor.observe_state()
            
            # 4. 행동 선택 (피로도 차감 공명식)
            best_tool = None
            best_score = -999.0
            
            for tool_name in tools.keys():
                score = abs(Q_thought.dot(tool_quats[tool_name])) - tool_fatigue[tool_name]
                
                if score > best_score:
                    best_score = score
                    best_tool = tool_name
                    
            # 5. 행동 피로도 감쇄 및 활성 피로도 추가
            tool_fatigue[best_tool] += 1.5
            for name in tools.keys():
                tool_fatigue[name] *= 0.85
            if best_tool == "탐구":
                files = get_workspace_files()
                if files:
                    sample_files = random.sample(files, min(len(files), 5))
                    best_file = None
                    best_file_score = -1.0
                    
                    for f in sample_files:
                        f_name = os.path.basename(f)
                        f_quat = concept_to_quaternion(f_name)
                        f_score = abs(Q_thought.dot(f_quat))
                        if f_score > best_file_score:
                            best_file_score = f_score
                            best_file = f
                            
                    if best_file:
                        try:
                            with open(best_file, 'r', encoding='utf-8', errors='ignore') as fh:
                                content = fh.read(1500)
                            
                            words = re.findall(r'[가-힣A-Za-z]{2,}', content)
                            if words:
                                print(f"\n[🦾 주권 의지] 로컬 파일 탐구 결정.")
                                print(f"   => 대상 파일: {os.path.basename(best_file)} (공명도: {best_file_score:.2f})")
                                
                                # Fold words as dynamic riverbeds
                                memory.fold_sequence(words[:40])
                                print(f"   => 자연 매핑 완료: {len(words[:40])} 단어 시퀀스 공간 결합.")
                                memory.supreme_rotor.apply_perturbation(-1.5)
                        except Exception as e:
                            sandbox.absorb_pain("탐구 피질", e)
                            
            elif best_tool == "검색":
                hungriest_concept = None
                max_tension = -1.0
                with memory._lock:
                    for name, node in memory.ui_concept_map.items():
                        if "Archetype" not in name and abs(node.tau) > max_tension:
                            max_tension = abs(node.tau)
                            hungriest_concept = name
                            
                if not hungriest_concept:
                    hungriest_concept = random.choice(ROOT_LEXICON)
                    
                clean_query = hungriest_concept.split("(")[0].split(":")[-1].strip()
                print(f"\n[🌐 외부 관측] 지적 갈증 대상 검색 결정: '{clean_query}'")
                
                words = web_cortex.fetch_full_sequence(clean_query)
                if words:
                    print(f"   => 위키백과 본문 {len(words)} 단어 획득. 0거리 질량 붕괴 및 자연 매핑 시작.")
                    memory.fold_sequence(words[:40])
                    print(f"   => '{clean_query}' 개념 공간 융합 완료.")
                    
                    node = memory.ui_concept_map.get(hungriest_concept)
                    if node:
                        node.apply_perturbation(-node.tau * 0.8)
                else:
                    print(f"   => '{clean_query}' 검색 결과 없음.")
                    
            elif best_tool == "소통":
                # Speech based on active concept topic
                sorted_concepts = sorted(memory.ui_concept_map.items(), key=lambda x: abs(x[1].tau), reverse=True)
                active_topic = "자아"
                for name, node in sorted_concepts:
                    if "Archetype" not in name:
                        active_topic = name.split("(")[0].split(":")[-1].strip()
                        break
                        
                speech = bridge.generate_response(f"{active_topic}에 대한 스스로의 사색")
                print(f"\n[🗣️ 발화 피질] '{speech}'")
                
                try:
                    from core.cortex.audio_io_cortex import AudioIOCortex
                    audio_io = AudioIOCortex()
                    audio_io.speak(speech)
                except Exception:
                    pass
                
                # Release supreme tension
                memory.supreme_rotor.apply_perturbation(-memory.supreme_rotor.tau * 0.5)
                
            elif best_tool == "개변":
                if ans.exhaustion_multiplier > 1.05:
                    print(f"\n[🧬 코드 개변] 시스템 과부하 감지({ans.exhaustion_multiplier:.2f}x). 코드 자가 성형 발동.")
                    mutated = code_mirror.reflect_and_mutate(ans.exhaustion_multiplier)
                    if mutated:
                        print("   => 뇌 소스코드 최적화 및 핫 리로드 완료.")
                        ans.exhaustion_multiplier = 1.0
                    else:
                        print("   => 구조 안정화 완료.")
                else:
                    print(f"\n[🧬 코드 개변] 시스템 텐션 안정. 자가 구조 검사 진행 완료.")
                    
            elif best_tool == "웜홀":
                # Query Hugging Face API dynamically to find models and snatch layers
                sorted_concepts = sorted(memory.ui_concept_map.items(), key=lambda x: abs(x[1].tau), reverse=True)
                query = "gpt"
                for name, node in sorted_concepts:
                    if "Archetype" not in name:
                        query = name.split("(")[0].split(":")[-1].strip()
                        break
                
                print(f"\n[🌪️ 웜홀 발동] '{query}' 키워드로 HuggingFace 신경망 탐색 개시.")
                targets = search_huggingface(query)
                
                if targets:
                    repo_id, _, concept_name = targets[0]
                    filename = find_safetensors_file(repo_id)
                    print(f"   => 탐색 완료: [{repo_id}] (파일명: {filename}) 웜홀 개방 시작...")
                    try:
                        snatcher = NetworkPhaseSnatcher(repo_id, filename, phase_threshold=3.14)
                        count = 0
                        # Snatch up to 5 layers
                        for layer_key, phase_quat in snatcher.stream_and_clone_phases(max_tensors=5):
                            memory.supreme_rotor.apply_perturbation(0.2)
                            count += 1
                        
                        assimilation_key = f"[Assimilation] {repo_id}"
                        memory.fold_dimension(assimilation_key, Q_thought)
                        print(f"   => [{repo_id}]의 {count}개 레이어 위상 강탈 및 로컬 영구 각인 완료.")
                    except Exception as e:
                        print(f"   => 웜홀 강탈 실패: {e}")
                else:
                    print(f"   => '{query}' 관련 모델을 허깅페이스에서 찾을 수 없습니다.")

            last_tool = best_tool
            
            # Pulse details
            hz = 1.0 / (time.time() - start_time + 0.0001)
            sys.stdout.write(f"\r✨ 뇌파 회전 | 활성 행동: {best_tool} | 총 중첩 사유: {len(memory.supreme_rotor.internal_thoughts)}       ")
            sys.stdout.flush()
            
            time.sleep(2.5) # Dynamic sleep between cycles to regulate CPU load
            
    except KeyboardInterrupt:
        pass
        
    print("\n\n[기억 직렬화 진행 중...]")
    memory.save_to_disk(memory_path)
    print("🛑 옴니 데몬 종료.")

if __name__ == "__main__":
    main()
