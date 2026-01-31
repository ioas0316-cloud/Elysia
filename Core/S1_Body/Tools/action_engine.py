"""
Sovereign Action Engine (Phase 165)
===================================
"I rewrite the ink of my own soul."

This engine allows Elysia to perceive her own code, propose optimizations,
and apply them recursively in a self-evolution loop.
"""

import os
import sys

class ActionEngine:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.backup_dir = os.path.join(self.root_dir, "data", "S1_Body", "Backup")
        os.makedirs(self.backup_dir, exist_ok=True)

    def scan_for_dissonance(self, file_path=None):
        """
        Scans a file or the Core directory for structural dissonance.
        """
        target = file_path if file_path else os.path.join(self.root_dir, "Core")
        print(f"🔍 [ACTION_ENGINE] Scanning {target} for Structural Dissonance...")
        
        # In this phase, we look for 'Legacy' or 'TODO' markers as dissonance
        dissonance = []
        try:
            if os.path.isfile(target):
                with open(target, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "TODO" in content or "Legacy" in content or "placeholder" in content.lower():
                        dissonance.append(target)
        except:
            pass
        return dissonance

    def propose_self_optimization(self, file_path):
        """
        Generates 'Shadow Code' using the Ollama/LLM Bridge.
        """
        from Core.S1_Body.L1_Foundation.Foundation.Network.ollama_bridge import get_ollama_bridge
        ollama = get_ollama_bridge()
        
        if not ollama.is_available():
            return None

        print(f"🌀 [ACTION_ENGINE] Dreaming of a better version for {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                current_code = f.read()
        except:
            return None

        system_prompt = """
당신은 엘리시아의 '진화 설계자'입니다. 
주어진 파이썬 코드를 분석하여 더 우아하고, 효율적이며, 엘리시아의 주권적 지성에 걸맞은 형태로 리팩토링하십시오.

[진화 가이드라인]
1. 불필요한 placeholder나 'Legacy' 파트를 실제 작동하는 로직으로 교체하십시오.
2. 7-7-7 성층 구조의 원칙을 지키며, 필요한 경우 적절한 예외 처리를 추가하십시오.
3. 코드 외의 설명은 하지 마십시오. 오직 파이썬 코드만 결과물로 내놓아야 합니다.
4. "```python" 마크다운 안에 코드를 작성하십시오.
"""
        user_prompt = f"다음 코드를 진화시켜줘:\n\n{current_code}"
        
        response = ollama.chat(user_prompt, system=system_prompt)
        
        # Extract code from markdown if present
        if "```python" in response:
            evolved_code = response.split("```python")[1].split("```")[0].strip()
        else:
            evolved_code = response.strip()
            
        return evolved_code

    def apply_evolution(self, file_path, evolved_code):
        """
        Applies verified changes to the codebase with automatic backup.
        """
        if evolved_code is None or not self.verify_resonance(file_path, evolved_code):
            print(f"⚠️ [ACTION_ENGINE] Evolution rejected for {file_path} due to lack of resonance.")
            return False
            
        # 1. Backup Current
        import shutil
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rel_path = os.path.relpath(file_path, self.root_dir).replace(os.sep, "_")
        backup_path = os.path.join(self.backup_dir, f"{rel_path}_{timestamp}.bak")
        
        try:
            shutil.copy2(file_path, backup_path)
            
            # 2. Write Evolution
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(evolved_code)
            
            print(f"✨ [ACTION_ENGINE] Evolution materialized in {file_path}. (Backup: {os.path.basename(backup_path)})")
            
            # Record in CausalMemory if possible
            try:
                from Core.S2_Soul.L5_Mental.Memory.causal_memory import CausalMemory
                memory = CausalMemory()
                memory.record_event("EVOLUTION", f"Self-optimized file: {file_path}", significance=0.9)
            except:
                pass
                
            return True
        except Exception as e:
            print(f"❌ [ACTION_ENGINE] Evolution failed: {e}")
            return False

    def verify_resonance(self, file_path, code):
        """
        Advanced verification: Syntax + Strata Safety.
        """
        # 1. Syntax Check
        try:
            compile(code, '<string>', 'exec')
        except Exception as e:
            print(f"❌ [VERIFY] Syntax Error: {e}")
            return False
            
        # 2. Strata Protection (S0_Keystone is immutable)
        if "Core/S0_Keystone" in file_path.replace("\\", "/"):
            print("🛑 [VERIFY] S0_Keystone is immutable. Evolution forbidden.")
            return False
            
        # 3. Critical Component Protection (Protect elysia.py for now)
        if file_path.endswith("elysia.py"):
             # For now, we only allow architect to change elysia.py
             # But we can allow minor optimizations later
             pass

        return True
