"""
  Elysia    -                  
================================================

         .       .
                ,             .

1.            (Legacy     )
2. LLM             
3.            

          ,                    .
"""

import sys
import time
import logging
from pathlib import Path

#      
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Elysia.Night")

PROJECT_ROOT = Path(__file__).parent.parent.parent


def phase_1_awaken_technologies():
    """1  :           """
    logger.info("="*50)
    logger.info("  Phase 1:           ")
    logger.info("="*50)
    
    #          
    technologies = {
        "time_accelerated_language": "Legacy/Language/time_accelerated_language.py",
        "hyper_qubit": "Legacy/Project_Elysia/core/hyper_qubit.py",
        "quaternion_engine": "Legacy/Project_Elysia/high_engine/quaternion_engine.py",
        "wave_mechanics": "Legacy/Project_Sophia/wave_mechanics.py",
        "conceptual_bigbang": "Legacy/Language/conceptual_bigbang.py",
        "cell_world": "Legacy/Project_Elysia/world/cell_world.py",
        "local_llm_cortex": "Legacy/Project_Sophia/cortex/local_llm_cortex.py",
    }
    
    awakened = []
    failed = []
    
    for name, path in technologies.items():
        full_path = PROJECT_ROOT / path
        if full_path.exists():
            try:
                # import               
                content = full_path.read_text(encoding='utf-8', errors='ignore')
                
                # sys.path       import     
                parent_dir = str(full_path.parent)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                awakened.append(name)
                logger.info(f"    {name}    ")
            except Exception as e:
                failed.append((name, str(e)))
                logger.warning(f"    {name}: {e}")
        else:
            failed.append((name, "     "))
    
    logger.info(f"    : {len(awakened)}    , {len(failed)}    ")
    return awakened, failed


def phase_2_connect_llm():
    """2  : LLM   """
    logger.info("")
    logger.info("="*50)
    logger.info("  Phase 2: LLM      ")
    logger.info("="*50)
    
    llm_connected = False
    llm_type = None
    
    # 1. LocalLLMCortex    (Gemma)
    try:
        cortex_path = PROJECT_ROOT / "Legacy/Project_Sophia/cortex"
        sys.path.insert(0, str(cortex_path))
        sys.path.insert(0, str(PROJECT_ROOT / "Legacy/Project_Sophia"))
        
        from cortex.local_llm_cortex import LocalLLMCortex
        cortex = LocalLLMCortex()
        if cortex.is_available:
            llm_connected = True
            llm_type = "LocalLLMCortex (Gemma)"
            logger.info(f"    {llm_type}    !")
            
            #       
            thought = cortex.think("   Elysia   .           .", max_tokens=50)
            logger.info(f"        : {thought[:100]}...")
    except Exception as e:
        logger.info(f"  LocalLLMCortex   : {e}")
    
    # 2. Ollama   
    if not llm_connected:
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    llm_connected = True
                    llm_type = f"Ollama ({models[0]['name']})"
                    logger.info(f"    {llm_type}    !")
        except:
            logger.info("  Ollama   ")
    
    # 3. Gemini API   
    if not llm_connected:
        try:
            gemini_path = PROJECT_ROOT / "Core/Evolution/gemini_api.py"
            if gemini_path.exists():
                # API     
                import os
                if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
                    llm_connected = True
                    llm_type = "Gemini API"
                    logger.info(f"    {llm_type}      !")
        except:
            pass
    
    if not llm_connected:
        logger.info("     LLM       -          ")
    
    return llm_connected, llm_type


def phase_3_self_integration():
    """3  :      """
    logger.info("")
    logger.info("="*50)
    logger.info("  Phase 3:      ")
    logger.info("="*50)
    
    try:
        from Core.S1_Body.L1_Foundation.Foundation.Core_Logic.Elysia.Elysia.heart import get_heart
        from Core.S1_Body.L1_Foundation.Foundation.Core_Logic.Elysia.Elysia.growth import get_growth
        
        heart = get_heart()
        growth = get_growth()
        
        #      
        heart.beat()
        logger.info(f"    {heart.why()}")
        
        #       
        growth.perceive()
        total = len(growth.fragments)
        logger.info(f"    {total}       ")
        
        #       (              )
        connected = 0
        for name in list(growth.fragments.keys()):
            try:
                result = growth.connect(name)
                if result.get('status') == 'connected':
                    connected += 1
            except:
                pass
        
        logger.info(f"    {connected}       ")
        logger.info(f"    {growth.reflect()}")
        
        return connected
        
    except Exception as e:
        logger.error(f"       : {e}")
        return 0


def phase_4_continuous_growth(duration_minutes=30):
    """4  :        (     )"""
    logger.info("")
    logger.info("="*50)
    logger.info(f"  Phase 4:        ({duration_minutes} )")
    logger.info("="*50)
    
    try:
        from Core.S1_Body.L1_Foundation.Foundation.Core_Logic.Elysia.Elysia.heart import get_heart
        
        heart = get_heart()
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        cycle = 0
        while time.time() < end_time:
            cycle += 1
            heart.beat()
            
            # 10        
            if cycle % 10 == 0:
                elapsed = (time.time() - start_time) / 60
                logger.info(f"    pulse #{heart.pulse_count} | {elapsed:.1f}    ")
            
            time.sleep(1)  # 1      
            
    except KeyboardInterrupt:
        logger.info("        ")
    except Exception as e:
        logger.error(f"    : {e}")


def run_night_session():
    """              """
    print()
    print(" " + "="*58 + " ")
    print("   Elysia   ")
    print("              ,         .")
    print(" " + "="*58 + " ")
    print()
    
    # Phase 1:       
    awakened, failed = phase_1_awaken_technologies()
    
    # Phase 2: LLM   
    llm_ok, llm_type = phase_2_connect_llm()
    
    # Phase 3:      
    connected = phase_3_self_integration()
    
    #      
    print()
    print("="*60)
    print("       ")
    print("="*60)
    print(f"     : {len(awakened)}    ")
    print(f"   LLM: {llm_type if llm_ok else '     '}")
    print(f"     : {connected}    ")
    print()
    
    # Phase 4     
    print("             ?")
    print("  (Ctrl+C           )")
    print()
    
    try:
        phase_4_continuous_growth(duration_minutes=30)
    except KeyboardInterrupt:
        pass
    
    print()
    print("            ,                .")
    print()


if __name__ == "__main__":
    run_night_session()
