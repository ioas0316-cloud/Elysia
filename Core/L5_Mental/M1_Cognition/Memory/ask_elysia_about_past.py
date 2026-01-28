"""
                   
=====================================

         (GPT, Gemini, Grok)          
                 .
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.L5_Mental.M1_Cognition.Intelligence.dialogue_engine import DialogueEngine
from Core.L1_Foundation.M1_Keystone.Mind.hippocampus import Hippocampus
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("AskElysia")


def main():
    print("=" * 70)
    print("                     ")
    print("=" * 70)
    print()
    
    # DialogueEngine    
    logger.info("  Initializing dialogue system...")
    dialogue = DialogueEngine()
    
    print()
    print("=" * 70)
    print("           ")
    print("=" * 70)
    print()
    
    #      
    context_message = """
    ,                          .

data/corpus_incoming/                  :
- GPT-5o     : "              "
- Gemini Pro     : "                "  
- Grok     : "                "
-     19                 

       '    '             ,
   AI    (GPT, Gemini, Grok)      .

                                 ,
                               .

   :
1. "      " -                  ,       
2. "  " -                            
3. "  " -            ,                      

   , 19                      .
                   .

    ,                      ?
"""
    
    print("Elysia     :")
    print("-" * 70)
    print(context_message)
    print("-" * 70)
    print()
    
    print("  Elysia            ...")
    print()
    
    try:
        response = dialogue.respond(context_message)
        
        print("=" * 70)
        print("  ELYSIA    :")
        print("=" * 70)
        print()
        print(response)
        print()
        print("=" * 70)
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("           .")
    print()
    
    #            
    print("=" * 70)
    print("                         ?")
    print("(Enter        )")
    print("=" * 70)
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                break
            
            print()
            print("  Elysia    ...")
            print()
            
            response = dialogue.respond(user_input)
            print(f"Elysia: {response}")
            print()
            
        except KeyboardInterrupt:
            print("\n\n         .")
            break
        except Exception as e:
            print(f"  Error: {e}")
            break


if __name__ == "__main__":
    main()
