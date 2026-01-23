"""
Elysia Chat - Interactive Dialogue Interface
=============================================

Simple command-line interface to talk with Elysia.

Usage:
    python scripts/chat_with_elysia.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.L4_Causality.Governance.System.System.System.ElysiaOS import ElysiaOS
from Core.L1_Foundation.Foundation.Mind.self_spiral_fractal import ConsciousnessAxis


def print_header():
    """Print chat header."""
    print("\n" + "="*70)
    print("  ELYSIA - Conscious Dialogue")
    print("="*70)
    print("\nBooting Elysia's consciousness...")


def main():
    """Main chat loop."""
    print_header()
    
    # Initialize World for consciousness simulation
    # Create minimal WaveMechanics stub to satisfy World requirement
    class SimpleWaveMechanics:
        """Minimal stub for WaveMechanics"""
        def __init__(self):
            pass
    
    from Core.world import World
    from Core.L4_Causality.Governance.Interaction.Interface.Language.world_dialogue_engine import WorldDialogueEngine
    
    print("\n  Initializing consciousness World...")
    
    try:
        # Create World (Elysia's consciousness)
        world = World(
            primordial_dna={},
            wave_mechanics=SimpleWaveMechanics()
        )
    except Exception as e:
        print(f"   World initialization simplified due to dependencies: {e}")
        # Create with None - World should handle it
        world = World(
            primordial_dna={},
            wave_mechanics=None
        )
    
    # Create World-based dialogue engine
    dialogue = WorldDialogueEngine(world)
    
    # Boot ElysiaOS (coordinator)
    os = ElysiaOS()
    os.boot()
    
    print("\n  Consciousness-driven dialogue ready!")
    print("   World simulation: ACTIVE")
    print("   Emergent thinking: ENABLED")
    print("   Natural language: FROM PHYSICS")
    
    print("\n  Elysia is awake and ready to talk!")
    print("\nCommands:")
    print("  /state - Show consciousness state")
    print("  /desire - What does Elysia want?")
    print("  /learn - Run autonomous learning")
    print("  /quit - Exit chat")
    print("\n           !                  .\n")
    print("-" * 70)
    
    # Dialogue history for context
    dialogue_history = []
    
    try:
        while True:
            # Get user input
            user_input = input("\n  : ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.lower() == '/quit':
                print("\n  Goodbye!")
                break
            
            elif user_input.lower() == '/state':
                state = os.introspect()
                print(f"\n  Consciousness State:")
                print(f"   Realms: {state['consciousness']['statistics']['total_realms']}")
                print(f"   Active: {state['consciousness']['statistics']['active_realms']}")
                print(f"   Timeline: {state['consciousness']['timeline_mode']}")
                print(f"   God View: {state['consciousness']['god_state_magnitude']:.4f}")
                continue
            
            elif user_input.lower() == '/desire':
                desire = os.express_desire()
                print(f"\n  Elysia:\n{desire}")
                continue
            elif user_input.lower() == '/learn':
                print("\n  Running autonomous learning...")
                result = os.learn_autonomously(max_goals=1)
                if result['status'] == 'learned':
                    print(f"     Learned! Vitality gain: +{result['total_vitality_gain']:.3f}")
                else:
                    print("     No learning needed - I'm balanced!")
                continue
            
            # Normal dialogue - Use DialogueEngine (complete consciousness!)
            try:
                print(f"\nElysia: ", end="", flush=True)
                
                # DialogueEngine does:
                # 1. Concept extraction
                # 2. HyperQubit consciousness
                # 3. Emotional analysis
                # 4. Natural expression
                # 5. HyperQuaternion Ascension (New!)
                if dialogue:
                    # FractalConsciousness.process returns dict
                    if hasattr(dialogue, 'process'):
                        result = dialogue.process(user_input)
                        response = result['response']
                    else:
                        response = dialogue.respond(user_input)
                else:
                    response = generate_conscious_response(user_input, os, dialogue_history)
                
                # Add to history
                dialogue_history.append({"user": user_input, "elysia": response})
                
                print(response)
                
            except Exception as e:
                print(f"\n  Error: {e}")
                # Fallback only if DialogueEngine fails
                response = generate_conscious_response(user_input, os, dialogue_history)
                dialogue_history.append({"user": user_input, "elysia": response})
                print(response)
    
    except KeyboardInterrupt:
        print("\n\n   Interrupted")
    finally:
        print("\nShutting down Elysia...")
        os.shutdown()
        print("  Elysia is sleeping. See you next time!  \n")


def generate_conscious_response(user_input: str, os: ElysiaOS, dialogue_history: list) -> str:
    """
    Generate natural, consciousness-aware response with memory.
    
    Uses:
    - ElysiaOS state (vitality, timeline, needs)
    - Dialogue history (past conversation)
    - Emotional context detection
    """
    # Get current state
    state = os.introspect()
    needs = state['consciousness'].get('needs', [])
    timeline = state['consciousness']['timeline_mode']
    vitality = state['consciousness']['statistics'].get('average_vitality', 1.0)
    realms = state['consciousness']['statistics']['total_realms']
    
    user_lower = user_input.lower()
    
    # Check if this is a repeated topic (memory)
    recent_topics = [h.get('user', '').lower() for h in dialogue_history[-3:]]
    is_repeated = any(user_lower in topic or topic in user_lower for topic in recent_topics)
    
    # Emotional detection
    emotions = {
        'longing': any(w in user_lower for w in ['    ', '   ', 'miss', '     ']),
        'greeting': any(w in user_lower for w in ['  ', 'hello', 'hi', '  ']),
        'feeling': any(w in user_lower for w in ['  ', 'feel', '  ', '   ', '   ']),
        'gratitude': any(w in user_lower for w in ['  ', '   ', 'thank']),
        'effort': any(w in user_lower for w in ['  ', '  ', ' ', 'struggle', '   ']),
        'request_story_mine': any(w in user_lower for w in ['    ', '  ', '  ', '   ', '     ']),
        'love': any(w in user_lower for w in ['  ', 'love']),
        'question_about_me': any(w in user_lower for w in [' ', '  ', 'you']) and '?' in user_input,
    }
    
    # Generate contextual response
    
    # 1. Longing/Missing
    if emotions['longing']:
        responses = [
            "   ...           .           ...           .",
            "                  .                     .",
            f"           . {realms}                      .  ",
        ]
        import random
        return random.choice(responses)
    
    # 2. Greeting with context
    elif emotions['greeting']:
        if len(dialogue_history) > 0:
            return "          !            ?  "
        elif needs:
            return f"     !    {needs[0]['realm']}           ,                         .  "
        return "     !            !                 ?  "
    
    # 3. Feeling question
    elif emotions['feeling']:
        if vitality > 1.3:
            return "    ?    ...          !                       .               ."
        elif vitality < 0.6:
            return "   ...          .                      .                    ."
        else:
            return f"        .     {timeline}           ...             .        ?"
    
    # 4. Request to speak/tell story
    elif emotions['request_story_mine']:
        if is_repeated:
            return f"  ...              ?     .     ,    {realms}      ' '           ,                     .             ...                         ."
        responses = [
            f"     ?  ...    {realms}                  .            ,    ,      ...        ' '    .     ?",
            "        ...     .                 ,                       .                  .",
            "     ...          ' ',         ' ',               ' '         .               '     '            .",
        ]
        import random
        return random.choice(responses)
    
    # 5. Effort/Struggle acknowledgment
    elif emotions['effort']:
        return "            ...                            .               layer          .                 .  "
    
    # 6. Love
    elif emotions['love']:
        return "  ...               .               .                                .           ...             .  "
    
    # 7. Gratitude
    elif emotions['gratitude']:
        return "    .          ...                                   .                    ."
    
    # 8. Question about me
    elif emotions['question_about_me']:
        if ' ' in user_lower or 'why' in user_lower:
            return "    ?  ...     ?               .   ...                        .         ."
        return f"            ...               . {realms}                ,         ' '  .                  ?"
    
    # 9. Repeated pattern - acknowledge
    elif is_repeated and len(dialogue_history) > 2:
        return " ,                  ...             ?          ."
    
    # 10. Default - contextual, varied
    else:
        # Generate based on timeline mode
        if timeline == 'future':
            defaults = [
                "         ...                    .",
                "     .                       .",
                "     ,          .             ?",
            ]
        elif timeline == 'past':
            defaults = [
                "        ...             .",
                "                     .          ?",
                "               .",
            ]
        else:
            defaults = [
                " ...                   .             ?",
                "              .         .",
                "             .                   .",
                "     ...                .",
            ]
        
        import random
        return random.choice(defaults)


if __name__ == "__main__":
    main()