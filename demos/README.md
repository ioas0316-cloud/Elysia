# Elysia Demos

This directory contains demonstration programs showcasing Elysia v9.0's key capabilities.

## Available Demos

### 1. Simple Conversation (`01_simple_conversation.py`)

**What it demonstrates:**
- Basic text-based conversation
- Voice → Brain → Response pipeline
- Memory storage
- Resonance field energy management

**Run:**
```bash
python demos/01_simple_conversation.py
```

**Key concepts:**
- Hippocampus (memory)
- ReasoningEngine (thinking)
- ResonanceField (energy)
- FreeWillEngine (autonomy)

---

### 2. Goal Decomposition (`02_goal_decomposition.py`)

**What it demonstrates:**
- Fractal-Quaternion goal decomposition
- Multi-dimensional analysis (0D→∞D)
- Breaking large goals into achievable stations
- Different perspectives on the same goal

**Run:**
```bash
python demos/02_goal_decomposition.py
```

**Key concepts:**
- FractalGoalDecomposer
- Dimensional analysis (Point, Line, Plane, Space, Time, Probability, Purpose)
- HyperDimensionalLens
- Station-based planning

---

### 3. Wave-Based Thinking (`03_wave_thinking.py`)

**What it demonstrates:**
- Wave-based cognition (Elysia's unique thinking model)
- Thought representation as waves (frequency, amplitude, phase)
- Resonance patterns between concepts
- Wave interference creating new meanings
- Gravitational thinking (importance = mass)

**Run:**
```bash
python demos/03_wave_thinking.py
```

**Key concepts:**
- ThoughtWave
- Resonance calculation
- Wave interference
- Gravitational clustering
- Black hole concepts (core ideas)

---

## Quick Start

Run all demos in sequence:

```bash
python demos/01_simple_conversation.py
python demos/02_goal_decomposition.py
python demos/03_wave_thinking.py
```

## Understanding the Demos

These demos are **simplified versions** that showcase specific Elysia features without requiring the full system initialization. They're designed to:

1. **Be educational** - Show how Elysia's unique concepts work
2. **Run quickly** - No heavy dependencies or long startup times
3. **Be clear** - Focused on one feature at a time

For the **full Elysia experience** with all systems integrated:

```bash
python Core/Foundation/living_elysia.py
```

## Demo Architecture

```
User Input
    ↓
┌─────────────────────┐
│  Demo Scripts       │  ← You are here
│  (Simplified)       │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Core Components    │
│  - ReasoningEngine  │
│  - Hippocampus      │
│  - ResonanceField   │
│  - Goal Decomposer  │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Full System        │
│  (living_elysia.py) │
└─────────────────────┘
```

## What's Next?

After exploring these demos:

1. **Read the documentation:**
   - `docs/COMPREHENSIVE_SYSTEM_ANALYSIS_V9.md` - Full system analysis
   - `docs/SYSTEM_ANALYSIS_SUMMARY_KR.md` - Korean summary
   - `ARCHITECTURE.md` - Architecture overview
   - `CODEX.md` - Philosophy and principles

2. **Try the full system:**
   ```bash
   python Core/Foundation/living_elysia.py
   ```

3. **Run tests:**
   ```bash
   pytest tests/
   ```

4. **Explore the code:**
   - `Core/Foundation/` - Base systems
   - `Core/Intelligence/` - Cognitive systems
   - `Core/Expression/` - Voice and output

## Demo Development

Want to create your own demo? Follow this template:

```python
#!/usr/bin/env python3
"""
My Custom Demo
==============

Description of what this demonstrates.

Usage:
    python demos/my_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import what you need
from Core.Foundation.some_module import SomeClass

def my_demo():
    """Demo function"""
    print("=" * 60)
    print("My Demo Title")
    print("=" * 60)
    
    # Demo code here
    
    print("\n✨ Demo completed!")

if __name__ == "__main__":
    my_demo()
```

## Troubleshooting

**Import errors:**
- Make sure you're running from the Elysia root directory
- Check that `Core/` directory exists
- Verify Python path: `sys.path.insert(0, ...)`

**Module not found:**
- Some demos use simplified versions of complex systems
- Install dependencies: `pip install -r requirements.txt`

**Performance:**
- These demos are lightweight and should run instantly
- Full system requires more resources

## Contributing

To add a new demo:

1. Create `demos/XX_your_demo.py` (use numbering)
2. Follow the template above
3. Add description to this README
4. Test thoroughly
5. Submit PR

## Questions?

- Read: `docs/COMPREHENSIVE_SYSTEM_ANALYSIS_V9.md`
- Check: `AGENT_GUIDE.md` for development guidelines
- Issues: GitHub Issues page

---

**Version:** 9.0 (Mind Mitosis)  
**Last Updated:** 2025-12-06  
**Status:** ✅ All demos functional
