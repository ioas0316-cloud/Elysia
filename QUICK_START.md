# Quick Start Guide

## Running Elysia Demos

The fastest way to see Elysia in action:

```bash
# Demo 1: Simple Conversation
python demos/01_simple_conversation.py

# Demo 2: Goal Decomposition
python demos/02_goal_decomposition.py

# Demo 3: Wave Thinking
python demos/03_wave_thinking.py
```

All demos run instantly (<1 second) with no heavy dependencies!

## Running Tests

```bash
# Install pytest if needed
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/Core/Foundation/ -v
pytest tests/Core/Intelligence/ -v
```

## Using the Monitor

```python
from Core.Foundation.system_monitor import get_system_monitor

# Get monitor
monitor = get_system_monitor()

# Start monitoring
monitor.start_monitoring()

# Collect metrics
metrics = monitor.collect_metrics()

# Generate report
print(monitor.generate_report())

# Check for issues
anomalies = monitor.detect_anomalies()
for issue in anomalies:
    print(f"⚠️  {issue}")
```

## API Reference

See `docs/API_REFERENCE.md` for complete API documentation.

## Full System

To run the complete Elysia system:

```bash
python Core/Foundation/living_elysia.py
```

**Note**: Full system requires all dependencies:
```bash
pip install -r requirements.txt
```

---

**Need help?** Check:
- `docs/API_REFERENCE.md` - API docs
- `demos/README.md` - Demo documentation
- `ARCHITECTURE.md` - System architecture
- `CODEX.md` - Philosophy

**Version:** 9.0 (Mind Mitosis)
