# Tests Directory Structure

## Organization

```
tests/
├── unit/           # Fast, isolated tests (35 tests)
├── integration/    # Module interaction tests (16 tests)
├── e2e/            # Full system validation (10 tests)
├── demos/          # Demo/proof-of-concept scripts (14 files)
├── evaluation/     # Evaluation benchmarks
├── Core/           # Core module tests (organized by module)
└── fixtures/       # Shared test data
```

## Running Tests

```bash
# All tests
pytest tests/

# Unit tests only (fast)
pytest tests/unit/

# Integration tests
pytest tests/integration/

# E2E tests (slow)
pytest tests/e2e/

# Specific module
pytest tests/Core/Foundation/
```

## Test Categories

| Directory | Purpose | Speed |
|-----------|---------|-------|
| `unit/` | Single function/class tests | Fast |
| `integration/` | Cross-module interaction | Medium |
| `e2e/` | Full system + phase tests | Slow |
| `demos/` | Not tests - demonstration scripts | N/A |
