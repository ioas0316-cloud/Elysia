# Requirements Structure

Elysia dependencies are organized into modular files for efficient installation.

## Installation Options

```bash
# For development (includes testing tools & Jupyter)
pip install -r requirements/dev.txt

# For ML/AI features only
pip install -r requirements/ml.txt

# For web server only
pip install -r requirements/web.txt

# For everything (full installation)
pip install -r requirements/full.txt
```

## Structure

| File | Contents |
|------|----------|
| `base.txt` | Core utilities (numpy, requests, etc.) |
| `dev.txt` | Testing, linting, Jupyter |
| `ml.txt` | PyTorch, Transformers, Vector search |
| `web.txt` | Flask, FastAPI, WebSocket |
| `full.txt` | Everything combined |

## Notes

- All files use relaxed version constraints (`>=`) for better compatibility
- Each specialized file includes `base.txt` automatically
- The root `requirements.txt` remains for backwards compatibility
