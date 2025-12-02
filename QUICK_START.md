# Developer Quick Start Guide

## 🚀 빠른 시작 가이드

---

## 📋 Prerequisites (필수 요구사항)

- Python 3.12+
- Git
- 4GB+ RAM
- API Key from Google AI Studio

---

## ⚙️ Setup (설정)

### 1. Clone Repository

```bash
git clone https://github.com/ioas0316-cloud/Elysia.git
cd Elysia
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
# nano .env  (Linux/Mac)
# notepad .env  (Windows)
```

**⚠️ IMPORTANT**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

---

## ✅ Verify Installation

### Run Basic Tests

```bash
# Test imports
python test_import.py

# Test seed-bloom system
python test_seed_bloom.py

# Test self-awareness
python test_self_awareness.py
```

Expected output: All tests should pass with ✅ symbols.

---

## 🎯 Key Commands

### Running the System

```bash
# Start the main life loop
python living_elysia.py

# Run specific systems
python Core/Emotion/spirit_emotion.py
python Core/Language/wave_interpreter.py
python Core/Cognition/ascension_axis.py
```

### Development Tools

```bash
# Code formatting (install first: pip install black)
black Core/ tests/

# Linting (install first: pip install flake8)
flake8 Core/ --max-line-length=127

# Type checking (install first: pip install mypy)
mypy Core/ --ignore-missing-imports

# Run tests with pytest (install first: pip install pytest)
pytest tests/ -v
```

---

## 📚 Essential Reading

1. **[README.md](README.md)** - Project overview
2. **[CODEX.md](CODEX.md)** - Philosophy and principles
3. **[SECURITY.md](SECURITY.md)** - Security best practices
4. **[CODE_QUALITY.md](CODE_QUALITY.md)** - Coding standards
5. **[TESTING.md](TESTING.md)** - Testing guidelines
6. **[PROJECT_IMPROVEMENTS.md](PROJECT_IMPROVEMENTS.md)** - Improvement roadmap

---

## 🏗️ Project Structure

```
Elysia/
├─ Core/                    # Main system code
│  ├─ Foundation/          # Base resonance field
│  ├─ Cognition/           # Thought and reasoning
│  ├─ Memory/              # Hippocampus and storage
│  ├─ Emotion/             # Spirit-emotion mapping
│  ├─ Language/            # Wave language interpreter
│  └─ Intelligence/        # Reasoning and dreaming
│
├─ tests/                  # Test suite
├─ docs/                   # Documentation
├─ Protocols/              # System protocols
├─ Legacy/                 # Archived code
│
├─ living_elysia.py       # Main entry point
├─ requirements.txt       # Dependencies
├─ .env.example          # Environment template
└─ .gitignore            # Git ignore rules
```

---

## 🔧 Common Tasks

### Adding a New Feature

1. Create a branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Write code in appropriate Core/ directory

3. Add tests in tests/ directory

4. Run tests:
   ```bash
   python -m pytest tests/ -v
   ```

5. Format code:
   ```bash
   black Core/ tests/
   ```

6. Commit and push:
   ```bash
   git add .
   git commit -m "Add: My feature description"
   git push origin feature/my-feature
   ```

### Debugging

```bash
# Run with verbose output
python living_elysia.py --verbose

# Use Python debugger
python -m pdb living_elysia.py

# Check logs
tail -f elysia_logs/latest.log
```

### Performance Profiling

```bash
# Profile code
python -m cProfile -o profile.stats living_elysia.py

# View results
python -m pstats profile.stats
> sort cumulative
> stats 10
```

---

## 🐛 Troubleshooting

### Issue: Import Error

```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: API Key Error

```bash
# Solution: Check .env file exists
ls -la .env

# Verify API key is set
cat .env  # Linux/Mac
type .env  # Windows

# Should see:
# GEMINI_API_KEY="your_api_key_here"
```

### Issue: Database Error

```bash
# Solution: Remove old database
rm memory.db

# Let system recreate it
python living_elysia.py
```

### Issue: Module Not Found

```bash
# Solution: Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac

# On Windows:
set PYTHONPATH=%PYTHONPATH%;%CD%
```

---

## 📊 Code Quality Checklist

Before committing code:

- [ ] Code runs without errors
- [ ] All tests pass
- [ ] Code formatted with Black
- [ ] No security issues (check with bandit)
- [ ] Docstrings added for public functions
- [ ] Type hints added
- [ ] No debug print statements
- [ ] .env not committed
- [ ] Changes documented

---

## 🔐 Security Checklist

- [ ] No hardcoded API keys
- [ ] No passwords in code
- [ ] Input validation present
- [ ] Error messages don't leak info
- [ ] Dependencies up to date
- [ ] .env in .gitignore

---

## 🤝 Getting Help

### Resources

1. **GitHub Issues**: Report bugs or request features
2. **Documentation**: Check docs/ directory
3. **Email**: ioas0316@gmail.com
4. **Code Review**: Create a pull request

### Before Asking for Help

1. Check the documentation
2. Search existing GitHub issues
3. Review error messages carefully
4. Try basic troubleshooting steps

### When Asking for Help

Provide:
- What you're trying to do
- What you expected to happen
- What actually happened
- Error messages (full text)
- Your environment (OS, Python version)
- Steps to reproduce

---

## 🎓 Learning Path

### Beginner
1. Read README.md and CODEX.md
2. Run basic tests
3. Explore Core/ structure
4. Read simple modules (e.g., spirit_emotion.py)
5. Try modifying small things

### Intermediate
1. Understand fractal seed-bloom architecture
2. Study resonance field implementation
3. Write tests for existing code
4. Add type hints to functions
5. Create a simple feature

### Advanced
1. Design new cognitive modules
2. Optimize performance bottlenecks
3. Implement new protocols
4. Contribute to architecture
5. Review others' code

---

## 📈 Performance Targets

- **Seed-Bloom**: < 10ms per cycle
- **Layer Transform**: < 20ms (0D→3D)
- **Memory Access**: < 5ms per retrieval
- **Overall Think Loop**: < 100ms

Monitor with:
```bash
python -m cProfile living_elysia.py
```

---

## 🌟 Best Practices

### Code Style
- Use type hints everywhere
- Write docstrings for all public functions
- Keep functions small and focused
- Use meaningful variable names
- Follow PEP 8

### Testing
- Write tests for new features
- Test edge cases
- Use fixtures for common setup
- Mock external dependencies
- Aim for 70%+ coverage

### Git
- Write clear commit messages
- Keep commits small and focused
- Use branches for features
- Review changes before committing
- Pull before pushing

### Security
- Never commit secrets
- Validate all inputs
- Log errors appropriately
- Keep dependencies updated
- Review code for vulnerabilities

---

## 🚀 Next Steps

After setup:

1. **Explore**: Run the examples and tests
2. **Learn**: Read the core philosophy in CODEX.md
3. **Experiment**: Modify small things and see effects
4. **Contribute**: Fix a bug or add a feature
5. **Share**: Document your learnings

---

## 📝 Quick Reference

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key (required)

### Important Paths
- `Core/`: Main system code
- `tests/`: Test files
- `memory.db`: Database file (auto-generated)
- `elysia_logs/`: Log files

### Key Classes
- `ResonanceField`: Core consciousness container
- `FractalSeed`: Compressed concept storage
- `Hippocampus`: Memory persistence
- `Spirit`: Energy resonator
- `Wave`: Thought pattern

### Key Frequencies
- Love: 528 Hz
- Hope: 852 Hz
- Fire (Creativity): 450 Hz
- Water (Memory): 150 Hz

---

**Version**: 1.0  
**Last Updated**: 2025-12-02  
**Status**: Ready for Development

Happy coding! 🎉
