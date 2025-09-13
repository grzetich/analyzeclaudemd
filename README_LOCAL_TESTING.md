# Local Testing Guide for Claude.md Analyzer

This guide helps you test the Claude.md analyzer locally when C++ dependencies cause installation issues.

## Problem: C++ Dependencies

The full analyzer requires these packages that may need C++ compilation:
- `numpy` (linear algebra)
- `scipy` (scientific computing)
- `scikit-learn` (machine learning/LDA)
- `pandas` (data processing)
- `psutil` (system monitoring)

## Solution: Multiple Testing Approaches

### Option 1: Full Installation (if C++ works)

```bash
# Use the correct Python interpreter
"/c/Users/edgrz/AppData/Local/Microsoft/WindowsApps/python.exe" -m pip install -r requirements.txt

# Run the full application
"/c/Users/edgrz/AppData/Local/Microsoft/WindowsApps/python.exe" app.py
```

### Option 2: Minimal Dependencies

```bash
# Install only basic web dependencies
pip install -r requirements-minimal.txt

# Use the mock version
python app_mock.py
```

### Option 3: Conda (better binary support)

```bash
# Create conda environment
conda create -n claudeanalyzer python=3.11
conda activate claudeanalyzer

# Install scientific packages via conda
conda install numpy scipy scikit-learn pandas flask requests nltk

# Run the full app
python app.py
```

## Testing Your Setup

### Quick Dependency Check
```bash
python test_simple.py
```

This will test:
- ✅ Basic Python modules
- ✅ Flask web framework
- ✅ Requests HTTP library
- ⚠️ Scientific stack (may fail with C++ issues)

### Test Mock Functionality
```bash
python -c "from app_mock import mock_lda_analysis; print('Mock LDA works!')"
```

## File Overview

### Created for Local Testing:

1. **`requirements-minimal.txt`** - Basic dependencies only
2. **`app_mock.py`** - Full mock version without C++ deps
3. **`test_simple.py`** - Dependency checker
4. **`README_LOCAL_TESTING.md`** - This guide

### Mock Features:

The mock version (`app_mock.py`) provides:
- ✅ **Mock LDA Analysis** - Simulates topic modeling without scikit-learn
- ✅ **Mock Memory Management** - No psutil dependency
- ✅ **Mock NLTK** - Fallback text processing if NLTK unavailable
- ✅ **Flask Web Interface** - Same routes as full version
- ✅ **HTML Visualization** - Simplified topic display

## Running the Mock Version

```bash
# Start mock server (runs on port 5001 to avoid conflicts)
python app_mock.py

# Test in browser
curl http://localhost:5001/
curl http://localhost:5001/test-mock-analysis
```

## Mock vs Real Comparison

| Feature | Real Version | Mock Version |
|---------|-------------|--------------|
| LDA Algorithm | scikit-learn LatentDirichletAllocation | Simple word-based topics |
| Memory Monitoring | psutil real metrics | Fake memory stats |
| Text Processing | Full NLTK pipeline | Basic tokenization |
| GitHub Integration | ✅ Same | ✅ Same |
| Visualization | ✅ Same HTML output | ✅ Simplified HTML |
| Performance | Production-ready | Development testing |

## Common Issues & Solutions

### Issue: `ImportError: No module named 'numpy'`
**Solution:** Use `app_mock.py` instead of `app.py`

### Issue: `Microsoft Visual C++ 14.0 is required`
**Solutions:**
1. Use mock version: `python app_mock.py`
2. Install via conda: `conda install numpy scipy scikit-learn`
3. Use pre-compiled wheels: `pip install --only-binary=all numpy scipy scikit-learn`

### Issue: Unicode errors on Windows console
**Solution:** The mock version removes emoji characters to avoid encoding issues

### Issue: NLTK data not found
**Solution:** 
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

## Production Deployment

For production, always use the full version with real dependencies:
- Real LDA provides accurate topic modeling
- Memory monitoring catches issues early
- Full NLTK gives better text preprocessing

The mock version is **only for local development testing** when C++ compilation fails.

## Verification Commands

```bash
# Test environment
python test_simple.py

# Test mock LDA
python -c "from app_mock import mock_lda_analysis; success, topics = mock_lda_analysis(['test doc'], 1); print(f'Works: {success}')"

# Test Flask routes
python -c "from app_mock import app; print('Mock app routes:', [r.rule for r in app.url_map.iter_rules()])"
```

## Next Steps

1. **Start with mock version** to verify Flask/web functionality
2. **Test GitHub API integration** (uses same code in both versions)
3. **Gradually add real dependencies** as C++ compilation issues are resolved
4. **Compare mock vs real outputs** to validate behavior

The mock version lets you develop and test the web interface, routing, and GitHub integration without wrestling with C++ compilation issues.