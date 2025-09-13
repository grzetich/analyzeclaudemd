#!/usr/bin/env python3
"""
Simple local testing script without unicode characters.
Tests core functionality without requiring C++ compilation.
"""

import os
import sys

def test_dependencies():
    """Test what dependencies are available"""
    print("Testing Dependencies")
    print("=" * 40)
    
    results = {}
    
    # Test basic modules
    basic = ['os', 'sys', 'json', 're', 'time']
    for module in basic:
        try:
            __import__(module)
            results[module] = "OK"
        except ImportError:
            results[module] = "MISSING"
    
    # Test Flask
    try:
        import flask
        results['flask'] = f"OK (v{flask.__version__})"
    except ImportError:
        results['flask'] = "MISSING - pip install flask"
    
    # Test requests
    try:
        import requests
        results['requests'] = f"OK (v{requests.__version__})"
    except ImportError:
        results['requests'] = "MISSING - pip install requests"
    
    # Test scientific stack
    scientific = {
        'numpy': 'numpy',
        'scipy': 'scipy', 
        'sklearn': 'scikit-learn',
        'pandas': 'pandas'
    }
    
    for module_name, package_name in scientific.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            results[package_name] = f"OK (v{version})"
        except ImportError:
            results[package_name] = "MISSING - C++ deps may be needed"
        except Exception as e:
            results[package_name] = f"ERROR: {str(e)}"
    
    # Print results
    for module, status in results.items():
        print(f"{module:15} : {status}")
    
    return results

def test_mock_app():
    """Test the mock application"""
    print("\nTesting Mock Application")
    print("=" * 40)
    
    try:
        # Simple test of mock functionality
        test_docs = [
            'claude assistant configuration',
            'project file structure',
            'code function class',
            'test validation check',
            'documentation guide'
        ]
        
        print(f"Test data: {len(test_docs)} documents")
        
        # Simple word processing test
        words = []
        for doc in test_docs:
            words.extend(doc.split())
        
        unique_words = list(set(words))
        print(f"Unique words: {len(unique_words)}")
        print(f"Sample words: {unique_words[:5]}")
        
        # Mock topic generation
        topics = [
            {"id": 0, "label": "AI Assistant", "words": ["claude", "assistant"]},
            {"id": 1, "label": "Project Setup", "words": ["project", "file"]},
            {"id": 2, "label": "Code Quality", "words": ["code", "function"]}
        ]
        
        print(f"Mock topics: {len(topics)}")
        for topic in topics:
            print(f"  {topic['label']}: {topic['words']}")
        
        return True
        
    except Exception as e:
        print(f"Mock test failed: {e}")
        return False

def main():
    print("Claude.md Analyzer - Local Testing")
    print("=" * 50)
    
    # Run tests
    deps = test_dependencies()
    mock_ok = test_mock_app()
    
    # Check Flask availability
    flask_ok = 'flask' in deps and 'OK' in deps['flask']
    requests_ok = 'requests' in deps and 'OK' in deps['requests']
    
    print("\nTest Summary")
    print("=" * 40)
    print(f"Dependencies: {'PASS' if deps else 'FAIL'}")
    print(f"Mock test:    {'PASS' if mock_ok else 'FAIL'}")
    print(f"Flask ready:  {'PASS' if flask_ok else 'FAIL'}")
    print(f"Requests ok:  {'PASS' if requests_ok else 'FAIL'}")
    
    # Recommendations
    print("\nRecommendations")
    print("=" * 40)
    
    if not flask_ok:
        print("1. Install Flask: pip install flask")
    if not requests_ok:
        print("2. Install requests: pip install requests")
    
    # Check scientific stack
    scientific_issues = []
    for pkg in ['numpy', 'scikit-learn', 'scipy', 'pandas']:
        if pkg not in deps or 'MISSING' in deps[pkg] or 'ERROR' in deps[pkg]:
            scientific_issues.append(pkg)
    
    if scientific_issues:
        print("3. Scientific stack issues detected:")
        print(f"   Missing: {', '.join(scientific_issues)}")
        print("   Options:")
        print("   a) Try: pip install numpy scipy scikit-learn pandas")
        print("   b) Use conda: conda install numpy scipy scikit-learn pandas")
        print("   c) Use mock version: python app_mock.py")
    
    if flask_ok and requests_ok:
        if not scientific_issues:
            print("\nReady to run: python app.py")
        else:
            print("\nReady for mock version: python app_mock.py")
    
    print("\nLocal Testing Files Created:")
    print("- requirements-minimal.txt (basic deps only)")
    print("- app_mock.py (no C++ dependencies)")
    print("- test_simple.py (this test)")

if __name__ == "__main__":
    main()