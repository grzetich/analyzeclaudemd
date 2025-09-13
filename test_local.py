#!/usr/bin/env python3
"""
Local testing script that can run with minimal dependencies.
Tests core functionality without requiring C++ compilation.
"""

import os
import sys
import json

def test_environment():
    """Test what dependencies are available"""
    print("Testing Local Environment")
    print("=" * 50)
    
    results = {}
    
    # Test basic Python modules
    basic_modules = ['os', 'sys', 'json', 're', 'time', 'datetime']
    for module in basic_modules:
        try:
            __import__(module)
            results[module] = "Available"
        except ImportError:
            results[module] = "Missing"
    
    # Test web framework
    try:
        import flask
        results['flask'] = f"[TEST] Available (v{flask.__version__})"
    except ImportError:
        results['flask'] = "[TEST] Missing - install with: pip install flask"
    
    try:
        import requests
        results['requests'] = f"[TEST] Available (v{requests.__version__})"
    except ImportError:
        results['requests'] = "[TEST] Missing - install with: pip install requests"
    
    # Test scientific stack (these might fail due to C++ deps)
    scientific_modules = {
        'numpy': 'numpy',
        'scipy': 'scipy', 
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'psutil': 'psutil'
    }
    
    for module_name, package_name in scientific_modules.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            results[package_name] = f"[TEST] Available (v{version})"
        except ImportError as e:
            results[package_name] = f"[TEST] Missing - C++ compilation may be required"
        except Exception as e:
            results[package_name] = f"[TEST][TEST]  Import error: {str(e)}"
    
    # Test NLTK
    try:
        import nltk
        results['nltk'] = f"[TEST] Available (v{nltk.__version__})"
        
        # Test NLTK data
        try:
            from nltk.corpus import stopwords
            stopwords.words('english')
            results['nltk-data'] = "[TEST] Stopwords available"
        except:
            results['nltk-data'] = "[TEST][TEST]  NLTK data not downloaded"
    except ImportError:
        results['nltk'] = "[TEST] Missing - install with: pip install nltk"
    
    # Print results
    for module, status in results.items():
        print(f"{module:15} : {status}")
    
    return results

def test_mock_lda():
    """Test the mock LDA implementation"""
    print("\n[TEST]¨ Testing Mock LDA Implementation")
    print("=" * 50)
    
    try:
        # Import mock implementation
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from app_mock import mock_lda_analysis, preprocess_text
        
        # Test data
        test_docs = [
            'claude assistant configuration prompt role system behavior instructions',
            'project file directory structure organization folder path management', 
            'code function class method style format convention programming',
            'test testing spec validation check verify quality assurance',
            'documentation readme docs guide tutorial example usage help support'
        ]
        
        print(f"[TEST]ù Testing with {len(test_docs)} documents")
        
        # Test preprocessing
        sample_text = test_docs[0]
        tokens = preprocess_text(sample_text)
        print(f"[TEST]§ Preprocessing test: '{sample_text[:30]}...' -> {len(tokens)} tokens")
        print(f"   Tokens: {tokens[:5]}...")
        
        # Test LDA
        success, topics_data = mock_lda_analysis(test_docs, num_topics=3)
        
        if success and topics_data:
            print(f"[TEST] Mock LDA successful: {len(topics_data)} topics generated")
            for i, topic in enumerate(topics_data):
                print(f"   Topic {i+1}: {topic['label']}")
                print(f"     Words: {topic['top_words'][:5]}")
            return True
        else:
            print("[TEST] Mock LDA failed")
            return False
            
    except Exception as e:
        print(f"[TEST] Mock LDA test failed: {e}")
        return False

def test_flask_minimal():
    """Test minimal Flask setup"""
    print("\n[TEST]ê Testing Minimal Flask Setup")
    print("=" * 50)
    
    try:
        from flask import Flask
        app = Flask(__name__)
        
        @app.route('/')
        def hello():
            return "Hello from minimal Flask!"
        
        @app.route('/test')
        def test():
            return {"status": "ok", "message": "Flask working"}
        
        print("[TEST] Flask app created successfully")
        print("   Routes registered: /, /test")
        print("   Ready to run with: app.run()")
        return True
        
    except Exception as e:
        print(f"[TEST] Flask test failed: {e}")
        return False

def suggest_installation_strategy(results):
    """Suggest installation strategy based on test results"""
    print("\n[TEST]° Installation Strategy Recommendations")
    print("=" * 50)
    
    missing_basic = []
    missing_scientific = []
    
    basic_required = ['flask', 'requests', 'nltk']
    scientific_optional = ['numpy', 'scikit-learn', 'scipy', 'pandas', 'psutil']
    
    for module in basic_required:
        if module not in results or '[TEST]' in results[module]:
            missing_basic.append(module)
    
    for module in scientific_optional:
        if module not in results or '[TEST]' in results[module] or '[TEST][TEST]' in results[module]:
            missing_scientific.append(module)
    
    if missing_basic:
        print("[TEST]® Critical missing dependencies (install these first):")
        for module in missing_basic:
            print(f"   pip install {module}")
    
    if missing_scientific:
        print("\n[TEST]ä Scientific stack issues (use mock version if these fail):")
        print("   Option 1 - Try installing with wheels:")
        for module in missing_scientific:
            print(f"     pip install {module}")
        
        print("\n   Option 2 - Use conda (may have better binary support):")
        print("     conda install numpy scipy scikit-learn pandas")
        
        print("\n   Option 3 - Use mock version (app_mock.py) to bypass C++ issues")
    
    if not missing_basic and not missing_scientific:
        print("[TEST] All dependencies available! You can run the full app.py")
    elif not missing_basic:
        print("[TEST] Basic dependencies available! Use app_mock.py for testing")
    
    print("\n[TEST]ã Testing Commands:")
    print("   Full version:  python app.py")
    print("   Mock version:  python app_mock.py")
    print("   This test:     python test_local.py")

def main():
    print("Local Testing for Claude.md Analyzer")
    print("=" * 60)
    print("This script helps you test the analyzer without C++ compilation issues\n")
    
    # Run tests
    results = test_environment()
    mock_success = test_mock_lda()
    flask_success = test_flask_minimal()
    
    # Provide recommendations
    suggest_installation_strategy(results)
    
    # Summary
    print(f"\n[TEST]ä Test Summary")
    print("=" * 50)
    print(f"Environment check: {'[TEST] Pass' if results else '[TEST] Fail'}")
    print(f"Mock LDA test:     {'[TEST] Pass' if mock_success else '[TEST] Fail'}")
    print(f"Flask test:        {'[TEST] Pass' if flask_success else '[TEST] Fail'}")
    
    if mock_success and flask_success:
        print("\n[TEST]â You can use the mock version for local testing!")
        print("   Run: python app_mock.py")
        print("   Then visit: http://localhost:5001")
    else:
        print("\n[TEST][TEST]  Some issues detected. Check the recommendations above.")

if __name__ == "__main__":
    main()