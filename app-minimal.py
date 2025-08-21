"""Minimal test version of the app without ML dependencies"""
from flask import Flask, render_template, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Mock analysis for testing
    return jsonify({
        'status': 'success',
        'message': 'Mock analysis completed (ML dependencies not available)',
        'topics_found': 3,
        'files_processed': 50
    })

@app.route('/visualization')
def visualization():
    # Mock visualization
    mock_html = """
    <html>
    <head><title>Mock Visualization</title></head>
    <body>
        <h1>Mock Topic Visualization</h1>
        <p>This is a test visualization. The real version requires scikit-learn.</p>
        <div>
            <h3>Topic 1: Development Patterns</h3>
            <p>Words: development, patterns, code, implementation</p>
        </div>
        <div>
            <h3>Topic 2: Documentation</h3>
            <p>Words: documentation, guide, instructions, setup</p>
        </div>
        <div>
            <h3>Topic 3: Integration</h3>
            <p>Words: integration, api, workflow, automation</p>
        </div>
    </body>
    </html>
    """
    return mock_html

if __name__ == '__main__':
    app.run(debug=True, port=5000)