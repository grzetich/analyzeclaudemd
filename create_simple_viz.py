#!/usr/bin/env python3

import json
import os

# Read the cached analysis data
with open('cache/last_analysis.json', 'r') as f:
    cache = json.load(f)

if not cache.get('success') or not cache.get('topics_data'):
    print("No valid analysis cache found.")
    exit(1)

topics_data = cache['topics_data']
files_count = cache.get('files_collected', 500)
vocab_size = len(set(word for topic in topics_data for word in topic['top_words']))

# Apply improved topic labeling
def generate_topic_label(top_words):
    top_5_words = top_words[:5]
    top_10_words = top_words[:10]
    
    if any(word in top_5_words for word in ['assistant', 'claude', 'ai']):
        return "AI Assistant Configuration"
    elif any(word in top_5_words for word in ['project', 'repo', 'repository']):
        return "Project Structure"
    elif any(word in top_5_words for word in ['npm', 'typescript', 'react', 'pnpm']):
        return "Frontend Development"
    elif any(word in top_5_words for word in ['server', 'database', 'service']) and any(word in top_10_words for word in ['api', 'integration', 'management']):
        return "Backend Services"
    elif any(word in top_5_words for word in ['code', 'function', 'class']):
        return "Code Guidelines"
    elif any(word in top_5_words for word in ['test', 'testing', 'spec']):
        return "Testing & Quality"
    elif any(word in top_5_words for word in ['data', 'database', 'api']):
        return "Data & APIs"
    elif any(word in top_5_words for word in ['user', 'interface', 'ui']):
        return "User Interface"
    elif any(word in top_5_words for word in ['deploy', 'build', 'production']):
        return "Deployment & Build"
    elif any(word in top_5_words for word in ['doc', 'documentation', 'readme']):
        return "Documentation"
    elif any(word in top_5_words for word in ['task', 'implementation', 'step']):
        return "Task & Implementation"
    else:
        return "General"

# Update topic labels
for i, topic in enumerate(topics_data):
    topic['label'] = generate_topic_label(topic['top_words'])

# Generate colors for topics
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Claude.md Topic Analysis - Simple Visualization</title>
    <meta charset="UTF-8">
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 12px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 2.2em; font-weight: 300; }}
        .header .subtitle {{ opacity: 0.9; font-size: 1.1em; }}
        .stats {{
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: #ecf0f1;
            border-bottom: 1px solid #ddd;
        }}
        .stat {{ text-align: center; }}
        .stat-number {{ 
            display: block; 
            font-size: 2em; 
            font-weight: bold; 
            color: #2c3e50; 
            margin-bottom: 5px; 
        }}
        .stat-label {{ color: #7f8c8d; font-size: 0.9em; }}
        .topics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
            padding: 30px; 
        }}
        .topic-card {{
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.2s ease;
        }}
        .topic-card:hover {{ transform: translateY(-5px); }}
        .topic-header {{
            padding: 20px;
            color: white;
            font-weight: bold;
            font-size: 1.1em;
        }}
        .topic-body {{ padding: 20px; background: white; }}
        .topic-words {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 15px;
        }}
        .word-tag {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.85em;
            color: #495057;
        }}
        .strength-bar {{
            height: 8px;
            background: rgba(255,255,255,0.3);
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }}
        .strength-fill {{
            height: 100%;
            background: rgba(255,255,255,0.8);
            border-radius: 4px;
            transition: width 0.8s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Claude.md Topic Analysis</h1>
            <div class="subtitle">Discovering patterns in Claude documentation across GitHub repositories</div>
        </div>
        
        <div class="stats">
            <div class="stat">
                <span class="stat-number">{files_count:,}</span>
                <div class="stat-label">Files Analyzed</div>
            </div>
            <div class="stat">
                <span class="stat-number">{len(topics_data)}</span>
                <div class="stat-label">Topics Discovered</div>
            </div>
            <div class="stat">
                <span class="stat-number">{vocab_size:,}</span>
                <div class="stat-label">Unique Words</div>
            </div>
            <div class="stat">
                <span class="stat-number">500+</span>
                <div class="stat-label">GitHub Repositories</div>
            </div>
        </div>
        
        <div class="topics-grid">
"""

# Add each topic
for i, topic in enumerate(topics_data):
    color = colors[i % len(colors)]
    strength_percent = (topic['topic_strength'] / max(t['topic_strength'] for t in topics_data)) * 100
    
    html_content += f"""
            <div class="topic-card">
                <div class="topic-header" style="background: linear-gradient(135deg, {color}, {color}88);">
                    {topic['label']}
                    <div class="strength-bar">
                        <div class="strength-fill" style="width: {strength_percent}%;"></div>
                    </div>
                </div>
                <div class="topic-body">
                    <strong>Top Words:</strong>
                    <div class="topic-words">
    """
    
    for word in topic['top_words'][:10]:  # Show top 10 words
        html_content += f'<span class="word-tag">{word}</span>'
    
    html_content += """
                    </div>
                </div>
            </div>
    """

html_content += """
        </div>
    </div>
</body>
</html>
"""

# Write the HTML file
output_path = "templates/lda_visualization.html"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Simple visualization generated: {output_path}")