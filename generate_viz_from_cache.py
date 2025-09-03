#!/usr/bin/env python3

import json
import os
import numpy as np

# Import only the functions we need, not the entire app which has cleanup handlers
import sys
sys.path.append('.')
from app import load_analysis_cache, VIS_HTML_PATH

def create_mock_lda_model_from_cache():
    """Create a mock LDA model from cached analysis data."""
    cache = load_analysis_cache()
    if not cache or not cache.get('success') or not cache.get('topics_data'):
        print("No valid analysis cache found.")
        return None, None, 0
    
    topics_data = cache['topics_data']
    num_topics = len(topics_data)
    
    # Create mock LDA model components from cached data
    class MockLDAModel:
        def __init__(self, topics_data):
            self.components_ = []
            self.feature_names = set()
            
            # Extract all unique words to create feature names
            for topic in topics_data:
                self.feature_names.update(topic['top_words'])
            
            self.feature_names = sorted(list(self.feature_names))
            
            # Create components matrix
            for topic in topics_data:
                component = np.zeros(len(self.feature_names))
                for i, word in enumerate(topic['top_words']):
                    if word in self.feature_names:
                        feature_idx = self.feature_names.index(word)
                        if i < len(topic['weights']):
                            component[feature_idx] = topic['weights'][i]
                self.components_.append(component)
    
    model = MockLDAModel(topics_data)
    analysis_stats = {
        'total_documents': cache.get('files_collected', 500),
        'processed_documents': cache.get('files_collected', 500),
        'vocabulary_size': len(model.feature_names),
        'topics_discovered': num_topics
    }
    
    return model, model.feature_names, analysis_stats

def main():
    print("Generating visualization from cached analysis data...")
    
    model, feature_names, analysis_stats = create_mock_lda_model_from_cache()
    if model is None:
        print("Failed to create model from cache data.")
        return
    
    print(f"Found {len(feature_names)} unique words in {analysis_stats['topics_discovered']} topics")
    print(f"Generating visualization for {analysis_stats['total_documents']} documents")
    
    # Generate the visualization
    create_enhanced_visualization(model, feature_names, analysis_stats['topics_discovered'], analysis_stats)
    
    if os.path.exists(VIS_HTML_PATH):
        print(f"✅ Visualization successfully generated: {VIS_HTML_PATH}")
    else:
        print("❌ Failed to generate visualization file")

if __name__ == "__main__":
    main()