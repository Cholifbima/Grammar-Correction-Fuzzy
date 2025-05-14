from flask import Flask, render_template, request, jsonify
import os
import json
from fuzzy_grammar.fuzzy_system import FuzzyGrammarSystem
from fuzzy_grammar.grammar_analyzer import GrammarAnalyzer
from fuzzy_grammar.feedback_generator import FeedbackGenerator

app = Flask(__name__)

# Initialize our components
fuzzy_system = FuzzyGrammarSystem()
grammar_analyzer = GrammarAnalyzer()
feedback_generator = FeedbackGenerator()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the provided text and return feedback"""
    data = request.get_json()
    text = data.get('text', '')
    tense = data.get('tense', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Step 1: Analyze grammar
    analysis_result = grammar_analyzer.analyze(text, tense)
    
    # If the text is not valid English, return early with error
    if not analysis_result.get('is_valid_english', True):
        return jsonify({
            'analysis': analysis_result,
            'feedback': {
                'severity_level': 'High',
                'overall_feedback': 'The input does not appear to be valid English.',
                'specific_feedback': [],
                'suggestions': ['Please enter valid English text.'],
                'resources': []
            }
        })
    
    # Step 2: Feed the analysis results to the fuzzy system
    fuzzy_result = fuzzy_system.evaluate(
        analysis_result['grammar_match'], 
        analysis_result['error_frequency'], 
        analysis_result['complexity']
    )
    
    # Step 3: Generate feedback based on analysis and fuzzy results
    feedback = feedback_generator.generate_feedback(analysis_result, fuzzy_result, tense)
    
    return jsonify({
        'analysis': analysis_result,
        'fuzzy_result': {
            'severity_score': fuzzy_result['severity_score'],
            'severity_level': fuzzy_result['severity_level']
        },
        'feedback': feedback
    })

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True) 