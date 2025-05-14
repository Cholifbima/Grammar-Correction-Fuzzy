# Fuzzy Grammar Tutor

A web application that analyzes English grammar using fuzzy inference system to provide nuanced feedback on grammar errors.

## Features

- Grammar error detection with severity level assessment
- Fuzzy logic-based evaluation of sentence quality
- Personalized feedback based on error types and severity
- Visual representation of fuzzy membership functions

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. Run the application:
   ```
   flask run
   ```
   
## Implementation Details

This application uses:
- Flask for the web application framework
- scikit-fuzzy for implementing the fuzzy inference system
- spaCy and NLTK for natural language processing
- TextBlob for additional grammar analysis

## Project Structure

- `app.py`: Main Flask application
- `fuzzy_grammar/`: Package containing the fuzzy logic implementation
  - `fuzzy_system.py`: Fuzzy inference system implementation
  - `grammar_analyzer.py`: Grammar analysis using NLP tools
  - `feedback_generator.py`: Generate feedback based on errors
- `templates/`: HTML templates
- `static/`: Static files (CSS, JS, images) 