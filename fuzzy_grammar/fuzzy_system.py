import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyGrammarSystem:
    """
    Fuzzy Inference System for Grammar Analysis
    
    This class implements a fuzzy inference system that evaluates the 
    quality of English sentences based on:
    - Grammar match score
    - Error frequency
    - Sentence complexity
    """
    
    def __init__(self):
        """Initialize the fuzzy inference system with variables and rules"""
        # Define fuzzy variables
        self.grammar_score = ctrl.Antecedent(np.arange(0, 101, 1), 'grammar_match')
        self.error_frequency = ctrl.Antecedent(np.arange(0, 101, 1), 'error_frequency')
        self.complexity = ctrl.Antecedent(np.arange(0, 101, 1), 'complexity')
        self.severity = ctrl.Consequent(np.arange(0, 101, 1), 'severity')
        
        # Define membership functions for inputs
        self.grammar_score['low'] = fuzz.trapmf(self.grammar_score.universe, [0, 0, 40, 60])
        self.grammar_score['medium'] = fuzz.trimf(self.grammar_score.universe, [40, 60, 80])
        self.grammar_score['high'] = fuzz.trapmf(self.grammar_score.universe, [60, 80, 100, 100])
        
        self.error_frequency['low'] = fuzz.trapmf(self.error_frequency.universe, [0, 0, 20, 40])
        self.error_frequency['medium'] = fuzz.trimf(self.error_frequency.universe, [20, 40, 60])
        self.error_frequency['high'] = fuzz.trapmf(self.error_frequency.universe, [40, 60, 100, 100])
        
        self.complexity['low'] = fuzz.trapmf(self.complexity.universe, [0, 0, 30, 50])
        self.complexity['medium'] = fuzz.trimf(self.complexity.universe, [30, 50, 70])
        self.complexity['high'] = fuzz.trapmf(self.complexity.universe, [50, 70, 100, 100])
        
        # Define membership functions for output
        self.severity['low'] = fuzz.trapmf(self.severity.universe, [0, 0, 30, 50])
        self.severity['medium'] = fuzz.trimf(self.severity.universe, [30, 50, 70])
        self.severity['high'] = fuzz.trapmf(self.severity.universe, [50, 70, 100, 100])
        
        # Define fuzzy rules
        rules = [
            # Rules related to grammar match score
            ctrl.Rule(self.grammar_score['high'] & self.error_frequency['low'], self.severity['low']),
            ctrl.Rule(self.grammar_score['high'] & self.error_frequency['medium'], self.severity['medium']),
            ctrl.Rule(self.grammar_score['high'] & self.error_frequency['high'], self.severity['medium']),
            
            ctrl.Rule(self.grammar_score['medium'] & self.error_frequency['low'], self.severity['low']),
            ctrl.Rule(self.grammar_score['medium'] & self.error_frequency['medium'], self.severity['medium']),
            ctrl.Rule(self.grammar_score['medium'] & self.error_frequency['high'], self.severity['high']),
            
            ctrl.Rule(self.grammar_score['low'] & self.error_frequency['low'], self.severity['medium']),
            ctrl.Rule(self.grammar_score['low'] & self.error_frequency['medium'], self.severity['high']),
            ctrl.Rule(self.grammar_score['low'] & self.error_frequency['high'], self.severity['high']),
            
            # Rules with complexity
            ctrl.Rule(self.grammar_score['low'] & self.complexity['high'], self.severity['high']),
            ctrl.Rule(self.grammar_score['medium'] & self.complexity['high'], self.severity['medium'])
        ]
        
        # Create control system
        self.grammar_ctrl = ctrl.ControlSystem(rules)
        self.grammar_simulator = ctrl.ControlSystemSimulation(self.grammar_ctrl)
        
    def evaluate(self, grammar_match, error_frequency, complexity):
        """
        Evaluate the grammar quality using fuzzy inference
        
        Args:
            grammar_match (float): Score from 0-100 indicating grammar correctness
            error_frequency (float): Score from 0-100 indicating frequency of errors
            complexity (float): Score from 0-100 indicating sentence complexity
        
        Returns:
            dict: Results of the fuzzy inference, including the severity score and level
        """
        # Input values to the fuzzy system
        self.grammar_simulator.input['grammar_match'] = grammar_match
        self.grammar_simulator.input['error_frequency'] = error_frequency
        self.grammar_simulator.input['complexity'] = complexity
        
        # Compute result
        self.grammar_simulator.compute()
        
        # Get the defuzzified output
        severity_score = self.grammar_simulator.output['severity']
        
        # Determine severity level
        severity_level = self._get_severity_level(severity_score)
        
        # Return the results
        return {
            'severity_score': severity_score,
            'severity_level': severity_level
        }
    
    def _get_severity_level(self, severity_score):
        """Convert numerical severity score to linguistic level"""
        if severity_score < 40:
            return 'Low'
        elif severity_score < 60:
            return 'Medium'
        else:
            return 'High'