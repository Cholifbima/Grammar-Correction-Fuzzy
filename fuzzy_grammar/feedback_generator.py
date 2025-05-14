class FeedbackGenerator:
    """
    Generates personalized feedback based on grammar analysis results
    
    This class takes the output from the GrammarAnalyzer and FuzzyGrammarSystem
    to generate appropriate feedback for the user.
    """
    
    def __init__(self):
        """Initialize the feedback generator"""
        # Define feedback templates based on severity level
        self.severity_templates = {
            'Low': [
                "Great job! Your grammar is good with only minor issues.",
                "Your sentence is well-structured with minimal errors.",
                "Your grammar is strong. Just a few small improvements to consider."
            ],
            'Medium': [
                "Your sentence has a few grammar issues that could be improved.",
                "There are some grammar mistakes in your text.",
                "This text contains several grammar issues to address."
            ],
            'High': [
                "There are significant grammar issues that need attention.",
                "Your text has several serious grammar mistakes to correct.",
                "This sentence needs considerable revision to improve its grammar."
            ]
        }
        
        # Resources for different grammar topics
        self.resources = {
            'Subject-verb agreement': {
                'url': 'https://owl.purdue.edu/owl/general_writing/grammar/subject_verb_agreement.html',
                'type': 'Subject-Verb Agreement'
            },
            'Verb form error': {
                'url': 'https://dictionary.cambridge.org/grammar/british-grammar/verbs-basic-forms',
                'type': 'Verb Forms'
            },
            'Article error': {
                'url': 'https://www.grammarly.com/blog/articles/',
                'type': 'Articles'
            },
            'Preposition error': {
                'url': 'https://www.grammarly.com/blog/prepositions/',
                'type': 'Prepositions'
            },
            'Sentence fragment': {
                'url': 'https://www.grammarly.com/blog/sentence-fragments/',
                'type': 'Sentence Fragments'
            },
            'Double negation': {
                'url': 'https://www.grammarly.com/blog/negatives/',
                'type': 'Double Negatives'
            },
            'Conditional error': {
                'url': 'https://www.englishpage.com/conditional/conditionalintro.html',
                'type': 'Conditional Sentences'
            },
            'Phrasal verb error': {
                'url': 'https://www.englishclub.com/vocabulary/phrasal-verbs-list.htm',
                'type': 'Phrasal Verbs'
            },
            'Modal verb error': {
                'url': 'https://www.englishpage.com/modals/modalintro.html',
                'type': 'Modal Verbs'
            },
            'Word usage error': {
                'url': 'https://www.grammarly.com/blog/common-word-errors/',
                'type': 'Word Usage'
            },
            'Irregular verb error': {
                'url': 'https://www.englishpage.com/irregularverbs/irregularverbs.html',
                'type': 'Irregular Verbs'
            },
            'Article with noun error': {
                'url': 'https://www.englishclub.com/grammar/articles-with-nouns.htm',
                'type': 'Articles with Nouns'
            },
            'Missing article/determiner': {
                'url': 'https://www.englishpage.com/articles/articles-omit.htm',
                'type': 'Articles and Determiners'
            },
            'Word repetition': {
                'url': 'https://www.grammarly.com/blog/concise-writing/',
                'type': 'Concise Writing'
            }
        }
        
        # Tense-specific resources
        self.tense_resources = {
            'Simple Present': {
                'url': 'https://www.englishpage.com/verbpage/simplepresent.html',
                'type': 'Simple Present Tense'
            },
            'Present Continuous': {
                'url': 'https://www.englishpage.com/verbpage/presentcontinuous.html',
                'type': 'Present Continuous Tense'
            },
            'Present Perfect': {
                'url': 'https://www.englishpage.com/verbpage/presentperfect.html',
                'type': 'Present Perfect Tense'
            },
            'Simple Past': {
                'url': 'https://www.englishpage.com/verbpage/simplepast.html',
                'type': 'Simple Past Tense'
            },
            'Past Continuous': {
                'url': 'https://www.englishpage.com/verbpage/pastcontinuous.html',
                'type': 'Past Continuous Tense'
            },
            'Past Perfect': {
                'url': 'https://www.englishpage.com/verbpage/pastperfect.html',
                'type': 'Past Perfect Tense'
            },
            'Future Simple': {
                'url': 'https://www.englishpage.com/verbpage/simplefuture.html',
                'type': 'Future Simple Tense'
            },
            'Future Continuous': {
                'url': 'https://www.englishpage.com/verbpage/futurecontinuous.html',
                'type': 'Future Continuous Tense'
            },
            'Future Perfect': {
                'url': 'https://www.englishpage.com/verbpage/futureperfect.html',
                'type': 'Future Perfect Tense'
            }
        }
        
        # General suggestions for each error type
        self.error_suggestions = {
            'Subject-verb agreement': [
                "Make sure your subject and verb agree in number.",
                "Singular subjects need singular verbs, plural subjects need plural verbs.",
                "Check that you're using the correct verb form with each subject."
            ],
            'Article error': [
                "Use 'a' before words that begin with a consonant sound.",
                "Use 'an' before words that begin with a vowel sound.",
                "Remember when to use 'the' versus 'a/an'."
            ],
            'Preposition error': [
                "Some verbs require specific prepositions.",
                "Check if you're using the correct preposition for the context.",
                "Many preposition errors happen with idiomatic expressions."
            ],
            'Sentence fragment': [
                "Make sure each sentence has a subject and a verb.",
                "Avoid incomplete thoughts or dependent clauses as complete sentences.",
                "Join fragments to complete sentences with appropriate punctuation."
            ],
            'Verb form error': [
                "Make sure you're using the correct form of the verb for your subject.",
                "Pay attention to singular vs. plural subjects when choosing verb forms.",
                "Be consistent with your verb tenses throughout your writing."
            ],
            'Double negation': [
                "Use only one negative word in a sentence.",
                "Double negatives can cause confusion in English.",
                "Choose the most appropriate single negative word for your meaning."
            ],
            'Conditional error': [
                "In first conditionals (if + present, will + base form), don't use 'will' in the if-clause.",
                "In second conditionals (if + past, would + base form), use past simple in the if-clause.",
                "In third conditionals (if + past perfect, would have + past participle), be consistent with the structure."
            ],
            'Phrasal verb error': [
                "Many phrasal verbs require specific prepositions or particles.",
                "Be careful with the word order in phrasal verbs.",
                "Some phrasal verbs have idiomatic meanings that must be memorized."
            ],
            'Modal verb error': [
                "Modal verbs (can, could, may, etc.) are followed by the base form of the verb without 'to'.",
                "Don't use 'to' after modal verbs like 'must', 'can', 'should', etc.",
                "Modal verbs don't change form with third-person singular subjects."
            ],
            'Word usage error': [
                "Some verbs have specific collocations with certain nouns.",
                "Pay attention to fixed expressions and common combinations of words.",
                "Use a collocation dictionary to check common word pairs."
            ],
            'Irregular verb error': [
                "Memorize the past tense and past participle forms of irregular verbs.",
                "Don't add -ed to irregular verbs to form the past tense.",
                "Review the most common irregular verbs and practice them regularly."
            ],
            'Article with noun error': [
                "Some nouns do not take articles (especially uncountable nouns and proper nouns).",
                "With some general concepts, use no article (e.g., 'life is short').",
                "Country names usually don't need articles, except for plural names or regions."
            ],
            'Missing article/determiner': [
                "Singular countable nouns usually need an article or determiner.",
                "Consider whether 'a', 'an', 'the', or another determiner is needed.",
                "Only plural or uncountable nouns can appear without articles in some contexts."
            ],
            'Word repetition': [
                "Avoid unnecessarily repeating the same word.",
                "Use pronouns or synonyms to avoid repetition.",
                "Check for accidental duplications of words."
            ]
        }
    
    def generate_feedback(self, analysis_result, fuzzy_result, tense=None):
        """
        Generate personalized feedback based on analysis results
        
        Args:
            analysis_result (dict): Result from the grammar analyzer
            fuzzy_result (dict): Result from the fuzzy inference system
            tense (str, optional): The specific tense being analyzed
            
        Returns:
            dict: Personalized feedback for the user
        """
        # Get severity level
        severity_level = fuzzy_result['severity_level']
        severity_score = fuzzy_result['severity_score']
        
        # Get errors
        errors = analysis_result.get('errors', [])
        
        # Generate overall feedback based on severity
        overall_feedback = self._generate_overall_feedback(severity_level)
        
        # Generate specific feedback for each error type
        specific_feedback = self._generate_specific_feedback(errors)
        
        # Generate suggestions based on errors
        suggestions = self._generate_suggestions(errors, tense)
        
        # Generate resources
        resources = self._generate_resources(errors, tense)
        
        return {
            'severity_level': severity_level,
            'severity_score': severity_score,
            'overall_feedback': overall_feedback,
            'specific_feedback': specific_feedback,
            'suggestions': suggestions,
            'resources': resources
        }
    
    def _generate_overall_feedback(self, severity_level):
        """Generate overall feedback based on severity level"""
        import random
        templates = self.severity_templates.get(severity_level, self.severity_templates['Medium'])
        return random.choice(templates)
    
    def _generate_specific_feedback(self, errors):
        """Generate specific feedback for each error type"""
        from collections import defaultdict
        
        # Group errors by type
        error_types = defaultdict(list)
        for error in errors:
            error_type = error['type']
            error_types[error_type].append(error)
        
        feedback = []
        
        # Generate feedback for each error type
        for error_type, type_errors in error_types.items():
            # Skip if too many of the same error type (just use the first few)
            example_errors = type_errors[:3]
            examples = [error['text'] for error in example_errors]
            
            # Get suggestions for this error type
            suggestions = [error['suggestion'] for error in example_errors]
            
            # Create feedback item
            feedback.append({
                'type': error_type,
                'feedback': f"Found {len(type_errors)} {error_type.lower()} issue{'s' if len(type_errors) > 1 else ''}.",
                'example': ", ".join(examples),
                'suggestion': " ".join(suggestions[:2])  # Limit to 2 suggestions
            })
        
        return feedback
    
    def _generate_suggestions(self, errors, tense=None):
        """Generate suggestions based on errors and tense"""
        import random
        
        suggestions = set()
        
        # Get error types
        error_types = {error['type'] for error in errors}
        
        # Add suggestions for each error type
        for error_type in error_types:
            if error_type in self.error_suggestions:
                # Add a random suggestion for this error type
                error_suggestion = random.choice(self.error_suggestions[error_type])
                suggestions.add(error_suggestion)
        
        # Add tense-specific suggestions if a tense was selected
        if tense:
            if "Subject-verb agreement" in error_types or "Verb form error" in error_types:
                suggestions.add(f"Review the correct verb forms for {tense} tense.")
            
            if tense == "Simple Present":
                suggestions.add("Remember: Use the base form for I/you/we/they and add -s/-es for he/she/it in simple present.")
            elif tense == "Simple Past":
                suggestions.add("For past tense, use the past form of the verb or 'did not' + base form (not past form) for negatives.")
            elif tense == "Present Continuous":
                suggestions.add("Present continuous should use am/is/are + verb-ing.")
            elif tense == "Present Perfect":
                suggestions.add("Present perfect uses have/has + past participle form of the verb.")
            elif tense == "Past Continuous":
                suggestions.add("Past continuous uses was/were + verb-ing.")
            elif tense == "Past Perfect":
                suggestions.add("Past perfect uses had + past participle form of the verb.")
            elif tense == "Future Simple":
                suggestions.add("Future simple uses will + base form of the verb (not -ing form).")
            elif tense == "Future Continuous":
                suggestions.add("Future continuous uses will be + verb-ing.")
            elif tense == "Future Perfect":
                suggestions.add("Future perfect uses will have + past participle form of the verb.")
                
        # Special handling for common error combinations
        if "Modal verb error" in error_types and "Verb form error" in error_types:
            suggestions.add("Remember that modal verbs (can, must, should) are followed directly by the base verb without 'to'.")
            
        if "Irregular verb error" in error_types and "Simple Past" in tense:
            suggestions.add("Pay special attention to irregular past tense forms - they don't follow the -ed pattern.")
            
        if "Article with noun error" in error_types and "Missing article/determiner" in error_types:
            suggestions.add("Review when to use articles (a, an, the) and when to omit them with different types of nouns.")
        
        return list(suggestions)
    
    def _generate_resources(self, errors, tense=None):
        """Generate resources based on errors and tense"""
        resources = []
        
        # Get unique error types
        error_types = {error['type'] for error in errors}
        
        # Add resources for each error type
        for error_type in error_types:
            resource = self.resources.get(error_type)
            if resource:
                resources.append(resource)
        
        # Add tense-specific resource if provided
        if tense and tense in self.tense_resources:
            resources.append(self.tense_resources[tense])
        
        # Always add a general grammar resource if there are errors
        if errors and len(resources) < 3:
            resources.append({
                'url': 'https://www.grammarly.com/grammar-check',
                'type': 'General Grammar'
            })
        
        # Limit to 3 most relevant resources
        return resources[:3] 