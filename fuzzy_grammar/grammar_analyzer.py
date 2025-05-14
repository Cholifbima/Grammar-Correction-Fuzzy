import spacy
import nltk
from textblob import TextBlob
import re
from spacy.matcher import Matcher, PhraseMatcher
from collections import Counter
import string
import enchant  # Library untuk memeriksa ejaan bahasa Inggris

# Ensure nltk data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

class GrammarAnalyzer:
    """
    Analyzes English sentences for grammatical correctness
    
    This class uses NLP tools like spaCy, NLTK and TextBlob to analyze
    English sentences for various grammar errors and provides metrics
    for the fuzzy inference system.
    """
    
    def __init__(self):
        """Initialize the grammar analyzer with necessary NLP components"""
        try:
            # Load spaCy model with exception handling
            self.nlp = spacy.load('en_core_web_sm', disable=['ner'])
        except:
            # Fallback if model loading fails
            import en_core_web_sm
            self.nlp = en_core_web_sm.load(disable=['ner'])
        
        # Initialize English dictionary for checking
        self.english_dict = enchant.Dict("en_US")
        
        # Initialize matcher for common error patterns
        self.matcher = Matcher(self.nlp.vocab)
        
        # Initialize phrase matcher for multi-token patterns
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
        
        # Add patterns for common grammar errors
        self._add_error_patterns()
        
        # Common article errors (a/an misuse)
        self.a_an_regex = re.compile(r'\b(a)\s+([aeiou])', re.IGNORECASE)
        self.an_a_regex = re.compile(r'\b(an)\s+([bcdfghjklmnpqrstvwxyz])', re.IGNORECASE)
        
        # Common preposition errors - expanded
        self.common_prep_errors = [
            (re.compile(r'\b(arrive) (to)\b', re.IGNORECASE), 'arrive at/in'),
            (re.compile(r'\b(different) (than)\b', re.IGNORECASE), 'different from'),
            (re.compile(r'\b(in) (the weekend)\b', re.IGNORECASE), 'on the weekend'),
            (re.compile(r'\b(depend) (of)\b', re.IGNORECASE), 'depend on'),
            (re.compile(r'\b(married) (with)\b', re.IGNORECASE), 'married to'),
            (re.compile(r'\b(in) (night)\b', re.IGNORECASE), 'at night'),
            (re.compile(r'\b(in) (morning)\b', re.IGNORECASE), 'in the morning'),
            (re.compile(r'\b(in) (evening)\b', re.IGNORECASE), 'in the evening'),
            (re.compile(r'\b(listen) (the)\b', re.IGNORECASE), 'listen to the'),
            (re.compile(r'\b(according) (with)\b', re.IGNORECASE), 'according to'),
            (re.compile(r'\b(agree) (to) (the opinion)\b', re.IGNORECASE), 'agree with the opinion'),
            (re.compile(r'\b(capable) (to)\b', re.IGNORECASE), 'capable of'),
        ]
        
        # Initialize subject-verb agreement patterns
        self.sv_agreement_patterns = []
        
        # Add basic patterns first to avoid overloading the matcher
        self._add_basic_sv_patterns()
        
        # Common tense errors and corrections in a more optimized format
        self.tense_corrections = self._initialize_tense_corrections()
        
        # Common word usage errors (expanded)
        self.word_usage_errors = {
            r'\b(make|doing) (a|an|the) (decision)\b': 'make a decision',
            r'\b(make|doing) (a|an|the) (mistake)\b': 'make a mistake',
            r'\b(take|taking) (a|an|the) (decision)\b': 'make a decision',
            r'\b(do|doing) (a|an|the) (mistake)\b': 'make a mistake',
            r'\b(make|making) (homework|research)\b': 'do homework/research',
            r'\b(very|much) (tall|short|big|small)\b': 'very tall/short/big/small',
            r'\b(much|many) (happy|sad|angry)\b': 'very happy/sad/angry',
            r'\b(little|few) (water|milk|sugar)\b': 'little water/milk/sugar',
            r'\b(little|few) (books|pens|students)\b': 'few books/pens/students',
        }
        
        # Modal verb errors with simplified regex
        self.modal_verb_errors = {
            r'\b(must) (to) \b': 'must (without "to")',
            r'\b(can) (to) \b': 'can (without "to")',
            r'\b(could) (to) \b': 'could (without "to")',
            r'\b(may) (to) \b': 'may (without "to")',
            r'\b(might) (to) \b': 'might (without "to")',
            r'\b(should) (to) \b': 'should (without "to")',
            r'\b(would) (to) \b': 'would (without "to")',
            r'\b(shall) (to) \b': 'shall (without "to")',
            r'\b(will) (to) \b': 'will (without "to")',
        }
        
        # Multi-word phrasal verb errors
        self.phrasal_verb_patterns = [
            # Incorrect: look at to, correct: look at
            (["look", "at", "to"], "look at"),
            # Incorrect: think about of, correct: think about
            (["think", "about", "of"], "think about"),
            # Incorrect: listen to for, correct: listen to
            (["listen", "to", "for"], "listen to"),
            # Incorrect: give up to, correct: give up
            (["give", "up", "to"], "give up"),
            # Incorrect: put off on, correct: put off
            (["put", "off", "on"], "put off"),
        ]
        
        # Add phrasal verb patterns to phrase matcher - using a more robust approach
        self._add_phrasal_verb_patterns()
            
        # Common irregular verb misuse patterns with simplified regex
        self.irregular_verb_errors = {
            r'\b(teached)\b': 'taught',
            r'\b(goed)\b': 'went',
            r'\b(thinked)\b': 'thought',
            r'\b(buyed)\b': 'bought',
            r'\b(selled)\b': 'sold',
            r'\b(catched)\b': 'caught',
            r'\b(fighted)\b': 'fought',
            r'\b(bringed)\b': 'brought',
            r'\b(readed)\b': 'read',
            r'\b(writed)\b': 'wrote',
            r'\b(sayed)\b': 'said',
            r'\b(maked)\b': 'made',
            r'\b(getted)\b': 'got',
            r'\b(putted)\b': 'put',
            r'\b(leaved)\b': 'left',
            r'\b(taked)\b': 'took',
            r'\b(finded)\b': 'found',
            r'\b(eated)\b': 'ate',
            r'\b(sleeped)\b': 'slept',
            r'\b(speaked)\b': 'spoke',
            r'\b(breaked)\b': 'broke',
            r'\b(feeled)\b': 'felt',
            r'\b(builded)\b': 'built',
        }
        
        # Article usage with specific nouns - simplified
        self.article_with_noun_errors = {
            r'\bthe (people|information|advice|furniture|homework)\b': 'people/information/advice/furniture/homework (no article needed)',
            r'\ba (people|furniture)\b': 'people/furniture (no article needed)',
            r'\b(the|a|an) (China|India|Japan|Brazil|Australia)\b': 'China/India/Japan/Brazil/Australia (no article needed)',
            r'\b(go to the|went to the) (home|school|church|bed|work)\b': 'go to/went to home/school/church/bed/work (no article)',
        }
        
        # Added for irregular plurals
        self.irregular_plurals = {
            "men": True, "women": True, "children": True, "people": True, 
            "feet": True, "teeth": True, "mice": True, "geese": True,
            "deer": True, "fish": True, "sheep": True, "species": True
        }
    
    def _add_basic_sv_patterns(self):
        """Add basic subject-verb agreement patterns to the matcher"""
        # Add most common subject-verb errors first
        basic_sv_patterns = [
            # Subject-verb agreement errors - first person
            [{'LOWER': 'i'}, {'LOWER': 'has'}],
            [{'LOWER': 'i'}, {'LOWER': 'is'}],
            [{'LOWER': 'i'}, {'LOWER': 'are'}],
            [{'LOWER': 'i'}, {'LOWER': "doesn't"}],
            # Third person singular errors
            [{'LOWER': 'he'}, {'LOWER': 'have'}],
            [{'LOWER': 'she'}, {'LOWER': 'have'}],
            [{'LOWER': 'it'}, {'LOWER': 'have'}],
            [{'LOWER': 'he'}, {'LOWER': 'are'}],
            [{'LOWER': 'she'}, {'LOWER': 'are'}],
            [{'LOWER': 'it'}, {'LOWER': 'are'}],
            [{'LOWER': 'he'}, {'LOWER': "don't"}],
            [{'LOWER': 'she'}, {'LOWER': "don't"}], 
            [{'LOWER': 'it'}, {'LOWER': "don't"}],
            # Third person singular without -s
            [{'LOWER': 'he'}, {'TAG': 'VB', 'OP': '+', 'IS_DIGIT': False}],
            [{'LOWER': 'she'}, {'TAG': 'VB', 'OP': '+', 'IS_DIGIT': False}],
            [{'LOWER': 'it'}, {'TAG': 'VB', 'OP': '+', 'IS_DIGIT': False}],
            # Plural subjects with singular verbs
            [{'LOWER': 'they'}, {'LOWER': 'has'}],
            [{'LOWER': 'they'}, {'LOWER': 'is'}],
            [{'LOWER': 'we'}, {'LOWER': 'has'}],
            [{'LOWER': 'we'}, {'LOWER': 'is'}],
            [{'LOWER': 'you'}, {'LOWER': 'has'}],
            [{'LOWER': 'you'}, {'LOWER': 'is'}],
            [{'LOWER': 'they'}, {'LOWER': "doesn't"}],
            [{'LOWER': 'we'}, {'LOWER': "doesn't"}],
            [{'LOWER': 'you'}, {'LOWER': "doesn't"}],
        ]
        
        # Add these patterns to the matcher
        for pattern in basic_sv_patterns:
            try:
                self.matcher.add('SV_AGREEMENT', [pattern])
            except Exception as e:
                print(f"Error adding pattern {pattern}: {e}")
    
    def _add_phrasal_verb_patterns(self):
        """Add phrasal verb patterns to the phrase matcher in batches"""
        try:
            for phrases, _ in self.phrasal_verb_patterns:
                pattern = [self.nlp(word.lower()) for word in phrases]
                self.phrase_matcher.add("PHRASAL_VERB", None, *pattern)
        except Exception as e:
            print(f"Warning: Error adding phrasal verb patterns: {e}")
    
    def _initialize_tense_corrections(self):
        """Initialize tense corrections in a more memory-efficient way"""
        # Create a structured dictionary for common tense errors
        return {
            'Simple Present': {
                # Subject-verb agreement errors
                'I has': 'I have',
                'You has': 'You have',
                'We has': 'We have',
                'They has': 'They have',
                'He have': 'He has',
                'She have': 'She has',
                'It have': 'It has',
                # To be errors
                'I is': 'I am',
                'You is': 'You are',
                'We is': 'We are',
                'They is': 'They are',
                'He are': 'He is',
                'She are': 'She is',
                'It are': 'It is',
                'I are': 'I am',
                # Contraction errors
                'I doesn\'t': 'I don\'t',
                'You doesn\'t': 'You don\'t',
                'We doesn\'t': 'We don\'t', 
                'They doesn\'t': 'They don\'t',
                'He don\'t': 'He doesn\'t',
                'She don\'t': 'She doesn\'t',
                'It don\'t': 'It doesn\'t',
                # First person with third-person verb
                'I plays': 'I play',
                'I writes': 'I write',
                'I reads': 'I read',
                'I watches': 'I watch',
                'I goes': 'I go',
                'I does': 'I do',
                'I makes': 'I make',
                'I says': 'I say',
                'I takes': 'I take',
                'I comes': 'I come',
                'I sees': 'I see',
                'I thinks': 'I think',
                'I tries': 'I try',
                'I studies': 'I study',
                # Third person singular
                'he go': 'he goes',
                'she go': 'she goes',
                'it go': 'it goes',
                'he play': 'he plays',
                'she play': 'she plays',
                'it play': 'it plays',
                'he write': 'he writes',
                'she write': 'she writes',
                'it write': 'it writes',
                'he do': 'he does',
                'she do': 'she does',
                'it do': 'it does',
                # Negation errors
                'I not have': 'I do not have',
                'he not have': 'he does not have',
                'she not have': 'she does not have',
                'it not have': 'it does not have',
                'we not have': 'we do not have',
                'they not have': 'they do not have',
                'you not have': 'you do not have',
                # Question form errors
                'have I': 'do I have',
                'have you': 'do you have',
                'have they': 'do they have',
                'have we': 'do we have',
                'has he': 'does he have',
                'has she': 'does she have',
                'has it': 'does it have',
            },
            'Simple Past': {
                # Regular past tense errors
                'I play yesterday': 'I played yesterday',
                'you play yesterday': 'you played yesterday',
                'he play yesterday': 'he played yesterday',
                'she play yesterday': 'she played yesterday',
                'it play yesterday': 'it played yesterday',
                'we play yesterday': 'we played yesterday',
                'they play yesterday': 'they played yesterday',
                # Irregular past tense errors
                'I goed': 'I went',
                'you goed': 'you went',
                'he goed': 'he went',
                'she goed': 'she went',
                'it goed': 'it went',
                'we goed': 'we went',
                'they goed': 'they went',
                # Negation errors
                'I did not went': 'I did not go',
                'you did not went': 'you did not go',
                'he did not went': 'he did not go',
                'she did not went': 'she did not go',
                'it did not went': 'it did not go',
                'we did not went': 'we did not go',
                'they did not went': 'they did not go',
                # To be errors in past
                'I were': 'I was',
                'he were': 'he was',
                'she were': 'she was',
                'it were': 'it was',
                'you was': 'you were',
                'we was': 'we were',
                'they was': 'they were',
            },
            'Present Continuous': {
                # Base form instead of -ing form
                'I am go': 'I am going',
                'you are go': 'you are going',
                'he is go': 'he is going',
                'she is go': 'she is going',
                'it is go': 'it is going',
                'we are go': 'we are going',
                'they are go': 'they are going',
                # Missing auxiliary
                'I going': 'I am going',
                'you going': 'you are going',
                'he going': 'he is going',
                'she going': 'she is going',
                'it going': 'it is going',
                'we going': 'we are going',
                'they going': 'they are going',
                # Wrong auxiliary
                'I is going': 'I am going',
                'you is going': 'you are going',
                'we is going': 'we are going',
                'they is going': 'they are going',
                'he are going': 'he is going',
                'she are going': 'she is going',
                'it are going': 'it is going',
            },
            'Present Perfect': {
                # Wrong participle form
                'I have went': 'I have gone',
                'you have went': 'you have gone',
                'he has went': 'he has gone',
                'she has went': 'she has gone',
                'it has went': 'it has gone',
                'we have went': 'we have gone',
                'they have went': 'they have gone',
                'I have ate': 'I have eaten',
                'you have ate': 'you have eaten',
                'he has ate': 'he has eaten',
                'she has ate': 'she has eaten',
                'it has ate': 'it has eaten',
                'we have ate': 'we have eaten',
                'they have ate': 'they have eaten',
                # Wrong auxiliary
                'I has gone': 'I have gone',
                'you has gone': 'you have gone',
                'we has gone': 'we have gone',
                'they has gone': 'they have gone',
                'he have gone': 'he has gone',
                'she have gone': 'she has gone',
                'it have gone': 'it has gone',
                # Using past simple instead of present perfect
                'I went already': 'I have gone already',
                'you went already': 'you have gone already',
                'he went already': 'he has gone already',
                'she went already': 'she has gone already',
                'it went already': 'it has gone already',
                'we went already': 'we have gone already',
                'they went already': 'they have gone already',
            },
            'Past Continuous': {
                # Base form instead of -ing form
                'I was go': 'I was going',
                'you were go': 'you were going',
                'he was go': 'he was going',
                'she was go': 'she was going',
                'it was go': 'it was going',
                'we were go': 'we were going',
                'they were go': 'they were going',
                'I was run': 'I was running',
                'you were run': 'you were running',
                'he was run': 'he was running',
                'she was run': 'she was running',
                'it was run': 'it was running',
                'we were run': 'we were running',
                'they were run': 'they were running',
                # Wrong to be form
                'I were going': 'I was going',
                'he were going': 'he was going',
                'she were going': 'she was going',
                'it were going': 'it was going',
                'you was going': 'you were going',
                'we was going': 'we were going',
                'they was going': 'they were going',
                # Missing auxiliary
                'I going yesterday': 'I was going yesterday',
                'you going yesterday': 'you were going yesterday',
                'he going yesterday': 'he was going yesterday',
                'she going yesterday': 'she was going yesterday',
                'it going yesterday': 'it was going yesterday',
                'we going yesterday': 'we were going yesterday',
                'they going yesterday': 'they were going yesterday',
            },
            'Past Perfect': {
                # Wrong participle form
                'I had went': 'I had gone',
                'you had went': 'you had gone',
                'he had went': 'he had gone',
                'she had went': 'she had gone',
                'it had went': 'it had gone',
                'we had went': 'we had gone',
                'they had went': 'they had gone',
                # Wrong auxiliary
                'I have had gone': 'I had gone',
                'you have had gone': 'you had gone',
                'he have had gone': 'he had gone',
                'she have had gone': 'she had gone',
                'I has had gone': 'I had gone',
                'he has had gone': 'he had gone',
                'she has had gone': 'she had gone',
                # Using past simple or present perfect instead of past perfect
                'I went before': 'I had gone before',
                'he went before': 'he had gone before',
                'she went before': 'she had gone before',
                'I have gone before': 'I had gone before',
                'he has gone before': 'he had gone before',
                'she has gone before': 'she had gone before',
            },
            'Future Simple': {
                # Using present continuous for future
                'I am going to go tomorrow': 'I will go tomorrow',
                'I will going': 'I will go',
                'you will going': 'you will go',
                'he will going': 'he will go',
                'she will going': 'she will go',
                'it will going': 'it will go',
                'we will going': 'we will go',
                'they will going': 'they will go',
                # Wrong form after will
                'I will goes': 'I will go',
                'you will goes': 'you will go',
                'he will goes': 'he will go',
                'she will goes': 'she will go',
                'it will goes': 'it will go',
                'we will goes': 'we will go',
                'they will goes': 'they will go',
                # Missing will
                'I go tomorrow': 'I will go tomorrow',
                'you go tomorrow': 'you will go tomorrow',
                'he go tomorrow': 'he will go tomorrow',
                'she go tomorrow': 'she will go tomorrow',
                'it go tomorrow': 'it will go tomorrow',
                'we go tomorrow': 'we will go tomorrow',
                'they go tomorrow': 'they will go tomorrow',
            },
            'Future Continuous': {
                # Base form instead of -ing form
                'I will be go': 'I will be going',
                'you will be go': 'you will be going',
                'he will be go': 'he will be going',
                'she will be go': 'she will be going',
                'it will be go': 'it will be going',
                'we will be go': 'we will be going',
                'they will be go': 'they will be going',
                # Wrong auxiliary structure
                'I will going': 'I will be going',
                'you will going': 'you will be going',
                'he will going': 'he will be going',
                'she will going': 'she will be going',
                'it will going': 'it will be going',
                'we will going': 'we will be going',
                'they will going': 'they will be going',
            },
            'Future Perfect': {
                # Wrong participle form
                'I will have went': 'I will have gone',
                'you will have went': 'you will have gone',
                'he will have went': 'he will have gone',
                'she will have went': 'she will have gone',
                'it will have went': 'it will have gone',
                'we will have went': 'we will have gone',
                'they will have went': 'they will have gone',
                # Wrong auxiliary structure
                'I will had gone': 'I will have gone',
                'you will had gone': 'you will have gone',
                'he will had gone': 'he will have gone',
                'she will had gone': 'she will have gone',
                'it will had gone': 'it will have gone',
                'we will had gone': 'we will have gone',
                'they will had gone': 'they will have gone',
                # Base form instead of participle
                'I will have go': 'I will have gone',
                'you will have go': 'you will have gone',
                'he will have go': 'he will have gone',
                'she will have go': 'she will have gone',
                'it will have go': 'it will have gone',
                'we will have go': 'we will have gone',
                'they will have go': 'they will have gone',
            },
        }
    
    def _add_error_patterns(self):
        """Add patterns for common grammar errors to the matcher in batches to prevent overloading"""
        # Double negation patterns
        double_negation = [
            [{'LOWER': 'not'}, {'IS_ALPHA': True, 'OP': '*'}, {'LOWER': 'no'}],
            [{'LOWER': 'not'}, {'IS_ALPHA': True, 'OP': '*'}, {'LOWER': 'nobody'}],
            [{'LOWER': 'not'}, {'IS_ALPHA': True, 'OP': '*'}, {'LOWER': 'nothing'}],
            [{'LOWER': 'never'}, {'IS_ALPHA': True, 'OP': '*'}, {'LOWER': 'not'}],
        ]
        
        # Incorrect gerund/infinitive usage - simplified list
        gerund_infinitive_errors = [
            # enjoy + infinitive (incorrect) - should be gerund
            [{'LOWER': 'enjoy'}, {'TAG': 'TO'}, {'TAG': 'VB'}],
            # finish + infinitive (incorrect) - should be gerund
            [{'LOWER': 'finish'}, {'TAG': 'TO'}, {'TAG': 'VB'}],
            # want + gerund (incorrect) - should be infinitive
            [{'LOWER': 'want'}, {'TAG': 'VBG'}],
            # hope + gerund (incorrect) - should be infinitive
            [{'LOWER': 'hope'}, {'TAG': 'VBG'}],
        ]
        
        # Conditional errors - simplified
        conditional_errors = [
            # Type 1 errors - If + will (incorrect)
            [{'LOWER': 'if'}, {'IS_ALPHA': True, 'OP': '+'}, {'LEMMA': 'will'}],
        ]
        
        # Add patterns to matcher in batches to avoid overloading
        try:
            # Add double negation patterns
            for pattern in double_negation:
                self.matcher.add('DOUBLE_NEGATION', [pattern])
                
            # Add gerund infinitive error patterns
            for pattern in gerund_infinitive_errors:
                self.matcher.add('GERUND_INFINITIVE_ERROR', [pattern])
                
            # Add conditional error patterns
            for pattern in conditional_errors:
                self.matcher.add('CONDITIONAL_ERROR', [pattern])
        except Exception as e:
            print(f"Warning: Error adding patterns to matcher: {e}")
    
    def analyze(self, text, tense=None):
        """
        Analyze the text for grammatical correctness
        
        Args:
            text (str): The English text to analyze
            tense (str, optional): The specific tense to check against
        
        Returns:
            dict: Analysis results including various metrics and detected errors
        """
        # Check if text is mostly English or nonsense
        is_valid_english, non_english_reason = self._is_valid_english(text)
        
        if not is_valid_english:
            return {
                'is_valid_english': False,
                'reason': non_english_reason,
                'grammar_match': 0,
                'error_frequency': 100,
                'complexity': 0,
                'errors': [{
                    'type': 'Invalid input',
                    'text': text,
                    'suggestion': 'Please enter valid English text.'
                }]
            }
        
        try:
            # Process text with spaCy with timeout protection
            doc = self.nlp(text)
            
            # Find the subject and determine if it's plural or singular
            subjects = []
            try:
                subjects = self._extract_subjects(doc)
            except Exception as e:
                print(f"Error extracting subjects: {e}")
                # Continue with empty subjects list rather than failing
            
            # Get TextBlob object for additional analysis
            blob = TextBlob(text)
            
            # Find grammar errors
            errors = []
            try:
                errors = self._detect_errors(doc, text, tense, subjects)
            except Exception as e:
                print(f"Error detecting errors: {e}")
                # Return a basic error if detection fails completely
                errors = [{
                    'type': 'Analysis error',
                    'text': text,
                    'suggestion': f'Error analyzing grammar: {str(e)}'
                }]
            
            # Calculate metrics
            try:
                grammar_match = self._calculate_grammar_match(doc, errors)
            except Exception as e:
                print(f"Error calculating grammar match: {e}")
                grammar_match = 50  # Default to medium score on error
                
            try:
                error_frequency = self._calculate_error_frequency(errors, len(text.split()))
            except Exception as e:
                print(f"Error calculating error frequency: {e}")
                error_frequency = 50  # Default to medium score on error
                
            try:
                complexity = self._calculate_complexity(doc)
            except Exception as e:
                print(f"Error calculating complexity: {e}")
                complexity = 50  # Default to medium score on error
            
            corrections = ""
            try:
                corrections = self._generate_corrections(text, errors)
            except Exception as e:
                print(f"Error generating corrections: {e}")
                corrections = text  # Return original text if corrections fail
            
            result = {
                'is_valid_english': True,
                'grammar_match': grammar_match,
                'error_frequency': error_frequency,
                'complexity': complexity,
                'errors': errors,
                'corrections': corrections
            }
            
            # Add subject information if available
            if subjects:
                result['subjects'] = subjects
            
            return result
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return {
                'is_valid_english': False,
                'reason': f"Error analyzing text: {str(e)}",
                'grammar_match': 0,
                'error_frequency': 100,
                'complexity': 0,
                'errors': [{
                    'type': 'Analysis error',
                    'text': text,
                    'suggestion': 'An error occurred while analyzing this text.'
                }]
            }
    
    def _is_valid_english(self, text):
        """Check if the text is likely to be valid English and not gibberish"""
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return False, "No valid words found."
        
        # Create a mapping of lowercase words to their original capitalization
        original_words = text.split()
        word_mapping = {}
        for orig_word in original_words:
            # Strip punctuation for comparison
            clean_word = re.sub(r'[^\w]', '', orig_word.lower())
            if clean_word:
                word_mapping[clean_word] = orig_word
        
        # Common English contractions that should be considered valid
        common_contractions = {
            "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", 
            "cannot", "couldn't", "shouldn't", "isn't", "aren't", "wasn't", 
            "weren't", "haven't", "hasn't", "hadn't", "i'm", "you're", 
            "he's", "she's", "it's", "we're", "they're", "i've", "you've", 
            "we've", "they've", "i'd", "you'd", "he'd", "she'd", "we'd", 
            "they'd", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll"
        }
        
        # Count words that are in the English dictionary or are valid contractions
        valid_words = 0
        for word in words:
            # Consider capitalized words as potentially valid proper names
            is_capitalized = False
            
            # Safely check if the word exists in our original text with capitalization
            if word in word_mapping and word_mapping[word]:
                is_capitalized = word_mapping[word][0].isupper() if word_mapping[word] else False
                
            # Check if word is a common contraction
            is_contraction = word in common_contractions
            
            if self.english_dict.check(word) or is_capitalized or is_contraction:
                valid_words += 1
        
        # Calculate the percentage of valid English words
        valid_percentage = valid_words / len(words) if words else 0
        
        # More lenient threshold for sentences with proper names
        if valid_percentage < 0.4:
            return False, f"Only {int(valid_percentage*100)}% of the words appear to be valid English."
            
        # Check for repeating characters (likely keyboard mashing)
        for word in words:
            if len(word) >= 4:
                for i in range(len(word) - 3):
                    if word[i] == word[i+1] == word[i+2] == word[i+3]:
                        return False, "The text contains keyboard mashing patterns."
        
        return True, ""
    
    def _extract_subjects(self, doc):
        """
        Extract subjects from the document and determine if they're singular or plural
        
        This performs a more advanced analysis to identify subjects including non-English names,
        compound subjects, and proper nouns.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            list: List of subject dictionaries with keys: text, is_plural, position
        """
        subjects = []
        
        try:
            # Step 1: Find all verbs in the sentence
            verbs = [token for token in doc if token.pos_ == "VERB" or token.pos_ == "AUX"]
            
            for verb in verbs:
                # Step 2: Find the subject for each verb
                subject = None
                compound_subject = []
                compound_start_idx = None
                compound_end_idx = None
                
                # Look for direct subject through syntactic dependencies
                for token in doc:
                    # Check for subject dependency or nominal subject
                    if token.dep_ in ["nsubj", "nsubjpass"] and token.head == verb:
                        subject = token
                        compound_start_idx = token.i
                        compound_end_idx = token.i + 1
                        
                        # Check for compound subjects - handle both "and" conjunctions and comma-separated lists
                        # First look for direct conjunctions
                        for child in token.children:
                            if child.dep_ == "conj":
                                compound_subject.append(child)
                                compound_end_idx = max(compound_end_idx, child.i + 1)
                        
                        # Look for comma and "and" patterns in the sentence before the verb
                        comma_and_pattern = False
                        for i in range(max(0, token.i-10), min(verb.i, len(doc))):
                            if doc[i].text == ',' or doc[i].text.lower() == 'and':
                                # Check if there's a noun after the comma/and
                                for j in range(i+1, min(verb.i, len(doc))):
                                    if doc[j].pos_ in ["NOUN", "PROPN", "PRON"]:
                                        compound_subject.append(doc[j])
                                        compound_start_idx = min(compound_start_idx, i)
                                        compound_end_idx = max(compound_end_idx, j + 1)
                                        comma_and_pattern = True
                                        break
                                
                        break
                
                # If no subject found through dependencies, use positional heuristic
                if not subject:
                    # Find the closest noun/pronoun before the verb
                    for token in doc[:verb.i]:
                        if token.pos_ in ["NOUN", "PROPN", "PRON"] and token.dep_ not in ["dobj", "pobj"]:
                            subject = token
                            compound_start_idx = token.i
                            compound_end_idx = token.i + 1
                            
                            # Check nearby tokens for potential compound subjects
                            for i in range(max(0, token.i-5), token.i):
                                if doc[i].text == ',' or doc[i].text.lower() == 'and':
                                    for j in range(i+1, token.i):
                                        if doc[j].pos_ in ["NOUN", "PROPN", "PRON"]:
                                            compound_subject.append(doc[j])
                                            compound_start_idx = min(compound_start_idx, j)
                    
                # Skip if no subject found
                if not subject:
                    continue
                
                # Step 3: Determine if the subject is plural
                is_plural = self._is_subject_plural(subject, doc, compound_subject)
                
                # Get the full subject text including modifiers and conjunctions
                if compound_subject and compound_start_idx is not None and compound_end_idx is not None:
                    # Get the full span including all parts of compound subject
                    subject_span = doc[compound_start_idx:compound_end_idx]
                else:
                    subject_span = self._get_subject_span(subject, doc, compound_subject)
                
                subjects.append({
                    'text': subject_span.text,
                    'is_plural': is_plural,
                    'position': subject.i,
                    'token': subject.text,
                    'pos': subject.pos_,
                    'compound': [cs.text for cs in compound_subject] if compound_subject else None
                })
        
        except Exception as e:
            print(f"Error extracting subjects: {e}")
        
        return subjects
    
    def _is_subject_plural(self, subject, doc, compound_subjects=None):
        """
        Determine if a subject is plural using multiple heuristics
        
        Args:
            subject: The subject token
            doc: The full spaCy Doc
            compound_subjects: List of additional subject tokens that are conjoined
            
        Returns:
            bool: True if the subject is plural, False otherwise
        """
        # Case 1: Compound subjects with "and" are plural
        if compound_subjects:
            return True
        
        # Case 2: Check for coordinating conjunction with commas
        # Look for patterns like "X, Y, and Z", "X, Y, Z, ..., and N"
        comma_and_pattern = re.compile(r'\w+\s*,\s*(\w+\s*,\s*)+(\w+\s+and\s+\w+)')
        text_around_subject = ' '.join([t.text for t in doc[max(0, subject.i-10):min(len(doc), subject.i+10)]])
        if comma_and_pattern.search(text_around_subject):
            return True
        
        # Case 3: Check for coordinating conjunction
        # Look for patterns like "X and Y", "X, Y, and Z"
        for token in doc[max(0, subject.i-5):min(len(doc), subject.i+5)]:
            if token.text.lower() == "and" and token.head == subject:
                return True
        
        # Case 4: Personal pronouns
        # Note: "I" follows plural verb conjugation rules though it's singular in meaning
        if subject.text.lower() == "i":
            # Special case: "I" is grammatically singular but follows plural verb conjugation rules
            # For subject-verb agreement purposes, we return True so it gets the correct verb form
            return True
        elif subject.text.lower() in ["he", "she", "it", "this", "that"]:
            return False
        elif subject.text.lower() in ["we", "they", "you", "these", "those"]:
            return True
        
        # Case 5: Check for explicit plural noun tag
        if subject.tag_ == "NNS":
            return True
        
        # Case 6: Check for irregular plurals
        if subject.text.lower() in self.irregular_plurals:
            return True
        
        # Case 7: Check for plural determiners
        for child in subject.children:
            if child.pos_ == "DET" and child.text.lower() in ["these", "those", "many", "several", "few"]:
                return True
            if child.text.lower() in ["a", "an", "this", "that", "each", "every"]:
                return False
        
        # Case 8: Check for numbers greater than 1
        for child in subject.children:
            if child.pos_ == "NUM" and child.text.isdigit() and int(child.text) > 1:
                return True
        
        # Case 9: Check for quantity words
        for child in subject.children:
            if child.text.lower() in ["many", "several", "few", "multiple", "various"]:
                return True
        
        # Default to singular for proper nouns and other cases
        if subject.pos_ == "PROPN":
            return False
        
        # Additional check for plurals ending in 's' not captured by NNS tag
        if subject.text.lower().endswith('s') and not subject.text.lower().endswith('ss'):
            # Common singular words ending in 's'
            singular_s_endings = ["news", "series", "species", "means", "physics", "politics", "mathematics"]
            if not any(subject.text.lower() == word for word in singular_s_endings):
                return True
        
        # Default to singular
        return False
    
    def _get_subject_span(self, subject, doc, compound_subjects=None):
        """
        Get the full text span of a subject including all of its modifiers
        
        Args:
            subject: The subject token
            doc: The full spaCy Doc
            compound_subjects: List of additional subject tokens that are conjoined
            
        Returns:
            spaCy Span: The full subject phrase
        """
        # Start with just the subject token
        start_idx = subject.i
        end_idx = subject.i + 1
        
        # Include compound subjects
        if compound_subjects:
            min_idx = start_idx
            max_idx = end_idx
            
            for cs in compound_subjects:
                min_idx = min(min_idx, cs.i)
                max_idx = max(max_idx, cs.i + 1)
            
            # Include all tokens between the first and last subject
            start_idx = min_idx
            end_idx = max_idx
            
            # Look for connecting "and", "or" between subjects
            for i in range(start_idx, end_idx):
                if doc[i].text.lower() in ["and", "or", ","] and i not in [cs.i for cs in compound_subjects]:
                    # Include this token in our span
                    pass
        
        # Include modifiers (adjectives, determiners, etc.)
        for token in doc:
            # Include descendant modifiers of the subject
            if token.head == subject and token.i < subject.i:
                start_idx = min(start_idx, token.i)
        
        return doc[start_idx:end_idx]
    
    def _detect_errors(self, doc, text, target_tense=None, subjects=None):
        """Detect various types of grammar errors"""
        errors = []
        
        # Direct contraction check - do this first as it's more reliable than subject-based checks
        try:
            # Simple pattern matching for common contraction errors
            for i in range(len(doc) - 1):
                try:
                    # Check for contraction errors first (simpler and more reliable)
                    if i < len(doc)-1 and doc[i].text and doc[i+1].text:
                        # Singular subjects with don't
                        if doc[i].text.lower() in ["he", "she", "it"] and doc[i+1].text.lower() == "don't":
                            errors.append({
                                'type': 'Contraction error',
                                'text': f"{doc[i].text} don't",
                                'suggestion': f"Use 'doesn't' with singular subjects: '{doc[i].text} doesn't'"
                            })
                        # Plural subjects with doesn't
                        elif doc[i].text.lower() in ["i", "we", "they", "you"] and doc[i+1].text.lower() == "doesn't":
                            errors.append({
                                'type': 'Contraction error',
                                'text': f"{doc[i].text} doesn't",
                                'suggestion': f"Use 'don't' with '{doc[i].text}'"
                            })
                    
                    # Check for incorrect verb forms after auxiliaries
                    if i < len(doc)-1 and doc[i].text and doc[i+1].text:
                        if doc[i].text.lower() in ["do", "does", "did", "don't", "doesn't", "didn't"] and i < len(doc)-1:
                            next_token = doc[i+1]
                            # If the next token is a verb but not in base form
                            if next_token.pos_ == "VERB" and next_token.tag_ != "VB":
                                # Get the base form - usually the lemma works for this
                                base_form = next_token.lemma_
                                
                                # Special handling for "to be" and other irregular verbs
                                if next_token.lemma_ == "be" and next_token.text.lower() in ["am", "is", "are", "was", "were"]:
                                    base_form = "be"
                                elif next_token.text.lower() == "has":
                                    base_form = "have"
                                
                                errors.append({
                                    'type': 'Auxiliary verb error',
                                    'text': f"{doc[i].text} {next_token.text}",
                                    'suggestion': f"Use base form of verb after '{doc[i].text}': '{doc[i].text} {base_form}'"
                                })
                except Exception as inner_e:
                    print(f"Error processing token at index {i}: {inner_e}")
                    continue  # Skip this token pair but continue with others
        except Exception as e:
            print(f"Error in direct pattern check: {e}")
        
        # Only do subject-verb agreement checks in simple present
        if target_tense == "Simple Present" and subjects:
            try:
                # Check subject-verb agreement for each subject
                for subject_info in subjects:
                    subject_token = doc[subject_info['position']] if subject_info['position'] < len(doc) else None
                    if not subject_token:
                        continue
                        
                    is_plural = subject_info['is_plural']
                    
                    # Find the associated verb
                    verb = None
                    # First look for direct dependency
                    if subject_token:
                        for token in doc:
                            if token.pos_ == "VERB" and token.head == subject_token:
                                verb = token
                                break
                    
                    # If no direct dependency, look for a verb after the subject
                    if not verb:
                        for token in doc[subject_info['position']+1:]:
                            if token.pos_ == "VERB":
                                verb = token
                                break
                    
                    if verb:
                        # Check for subject-verb agreement errors
                        has_error, correct_form = self._check_sv_agreement_simple_present(
                            subject_info, verb, doc)
                        
                        if has_error and correct_form:
                            errors.append({
                                'type': 'Subject-verb agreement',
                                'text': f"{subject_info['text']} {verb.text}",
                                'suggestion': f"Use '{correct_form}' instead of '{verb.text}' with {subject_info['text']}"
                            })
                    
                    try:
                        # Check for contraction errors (don't/doesn't)
                        contraction_errors = self._check_contraction_errors(subject_info, doc)
                        if contraction_errors:
                            errors.extend(contraction_errors)
                    except Exception as e:
                        print(f"Error checking contractions: {e}")
                    
            except Exception as e:
                print(f"Error in subject-verb agreement check: {e}")
        
        # Check for subject-verb agreement errors using the matcher
        try:
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                error_span = doc[start:end].text
                rule_id = self.nlp.vocab.strings[match_id]
                
                if rule_id == 'SV_AGREEMENT':
                    # Determine the correction based on the error
                    correction = self._get_sv_agreement_correction(error_span)
                    
                    errors.append({
                        'type': 'Subject-verb agreement',
                        'text': error_span,
                        'suggestion': f"Use '{correction}' instead" if correction else "Check subject-verb agreement"
                    })
                elif rule_id == 'DOUBLE_NEGATION':
                    errors.append({
                        'type': 'Double negation',
                        'text': error_span,
                        'suggestion': "Avoid using double negatives; use only one negative word"
                    })
                elif rule_id == 'GERUND_INFINITIVE_ERROR':
                    if 'enjoy' in error_span.lower() or 'finish' in error_span.lower():
                        errors.append({
                            'type': 'Verb form error',
                            'text': error_span,
                            'suggestion': f"Use gerund (-ing form) after {error_span.split()[0]}, not infinitive"
                        })
                    else:
                        errors.append({
                            'type': 'Verb form error',
                            'text': error_span,
                            'suggestion': f"Use infinitive (to + verb) after {error_span.split()[0]}, not gerund"
                        })
                elif rule_id == 'CONDITIONAL_ERROR':
                    errors.append({
                        'type': 'Conditional error',
                        'text': error_span,
                        'suggestion': "Check conditional clause construction"
                    })
        except Exception as e:
            print(f"Error in matcher: {e}")
        
        # Check for phrasal verb errors
        try:
            phrase_matches = self.phrase_matcher(doc)
            for match_id, start, end in phrase_matches:
                phrase_span = doc[start:end].text
                for phrases, correction in self.phrasal_verb_patterns:
                    if all(word.lower() in phrase_span.lower() for word in phrases):
                        errors.append({
                            'type': 'Phrasal verb error',
                            'text': phrase_span,
                            'suggestion': f"Use '{correction}' instead"
                        })
        except Exception as e:
            print(f"Error in phrase matcher: {e}")
        
        # Check for article errors (a/an)
        try:
            a_an_errors = self.a_an_regex.finditer(text)
            for match in a_an_errors:
                errors.append({
                    'type': 'Article error',
                    'text': match.group(0),
                    'suggestion': f'Use "an" before vowel sounds: "an {match.group(2)}"'
                })
            
            an_a_errors = self.an_a_regex.finditer(text)
            for match in an_a_errors:
                errors.append({
                    'type': 'Article error',
                    'text': match.group(0),
                    'suggestion': f'Use "a" before consonant sounds: "a {match.group(2)}"'
                })
        except Exception as e:
            print(f"Error checking article errors: {e}")
        
        # Check for common preposition errors
        try:
            for regex_pattern, correct_form in self.common_prep_errors:
                for match in regex_pattern.finditer(text):
                    errors.append({
                        'type': 'Preposition error',
                        'text': match.group(0),
                        'suggestion': f'Use "{correct_form}" instead'
                    })
        except Exception as e:
            print(f"Error checking preposition errors: {e}")
        
        # Check for word usage errors
        try:
            for pattern, suggestion in self.word_usage_errors.items():
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    errors.append({
                        'type': 'Word usage error',
                        'text': match.group(0),
                        'suggestion': f'Use "{suggestion}" instead'
                    })
        except Exception as e:
            print(f"Error checking word usage errors: {e}")
        
        # Check for modal verb errors
        try:
            for pattern, suggestion in self.modal_verb_errors.items():
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    errors.append({
                        'type': 'Modal verb error',
                        'text': match.group(0),
                        'suggestion': f'Use {suggestion}'
                    })
        except Exception as e:
            print(f"Error checking modal verb errors: {e}")
        
        # Check for irregular verb errors
        try:
            for pattern, correction in self.irregular_verb_errors.items():
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    errors.append({
                        'type': 'Irregular verb error',
                        'text': match.group(0),
                        'suggestion': f'Use "{correction}" instead'
                    })
        except Exception as e:
            print(f"Error checking irregular verb errors: {e}")
        
        # Check for article with noun errors
        try:
            for pattern, suggestion in self.article_with_noun_errors.items():
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    errors.append({
                        'type': 'Article with noun error',
                        'text': match.group(0),
                        'suggestion': f'Use {suggestion}'
                    })
        except Exception as e:
            print(f"Error checking article with noun errors: {e}")
        
        # Check for sentence fragments (simplified)
        try:
            for sent in doc.sents:
                has_verb = any(token.pos_ == "VERB" for token in sent)
                if not has_verb and len(sent) > 3:  # Only flag longer fragments
                    errors.append({
                        'type': 'Sentence fragment',
                        'text': sent.text,
                        'suggestion': 'This may be a sentence fragment. Consider adding a verb.'
                    })
        except Exception as e:
            print(f"Error checking sentence fragments: {e}")
        
        # Check for specific tense errors if a target tense is provided
        try:
            if target_tense and target_tense in self.tense_corrections:
                for error_pattern, correction in self.tense_corrections[target_tense].items():
                    # Use a more flexible matching approach
                    pattern_words = error_pattern.lower().split()
                    text_words = text.lower().split()
                    
                    # Check for consecutive matches
                    for i in range(len(text_words) - len(pattern_words) + 1):
                        if all(text_words[i+j] == pattern_words[j] for j in range(len(pattern_words))):
                            # Get the actual text from the original case
                            actual_text = ' '.join(text.split()[i:i+len(pattern_words)])
                            errors.append({
                                'type': f'{target_tense} tense error',
                                'text': actual_text,
                                'suggestion': f'Use "{correction}" for correct {target_tense} tense'
                            })
        except Exception as e:
            print(f"Error checking tense errors: {e}")
        
        # Detect additional errors based on context (simplified to reduce processing)
        try:
            for i, token in enumerate(doc):
                if i > 0 and token.text.lower() == doc[i-1].text.lower() and token.is_alpha:
                    errors.append({
                        'type': 'Word repetition',
                        'text': f"{doc[i-1].text} {token.text}",
                        'suggestion': f'Remove the repeated word "{token.text}"'
                    })
        except Exception as e:
            print(f"Error checking token-based errors: {e}")
        
        # Add article error check for "a" before vowel sounds and "an" before consonant sounds
        try:
            words = text.split()
            for i in range(len(words) - 1):
                if words[i].lower() == 'a' and words[i+1] and words[i+1][0].lower() in 'aeiou':
                    errors.append({
                        'type': 'Article error',
                        'text': f"{words[i]} {words[i+1]}",
                        'suggestion': f'Use "an" before words starting with vowel sounds: "an {words[i+1]}"'
                    })
                elif words[i].lower() == 'an' and words[i+1] and words[i+1][0].lower() not in 'aeiou':
                    errors.append({
                        'type': 'Article error',
                        'text': f"{words[i]} {words[i+1]}",
                        'suggestion': f'Use "a" before words starting with consonant sounds: "a {words[i+1]}"'
                    })
        except Exception as e:
            print(f"Error checking a/an usage: {e}")
            
        return errors
    
    def _generate_corrections(self, text, errors):
        """Generate corrected version of the text based on detected errors"""
        corrected_text = text
        
        # Sort errors by their position in text (if available), otherwise just use as is
        try:
            sorted_errors = sorted(errors, key=lambda e: text.find(e['text']) if text.find(e['text']) != -1 else float('inf'))
            
            # Apply corrections in reverse order to avoid changing positions
            for error in reversed(sorted_errors):
                if 'text' in error and 'suggestion' in error:
                    error_text = error['text']
                    suggestion = error['suggestion']
                    corrected_part = None
                    
                    # Handle auxiliary verb errors
                    if error['type'] == 'Auxiliary verb error':
                        match = re.search(r"'([^']+)'$", suggestion)
                        if match:
                            corrected_part = match.group(1)
                    
                    # Handle contraction errors specifically
                    if not corrected_part and error['type'] == 'Contraction error':
                        # Extract the correction directly from the suggestion
                        if "Use 'don't'" in suggestion:
                            # Replace doesn't with don't
                            corrected_part = error_text.replace("doesn't", "don't")
                        elif "Use 'doesn't'" in suggestion:
                            # Replace don't with doesn't
                            corrected_part = error_text.replace("don't", "doesn't")
                        elif "Use 'haven't'" in suggestion:
                            # Replace hasn't with haven't
                            corrected_part = error_text.replace("hasn't", "haven't")
                        elif "Use 'hasn't'" in suggestion:
                            # Replace haven't with hasn't
                            corrected_part = error_text.replace("haven't", "hasn't")
                        elif "Use 'aren't'" in suggestion:
                            # Replace isn't with aren't
                            corrected_part = error_text.replace("isn't", "aren't")
                        elif "Use 'isn't'" in suggestion:
                            # Replace aren't with isn't
                            corrected_part = error_text.replace("aren't", "isn't")
                        elif "I amn't" in error_text:
                            # Replace amn't with 'm not
                            corrected_part = "I'm not"
                    
                    # [Rest of the method remains unchanged]
                    
                    # Try to extract the correction from the suggestion with proper format
                    if not corrected_part:
                        match = re.search(r'"([^"]+)"', suggestion)
                        if match:
                            corrected_part = match.group(1)
                    
                    # For subject-verb agreement errors without quoted suggestions
                    if not corrected_part and error['type'] == 'Subject-verb agreement':
                        # Extract the recommended form directly from the suggestion
                        match = re.search(r"Use '([^']+)' instead", suggestion)
                        if match:
                            # Get the subject from the original text
                            words = error_text.split()
                            if len(words) >= 2:
                                # For compound subjects like "Fathoni and Bima plays"
                                # We need to preserve the full subject
                                if " and " in error_text or "," in error_text:
                                    # Find where the verb starts (last word)
                                    verb = words[-1]
                                    # Get subject part (everything before the verb)
                                    subject_part = " ".join(words[:-1])
                                    corrected_part = subject_part + " " + match.group(1)
                                else:
                                    # Single subject case
                                    subject = words[0]
                                    corrected_part = subject + " " + match.group(1)
                        
                        if not corrected_part:
                            # Handle special case for "I plays" type errors
                            words = error_text.split()
                            if len(words) >= 2:
                                # Get the verb (last word)
                                verb = words[-1].lower()
                                
                                # Get the full subject (everything except the last word)
                                subject = " ".join(words[:-1])
                                
                                # First person + -s verb
                                if subject.lower() == "i" and verb.endswith('s') and verb != "is":
                                    if verb.endswith('ies') and len(verb) > 3:
                                        corrected_part = f"I {verb[:-3]}y"
                                    elif verb.endswith('es') and any(verb.endswith(x+'es') for x in ['sh', 'ch', 'x', 'ss', 'zz', 'o']):
                                        corrected_part = f"I {verb[:-2]}"
                                    else:
                                        corrected_part = f"I {verb[:-1]}"
                                
                                # Plural subject + -s verb
                                elif (" and " in subject or "," in subject or self.is_plural_subject(subject)) and verb.endswith('s') and verb != "is":
                                    if verb.endswith('ies') and len(verb) > 3:
                                        corrected_part = f"{subject} {verb[:-3]}y"
                                    elif verb.endswith('es') and any(verb.endswith(x+'es') for x in ['sh', 'ch', 'x', 'ss', 'zz', 'o']):
                                        corrected_part = f"{subject} {verb[:-2]}"
                                    else:
                                        corrected_part = f"{subject} {verb[:-1]}"
                    
                    # Handle tense errors directly
                    if not corrected_part and error['type'].endswith('tense error'):
                        for tense_name, corrections in self.tense_corrections.items():
                            if error['type'].startswith(tense_name):
                                for error_pattern, correction in corrections.items():
                                    if error_text.lower() == error_pattern.lower():
                                        corrected_part = correction
                                        break
                    
                    # Handle irregular verb errors
                    if not corrected_part and error['type'] == 'Irregular verb error':
                        for pattern, correction in self.irregular_verb_errors.items():
                            pattern = pattern.replace(r'\b', '').replace(r'\b', '')
                            pattern = pattern.replace('(', '').replace(')', '')
                            if pattern in error_text.lower():
                                # Replace only the irregular verb part
                                corrected_part = error_text.lower().replace(pattern, correction)
                    
                    # If we found a correction to apply
                    if corrected_part:
                        # Make sure we replace the exact text
                        corrected_text = corrected_text.replace(error_text, corrected_part)
        
            # Apply a/an corrections after all other corrections
            # This helps ensure we have the right articles after other replacements
            words = corrected_text.split()
            for i in range(len(words) - 1):
                if words[i].lower() == 'a' and words[i+1] and words[i+1][0].lower() in 'aeiou':
                    words[i] = 'an'
                elif words[i].lower() == 'an' and words[i+1] and words[i+1][0].lower() not in 'aeiou':
                    words[i] = 'a'
            
            corrected_text = ' '.join(words)
                
        except Exception as e:
            print(f"Error generating corrections: {e}")
        
        return corrected_text
    
    def is_plural_subject(self, subject):
        """Helper function to determine if a subject is plural"""
        subject = subject.lower()
        # Directly plural subjects
        if subject in ['we', 'they', 'you', 'these', 'those', 'i']:
            return True
        # Singular subjects
        if subject in ['he', 'she', 'it', 'this', 'that']:
            return False
        # Check for irregular plurals
        if subject in self.irregular_plurals:
            return True
        # Check for -s ending as default heuristic
        if subject.endswith('s') and not subject.endswith('ss'):
            return True
        return False
    
    def _calculate_grammar_match(self, doc, errors):
        """Calculate a grammar match score from 0-100"""
        # Base score starts at 100, deduct for each error
        base_score = 100
        
        # Deduct points based on error type and frequency
        error_types = Counter([error['type'] for error in errors])
        
        # More serious errors have higher deductions
        deductions = {
            'Subject-verb agreement': 15,
            'Verb form error': 15,
            'Article error': 10,
            'Preposition error': 10,
            'Sentence fragment': 20,
            'Double negation': 15,
            'Gerund/Infinitive error': 12,
            'Conditional error': 15,
            'Phrasal verb error': 10,
            'Modal verb error': 12,
            'Word usage error': 8,
            'Irregular verb error': 12,
            'Article with noun error': 10,
            'Missing article/determiner': 8,
            'Word repetition': 5
        }
        
        # Calculate total deduction
        total_deduction = 0
        for error_type, count in error_types.items():
            # Apply diminishing returns for multiple errors of the same type
            deduction_value = deductions.get(error_type, 10)
            total_deduction += deduction_value * min(count, 3) * (0.8 if count > 3 else 1.0)
        
        # Ensure score doesn't go below 0
        grammar_match = max(0, base_score - total_deduction)
        
        return grammar_match
    
    def _calculate_error_frequency(self, errors, word_count):
        """Calculate error frequency as a percentage (0-100)"""
        if word_count == 0:
            return 0
            
        error_count = len(errors)
        error_ratio = min(1.0, error_count / max(1, word_count / 5))  # Expect roughly 1 error per 5 words at most
        
        # Convert to 0-100 scale
        return error_ratio * 100
    
    def _calculate_complexity(self, doc):
        """Calculate sentence complexity on a scale of 0-100"""
        # Count various complexity indicators
        try:
            avg_token_length = sum(len(token.text) for token in doc) / max(1, len(doc))
            avg_sentence_length = sum(len(sent) for sent in doc.sents) / max(1, len(list(doc.sents)))
            
            # Count subordinate clauses (simplified approximation)
            subordinate_markers = ['although', 'though', 'because', 'since', 'when', 'while', 'if', 'unless']
            subordinate_count = sum(1 for token in doc if token.text.lower() in subordinate_markers)
            
            # Normalize and combine factors (with simpler calculation)
            norm_token_length = min(1.0, avg_token_length / 8.0) * 20
            norm_sent_length = min(1.0, avg_sentence_length / 25.0) * 40
            norm_subordinate = min(1.0, subordinate_count / 5.0) * 40
            
            complexity = norm_token_length + norm_sent_length + norm_subordinate
            
            return min(100, complexity)
        except Exception as e:
            print(f"Error calculating complexity: {e}")
            return 50  # Return medium complexity on error 

    def detect_subject_number(self, doc):
        """Detect if the subject is singular or plural"""
        subject_is_plural = False
        subject_text = ""
        subject_pos = -1
        
        # First look for the subject NP
        for i, token in enumerate(doc):
            # Main subject is often the first noun phrase before a verb
            if token.pos_ == "NOUN" or token.pos_ == "PROPN" or token.pos_ == "PRON":
                subject_text = token.text
                subject_pos = i
                
                # Check if it's a conjunction phrase like "John and Mary"
                if i + 2 < len(doc) and doc[i+1].text.lower() == "and":
                    subject_is_plural = True
                    subject_text = f"{token.text} and {doc[i+2].text}"
                    
                # Check if it's a known plural pronoun
                if token.text.lower() in ["we", "they", "you", "these", "those"]:
                    subject_is_plural = True
                elif token.text.lower() in ["i", "he", "she", "it", "this", "that"]:
                    subject_is_plural = False
                # Check for plural nouns ending with -s (simplified)
                elif token.pos_ == "NOUN" and token.tag_ == "NNS":
                    subject_is_plural = True
                # Check irregular plurals
                elif token.text.lower() in self.irregular_plurals:
                    subject_is_plural = True
                break
        
        return subject_is_plural, subject_text, subject_pos

    def detect_intended_tense(self, doc, text):
        """Detect the intended tense based on the text structure"""
        
        # Look for time markers
        time_markers = {
            'Simple Present': ['always', 'usually', 'regularly', 'often', 'every day'],
            'Simple Past': ['yesterday', 'last week', 'ago', 'in 2020'],
            'Present Continuous': ['now', 'right now', 'at the moment', 'currently'],
            'Present Perfect': ['already', 'yet', 'just', 'ever', 'never', 'since', 'for'],
            'Past Continuous': ['while', 'when', 'as'],
            'Future Simple': ['tomorrow', 'next week', 'later'],
        }
        
        # Check for auxiliary verbs and their forms
        aux_patterns = {
            'Present Perfect': [('have', 'VB'), ('has', 'VB')],
            'Past Perfect': [('had', 'VB')],
            'Present Continuous': [('am', 'VBG'), ('is', 'VBG'), ('are', 'VBG')],
            'Past Continuous': [('was', 'VBG'), ('were', 'VBG')],
            'Future Simple': [('will', 'VB'), ('shall', 'VB')],
        }
        
        # Check time markers
        for tense, markers in time_markers.items():
            for marker in markers:
                if marker in text.lower():
                    return tense
        
        # Check verb patterns
        for tense, patterns in aux_patterns.items():
            for aux, verb_tag in patterns:
                if any(token.text.lower() == aux and i+1 < len(doc) and doc[i+1].tag_ == verb_tag 
                       for i, token in enumerate(doc)):
                    return tense
        
        # Default to Simple Present if we can't determine
        return 'Simple Present'

    def analyze_with_tense_suggestion(self, text):
        """Analyze text with tense detection and suggestions"""
        doc = self.nlp(text)
        
        # 1. Detect subject number (singular/plural)
        is_plural, subject, subject_pos = self.detect_subject_number(doc)
        
        # 2. Detect intended tense
        intended_tense = self.detect_intended_tense(doc, text)
        
        # 3. Run normal analysis
        analysis_result = self.analyze(text, intended_tense)
        
        # 4. Add tense suggestion to results
        analysis_result['subject_info'] = {
            'text': subject,
            'is_plural': is_plural
        }
        analysis_result['suggested_tense'] = intended_tense
        
        return analysis_result 

    def _get_sv_agreement_correction(self, error_text):
        """Provide a correction for subject-verb agreement errors"""
        parts = error_text.lower().split()
        if len(parts) >= 2:
            subject = parts[0]
            verb = parts[1]
            
            # For "I/you/we/they has" -> "I/you/we/they have"
            if subject in ["i", "you", "we", "they"] and verb == "has":
                return f"{subject} have"
                
            # For "he/she/it have" -> "he/she/it has"
            if subject in ["he", "she", "it"] and verb == "have":
                return f"{subject} has"
                
            # For "I is" -> "I am"
            if subject == "i" and verb == "is":
                return "I am"
                
            # For "you/we/they is" -> "you/we/they are"
            if subject in ["you", "we", "they"] and verb == "is":
                return f"{subject} are"
                
            # For "I/he/she/it are" -> "I am" or "he/she/it is"
            if subject == "i" and verb == "are":
                return "I am"
            elif subject in ["he", "she", "it"] and verb == "are":
                return f"{subject} is"
        
        return None 

    def _check_contraction_errors(self, subject_info, doc):
        """Check for errors with contractions like don't/doesn't"""
        errors = []
        
        try:
            # Extract necessary info with proper error checking
            if 'is_plural' not in subject_info or 'text' not in subject_info or 'position' not in subject_info:
                return errors
                
            is_plural = subject_info['is_plural']
            subject_text = subject_info['text']
            subject_position = subject_info['position']
            
            if subject_position >= len(doc) or subject_text == "":
                return errors
            
            # Look for contractions in the entire sentence, not just after subject
            # This is more robust than relying on exact positioning
            for token in doc:
                if token.text.lower() == "don't":
                    # Check if this token is associated with our subject
                    if token.i > subject_position and not is_plural:
                        # For singular subjects like "he, she, it" - should use "doesn't"
                        if not any(subj.lower() in subject_text.lower() for subj in ["i", "you", "we", "they"]):
                            errors.append({
                                'type': 'Contraction error',
                                'text': f"{subject_text} don't",
                                'suggestion': f"Use 'doesn't' with singular subjects: '{subject_text} doesn't'"
                            })
                
                elif token.text.lower() == "doesn't":
                    # Check if this token is associated with our subject
                    if token.i > subject_position and (is_plural or any(subj.lower() in subject_text.lower() for subj in ["i", "you", "we", "they"])):
                        errors.append({
                            'type': 'Contraction error',
                            'text': f"{subject_text} doesn't",
                            'suggestion': f"Use 'don't' with '{subject_text}'"
                        })
                
                # Similar checks for other contractions
                if token.text.lower() == "haven't":
                    if token.i > subject_position and not is_plural and not any(subj.lower() in subject_text.lower() for subj in ["i", "you", "we", "they"]):
                        errors.append({
                            'type': 'Contraction error',
                            'text': f"{subject_text} haven't",
                            'suggestion': f"Use 'hasn't' with singular subjects: '{subject_text} hasn't'"
                        })
                
                elif token.text.lower() == "hasn't":
                    if token.i > subject_position and (is_plural or any(subj.lower() in subject_text.lower() for subj in ["i", "you", "we", "they"])):
                        errors.append({
                            'type': 'Contraction error',
                            'text': f"{subject_text} hasn't",
                            'suggestion': f"Use 'haven't' with '{subject_text}'"
                        })
                
                # Check for "isn't" vs "aren't" errors
                if token.text.lower() == "isn't":
                    if token.i > subject_position and (is_plural or any(subj.lower() in subject_text.lower() for subj in ["you", "we", "they"])):
                        errors.append({
                            'type': 'Contraction error',
                            'text': f"{subject_text} isn't",
                            'suggestion': f"Use 'aren't' with '{subject_text}'"
                        })
                
                elif token.text.lower() == "aren't":
                    if token.i > subject_position and any(subj.lower() in subject_text.lower() for subj in ["he", "she", "it"]):
                        errors.append({
                            'type': 'Contraction error',
                            'text': f"{subject_text} aren't",
                            'suggestion': f"Use 'isn't' with '{subject_text}'"
                        })
                
                # Special case for "I'm not" vs "I am not"
                if token.text.lower() == "amn't" and subject_text.lower() == "i":
                    errors.append({
                        'type': 'Contraction error',
                        'text': "I amn't",
                        'suggestion': "Use 'I'm not' or 'I am not' instead"
                    })
        
        except Exception as e:
            print(f"Error in contraction check: {e}")
            # Don't raise exception, just return empty errors
            return []
            
        return errors

    def _check_sv_agreement_simple_present(self, subject_info, verb, doc):
        """
        Check for subject-verb agreement errors specifically in Simple Present tense
        
        Args:
            subject_info: Dictionary with subject information
            verb: The verb token associated with the subject
            doc: The full spaCy document
            
        Returns:
            tuple: (has_error, correct_form)
        """
        # Skip auxiliaries and modals
        if verb.pos_ == "AUX" or verb.tag_ == "MD":
            return False, None
        
        # Only work with present tense base form verbs
        if verb.tag_ not in ["VB", "VBP", "VBZ"]:
            return False, None
            
        verb_text = verb.text.lower()
        is_plural = subject_info['is_plural']
        subject_text = subject_info['text'].lower()
        
        # Special case for the verb "be"
        if verb_text in ["am", "is", "are"]:
            # First person singular - I am
            if subject_text == "i" and verb_text != "am":
                return True, "am"
            # Third person singular - he/she/it is
            elif not is_plural and subject_text not in ["i", "you", "we", "they"] and verb_text != "is":
                return True, "is"
            # Plural and second person - they/you/we are
            elif (is_plural or subject_text in ["you", "we", "they"]) and verb_text != "are":
                return True, "are"
            return False, None
                
        # Handle "have"/"has"
        if verb_text in ["have", "has"]:
            # "I/we/you/they" should use "have"
            if (is_plural or subject_text in ["i", "you", "we", "they"]) and verb_text == "has":
                return True, "have"
            # Third person singular (except I/you/we/they) should use "has"
            elif not is_plural and subject_text not in ["i", "you", "we", "they"] and verb_text == "have":
                return True, "has"
            return False, None
            
        # Special case for first person (I) and plural subjects - never use -s/-es ending
        if (subject_text == "i" or is_plural or subject_text in ["you", "we", "they"]) and verb_text.endswith('s'):
            # Remove -s or -es ending
            if verb_text.endswith('ies') and len(verb_text) > 3:
                # For words ending in consonant + y -> ies (like "fly" -> "flies")
                return True, verb_text[:-3] + 'y'
            elif verb_text.endswith('es') and any(verb_text.endswith(x+'es') for x in ['sh', 'ch', 'x', 'ss', 'zz', 'o']):
                # For special -es endings like watches, rushes, boxes, passes
                return True, verb_text[:-2]
            elif verb_text.endswith('ves') and len(verb_text) > 3:
                # For words like "leaves" -> "leave"
                return True, verb_text[:-1]
            else:
                # For regular -s endings (likes -> like)
                return True, verb_text[:-1]
        
        # Regular verbs - third person singular should end in -s or -es
        if not is_plural and subject_text not in ["i", "you", "we", "they"]:
            # If verb doesn't end with s, it's an error for singular subjects
            if not verb_text.endswith('s'):
                # Add correct -s form
                if verb_text.endswith('o') or verb_text.endswith('ch') or \
                   verb_text.endswith('sh') or verb_text.endswith('ss') or \
                   verb_text.endswith('x') or verb_text.endswith('z'):
                    return True, verb_text + 'es'
                elif verb_text.endswith('y') and verb_text[-2] not in 'aeiou':
                    # Change y to ies for consonant + y
                    return True, verb_text[:-1] + 'ies'
                else:
                    return True, verb_text + 's'
        else:
            # Plural subjects shouldn't have -s ending verbs (except be/have)
            if verb_text.endswith('s') and not verb_text in ["is", "was"]:
                # Remove -s or -es ending for plural subjects
                if verb_text.endswith('ies') and len(verb_text) > 3:
                    return True, verb_text[:-3] + 'y'
                elif verb_text.endswith('es') and any(verb_text.endswith(x+'es') for x in ['sh', 'ch', 'x', 'ss', 'zz', 'o']):
                    return True, verb_text[:-2]
                else:
                    return True, verb_text[:-1]