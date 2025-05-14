// Main JavaScript for Fuzzy Grammar Tutor

// Global chart variable
let accuracyChart = null;

document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const grammarForm = document.getElementById('grammar-form');
    const textInput = document.getElementById('text-input');
    const tenseSelector = document.getElementById('tense-selector');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analyzeSpinner = document.getElementById('analyze-spinner');
    const exampleBtn = document.getElementById('example-btn');
    const resultsSection = document.getElementById('results-section');
    const invalidEnglishAlert = document.getElementById('invalid-english-alert');
    const invalidEnglishReason = document.getElementById('invalid-english-reason');
    const validResults = document.getElementById('valid-results');
    const severityBadge = document.getElementById('severity-badge');
    const overallFeedback = document.getElementById('overall-feedback');
    const correctedText = document.getElementById('corrected-text');
    const interactiveText = document.getElementById('interactive-text');
    const specificFeedback = document.getElementById('specific-feedback');
    const suggestionsList = document.getElementById('suggestions-list');
    const resourcesList = document.getElementById('resources-list');
    const accuracyChartElement = document.getElementById('accuracyChart');
    const accuracyPercentage = document.querySelector('.grammar-accuracy-percentage');

    // Initialize empty accuracy chart
    initAccuracyChart(0);

    // Example sentences with various grammar errors for different tenses
    const examples = {
        "Simple Present": "I has apple. She don't like coffee. They goes to school every day.",
        "Simple Past": "Yesterday I goed to the store. She did not went to the party.",
        "Present Continuous": "I am eat dinner right now. He are playing football.",
        "Present Perfect": "I has seen that movie already. She have never been to Paris.",
        "Future": "I will goes to the beach tomorrow. She will meeting him next week.",
        "general": "I has a apple. She don't have nothing to do yesterday. Although she was tired."
    };

    // Show a relevant example when clicking the example button
    exampleBtn.addEventListener('click', () => {
        // Get the selected tense or use general if none selected
        const selectedTense = tenseSelector.value || "general";
        // Use the example for the selected tense or fall back to general
        const example = examples[selectedTense] || examples["general"];
        textInput.value = example;
    });

    // Handle form submission
    grammarForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Get the input text and selected tense
        const text = textInput.value.trim();
        const tense = tenseSelector.value;
        
        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }
        
        // Show loading spinner
        analyzeBtn.disabled = true;
        analyzeSpinner.classList.remove('d-none');
        
        try {
            // Send the text to the server for analysis
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text, tense }),
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Display the results
            displayResults(data);
            
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis. Please try again.');
        } finally {
            // Hide loading spinner
            analyzeBtn.disabled = false;
            analyzeSpinner.classList.add('d-none');
        }
    });
    
    /**
     * Initialize the accuracy chart with a given value
     */
    function initAccuracyChart(accuracy) {
        // Destroy existing chart if it exists
        if (accuracyChart) {
            accuracyChart.destroy();
        }
        
        // Create new chart
        accuracyChart = new Chart(accuracyChartElement, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [accuracy, 100 - accuracy],
                    backgroundColor: [
                        getColorForAccuracy(accuracy),
                        'rgba(200, 200, 200, 0.2)'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                cutout: '70%',
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                },
                animation: {
                    animateRotate: true,
                    animateScale: true
                }
            }
        });
        
        // Update the percentage text
        if (accuracyPercentage) {
            accuracyPercentage.textContent = `${Math.round(accuracy)}%`;
        }
    }
    
    /**
     * Get color based on accuracy percentage
     */
    function getColorForAccuracy(accuracy) {
        if (accuracy >= 80) {
            return 'rgba(25, 135, 84, 0.8)'; // Green for high accuracy
        } else if (accuracy >= 50) {
            return 'rgba(255, 193, 7, 0.8)'; // Yellow for medium accuracy
        } else {
            return 'rgba(220, 53, 69, 0.8)'; // Red for low accuracy
        }
    }
    
    /**
     * Display the analysis results in the UI
     */
    function displayResults(data) {
        // Show results section
        resultsSection.classList.remove('d-none');
        
        // Check if the input was valid English
        if (!data.analysis.is_valid_english) {
            // Show invalid English alert and hide the valid results
            invalidEnglishAlert.classList.remove('d-none');
            invalidEnglishReason.textContent = data.analysis.reason;
            validResults.classList.add('d-none');
            return;
        }
        
        // Valid English - hide alert and show results
        invalidEnglishAlert.classList.add('d-none');
        validResults.classList.remove('d-none');
        
        // Get the data
        const analysis = data.analysis;
        const feedback = data.feedback;
        
        // Update accuracy chart with grammar match percentage
        initAccuracyChart(analysis.grammar_match);
        
        // Update overall feedback
        overallFeedback.textContent = feedback.overall_feedback;
        
        // Update corrected text
        correctedText.textContent = analysis.corrections || textInput.value;
        
        // Update interactive text with error highlighting
        interactiveText.innerHTML = createInteractiveText(textInput.value, analysis.errors);
        
        // Initialize tooltips for error highlights
        initializeTooltips();
        
        // Update specific feedback
        specificFeedback.innerHTML = '';
        if (feedback.specific_feedback.length > 0) {
            feedback.specific_feedback.forEach(item => {
                const itemClass = item.type.replace(/\s+/g, '-');
                specificFeedback.innerHTML += `
                    <div class="feedback-item ${itemClass}">
                        <h4 class="h6">${item.type}</h4>
                        <p>${item.feedback}</p>
                        <p><strong>Example:</strong> <span class="error-text">${item.example}</span></p>
                        <p><strong>Suggestion:</strong> ${item.suggestion}</p>
                    </div>
                `;
            });
        } else {
            specificFeedback.innerHTML = '<p>No specific grammar issues detected.</p>';
        }
        
        // Update suggestions
        suggestionsList.innerHTML = '';
        if (feedback.suggestions.length > 0) {
            feedback.suggestions.forEach(suggestion => {
                suggestionsList.innerHTML += `<li>${suggestion}</li>`;
            });
        } else {
            suggestionsList.innerHTML = '<li>No specific suggestions at this time.</li>';
        }
        
        // Update resources
        resourcesList.innerHTML = '';
        if (feedback.resources.length > 0) {
            feedback.resources.forEach(resource => {
                resourcesList.innerHTML += `
                    <li><a href="${resource.url}" target="_blank">${resource.type} Guide</a></li>
                `;
            });
        } else {
            resourcesList.innerHTML = '<li>No specific resources available.</li>';
        }
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    /**
     * Create interactive text with error highlighting and tooltips
     */
    function createInteractiveText(text, errors) {
        if (!errors || errors.length === 0) {
            return escapeHtml(text);
        }
        
        // Sort errors by position in text to process them in order
        const sortedErrors = [...errors].sort((a, b) => {
            return text.indexOf(a.text) - text.indexOf(b.text);
        });
        
        let result = text;
        
        // Replace errors with highlighted spans with tooltips
        // Process in reverse order to avoid changing positions as we modify the text
        for (let i = sortedErrors.length - 1; i >= 0; i--) {
            const error = sortedErrors[i];
            const errorText = error.text;
            const errorIndex = result.indexOf(errorText);
            
            if (errorIndex !== -1) {
                const before = result.substring(0, errorIndex);
                const after = result.substring(errorIndex + errorText.length);
                const highlightedError = `
                    <span class="error-highlight" data-error-type="${error.type}">
                        ${escapeHtml(errorText)}
                        <span class="grammar-tooltip">
                            <strong>${escapeHtml(error.type)}</strong>
                            <div class="tooltip-suggestion">${escapeHtml(error.suggestion)}</div>
                        </span>
                    </span>
                `;
                result = before + highlightedError + after;
            }
        }
        
        // Replace new lines with space for horizontal display
        return result;
    }
    
    /**
     * Escape HTML special characters to prevent XSS
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    /**
     * Initialize tooltips for error highlights using Bootstrap tooltips
     * This is an alternative approach if the CSS-only tooltips aren't working well
     */
    function initializeTooltips() {
        // Using native browser hover behavior for tooltips
        // If you prefer Bootstrap tooltips, uncomment below:
        /*
        const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]')
        const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl))
        */
    }
});