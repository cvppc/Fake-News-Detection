// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const newsTextArea = document.getElementById('news-text');
    const predictBtn = document.getElementById('predict-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultsContainer = document.getElementById('results-container');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('result-content');
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');

    // Verdict Elements
    const predictionElement = document.getElementById('prediction');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceLevel = document.getElementById('confidence-level');
    const verdictIconElement = document.getElementById('verdict-icon-element');
    const rawScoresElement = document.getElementById('raw-scores'); // For displaying raw scores

    // LIME Elements
    const limePlot = document.getElementById('lime-plot');
    const limeHighlightedText = document.getElementById('lime-highlighted-text');
    const limeFeatureList = document.getElementById('lime-feature-list');
    const limeTabButtons = document.querySelectorAll('#lime-explanation-container .tab-btn');
    const limeTabPanes = document.querySelectorAll('#lime-explanation-container .tab-pane');

    // SHAP Elements (New)
    const shapPlotContainer = document.getElementById('shap-plot-container');
    const shapExplanationContainer = document.getElementById('shap-explanation-container'); // To show/hide the whole section

    // Check if predict button should be initially disabled (based on template variable)
    // Note: This check won't work perfectly with external JS as it relies on Jinja templating
    // The button disabling logic in the HTML template itself is more reliable
    if (predictBtn && predictBtn.disabled) {
         console.warn("Model not loaded, analysis button is disabled.");
         // Optionally show a message near the button or text area
    }


    // --- Event Listeners ---
    // Ensure elements exist before adding listeners
    if (predictBtn) {
        predictBtn.addEventListener('click', analyzeText);
    }
    if (resetBtn) {
        resetBtn.addEventListener('click', resetForm);
    }

    // LIME Tab Navigation
    limeTabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Handle only LIME tabs
            limeTabButtons.forEach(btn => btn.classList.remove('active'));
            limeTabPanes.forEach(pane => pane.classList.remove('active'));

            button.classList.add('active');
            const tabId = button.getAttribute('data-tab');
            const targetPane = document.getElementById(`${tabId}-tab`);
            if (targetPane) {
                 targetPane.classList.add('active');
            } else {
                 console.error("Target pane not found for tab:", tabId);
            }
        });
    });

    // --- Core Functions ---

    function analyzeText() {
        // Guard against disabled button click
        if (predictBtn.disabled) {
            showError('Analysis is disabled because the backend model is not available.');
            return;
        }

        const text = newsTextArea.value.trim();

        if (!text) {
            showError('Please enter some text to analyze.');
            // Optionally add a visual cue to the textarea
            if (newsTextArea) {
                newsTextArea.style.borderColor = 'var(--danger-color)';
                setTimeout(() => { newsTextArea.style.borderColor = ''; }, 2000);
            }
            return;
        }

        // --- UI Update: Start Analysis ---
        if (resultsContainer) resultsContainer.classList.remove('hidden');
        if (loader) loader.classList.remove('hidden');
        if (resultContent) resultContent.classList.add('hidden');
        if (errorMessage) errorMessage.classList.add('hidden');
        if (predictBtn) {
            predictBtn.disabled = true; // Disable button during analysis
            predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...'; // Change button text
        }

        // Scroll to results smoothly
        if (resultsContainer) {
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        // --- API Call ---
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => {
            if (!response.ok) {
                // Handle HTTP errors (e.g., 500 Internal Server Error)
                return response.json().then(errData => {
                   // Use errData.error if available, otherwise construct message
                   throw new Error(errData?.error || `Server responded with status ${response.status}`);
                }).catch(() => {
                    // If parsing JSON fails after non-ok response, throw generic status error
                    throw new Error(`Server responded with status ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            // --- UI Update: Hide Loader ---
            if (loader) loader.classList.add('hidden');

            if (data.error) {
                // Handle errors reported in the JSON payload
                showError(data.error);
            } else {
                // --- UI Update: Display Results ---
                displayResults(data);
                if (resultContent) resultContent.classList.remove('hidden'); // Show results content
                // Make sure error message is hidden if previous run had error
                if (errorMessage) errorMessage.classList.add('hidden');
            }
        })
        .catch(error => {
            // Handle network errors or errors thrown from response check
            console.error('Fetch Error:', error);
            if (loader) loader.classList.add('hidden');
            // Provide a user-friendly error message
            showError(`Analysis failed: ${error.message || 'Check network connection or server status.'}`);
        })
        .finally(() => {
             // --- UI Update: Re-enable Button ---
            if (predictBtn) {
                predictBtn.disabled = false; // Re-enable button
                predictBtn.innerHTML = '<i class="fas fa-search"></i> Analyze'; // Restore original button text
            }
        });
    }

    function displayResults(data) {
        // --- 1. Update Verdict Section ---
        const prediction = data.prediction?.toUpperCase() || 'N/A'; // Ensure consistent case, handle missing
        const confidencePercent = typeof data.confidence === 'number' ? (data.confidence * 100).toFixed(1) : 'N/A';
        const isFake = prediction === 'FAKE';

        if (predictionElement) {
            predictionElement.textContent = prediction;
            predictionElement.className = `prediction-label ${prediction.toLowerCase()}`;
        }
        if (confidenceValue) confidenceValue.textContent = confidencePercent;
        if (confidenceLevel) {
            confidenceLevel.style.width = prediction !== 'N/A' ? `${confidencePercent}%` : '0%';
            confidenceLevel.className = `confidence-level ${prediction.toLowerCase()}`;
        }
        if (verdictIconElement) {
             verdictIconElement.className = `fas ${isFake ? 'fa-times-circle' : 'fa-check-circle'}`;
             if (verdictIconElement.parentElement) {
                verdictIconElement.parentElement.className = `verdict-icon ${prediction.toLowerCase()}`;
             }
        }

         // Display raw scores (optional)
         if (rawScoresElement) {
             if (data.scores) {
                let scoreText = Object.entries(data.scores)
                                    .map(([label, score]) => `${label}: ${score.toFixed(3)}`)
                                    .join(' | ');
                rawScoresElement.textContent = `Raw Scores: ${scoreText}`;
             } else {
                rawScoresElement.textContent = '';
             }
         }


        // --- 2. Update LIME Section ---
        // LIME Plot
        if (limePlot){
            limePlot.src = data.lime_plot ? `data:image/png;base64,${data.lime_plot}` : '';
            limePlot.alt = `LIME Explanation Plot for ${prediction}`;
        }

        // LIME Highlighted Text
        if (limeHighlightedText){
            limeHighlightedText.innerHTML = data.lime_highlighted_text || '<p style="color: var(--text-secondary);">Could not generate highlighted text.</p>';
        }

        // LIME Feature List
        if (limeFeatureList){
            limeFeatureList.innerHTML = ''; // Clear previous list
            if (data.lime_feature_weights && data.lime_feature_weights.length > 0) {
                data.lime_feature_weights.forEach(feature => {
                    const featureItem = document.createElement('div');
                    // Add the class 'positive' or 'negative' to the item itself for border styling
                    featureItem.className = `feature-item ${feature.class || 'neutral'}`; // Add default class
                    featureItem.innerHTML = `
                        <span class="feature-word" title="${feature.word || ''}">${feature.word || ''}</span>
                        <span class="feature-weight ${feature.class || 'neutral'}" title="Weight: ${typeof feature.weight === 'number' ? feature.weight.toFixed(4) : ''}">
                            ${typeof feature.weight === 'number' ? feature.weight.toFixed(3) : '?'}
                        </span>
                    `;
                    limeFeatureList.appendChild(featureItem);
                });
            } else {
                limeFeatureList.innerHTML = '<p style="color: var(--text-secondary);">No significant features identified by LIME.</p>';
            }
        }

        // --- 3. Update SHAP Section (New) ---
 
        if (shapPlotContainer && shapExplanationContainer) {
            if (data.shap_plot_png) {
                shapPlotContainer.innerHTML = `<img src="data:image/png;base64,${data.shap_plot_png}" alt="SHAP Force Plot" style="max-width: 100%; height: auto;">`;
                shapExplanationContainer.classList.remove('hidden');
            } else {
                shapPlotContainer.innerHTML = '<p style="color: var(--text-secondary);">Could not generate SHAP plot.</p>';
            }
        }
        // --- 4. Reset to Default Tab (LIME Text Highlight) ---
        limeTabButtons.forEach(btn => btn.classList.remove('active'));
        limeTabPanes.forEach(pane => pane.classList.remove('active'));
        const defaultLimeTab = document.querySelector('#lime-explanation-container .tab-btn[data-tab="lime-text"]');
        const defaultLimePane = document.getElementById('lime-text-tab');
        if (defaultLimeTab && defaultLimePane) {
            defaultLimeTab.classList.add('active');
            defaultLimePane.classList.add('active');
        }

        // Ensure results are visible
        if (resultContent) resultContent.classList.remove('hidden');
        if (errorMessage) errorMessage.classList.add('hidden'); // Hide any previous error

    }

    function showError(message) {
        // Make sure results container is visible to show the error message inside it
        if (resultsContainer) resultsContainer.classList.remove('hidden');
        if (loader) loader.classList.add('hidden'); // Ensure loader is hidden
        if (resultContent) resultContent.classList.add('hidden'); // Hide normal results content
        if (errorMessage) errorMessage.classList.remove('hidden'); // Show error message block
        if (errorText) errorText.textContent = message || 'An unknown error occurred.'; // Set the error text
    }

    function resetForm() {
        if (newsTextArea) newsTextArea.value = ''; // Clear textarea
        if (resultsContainer) resultsContainer.classList.add('hidden'); // Hide results section
        // Clear previous results visually (optional but good practice)
        if (limeHighlightedText) limeHighlightedText.innerHTML = '';
        if (limeFeatureList) limeFeatureList.innerHTML = '';
        if (limePlot) limePlot.src = '';
        if (shapPlotContainer) shapPlotContainer.innerHTML = '<p style="padding: 20px; text-align: center; color: var(--text-secondary);">SHAP plot will appear here...</p>'; // Reset SHAP placeholder
        if (errorMessage) errorMessage.classList.add('hidden'); // Hide any error message
        if (resultContent) resultContent.classList.add('hidden'); // Hide result content block

        if (newsTextArea) newsTextArea.focus(); // Set focus back to textarea
    }

    // --- Placeholder Cycling ---
    const sampleTexts = [
        "Scientists discover new renewable energy source that could revolutionize power generation using quantum entanglement.",
        "Breaking: Documents reveal famous actor secretly funded anonymous online troll farms to manipulate public opinion during election.",
        "Controversial study claims daily consumption of processed cheese drastically improves cognitive function in adults over 50.",
        "Government report confirms unexpected surge in exports leads to record economic growth, defying earlier pessimistic forecasts.",
        "Investigation uncovers network of deepfake videos targeting political figures ahead of major summit."
    ];
    let currentPlaceholderIndex = 0;
    function cyclePlaceholder() {
        if (newsTextArea) {
            // Set placeholder with line breaks for better readability
            newsTextArea.setAttribute('placeholder', `Enter text...\n\nE.g., "${sampleTexts[currentPlaceholderIndex]}"`);
            currentPlaceholderIndex = (currentPlaceholderIndex + 1) % sampleTexts.length;
        }
    }

    // Only run placeholder logic if textarea exists
    if (newsTextArea) {
        cyclePlaceholder(); // Initial call
        setInterval(cyclePlaceholder, 8000); // Cycle every 8 seconds
    }

});