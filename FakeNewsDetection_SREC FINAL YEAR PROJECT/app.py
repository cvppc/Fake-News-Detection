# app.py - Combined LIME and SHAP Explanations
from flask import Flask, render_template, request, jsonify
from markupsafe import Markup 
import numpy as np
from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import shap
import base64
from io import BytesIO, StringIO
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for Matplotlib in Flask
import matplotlib.pyplot as plt
import html
import torch # Required by newer SHAP versions with transformers
import logging # For better error logging

# --- Flask App Setup ---
app = Flask(__name__)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading ---
model_name = "therealcyberlord/fake-news-classification-distilbert"
classifier = None
tokenizer = None
model_loaded = False
try:
    # Load pipeline (includes model and tokenizer)
    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name, # Pass tokenizer name explicitly if needed
        return_all_scores=True,
        # Use GPU if available, otherwise CPU
        # device=0 if torch.cuda.is_available() else -1
    )
    # Explicitly get tokenizer for SHAP
    tokenizer = classifier.tokenizer
    model_loaded = True
    logger.info(f"Model '{model_name}' loaded successfully on device: {classifier.device}")
    # Get label map directly from model config for consistency
    label_map = classifier.model.config.id2label
    # SHAP needs class names in the order defined by the model's predictor output
    # The predictor below ensures [score_for_label_0, score_for_label_1] order
    class_names = [label_map[i] for i in range(len(label_map))]
    logger.info(f"Detected class names: {class_names}") # Should be ['Fake', 'Real'] or ['Real', 'Fake'] depending on model
    # Find index for 'FAKE' and 'REAL' based on loaded map
    FAKE_INDEX = class_names.index('Fake') # Use 'Fake' as label name from model config
    REAL_INDEX = class_names.index('Real') # Use 'Real' as label name from model config
    logger.info(f"Index for 'Fake': {FAKE_INDEX}, Index for 'Real': {REAL_INDEX}")

except Exception as e:
    logger.error(f"FATAL: Error loading model '{model_name}': {e}", exc_info=True)
    # Optionally: exit() or provide a fallback mechanism

# --- LIME Predictor Function ---
# Takes a list of texts, returns numpy array of probabilities for each class [prob_class_0, prob_class_1, ...]
def lime_predictor(texts):
    if not model_loaded:
        raise RuntimeError("Model is not loaded.")
    try:
        with torch.no_grad(): # Disable gradient calculations for inference
             outputs = classifier(texts, truncation=True, padding=True)
        # Ensure scores are returned in the order defined by label_map [score_label_0, score_label_1]
        ordered_scores = []
        for sample_output in outputs:
            score_dict = {item['label']: item['score'] for item in sample_output}
            scores = [score_dict.get(label_map[i], 0.0) for i in range(len(label_map))]
            ordered_scores.append(scores)
        return np.array(ordered_scores)
    except Exception as e:
        logger.error(f"Error in LIME predictor: {e}", exc_info=True)
        # Return empty array with correct shape on error
        num_classes = len(label_map) if label_map else 2
        return np.empty((len(texts), num_classes))

# --- SHAP Predictor Function ---
# Similar to LIME's but handles potential SHAP masking artifacts
def shap_predictor(texts):
    if not model_loaded:
        raise RuntimeError("Model is not loaded.")

    # Input `texts` from SHAP can sometimes be NumPy array or contain None/masked tokens.
    if isinstance(texts, np.ndarray):
        input_texts = [str(item) if item is not None else "" for item in texts]
    elif isinstance(texts, (list, tuple)):
        input_texts = [str(item) if item is not None else "" for item in texts]
    elif isinstance(texts, str):
        input_texts = [texts]
    else:
        logger.warning(f"SHAP predictor received unexpected type: {type(texts)}. Converting to string.")
        try:
            input_texts = [str(texts)]
        except Exception:
            logger.error("Failed to convert SHAP input to string list.")
            num_classes = len(label_map) if label_map else 2
            return np.empty((0, num_classes))

    # Filter out empty strings that might result from masking
    valid_texts = [t for t in input_texts if isinstance(t, str) and len(t.strip()) > 0]

    if not valid_texts:
         num_classes = len(label_map) if label_map else 2
         return np.zeros((len(texts), num_classes)) # Return array of zeros matching input length

    try:
        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = classifier(valid_texts, truncation=True, padding=True, max_length=tokenizer.model_max_length)

        # --- Process outputs ---
        results_dict = {text: None for text in valid_texts}
        for i, sample_output in enumerate(outputs):
             score_dict = {item['label']: item['score'] for item in sample_output}
             # Ensure scores are returned in the order defined by label_map [score_label_0, score_label_1]
             scores = [score_dict.get(label_map[i], 0.0) for i in range(len(label_map))]
             results_dict[valid_texts[i]] = scores

        # Map results back to the original input order, handling empty strings
        final_scores = []
        num_classes = len(label_map) if label_map else 2
        idx_valid = 0
        for text in input_texts:
            if isinstance(text, str) and len(text.strip()) > 0:
                final_scores.append(results_dict[text])
                idx_valid += 1
            else:
                # Return neutral probability (0.5) for masked/empty inputs
                final_scores.append([0.5] * num_classes)

        return np.array(final_scores) # Shape: (num_input_texts, num_classes)

    except Exception as e:
        logger.error(f"Error during SHAP prediction: {e}", exc_info=True)
        num_classes = len(label_map) if label_map else 2
        return np.zeros((len(texts), num_classes)) # Return zeros on error

# --- LIME Explainer Setup ---
if model_loaded:
    lime_explainer = LimeTextExplainer(class_names=class_names)
else:
    lime_explainer = None
    logger.warning("LIME explainer not initialized because model failed to load.")

# --- SHAP Explainer Setup ---
shap_explainer = None
if model_loaded and tokenizer:
    try:
        # Use SHAP's text masker with the HF tokenizer
        shap_masker = shap.maskers.Text(tokenizer)
        # Initialize the SHAP explainer with the predictor and masker
        shap_explainer = shap.Explainer(shap_predictor, shap_masker, output_names=class_names)
        logger.info("SHAP explainer initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing SHAP explainer: {e}", exc_info=True)
else:
    logger.warning("SHAP explainer not initialized because model or tokenizer failed to load.")

# --- Helper: Create Custom Highlights ---
def create_custom_highlights(text, feature_list):
    """Create custom HTML highlights for LIME words."""
    highlighted_text_escaped = html.escape(text) # Escape original text first

    # Sort features by weight magnitude (descending) then length (descending) for better replacement
    # This helps prevent smaller words within larger ones from being replaced first incorrectly
    sorted_features = sorted(feature_list, key=lambda x: (abs(x[1]), len(x[0])), reverse=True)

    replacements = {} # Store planned replacements to avoid modifying string during iteration

    # Iterate through text to find positions of words and mark for replacement
    current_pos = 0
    temp_text = text.lower() # Use lower case for matching
    for word, weight in sorted_features:
        search_word = word.lower()
        start_index = temp_text.find(search_word, current_pos)
        while start_index != -1:
             # Very basic check: ensure it's likely a whole word (surrounded by space or punctuation)
             # This is imperfect but better than nothing. A regex approach would be more robust.
            is_word_boundary_start = start_index == 0 or not text[start_index-1].isalnum()
            is_word_boundary_end = (start_index + len(word)) == len(text) or not text[start_index + len(word)].isalnum()

            if is_word_boundary_start and is_word_boundary_end:
                 # Check if this segment is already marked for replacement by a larger word
                 already_marked = False
                 for r_start, r_end in replacements.keys():
                     if not (start_index + len(word) <= r_start or start_index >= r_end):
                         already_marked = True
                         break

                 if not already_marked:
                    original_word_segment = text[start_index : start_index + len(word)]
                    # Use FAKE_INDEX and REAL_INDEX to determine color correctly based on model output order
                    # Assuming positive weight means REAL (index REAL_INDEX) and negative means FAKE (index FAKE_INDEX)
                    # Check LIME docs if this assumption changes based on explain_instance parameters
                    if REAL_INDEX == 1 : # If Real is class 1 (common case)
                        color = "rgba(0, 255, 0, 0.4)" if weight > 0 else "rgba(255, 0, 0, 0.4)" # Green for Real, Red for Fake
                    else: # If Real is class 0
                         color = "rgba(255, 0, 0, 0.4)" if weight > 0 else "rgba(0, 255, 0, 0.4)" # Red for Fake, Green for Real

                    highlighted_word = f'<span style="background-color:{color}; padding: 1px 2px; border-radius: 3px;" title="LIME Weight: {weight:.3f}">{html.escape(original_word_segment)}</span>'
                    replacements[(start_index, start_index + len(word))] = highlighted_word
                    # Rough "mark as used" - advance position marker (imperfect)
                    # current_pos = start_index + len(word)
                    # Break inner loop after first valid match for this word instance to avoid multiple replacements? Or find all? Let's find all for now.

            # Continue search from next position
            start_index = temp_text.find(search_word, start_index + 1)

    # Apply replacements from end to start to avoid messing up indices
    sorted_indices = sorted(replacements.keys(), key=lambda x: x[0], reverse=True)
    final_html = html.escape(text) # Start with escaped text
    for start, end in sorted_indices:
        final_html = final_html[:start] + replacements[(start, end)] + final_html[end:]

    return f'<div style="white-space: pre-wrap; line-height: 1.8;">{final_html}</div>'


# --- Routes ---
@app.route('/')
def index():
    """Renders the main page."""
    logger.info("--- Entering index() route [Simplified] ---")
    shap_init_js = shap.initjs()
    # Render the template, passing model status and SHAP JS initialization
    return render_template('index.html',
                           model_loaded=model_loaded)  # Use

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the text analysis request."""
    if not model_loaded or not lime_explainer or not shap_explainer:
        logger.error("Prediction endpoint called but model/explainers are not ready.")
        return jsonify({'error': 'Server not ready: Model or explainers failed to load.'}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid request data.'}), 400

    text = data['text']
    if not text or len(text.strip()) == 0:
        return jsonify({'error': 'Please enter some text for analysis.'}), 400

    logger.info(f"Received text for analysis (length: {len(text)})")

    try:
        # 1. Classification
        logger.info("Performing classification...")
        with torch.no_grad():
            raw_result = classifier(text, truncation=True, padding=True)[0] # Get result for the first text
        logger.info(f"Raw classification result: {raw_result}")

        # Extract scores based on configured label names ('Fake', 'Real')
        score_dict = {item['label']: item['score'] for item in raw_result}
        fake_score = score_dict.get('Fake', 0.0)
        real_score = score_dict.get('Real', 0.0)

        # Determine prediction and confidence
        prediction = 'FAKE' if fake_score > real_score else 'REAL'
        confidence = fake_score if prediction == 'FAKE' else real_score
        predicted_class_index = FAKE_INDEX if prediction == 'FAKE' else REAL_INDEX
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}, Predicted Index: {predicted_class_index}")

        # 2. LIME Explanation
        logger.info("Generating LIME explanation...")
        lime_exp = lime_explainer.explain_instance(
            text,
            lime_predictor,
            num_features=15, # Increased features slightly
            num_samples=1500, # Slightly reduced samples for speed
            labels=(predicted_class_index,) # Explain only the predicted class
        )

        # LIME Feature Weights (for the predicted class)
        lime_feature_weights = []
        lime_raw_features = lime_exp.as_list(label=predicted_class_index)
        for feature, weight in lime_raw_features:
             # Determine class based on sign and which index means REAL
             is_positive_for_real = (REAL_INDEX == 1 and weight > 0) or (REAL_INDEX == 0 and weight < 0)
             lime_feature_weights.append({
                 'word': feature,
                 'weight': weight,
                 # 'positive' means contributes to REAL, 'negative' means contributes to FAKE
                 'class': 'positive' if is_positive_for_real else 'negative'
             })

        # LIME Plot (Matplotlib figure)
        logger.info("Generating LIME plot...")
        fig = lime_exp.as_pyplot_figure(label=predicted_class_index)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        lime_plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig) # Close the figure to free memory
        logger.info("LIME plot generated.")

        # LIME Highlighted Text
        logger.info("Generating LIME highlighted text...")
        # Use the custom highlighter which relies on the feature weights
        lime_highlighted_text = create_custom_highlights(text, lime_raw_features)
        logger.info("LIME highlighted text generated.")


        # 3. SHAP Explanation
        logger.info("Calculating SHAP values...")
        logger.info("Calculating SHAP values...")
        shap_values = shap_explainer([text])
        logger.info("SHAP values calculated.")

        logger.info("Generating SHAP force plot as static image...")
        try:
            shap_vals_for_class = shap_values[0].values[:, predicted_class_index]  # SHAP values for the predicted class
            shap_features = shap_values[0].feature_names  # Feature names (tokens)
            logger.info(f"SHAP values shape for plot: {shap_vals_for_class.shape}")
            logger.info(f"SHAP features length for plot: {len(shap_features)}")

    # Check for length mismatch
            if len(shap_features) != len(shap_vals_for_class):
                logger.error(f"SHAP length mismatch DETECTED! Features: {len(shap_features)}, Values: {len(shap_vals_for_class)}.")
                shap_plot_data = ""
            else:
                plt.figure(figsize=(10, 4))
                shap.force_plot(
                    base_value=shap_values[0].base_values[predicted_class_index],
                    shap_values=shap_vals_for_class,
                    features=shap_features,
                    out_names=class_names[predicted_class_index],
                    link="logit",
                    matplotlib=True
                 )
                plt.tight_layout()

        # Save the plot to a BytesIO buffer as PNG
                buffer = BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                shap_plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                logger.info("SHAP force plot PNG generated.")
        except Exception as e:
            logger.error(f"Error generating SHAP plot: {e}", exc_info=True)
            shap_plot_data = ""


         
# --- Remove or comment out the old save_html location if it exists outside the else ---
# (The previous code snippet might have left these lines outside, ensure they are removed or commented)
# # Save SHAP plot to an HTML string
# shap_html_buffer = StringIO()
# shap.save_html(shap_html_buffer, force_plot) # <-- REMOVE/COMMENT THIS if outside else
# shap_plot_html = shap_html_buffer.getvalue()

        # 4. Prepare Response
        logger.info("Preparing JSON response...")
        response_data = {
            'prediction': prediction,
            'confidence': float(confidence),
            # LIME Results
            'lime_feature_weights': lime_feature_weights,
            'lime_plot': lime_plot_data,
            'lime_highlighted_text': lime_highlighted_text,
            # SHAP Results
            'shap_plot_png': shap_plot_data,
            # Add raw scores if needed for debugging/display
            'scores': {label: score_dict.get(label, 0.0) for label in class_names}
        }
        logger.info("Analysis complete. Sending response.")
        return jsonify(response_data)



    except Exception as e:
        logger.error(f"Error during prediction processing: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred during analysis: {str(e)}'}), 500

# --- Main Execution ---
if __name__ == '__main__':
    if not model_loaded:
        print("\n" + "="*50)
        print("WARNING: Model failed to load. Running Flask app in limited mode.")
        print("Prediction endpoint will return errors.")
        print("Please check model name, network connection, and dependencies.")
        print("="*50 + "\n")
    app.run(debug=True,use_reloader=False) # debug=True enables auto-reloading and detailed errors