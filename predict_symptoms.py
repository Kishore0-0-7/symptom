#!/usr/bin/env python3
"""
Symptom Analyzer - Disease Prediction Tool
Uses trained models to predict disease from symptoms
"""

import os
import re
import sys
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Check if models exist
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    print("ERROR: Models directory not found.")
    print("Please run predict_disease.py first to train models.")
    sys.exit(1)

# Define the same stopwords set used in training
STOPWORDS = {
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through',
    'during', 'before', 'after', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off'
}

def preprocess_text(text):
    """Process text by removing stopwords and punctuation"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in STOPWORDS]
    return ' '.join(tokens)

def load_models():
    """Load trained models from disk"""
    try:
        print("Loading models...")
        vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.pkl'))
        nb_model = joblib.load(os.path.join(MODEL_DIR, 'naive_bayes_model.pkl'))
        rf_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
        le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        
        # Check if deep learning model exists
        dl_model = None
        dl_model_path = os.path.join(MODEL_DIR, 'deep_learning_model')
        if os.path.exists(dl_model_path):
            try:
                from tensorflow import keras
                dl_model = keras.models.load_model(dl_model_path)
                print("Deep learning model loaded successfully.")
            except (ImportError, Exception) as e:
                print(f"Could not load deep learning model: {str(e)}")
        
        return vectorizer, nb_model, rf_model, dl_model, le
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        sys.exit(1)

def predict_disease(symptoms, vectorizer, models, le, verbose=True):
    """Predict disease based on symptoms"""
    if verbose:
        print(f"\nAnalyzing symptoms: '{symptoms}'")
    
    # Preprocess input
    processed = preprocess_text(symptoms)
    if verbose:
        print(f"Processed symptoms: '{processed}'")
    
    # Vectorize input
    X = vectorizer.transform([processed])
    
    # Make predictions with each model
    results = []
    confidence_levels = []
    
    # Naive Bayes prediction
    nb_proba = models['nb'].predict_proba(X)[0]
    nb_top_idx = np.argsort(nb_proba)[-1]
    nb_pred = le.inverse_transform([nb_top_idx])[0]
    nb_conf = nb_proba[nb_top_idx]
    results.append((nb_pred, nb_conf, "Naive Bayes"))
    
    # Random Forest prediction
    rf_proba = models['rf'].predict_proba(X)[0]
    rf_top_idx = np.argsort(rf_proba)[-1]
    rf_pred = le.inverse_transform([rf_top_idx])[0]
    rf_conf = rf_proba[rf_top_idx]
    results.append((rf_pred, rf_conf, "Random Forest"))
    
    # Deep Learning prediction (if available)
    if 'dl' in models and models['dl'] is not None:
        try:
            if hasattr(X, 'toarray'):
                X_dense = X.toarray()
            else:
                X_dense = X
            dl_proba = models['dl'].predict(X_dense, verbose=0)[0]
            dl_top_idx = np.argmax(dl_proba)
            dl_pred = le.inverse_transform([dl_top_idx])[0]
            dl_conf = dl_proba[dl_top_idx]
            results.append((dl_pred, dl_conf, "Deep Learning"))
        except Exception as e:
            if verbose:
                print(f"Deep learning prediction error: {str(e)}")
    
    # Determine final prediction (voting)
    predictions = [r[0] for r in results]
    final_pred = max(set(predictions), key=predictions.count)
    
    # Calculate average confidence for final prediction
    matching_confs = [r[1] for r in results if r[0] == final_pred]
    avg_conf = sum(matching_confs) / len(matching_confs) if matching_confs else 0
    
    # Determine confidence level
    if avg_conf >= 0.7:
        confidence_level = "High"
    elif avg_conf >= 0.4:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    if verbose:
        print("\nResults:")
        for disease, conf, model_name in results:
            conf_level = "High" if conf >= 0.7 else "Medium" if conf >= 0.4 else "Low"
            print(f"  {model_name}: {disease} ({conf:.2%}, {conf_level} confidence)")
        
        print(f"\nFINAL PREDICTION: {final_pred}")
        print(f"Confidence: {avg_conf:.2%} ({confidence_level})")
        
        # Provide additional information
        if confidence_level == "Low":
            print("\nNote: Low confidence prediction. Consider providing more detailed symptoms.")
    
    return {
        'disease': final_pred,
        'confidence': avg_conf,
        'confidence_level': confidence_level,
        'model_predictions': results
    }

def interactive_mode():
    """Run the prediction tool in interactive mode"""
    print("=" * 60)
    print("SYMPTOM ANALYZER - DISEASE PREDICTION TOOL")
    print("=" * 60)
    
    # Load models
    vectorizer, nb_model, rf_model, dl_model, le = load_models()
    
    # Create models dictionary
    models = {
        'nb': nb_model,
        'rf': rf_model
    }
    if dl_model is not None:
        models['dl'] = dl_model
    
    print("\nReady to analyze symptoms!")
    print("Enter 'quit', 'exit', or 'q' to exit the program.")
    
    # Define example symptoms to help users
    examples = [
        "fever headache cough",
        "chest pain shortness of breath sweating",
        "stiff neck fever headache light sensitivity",
        "abdominal pain diarrhea nausea vomiting"
    ]
    
    print("\nExample inputs:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    
    # Main interaction loop
    while True:
        print("\nEnter symptoms (or type a number 1-4 to use an example):")
        user_input = input("> ").strip()
        
        # Check for exit commands
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Thank you for using the Symptom Analyzer!")
            break
        
        # Check if user entered a number to use an example
        if user_input.isdigit() and 1 <= int(user_input) <= len(examples):
            user_input = examples[int(user_input) - 1]
            print(f"Using example: '{user_input}'")
        
        # Make prediction
        if user_input:
            predict_disease(user_input, vectorizer, models, le)
        else:
            print("Please enter symptoms or select an example.")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Command line mode
        symptoms = ' '.join(sys.argv[1:])
        vectorizer, nb_model, rf_model, dl_model, le = load_models()
        models = {'nb': nb_model, 'rf': rf_model}
        if dl_model is not None:
            models['dl'] = dl_model
        predict_disease(symptoms, vectorizer, models, le)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main() 