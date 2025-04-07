#!/usr/bin/env python3
"""Script to generate a compact ML training notebook for Symptom Analyzer"""

import nbformat as nbf
import os

# Create a new notebook
nb = nbf.v4.new_notebook()

# Title and introduction
title_cell = nbf.v4.new_markdown_cell("""# Symptom Analyzer: ML Training
A compact ML system for disease prediction from symptoms.

This notebook demonstrates:
1. Loading and preprocessing symptom data
2. Training ML models (Random Forest)
3. Model evaluation and prediction""")

# Setup and imports
setup_cell = nbf.v4.new_code_cell("""# Essential imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

# Create directories
os.makedirs('models', exist_ok=True)

# Text preprocessing
def preprocess_text(text):
    '''Clean text by removing special characters and converting to lowercase'''
    return ' '.join(text.lower().split())

# Load and preprocess data
print("Loading dataset...")
df = pd.read_csv('data/symptom_disease.csv')
df['Processed_Symptoms'] = df['Symptoms'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(df['Processed_Symptoms'])
le = LabelEncoder()
y = le.fit_transform(df['Disease'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save models
joblib.dump(vectorizer, 'models/vectorizer.pkl')
joblib.dump(rf_model, 'models/random_forest_model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
print("\nModels saved successfully!")""")

# Prediction function
prediction_cell = nbf.v4.new_code_cell("""# Prediction function
def predict_disease(symptoms):
    '''Predict disease based on symptoms'''
    # Process and predict
    X = vectorizer.transform([preprocess_text(symptoms)])
    proba = rf_model.predict_proba(X)[0]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(proba)[-3:][::-1]
    results = [(le.inverse_transform([i])[0], proba[i]) for i in top_3_idx]
    
    # Display results
    print(f"Analyzing: '{symptoms}'")
    print("\nTop 3 Predictions:")
    for disease, prob in results:
        print(f"  - {disease}: {prob:.2%}")
    
    return results[0][0]

# Test prediction
print("Testing prediction:")
predict_disease("fever headache cough")""")

# Add cells to notebook
nb.cells.extend([title_cell, setup_cell, prediction_cell])

# Write the notebook
with open('Symptom_Analyzer.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Compact ML training notebook 'Symptom_Analyzer.ipynb' created successfully!") 