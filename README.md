# Symptom Analyzer: Disease Prediction System

A machine learning-based system for predicting diseases from symptoms. This project includes a training program to build machine learning models and a prediction tool for analyzing symptoms.

## Features

- **Model Training:** Train multiple ML models (Naive Bayes, Random Forest, Deep Learning) on symptom-disease data
- **Visualization:** Generate visualizations of disease distributions, model performance, and feature importance
- **Interactive Prediction:** Use trained models to predict diseases from symptoms through command-line or interactive mode
- **Confidence Levels:** Get prediction confidence ratings (High/Medium/Low) to assess reliability
- **Model Ensemble:** Combines predictions from multiple models for more robust results

## Requirements

- Python 3.6+
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- joblib
- TensorFlow (optional, for deep learning model)

## Getting Started

### Step 1: Train the models

```bash
python predict_disease.py
```

This will:
1. Create a sample dataset of symptoms and diseases
2. Preprocess the data and extract features
3. Train Naive Bayes and Random Forest models
4. Train a deep learning model (if TensorFlow is available)
5. Generate visualizations of model performance and feature importance
6. Save the trained models to the 'models/' directory

### Step 2: Use the prediction tool

#### Command-line mode:

```bash
python predict_symptoms.py "fever headache cough"
```

#### Interactive mode:

```bash
python predict_symptoms.py
```

Follow the prompts to enter symptoms or select from example inputs.

## Program Components

### predict_disease.py

The main training program that:
- Creates and processes a symptom-disease dataset
- Trains multiple machine learning models
- Evaluates model performance
- Visualizes results
- Saves trained models for later use

### predict_symptoms.py

The prediction tool that:
- Loads trained models
- Processes user-input symptoms
- Makes predictions using all available models
- Provides a final prediction with confidence level
- Supports both command-line and interactive modes

## Example Diseases and Symptoms

The system can predict several diseases based on symptoms:

- **Common Cold:** fever, headache, cough, runny nose, congestion
- **Heart Attack:** chest pain, shortness of breath, sweating, pain radiating to arm
- **Meningitis:** high fever, severe headache, stiff neck, light sensitivity
- **Tuberculosis:** fatigue, weight loss, night sweats, persistent cough
- **Gastroenteritis:** abdominal pain, diarrhea, nausea, vomiting
- **Diabetes:** frequent urination, excessive thirst, hunger, weight loss
- **Asthma:** wheezing, shortness of breath, chest tightness, coughing
- **Dengue:** fever, rash, joint pain, pain behind eyes
- **Mononucleosis:** fatigue, sore throat, swollen lymph nodes

## Model Performance

- **Naive Bayes:** Simple probabilistic classifier, typically ~85-90% accuracy
- **Random Forest:** Ensemble method, typically ~85-95% accuracy
- **Deep Learning:** Neural network approach, performance varies with dataset size

## Notes

- This is a simplified demonstration system and not a substitute for professional medical advice
- Prediction accuracy depends on the quality and size of the training dataset
- Consider adding more symptom-disease examples for improved accuracy
- The confidence level helps assess the reliability of predictions

## License

This project is available under the MIT License. 